"""Benchmark for BatchNormFwdOp and BatchNormBwdOp.

Compares TileOPs vs PyTorch cuDNN batch norm on common ResNet-style shapes. The
forward row adds flag_gems' batch_norm and cuDNN through inductor. The backward row
carries two torch tags: an autograd node driven on this thread, and aten's backward
kernel by itself. The difference between them is the forward the autograd one rebuilds.
"""

import pytest
import torch

from benchmarks.baselines import FLAGGEMS_TAG, TORCH_COMPILE_TAG, compiled_reference, flaggems_op
from benchmarks.benchmark_base import ManifestBenchmark, backward_of, manifest_calls
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.norm import BatchNormBwdCall, RunningStatsCall


def _flaggems_bn_fwd(running_mean, running_var, training: bool, momentum: float, eps: float):
    """flag_gems' batch_norm on its own running statistics, output only.

    Training mode updates them in place, and the cuDNN reference clones before it
    does, so this gets copies rather than the tensors the other tags read.
    """
    fn = flaggems_op("batch_norm")
    private_mean, private_var = running_mean.clone(), running_var.clone()

    def baseline_fn(x, _running_mean, _running_var, weight, bias):
        out = fn(x.float(), weight, bias, private_mean, private_var, training, momentum, eps)
        return out[0].to(x.dtype)

    return baseline_fn


def _torch_bn_bwd(grad_out, x, weight, mean, rstd):
    """PyTorch reference backward, driven on this thread. Recomputes the forward."""
    with torch.enable_grad():
        x32 = x.float().requires_grad_(True)
        w32 = weight.float().requires_grad_(True)
        b32 = torch.zeros(x.shape[1], device=x.device, dtype=torch.float32, requires_grad=True)
        rm = torch.zeros(x.shape[1], device=x.device, dtype=torch.float32)
        rv = torch.ones(x.shape[1], device=x.device, dtype=torch.float32)
        y = torch.nn.functional.batch_norm(x32, rm, rv, w32, b32, training=True, eps=1e-5)
    dx, dw, db = backward_of(y)(grad_out.float())
    return dx.to(x.dtype), dw, db


def _aten_bn_bwd(grad_out, x, weight, mean, rstd):
    """aten's batch-norm backward, run by itself on the saved statistics.

    The float32 casts are load-bearing rather than overhead: handed float16 this kernel
    forms the channel gradients in float16 too, and the reduction over every spatial
    element then lands far outside tolerance.
    """
    dx, dw, db = torch.ops.aten.native_batch_norm_backward(
        grad_out.float(),
        x.float(),
        weight.float(),
        None,
        None,
        mean,
        rstd,
        True,
        1e-5,
        [True, True, True],
    )
    return dx.to(x.dtype), dw, db


@pytest.mark.parametrize("call", manifest_calls(BatchNormFwdOp))
def test_batch_norm_fwd_bench(call):
    workload = RunningStatsCall(call)
    inputs = workload.gen_inputs()
    op = BatchNormFwdOp(**workload.arguments())
    training, momentum, eps = (call.params[k] for k in ("training", "momentum", "eps"))
    torch_fn = workload.ref_program
    functors = {"tileops": op}
    if all((t is not None for t in inputs)):
        flaggems_fn = _flaggems_bn_fwd(inputs[1], inputs[2], training, momentum, eps)
        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch-cudnn"] = torch_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(torch_fn)
    ManifestBenchmark(op, workload).compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(BatchNormBwdOp))
def test_batch_norm_bwd_bench(call):
    workload = BatchNormBwdCall(call)
    inputs = workload.gen_inputs()
    op = BatchNormBwdOp(**workload.arguments())
    ManifestBenchmark(op, workload).compare(
        {"tileops": op, "torch-autograd": _torch_bn_bwd, "torch-native-batch-norm": _aten_bn_bwd},
        *inputs,
    )
