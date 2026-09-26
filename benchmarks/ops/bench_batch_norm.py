"""Benchmark for BatchNormFwdOp and BatchNormBwdOp.

Compares TileOPs vs PyTorch cuDNN batch norm on common ResNet-style shapes. The
forward row adds flag_gems' batch_norm and cuDNN through inductor. The backward row
carries two torch tags: an autograd node driven on this thread, and aten's backward
kernel by itself. The difference between them is the forward the autograd one rebuilds.
"""

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, backward_of, manifest_calls
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.normalization import BatchNormBwdCall, RunningStatsCall


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
    return backward_of(y)(grad_out.float())


def _aten_bn_bwd(grad_out, x, weight, mean, rstd):
    """aten's batch-norm backward, run by itself on the saved statistics.

    The float32 casts are load-bearing rather than overhead: handed float16 this kernel
    forms the channel gradients in float16 too, and the reduction over every spatial
    element then lands far outside tolerance.
    """
    return torch.ops.aten.native_batch_norm_backward(
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


@pytest.mark.parametrize("call", manifest_calls(BatchNormFwdOp))
def test_batch_norm_fwd_bench(call):
    workload = RunningStatsCall(call)
    inputs = workload.gen_inputs()
    op = BatchNormFwdOp(**workload.arguments())
    training, momentum, eps = (call.params[k] for k in ("training", "momentum", "eps"))

    def torch_fn(x, rm, rv, w, b):
        rm, rv = (None, None) if rm is None else (rm.clone(), rv.clone())
        return torch.nn.functional.batch_norm(
            x.float(), rm, rv, w, b, training=training, momentum=momentum, eps=eps
        ).to(x.dtype)

    # cuDNN and the kernels reduce over N*H*W in fp32; agreement is at the storage dtype's.
    tolerance = reference_tolerance(inputs[0].dtype)
    reference_inputs = tuple(t if t is None else t.clone() for t in inputs)
    assert_matches_reference(op, torch_fn, *reference_inputs, **tolerance)
    functors = {"tileops": op}
    # flag_gems' entry point takes every tensor; a row omitting one has no tag.
    if all(t is not None for t in inputs):
        flaggems_fn = _flaggems_bn_fwd(inputs[1], inputs[2], training, momentum, eps)
        assert_matches_reference(flaggems_fn, torch_fn, *inputs, **tolerance)
        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch-cudnn"] = torch_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(torch_fn)
    ManifestBenchmark(op, workload).compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(BatchNormBwdOp))
def test_batch_norm_bwd_bench(call):
    workload = BatchNormBwdCall(call)
    inputs = workload.gen_inputs()
    op = BatchNormBwdOp(**workload.arguments())

    # A reduction this long disagrees with the reference's order past float32's tolerance.
    assert_matches_reference(_aten_bn_bwd, _torch_bn_bwd, *inputs, rtol=1e-3, atol=1e-3)
    assert_matches_reference(
        op, workload.ref_program, *inputs, **reference_tolerance(inputs[0].dtype)
    )

    ManifestBenchmark(op, workload).compare(
        {
            "tileops": op,
            "torch-autograd": _torch_bn_bwd,
            "torch-native-batch-norm": _aten_bn_bwd,
        },
        *inputs,
    )
