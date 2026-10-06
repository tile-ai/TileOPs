"""Benchmark for BatchNormFwdOp and BatchNormBwdOp.

Compares TileOPs vs PyTorch cuDNN batch norm on common ResNet-style shapes. The
forward row adds flag_gems' batch_norm and cuDNN through inductor. The backward row
carries two torch tags: an autograd node driven on this thread, and aten's backward
kernel by itself. The difference between them is the forward the autograd one rebuilds.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    backward_of,
    compiled_reference,
    flaggems_op,
    private_inputs,
)
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp


def _flaggems_bn_fwd(training: bool, momentum: float, eps: float):
    """Expose the output and the running statistics it updates in place."""
    fn = flaggems_op("batch_norm")

    def baseline_fn(x, running_mean, running_var, weight, bias):
        out = fn(x.float(), weight, bias, running_mean, running_var, training, momentum, eps)
        return out[0].to(x.dtype), running_mean, running_var

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


@pytest.mark.parametrize("case", bench.cases(BatchNormFwdOp), ids=lambda case: case.id)
def test_batch_norm_fwd_bench(case):
    inputs = case.inputs
    op = BatchNormFwdOp(**case.arguments)
    training, momentum, eps = (case.params[k] for k in ("training", "momentum", "eps"))
    torch_fn = case.reference
    functors = {}
    if all((t is not None for t in inputs)):
        flaggems_fn = _flaggems_bn_fwd(training, momentum, eps)
        functors[FLAGGEMS_TAG] = private_inputs(flaggems_fn, inputs, 1, 2)
    functors["torch-cudnn"] = torch_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(torch_fn)
    bench.Runner(op, case).compare(functors)


@pytest.mark.parametrize("case", bench.cases(BatchNormBwdOp), ids=lambda case: case.id)
def test_batch_norm_bwd_bench(case):
    op = BatchNormBwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {"torch-autograd": _torch_bn_bwd, "torch-native-batch-norm": _aten_bn_bwd}
    )
