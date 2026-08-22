"""Benchmark for BatchNormFwdOp and BatchNormBwdOp.

Compares TileOPs vs PyTorch cuDNN batch norm on common ResNet-style shapes. The
forward row adds flag_gems' batch_norm and cuDNN through inductor; the backward row
stays on the autograd reference alone, which is a node driven on this thread and not
work either of those performs.
"""

import math

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, backward_of, workload_params
from tileops.manifest import load_workloads
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.normalization import BatchNormBwdWorkload, BatchNormFwdWorkload

_FWD_OP_NAME = "BatchNormFwdOp"
_BWD_OP_NAME = "BatchNormBwdOp"

# Benchmark classes


# Benchmark helpers


def _make_inputs(N, C, spatial, dtype, device="cuda"):
    shape = (N, C, *spatial)
    x = torch.randn(*shape, device=device, dtype=dtype)
    weight = torch.randn(C, device=device, dtype=torch.float32)
    bias = torch.randn(C, device=device, dtype=torch.float32)
    running_mean = torch.zeros(C, device=device, dtype=torch.float32)
    running_var = torch.ones(C, device=device, dtype=torch.float32)
    return x, weight, bias, running_mean, running_var


def _make_bwd_inputs(N, C, spatial, dtype, device="cuda"):
    x, weight, bias, running_mean, running_var = _make_inputs(N, C, spatial, dtype, device)
    grad_out = torch.randn_like(x)
    L = N * math.prod(spatial) if spatial else N
    x_cl = x.float().permute(1, 0, *range(2, x.ndim)).reshape(C, L).contiguous()
    mean = x_cl.mean(dim=1)
    var = x_cl.var(dim=1, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + 1e-5)
    return grad_out, x, weight, mean, rstd


def _torch_bn_fwd(x, weight, bias, running_mean, running_var):
    return torch.nn.functional.batch_norm(
        x.float(),
        running_mean.clone(),
        running_var.clone(),
        weight.float(),
        bias.float(),
        training=True,
    )


def _flaggems_bn_fwd(running_mean: torch.Tensor, running_var: torch.Tensor):
    """flag_gems' batch_norm on its own running statistics, output only.

    Training mode updates them in place, and the cuDNN reference clones before it
    does, so this gets copies rather than the tensors the other tags read.
    """
    fn = flaggems_op("batch_norm")
    private_mean, private_var = running_mean.clone(), running_var.clone()

    def baseline_fn(x, _running_mean, _running_var, weight, bias):
        return fn(x.float(), weight, bias, private_mean, private_var, True, 0.1, 1e-5)[0]

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


# Manifest-driven params


def _fwd_args(w: dict, dtype: torch.dtype) -> tuple:
    """``(N, C, spatial, dtype, tune, training)``."""
    n, c, *spatial = w["x_shape"]
    return (n, c, tuple(spatial), dtype, True, False)


def _bwd_args(w: dict, dtype: torch.dtype) -> tuple:
    """``(N, C, spatial, dtype)``."""
    n, c, *spatial = w["x_shape"]
    return (n, c, tuple(spatial), dtype)


# Benchmark tests


@pytest.mark.parametrize(
    "N, C, spatial, dtype, training, tune", workload_params(load_workloads(_FWD_OP_NAME), _fwd_args)
)
def test_batch_norm_fwd_bench(N, C, spatial, dtype, training, tune):
    x, weight, bias, running_mean, running_var = _make_inputs(N, C, spatial, dtype)
    # Manifest input order: (x, running_mean, running_var, weight, bias).
    inputs = (x, running_mean, running_var, weight, bias)

    op = BatchNormFwdOp(training=training, tune=tune)

    test = BatchNormFwdWorkload(N, C, spatial, dtype, training)
    bm = ManifestBenchmark(_FWD_OP_NAME, op, test)

    spatial = str(spatial)  # stringify tuple so it survives BenchmarkReport.record filtering

    def torch_fn(x, rm, rv, w, b):
        return _torch_bn_fwd(x, w, b, rm, rv)

    flaggems_fn = _flaggems_bn_fwd(running_mean, running_var)
    # cuDNN and Triton both reduce over N*H*W in fp32; agreement is at fp32 strength.
    assert_matches_reference(flaggems_fn, torch_fn, *inputs, rtol=1e-4, atol=1e-4)

    bm.compare(
        {
            "tileops": lambda *a: op(*a),
            FLAGGEMS_TAG: flaggems_fn,
            "torch-cudnn": torch_fn,
            TORCH_COMPILE_TAG: compiled_reference(torch_fn),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


@pytest.mark.parametrize(
    "N, C, spatial, dtype", workload_params(load_workloads(_BWD_OP_NAME), _bwd_args)
)
def test_batch_norm_bwd_bench(N, C, spatial, dtype):
    inputs = _make_bwd_inputs(N, C, spatial, dtype)

    op = BatchNormBwdOp()

    test = BatchNormBwdWorkload(N, C, spatial, dtype)
    bm = ManifestBenchmark(_BWD_OP_NAME, op, test)

    spatial = str(spatial)  # stringify tuple so it survives BenchmarkReport.record filtering

    bm.compare(
        {"tileops": op, "torch-autograd": _torch_bn_bwd}, *inputs, record_as=op, params=locals()
    )
