"""Benchmark for BatchNormFwdOp and BatchNormBwdOp.

Compares TileOPs vs PyTorch cuDNN batch norm on common ResNet-style shapes.

Run:
    conda run -n tileops python -m pytest benchmarks/ops/bench_batch_norm.py -vvs
"""

import math
from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.batch_norm import BatchNormBwdTest, BatchNormFwdTest

_FWD_OP_NAME = "BatchNormFwdOp"
_BWD_OP_NAME = "BatchNormBwdOp"

# ---------------------------------------------------------------------------
# Benchmark classes
# ---------------------------------------------------------------------------

class BatchNormFwdBenchmark(BenchmarkBase[BatchNormFwdTest]):

    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test, op):
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class BatchNormBwdBenchmark(BenchmarkBase[BatchNormBwdTest]):

    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test, op):
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

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
        x.float(), running_mean.clone(), running_var.clone(),
        weight.float(), bias.float(), training=True)


def _torch_bn_bwd(grad_out, x, weight, mean, rstd):
    """PyTorch reference backward via autograd."""
    with torch.enable_grad():
        x32 = x.float().requires_grad_(True)
        w32 = weight.float().requires_grad_(True)
        b32 = torch.zeros(x.shape[1], device=x.device, dtype=torch.float32, requires_grad=True)
        rm = torch.zeros(x.shape[1], device=x.device, dtype=torch.float32)
        rv = torch.ones(x.shape[1], device=x.device, dtype=torch.float32)
        y = torch.nn.functional.batch_norm(
            x32, rm, rv, w32, b32, training=True, eps=1e-5)
        y.backward(grad_out.float())
    return x32.grad, w32.grad, b32.grad


# ---------------------------------------------------------------------------
# Manifest-driven params
# ---------------------------------------------------------------------------

def _manifest_fwd_params():
    params = []
    for w in load_workloads(_FWD_OP_NAME):
        shape = w["x_shape"]
        N, C, spatial = shape[0], shape[1], tuple(shape[2:])
        label = w.get("label", f"{N}x{C}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(N, C, spatial, dtype, True, False,
                                       id=f"{label}-{dtype_str}"))
    return params


def _manifest_bwd_params():
    params = []
    for w in load_workloads(_BWD_OP_NAME):
        shape = w["x_shape"]
        N, C, spatial = shape[0], shape[1], tuple(shape[2:])
        label = w.get("label", f"{N}x{C}")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(N, C, spatial, dtype,
                                       id=f"{label}-{dtype_str}"))
    return params


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N, C, spatial, dtype, training, tune", _manifest_fwd_params())
def test_batch_norm_fwd_bench(N, C, spatial, dtype, training, tune):
    x, weight, bias, running_mean, running_var = _make_inputs(N, C, spatial, dtype)
    # Manifest input order: (x, running_mean, running_var, weight, bias).
    inputs = (x, running_mean, running_var, weight, bias)

    op = BatchNormFwdOp(N, C, *spatial, dtype=dtype, training=training, tune=tune)

    test = BatchNormFwdTest(N, C, spatial, dtype, training)
    bm = BatchNormFwdBenchmark(test, op)

    result = bm.profile(lambda *a: op(*a), *inputs)
    spatial = str(spatial)  # stringify tuple so it survives BenchmarkReport.record filtering
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(
        lambda x, rm, rv, w, b: _torch_bn_fwd(x, w, b, rm, rv), *inputs,
    )
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-cudnn")


@pytest.mark.parametrize("N, C, spatial, dtype", _manifest_bwd_params())
def test_batch_norm_bwd_bench(N, C, spatial, dtype):
    inputs = _make_bwd_inputs(N, C, spatial, dtype)

    op = BatchNormBwdOp(N, C, *spatial, dtype=dtype)

    test = BatchNormBwdTest(N, C, spatial, dtype)
    bm = BatchNormBwdBenchmark(test, op)

    result = bm.profile(op, *inputs)
    spatial = str(spatial)  # stringify tuple so it survives BenchmarkReport.record filtering
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(_torch_bn_bwd, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-autograd")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
