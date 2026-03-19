"""Benchmark for BatchNormFwdOp and BatchNormBwdOp.

Compares TileOPs vs PyTorch cuDNN batch norm on common ResNet-style shapes.

Run:
    conda run -n tileops python -m pytest benchmarks/ops/bench_batch_norm.py -vvs
"""

import math
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_batch_norm import BatchNormBwdTest, BatchNormFwdTest
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp

# ---------------------------------------------------------------------------
# Benchmark classes
# ---------------------------------------------------------------------------

class BatchNormFwdBenchmark(BenchmarkBase):

    def __init__(self, test, N, C, spatial):
        super().__init__(test)
        self.N = N
        self.C = C
        self.spatial = spatial

    def calculate_flops(self) -> Optional[float]:
        # 2 passes × L elements × C channels × ~5 ops (mean/var/norm/scale/shift)
        L = self.N * math.prod(self.spatial) if self.spatial else self.N
        return 5.0 * self.C * L * 2

    def calculate_memory(self) -> Optional[float]:
        # Read x + write y + params (weight, bias, running stats)
        L = self.N * math.prod(self.spatial) if self.spatial else self.N
        elem_bytes = 2  # float16 / bfloat16
        return (2 * self.C * L + 4 * self.C) * elem_bytes


class BatchNormBwdBenchmark(BenchmarkBase):

    def __init__(self, test, N, C, spatial):
        super().__init__(test)
        self.N = N
        self.C = C
        self.spatial = spatial

    def calculate_flops(self) -> Optional[float]:
        L = self.N * math.prod(self.spatial) if self.spatial else self.N
        return 8.0 * self.C * L

    def calculate_memory(self) -> Optional[float]:
        L = self.N * math.prod(self.spatial) if self.spatial else self.N
        elem_bytes = 2
        return (3 * self.C * L + 3 * self.C) * elem_bytes


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
# Benchmark tests
# ---------------------------------------------------------------------------

# Use a reduced fixture for benchmarks (avoid OOM on large spatial dims).
_FWD_BENCH_PARAMS = [
    (32, 64,  (),        torch.float16, True, False),
    (8,  64,  (32, 32),  torch.float16, True, False),
    (4,  128, (32, 32),  torch.float16, True, False),
    (4,  256, (28, 28),  torch.float16, True, False),
    (4,  128, (1024, 1024),  torch.float16, True, False),
    (4,  256, (1024, 1024),  torch.float16, True, False),
]

_BWD_BENCH_PARAMS = [
    (32, 64,  (),        torch.float16),
    (8,  64,  (32, 32),  torch.float16),
    (4,  128, (32, 32),  torch.float16),
    (4,  256, (28, 28),  torch.float16),
    (4,  128, (1024, 1024),  torch.float16),
    (4,  256, (1024, 1024),  torch.float16),
]


@pytest.mark.parametrize("N, C, spatial, dtype, training, tune", _FWD_BENCH_PARAMS)
def test_batch_norm_fwd_bench(N, C, spatial, dtype, training, tune):
    inputs = _make_inputs(N, C, spatial, dtype)

    op = BatchNormFwdOp(N, C, *spatial, dtype=dtype, tune=tune)

    test = BatchNormFwdTest(N, C, spatial, dtype, training)
    bm = BatchNormFwdBenchmark(test, N, C, spatial)

    result = bm.profile(lambda *a: op(*a, training=training), *inputs)
    spatial = str(spatial)  # stringify tuple so it survives BenchmarkReport.record filtering
    BenchmarkReport.record("batch_norm_fwd", locals(), result, tag="tileops")

    result_bl = bm.profile(_torch_bn_fwd, *inputs)
    BenchmarkReport.record("batch_norm_fwd", locals(), result_bl, tag="torch_cudnn")


@pytest.mark.parametrize("N, C, spatial, dtype", _BWD_BENCH_PARAMS)
def test_batch_norm_bwd_bench(N, C, spatial, dtype):
    inputs = _make_bwd_inputs(N, C, spatial, dtype)

    op = BatchNormBwdOp(N, C, *spatial, dtype=dtype)

    test = BatchNormBwdTest(N, C, spatial, dtype)
    bm = BatchNormBwdBenchmark(test, N, C, spatial)

    result = bm.profile(op, *inputs)
    spatial = str(spatial)  # stringify tuple so it survives BenchmarkReport.record filtering
    BenchmarkReport.record("batch_norm_bwd", locals(), result, tag="tileops")

    result_bl = bm.profile(_torch_bn_bwd, *inputs)
    BenchmarkReport.record("batch_norm_bwd", locals(), result_bl, tag="torch_autograd")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
