"""Benchmarks for fp8 elementwise ops (e4m3fn, e5m2).

Profiles TileOPs fp8 kernels vs PyTorch fp16-compute-then-cast baselines
for unary (relu, exp), binary (add), and fused gated (silu_and_mul) ops
across three shapes and both fp8 dtypes.
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops.elementwise import AddFwdOp, ExpFwdOp, ReluFwdOp, SiluAndMulFwdOp
from workloads.workload_base import FixtureBase

# Shapes modeled on real LLM workloads: batch × seq_len × hidden_dim
# Small:  1 × 2048 × 4096  =  8,388,608  (single-batch inference, LLaMA-7B hidden)
# Medium: 8 × 2048 × 4096  = 67,108,864  (multi-batch inference)
# Large:  4 × 4096 × 8192  = 134,217,728 (training, LLaMA-70B hidden)
_SHAPES_1D = (1 * 2048 * 4096, 8 * 2048 * 4096, 4 * 4096 * 8192)
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Fp8UnaryBenchCase:
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = (torch.randn(self.n_total, dtype=torch.float16, device="cuda") * 2.0)
        return (x.to(self.dtype),)


class Fp8UnaryBenchmark(BenchmarkBase[Fp8UnaryBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        # fp8 in (1B) + fp8 out (1B) per element
        return self.workload.n_total * 2


class Fp8BinaryBenchCase:
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = (torch.randn(self.n_total, dtype=torch.float16, device="cuda") * 0.5).to(self.dtype)
        b = (torch.randn(self.n_total, dtype=torch.float16, device="cuda") * 0.5).to(self.dtype)
        return a, b


class Fp8BinaryBenchmark(BenchmarkBase[Fp8BinaryBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        # fp8 in a (1B) + fp8 in b (1B) + fp8 out (1B)
        return self.workload.n_total * 3


class Fp8FusedGatedBenchCase:
    def __init__(self, M: int, N: int, dtype: torch.dtype):
        self.M = M
        self.N = N
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = (torch.randn(self.M, 2 * self.N, dtype=torch.float16, device="cuda") * 0.5)
        return (x.to(self.dtype),)


class Fp8FusedGatedBenchmark(BenchmarkBase[Fp8FusedGatedBenchCase]):
    def calculate_flops(self) -> Optional[float]:
        # FIXME(ying): hardcoded for silu (4 FLOPs/elem + 1 mul with value = 5).
        # Must update when benchmarking other activations (e.g. gelu).
        return self.workload.M * self.workload.N * 5

    def calculate_memory(self) -> Optional[float]:
        # Read x (M*2N*1B) + write y (M*N*1B)
        return (self.workload.M * 2 * self.workload.N + self.workload.M * self.workload.N)


# ---------------------------------------------------------------------------
# Unary fp8 benchmarks: relu, exp
# ---------------------------------------------------------------------------

_unary_params = []
for _op_name, _op_cls, _bl_fn in [
    ("relu_fp8", ReluFwdOp, torch.relu),
    ("exp_fp8", ExpFwdOp, torch.exp),
]:
    for _shape in _SHAPES_1D:
        for _dt in _FP8_DTYPES:
            _unary_params.append(pytest.param(
                _op_name, _shape, _dt, _op_cls, _bl_fn,
                id=f"{_op_name}-{_shape}-{_dt}",
            ))


class Fp8UnaryBenchFixture(FixtureBase):
    PARAMS = [("op_name, n_total, dtype, op_cls, baseline_fn", _unary_params)]


@Fp8UnaryBenchFixture
def test_fp8_unary_bench(op_name, n_total, dtype, op_cls, baseline_fn):
    test = Fp8UnaryBenchCase(n_total=n_total, dtype=dtype)
    bm = Fp8UnaryBenchmark(test)
    inputs = test.gen_inputs()

    op = op_cls(N_total=n_total, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    # Baseline: PyTorch fp16 compute then cast back to fp8
    def baseline(*args):
        return baseline_fn(args[0].to(torch.float16)).to(dtype)

    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="torch")


# ---------------------------------------------------------------------------
# Binary fp8 benchmark: add
# ---------------------------------------------------------------------------

_binary_params = []
for _shape in _SHAPES_1D:
    for _dt in _FP8_DTYPES:
        _binary_params.append(pytest.param(
            "add_fp8", _shape, _dt,
            id=f"add_fp8-{_shape}-{_dt}",
        ))


class Fp8BinaryBenchFixture(FixtureBase):
    PARAMS = [("op_name, n_total, dtype", _binary_params)]


@Fp8BinaryBenchFixture
def test_fp8_binary_bench(op_name, n_total, dtype):
    test = Fp8BinaryBenchCase(n_total=n_total, dtype=dtype)
    bm = Fp8BinaryBenchmark(test)
    inputs = test.gen_inputs()

    op = AddFwdOp(a_shape=(n_total,), b_shape=(n_total,), dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    def baseline(a, b):
        return (a.to(torch.float16) + b.to(torch.float16)).to(dtype)

    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="torch")


# ---------------------------------------------------------------------------
# Fused gated fp8 benchmark: silu_and_mul
# ---------------------------------------------------------------------------

# Fused gated shapes: (batch × seq_len, intermediate_dim)
# LLaMA-7B:  hidden=4096,  intermediate=11008
# LLaMA-13B: hidden=5120,  intermediate=13824
# LLaMA-70B: hidden=8192,  intermediate=28672
_GATED_SHAPES = [
    (1 * 2048, 11008),   # LLaMA-7B single-batch inference
    (8 * 2048, 11008),   # LLaMA-7B multi-batch inference
    (4 * 4096, 28672),   # LLaMA-70B training
]
_gated_params = []
for _M, _N in _GATED_SHAPES:
    for _dt in _FP8_DTYPES:
        _gated_params.append(pytest.param(
            "silu_and_mul_fp8", _M, _N, _dt,
            id=f"silu_and_mul_fp8-{_M}x{_N}-{_dt}",
        ))


class Fp8FusedGatedBenchFixture(FixtureBase):
    PARAMS = [("op_name, M, N, dtype", _gated_params)]


@Fp8FusedGatedBenchFixture
def test_fp8_fused_gated_bench(op_name, M, N, dtype):
    test = Fp8FusedGatedBenchCase(M=M, N=N, dtype=dtype)
    bm = Fp8FusedGatedBenchmark(test)
    inputs = test.gen_inputs()

    op = SiluAndMulFwdOp(M=M, N=N, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    def baseline(x):
        x_fp16 = x.to(torch.float16)
        gate = x_fp16[:, :N]
        value = x_fp16[:, N:]
        return (F.silu(gate) * value).to(dtype)

    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
