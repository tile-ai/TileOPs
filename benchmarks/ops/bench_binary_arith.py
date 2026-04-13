"""Benchmarks for binary arithmetic ops covering risk points R1, R2, R4.

Risk points covered:
- R1: Stride-based load vectorization (add x explicit_parallel x fp16 x
      {1D same-shape, 2D bias-add, 3D interleaved})
- R2: Divmod overhead on small tensors (add same-shape/3D-broadcast x fp16 x 4K)
- R4: DEFAULT_STRATEGY confirmation (add x 2 strategies x 3 dtypes x 3 sizes x
      {same-shape, 2D bias-add, 3D interleaved})

Profiles both binary strategies (direct, explicit_parallel) and compares
against PyTorch baseline.
"""

from math import prod
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.elementwise import AddOp, WhereOp
from workloads.base import FixtureBase
from workloads.binary_arith import AddSameShapeTest

# ---------------------------------------------------------------------------
# LLM-realistic shapes (LLaMA-family defaults)
# ---------------------------------------------------------------------------

_SIZES = {
    "4K": 4096,
    "1M": 1_048_576,        # 1024 * 1024
    "16M": 16_777_216,      # 1024 * 16384
}

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BINARY_STRATEGIES = ("direct", "explicit_parallel")

def _make_interleaved_3d(n: int) -> tuple[tuple, tuple]:
    """Build (A,1,C) + (1,B,1) -> (A,B,C) with A*B*C == n exactly.

    Uses A=8 (or 1 for very small n). Finds the largest B <= sqrt(n/A)
    that divides n/A evenly, then C = n/(A*B).
    """
    if n < 8:
        return (1, 1, n), (1, n, 1)
    a_dim = 8
    remainder = n // a_dim
    b_dim = int(remainder ** 0.5)
    while b_dim > 1 and remainder % b_dim != 0:
        b_dim -= 1
    c_dim = remainder // b_dim
    return (a_dim, 1, c_dim), (1, b_dim, 1)


# Broadcast patterns for binary ops
_BROADCAST_PATTERNS = {
    "same_shape": lambda n: ((n,), (n,)),
    "bias_add_2d": lambda n: (
        (1024, n // 1024) if n >= 1024 else (1, n),
        (1, n // 1024) if n >= 1024 else (1, n),
    ),
    "interleaved_3d": lambda n: _make_interleaved_3d(n),
}


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class BinaryBenchCase:
    """Minimal test harness for binary benchmarks."""

    def __init__(
        self, a_shape: tuple, b_shape: tuple, dtype: torch.dtype,
    ):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype
        self.n_total = prod(torch.broadcast_shapes(a_shape, b_shape))

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.a_shape, device="cuda", dtype=self.dtype)
        b = torch.randn(self.b_shape, device="cuda", dtype=self.dtype)
        return a, b


class BinaryBenchmark(BenchmarkBase):
    """Bandwidth-oriented benchmark for binary elementwise ops."""

    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem_bytes = t.dtype.itemsize
        # Read a + read b + write output
        a_elems = prod(t.a_shape) if hasattr(t, "a_shape") else t.n_total
        b_elems = prod(t.b_shape) if hasattr(t, "b_shape") else t.n_total
        return (a_elems + b_elems + t.n_total) * elem_bytes


class WhereBenchCase:
    """Test harness for where op benchmarks."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = torch.randint(0, 2, (self.n_total,), device="cuda", dtype=torch.bool)
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        y = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        return cond, x, y


class WhereBenchmark(BenchmarkBase):
    """Benchmark for where op."""

    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem_bytes = t.dtype.itemsize
        # Read cond (1 byte) + read x + read y + write output
        return t.n_total * (1 + 3 * elem_bytes)


# ---------------------------------------------------------------------------
# R1: Stride-based load vectorization
# ---------------------------------------------------------------------------

_R1_PATTERNS = [
    ("same_shape_1d", (1_000_000,), (1_000_000,)),
    # bias-add: (1000, 1000) + (1, 1000) -> 1,000,000 output elements
    ("bias_add_2d", (1000, 1000), (1, 1000)),
    # interleaved: (8,1,1024) + (1,128,1) -> (8,128,1024) = 1,048,576 output
    ("interleaved_3d", (8, 1, 1024), (1, 128, 1)),
]


class R1VectorizationFixture(FixtureBase):
    PARAMS = [
        ("pattern_name, a_shape, b_shape", [
            pytest.param(name, a, b, marks=pytest.mark.smoke if name == "same_shape_1d"
                         else pytest.mark.full)
            for name, a, b in _R1_PATTERNS
        ]),
    ]


@R1VectorizationFixture
def test_r1_vectorization(
    pattern_name: str,
    a_shape: tuple,
    b_shape: tuple,
) -> None:
    """R1: Benchmark stride-based load vectorization.

    Binary divmod offset may prevent uint4 vectorized loads.
    Compares same-shape (no divmod) vs broadcast patterns (divmod required).
    """
    dtype = torch.float16
    test = BinaryBenchCase(a_shape, b_shape, dtype)
    bm = BinaryBenchmark(test)
    inputs = test.gen_inputs()

    op = AddOp(
        a_shape=a_shape, b_shape=b_shape, dtype=dtype,
        strategy="explicit_parallel",
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "r1_vectorization",
        {"pattern_name": pattern_name, "n_total": test.n_total},
        result,
        tag=f"add_{pattern_name}",
    )

    # Baseline: PyTorch add with broadcast
    a, b = inputs

    def baseline_fn(a, b):
        return a + b

    result_bl = bm.profile(baseline_fn, a, b)
    BenchmarkReport.record(
        "r1_vectorization",
        {"pattern_name": pattern_name, "n_total": test.n_total},
        result_bl,
        tag=f"torch-{pattern_name}",
    )


# ---------------------------------------------------------------------------
# R2: Divmod overhead on small tensors (binary)
# ---------------------------------------------------------------------------


class R2BinaryFixture(FixtureBase):
    PARAMS = [
        ("pattern_name, a_shape, b_shape", [
            pytest.param("same_shape", (4096,), (4096,), marks=pytest.mark.smoke),
            pytest.param(
                "broadcast_3d", (4, 1, 32), (1, 32, 1),
                marks=pytest.mark.full,
            ),
        ]),
    ]


@R2BinaryFixture
def test_r2_small_tensor_binary(
    pattern_name: str,
    a_shape: tuple,
    b_shape: tuple,
) -> None:
    """R2: Benchmark divmod overhead on small tensors (binary add, 4K)."""
    dtype = torch.float16
    test = BinaryBenchCase(a_shape, b_shape, dtype)
    bm = BinaryBenchmark(test)
    inputs = test.gen_inputs()

    op = AddOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "r2_small_tensor_binary",
        {"pattern_name": pattern_name, "n_total": test.n_total},
        result,
        tag=f"add_{pattern_name}",
    )

    a, b = inputs

    def baseline_fn(a, b):
        return a + b

    result_bl = bm.profile(baseline_fn, a, b)
    BenchmarkReport.record(
        "r2_small_tensor_binary",
        {"pattern_name": pattern_name, "n_total": test.n_total},
        result_bl,
        tag=f"torch-{pattern_name}",
    )


# ---------------------------------------------------------------------------
# R4: DEFAULT_STRATEGY confirmation (binary full matrix)
# ---------------------------------------------------------------------------


_R4_BINARY_PARAMS = []
for size_label, n in _SIZES.items():
    for dt in _DTYPES:
        for strategy in _BINARY_STRATEGIES:
            for pat_name, pat_fn in _BROADCAST_PATTERNS.items():
                a_shape, b_shape = pat_fn(n)
                mark = pytest.mark.smoke if (
                    size_label == "1M" and dt == torch.float16
                    and strategy == "explicit_parallel"
                    and pat_name == "same_shape"
                ) else pytest.mark.full
                _R4_BINARY_PARAMS.append(
                    pytest.param(
                        a_shape, b_shape, dt, strategy, size_label, pat_name,
                        id=f"{size_label}-{dt}-{strategy}-{pat_name}",
                        marks=mark,
                    )
                )


class R4BinaryStrategyFixture(FixtureBase):
    PARAMS = [
        ("a_shape, b_shape, dtype, strategy, size_label, pattern_name",
         _R4_BINARY_PARAMS),
    ]


@R4BinaryStrategyFixture
def test_r4_default_strategy_binary(
    a_shape: tuple,
    b_shape: tuple,
    dtype: torch.dtype,
    strategy: str,
    size_label: str,
    pattern_name: str,
) -> None:
    """R4: Benchmark both binary strategies across full matrix.

    Covers: add x {direct, explicit_parallel} x {fp32, fp16, bf16}
            x {4K, 1M, 16M} x {same-shape, bias-add, interleaved-3D}
    """
    test = BinaryBenchCase(a_shape, b_shape, dtype)
    bm = BinaryBenchmark(test)
    inputs = test.gen_inputs()

    op = AddOp(
        a_shape=a_shape, b_shape=b_shape, dtype=dtype, strategy=strategy,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "r4_strategy_binary",
        {
            "size_label": size_label,
            "pattern_name": pattern_name,
            "strategy": strategy,
            "n_total": test.n_total,
        },
        result,
        tag=f"add_{strategy}_{pattern_name}",
    )


# ---------------------------------------------------------------------------
# R4: Where op strategy comparison (3-input op)
# ---------------------------------------------------------------------------


_R4_WHERE_PARAMS = []
for size_label, n in _SIZES.items():
    _R4_WHERE_PARAMS.append(
        pytest.param(
            n, size_label, torch.float16,
            id=f"where-{size_label}-fp16",
            marks=pytest.mark.full,
        )
    )


class R4WhereFixture(FixtureBase):
    PARAMS = [
        ("n_total, size_label, dtype", _R4_WHERE_PARAMS),
    ]


@R4WhereFixture
def test_r4_where_bench(
    n_total: int,
    size_label: str,
    dtype: torch.dtype,
) -> None:
    """R4: Benchmark where op across sizes."""
    test = WhereBenchCase(n_total, dtype)
    bm = WhereBenchmark(test)
    inputs = test.gen_inputs()

    op = WhereOp(N_total=n_total, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "r4_where",
        {"n_total": n_total, "size_label": size_label},
        result,
        tag="tileops-where",
    )

    cond, x, y = inputs

    def baseline_fn(cond, x, y):
        return torch.where(cond, x, y)

    result_bl = bm.profile(baseline_fn, cond, x, y)
    BenchmarkReport.record(
        "r4_where",
        {"n_total": n_total, "size_label": size_label},
        result_bl,
        tag="torch",
    )


# ---------------------------------------------------------------------------
# Baseline throughput benchmarks (existing, refined with LLaMA shapes)
# ---------------------------------------------------------------------------


_ADD_BENCH_PARAMS = [
    pytest.param(prod((1024, 4096)), torch.float16, id="throughput-fp16"),
    pytest.param(prod((1024, 4096)), torch.bfloat16, id="throughput-bf16"),
    pytest.param(prod((1024, 4096)), torch.float32, id="baseline-fp32"),
]


@pytest.mark.parametrize("n_total, dtype", _ADD_BENCH_PARAMS)
def test_add_bench(n_total: int, dtype: torch.dtype) -> None:
    test = AddSameShapeTest(n_total, dtype)
    bm = BinaryBenchmark(test)
    inputs = test.gen_inputs()

    shape = (n_total,)
    op = AddOp(a_shape=shape, b_shape=shape, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(a, b):
        return a + b

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
