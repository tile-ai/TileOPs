"""Benchmarks for unary elementwise math ops in the elementwise_unary_math family.

Profiles TileOPs vs PyTorch baselines for each op routed to this bench file by
``tileops/manifest/elementwise_unary_math.yaml``. Coverage spans the manifest-
routed unary_math ops that point ``source.bench`` at this file, including
several entries currently flagged ``status: spec-only`` because the
implementation is reachable via the float code path even though the manifest
declares additional dtypes the kernel does not yet support. Specifically:

* ``AbsFwdOp``, ``NegFwdOp``, ``ReciprocalFwdOp``, ``SignFwdOp``,
  ``FloorFwdOp``, ``CeilFwdOp``, ``TruncFwdOp``, ``IsnanFwdOp``,
  ``IsinfFwdOp``, and ``IsfiniteFwdOp`` route through ``FloatUnaryKernel`` /
  ``FloatPredicateKernel``. They are benched at float16/bfloat16/float32
  inputs (the supported subset). The declared int/uint8 dtype gap is tracked
  in #1171 and is the reason these ops are ``spec-only``; it does not affect
  bench reachability.
* ``SigmoidFwdOp`` and ``TanhFwdOp`` are benched in
  ``benchmarks/ops/bench_activation.py`` instead.
* ``RoundFwdOp`` is ``status: spec-only`` because the kernel only supports
  ``decimals=0`` (full ``torch.round`` semantics tracked in #1170) and is
  excluded here.

Each op is parametrized at one representative bandwidth-bound shape and dtype
to keep wall-clock cost bounded while still emitting numeric rows in
``profile_run.log`` for every benched op (AC-4 of issue #1162).
"""

from typing import Callable, Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops.elementwise import (
    AbsFwdOp,
    BitwiseNotFwdOp,
    CeilFwdOp,
    CosFwdOp,
    ErfFwdOp,
    ExpFwdOp,
    Expm1FwdOp,
    FloorFwdOp,
    IsfiniteFwdOp,
    IsinfFwdOp,
    IsnanFwdOp,
    Log1pFwdOp,
    LogFwdOp,
    LogicalNotFwdOp,
    NegFwdOp,
    ReciprocalFwdOp,
    RsqrtFwdOp,
    SignFwdOp,
    SinFwdOp,
    SqrtFwdOp,
    TruncFwdOp,
)
from workloads.workload_base import FixtureBase

_SHAPES = (262_144, 1_048_576, 4_000_000)
_BENCH_N = _SHAPES[1]


class UnaryElementwiseBenchCase:
    """Minimal test harness shared by BenchmarkBase."""

    def __init__(
        self,
        n_total: int,
        dtype: torch.dtype,
        output_dtype: torch.dtype,
        gen_inputs: Callable[[int, torch.dtype], tuple[torch.Tensor]],
    ):
        self.n_total = n_total
        self.dtype = dtype
        self.output_dtype = output_dtype
        self._gen_inputs = gen_inputs

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return self._gen_inputs(self.n_total, self.dtype)


class UnaryElementwiseBenchmark(BenchmarkBase[UnaryElementwiseBenchCase]):
    """Bandwidth-oriented benchmark for unary elementwise ops."""

    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        return self.workload.n_total * (
            self.workload.dtype.itemsize + self.workload.output_dtype.itemsize
        )


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------


def _randn(n_total: int, dtype: torch.dtype) -> tuple[torch.Tensor]:
    """Standard normal input."""
    return (torch.randn(n_total, device="cuda", dtype=dtype),)


def _positive(n_total: int, dtype: torch.dtype) -> tuple[torch.Tensor]:
    """Strictly positive input for log/sqrt/rsqrt/log1p."""
    x = torch.rand(n_total, device="cuda", dtype=dtype).clamp(min=0.01) + 0.01
    return (x,)


def _nonzero(n_total: int, dtype: torch.dtype) -> tuple[torch.Tensor]:
    """Nonzero input for reciprocal.

    Generates samples in fp32 so the magnitude floor is enforced before any
    fp16/bf16 quantization, then casts to the target dtype. The floor is
    chosen relative to ``finfo(dtype).tiny`` so that even after rounding to
    the lower-precision dtype the value is guaranteed nonzero, avoiding
    ``inf`` outputs from ``reciprocal`` skewing the benchmark.
    """
    finfo = torch.finfo(dtype)
    floor_mag = max(1e-2, finfo.tiny * 1024)
    x = torch.randn(n_total, device="cuda", dtype=torch.float32)
    sign = torch.where(x >= 0, 1.0, -1.0)
    x = sign * x.abs().clamp(min=floor_mag)
    return (x.to(dtype),)


def _logical_inputs(n_total: int, dtype: torch.dtype) -> tuple[torch.Tensor]:
    """Half-zero / half-nonzero input for logical_not."""
    x = torch.randn(n_total, device="cuda", dtype=dtype)
    mask = torch.rand(n_total, device="cuda") > 0.5
    x[mask] = 0
    return (x,)


def _bitwise_inputs(n_total: int, dtype: torch.dtype) -> tuple[torch.Tensor]:
    """Integer input for bitwise_not."""
    x = torch.randint(-128, 128, (n_total,), device="cuda", dtype=dtype)
    return (x,)


def _special_inputs(n_total: int, dtype: torch.dtype) -> tuple[torch.Tensor]:
    """Mix of NaN, +inf, -inf, finite for isnan/isinf/isfinite."""
    x = torch.randn(n_total, device="cuda", dtype=dtype)
    quarter = n_total // 4
    x[:quarter] = float("nan")
    x[quarter:2 * quarter] = float("inf")
    x[2 * quarter:3 * quarter] = float("-inf")
    return (x,)


# ---------------------------------------------------------------------------
# Baselines that need dtype upcast (fp16 floor/ceil/round/trunc are not native)
# ---------------------------------------------------------------------------


def _baseline_floor(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x.float()).to(x.dtype)


def _baseline_ceil(x: torch.Tensor) -> torch.Tensor:
    return torch.ceil(x.float()).to(x.dtype)


def _baseline_trunc(x: torch.Tensor) -> torch.Tensor:
    return torch.trunc(x.float()).to(x.dtype)


# ---------------------------------------------------------------------------
# Per-op parametrization. One representative shape/dtype per op so each
# implemented op in the family produces numeric rows in profile_run.log.
# ---------------------------------------------------------------------------


_FLOAT_OP_CASES = [
    # (op_name, op_cls, baseline_fn, gen_inputs, dtype, output_dtype)
    ("exp", ExpFwdOp, torch.exp, _randn, torch.float16, torch.float16),
    ("log", LogFwdOp, torch.log, _positive, torch.float16, torch.float16),
    ("sqrt", SqrtFwdOp, torch.sqrt, _positive, torch.float16, torch.float16),
    ("rsqrt", RsqrtFwdOp, torch.rsqrt, _positive, torch.float16, torch.float16),
    ("abs", AbsFwdOp, torch.abs, _randn, torch.float16, torch.float16),
    ("neg", NegFwdOp, torch.neg, _randn, torch.float16, torch.float16),
    ("reciprocal", ReciprocalFwdOp, torch.reciprocal, _nonzero,
     torch.float16, torch.float16),
    ("sign", SignFwdOp, torch.sign, _randn, torch.float16, torch.float16),
    ("sin", SinFwdOp, torch.sin, _randn, torch.float16, torch.float16),
    ("cos", CosFwdOp, torch.cos, _randn, torch.float16, torch.float16),
    ("floor", FloorFwdOp, _baseline_floor, _randn, torch.float16, torch.float16),
    ("ceil", CeilFwdOp, _baseline_ceil, _randn, torch.float16, torch.float16),
    ("trunc", TruncFwdOp, _baseline_trunc, _randn, torch.float16, torch.float16),
    ("erf", ErfFwdOp, torch.erf, _randn, torch.float16, torch.float16),
    ("log1p", Log1pFwdOp, torch.log1p, _positive, torch.float16, torch.float16),
    ("expm1", Expm1FwdOp, torch.expm1, _randn, torch.float16, torch.float16),
]

_LOGICAL_OP_CASES = [
    ("logical_not", LogicalNotFwdOp, torch.logical_not, _logical_inputs,
     torch.float16, torch.bool),
]

_BITWISE_OP_CASES = [
    ("bitwise_not", BitwiseNotFwdOp, torch.bitwise_not, _bitwise_inputs,
     torch.int32, torch.int32),
]

_SPECIAL_OP_CASES = [
    ("isnan", IsnanFwdOp, torch.isnan, _special_inputs,
     torch.float16, torch.bool),
    ("isinf", IsinfFwdOp, torch.isinf, _special_inputs,
     torch.float16, torch.bool),
    ("isfinite", IsfiniteFwdOp, torch.isfinite, _special_inputs,
     torch.float16, torch.bool),
]


_ALL_CASES = (
    _FLOAT_OP_CASES + _LOGICAL_OP_CASES + _BITWISE_OP_CASES + _SPECIAL_OP_CASES
)


_BENCH_PARAMS = [
    pytest.param(
        op_name, _BENCH_N, dtype, output_dtype, op_cls, baseline_fn, gen_inputs,
        id=f"{op_name}-{str(dtype).replace('torch.', '')}",
    )
    for op_name, op_cls, baseline_fn, gen_inputs, dtype, output_dtype in _ALL_CASES
]


class UnaryElementwiseBenchFixture(FixtureBase):
    PARAMS = [
        ("op_name, n_total, dtype, output_dtype, op_cls, baseline_fn, gen_inputs",
         _BENCH_PARAMS),
    ]


@UnaryElementwiseBenchFixture
def test_unary_elementwise_bench(
    op_name: str,
    n_total: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    op_cls,
    baseline_fn,
    gen_inputs,
) -> None:
    """Profile one op from elementwise_unary_math at a representative shape."""
    test = UnaryElementwiseBenchCase(
        n_total=n_total,
        dtype=dtype,
        output_dtype=output_dtype,
        gen_inputs=gen_inputs,
    )
    bm = UnaryElementwiseBenchmark(test)
    inputs = test.gen_inputs()

    op = op_cls(N_total=n_total, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op_name, locals(), result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op_name, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
