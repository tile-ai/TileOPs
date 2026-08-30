"""Benchmarks for the elementwise unary_math op family.

Measures latency, FLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes, dtypes, and roofline formulas are loaded from the ops
manifest (``src/tileops/manifest/elementwise_unary_math.yaml``).

One ``test_*_bench`` per op, so every op this file is declared the benchmark
of records a row of its own. A shared
``_profile_and_record`` helper handles the profile + record pair so the
per-op functions stay tiny and intentional.

Baselines: torch eager and the same reference through inductor. flag_gems covers
most of these ops but cannot be timed on them; ``flaggems_op`` says why.
"""

from typing import Callable

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, workloads_to_params
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
    RoundFwdOp,
    RsqrtFwdOp,
    SignFwdOp,
    SinFwdOp,
    SqrtFwdOp,
    TruncFwdOp,
)
from workloads.elementwise import (
    draw_bool,
    draw_int,
    draw_normal,
    draw_positive_away_from_zero,
    draw_special_floats,
)

# Workload + input generation


class UnaryWorkload:
    """Minimal shape/dtype descriptor for unary elementwise ops.

    Holds ``shape`` and ``dtype`` so that :class:`ManifestBenchmark` can call
    ``op.eval_roofline()`` after ``forward()`` has bound the dynamic vars.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


# Shared workload and profiling helpers


def _profile_and_record(
    op,
    bm: ManifestBenchmark,
    inputs: tuple,
    baseline_fn: Callable,
) -> None:
    """Profile op and torch baseline against the same inputs and record both.

    ``ManifestBenchmark`` is constructed at the call site of each per-op test,
    over the op that test builds. This helper only handles the profile + record
    pair.
    """
    functors = {
        "tileops": op,
        "torch": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }
    try:
        bm.compare(functors, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# Per-op constants and tests — one block per manifest entry, so each op loads
# its own workloads and records a row of its own.


@pytest.mark.parametrize("shape, dtype", workloads_to_params(ExpFwdOp))
def test_exp_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = ExpFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.exp)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(LogFwdOp))
def test_log_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_positive_away_from_zero(shape, dtype)
    op = LogFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.log)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(SqrtFwdOp))
def test_sqrt_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_positive_away_from_zero(shape, dtype)
    op = SqrtFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.sqrt)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(RsqrtFwdOp))
def test_rsqrt_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_positive_away_from_zero(shape, dtype)
    op = RsqrtFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.rsqrt)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(AbsFwdOp))
def test_abs_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = AbsFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.abs)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(NegFwdOp))
def test_neg_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = NegFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.neg)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(ReciprocalFwdOp))
def test_reciprocal_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_positive_away_from_zero(shape, dtype)
    op = ReciprocalFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.reciprocal)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(SignFwdOp))
def test_sign_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = SignFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.sign)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(SinFwdOp))
def test_sin_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = SinFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.sin)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(CosFwdOp))
def test_cos_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = CosFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.cos)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(FloorFwdOp))
def test_floor_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = FloorFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.floor)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(CeilFwdOp))
def test_ceil_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = CeilFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.ceil)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(RoundFwdOp))
def test_round_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = RoundFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.round)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(TruncFwdOp))
def test_trunc_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = TruncFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.trunc)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(ErfFwdOp))
def test_erf_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = ErfFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.erf)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(Log1pFwdOp))
def test_log1p_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_positive_away_from_zero(shape, dtype)
    op = Log1pFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.log1p)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(Expm1FwdOp))
def test_expm1_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    op = Expm1FwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.expm1)


# SigmoidFwdOp / TanhFwdOp are activation ops; their manifest source.bench
# points to ``benchmarks/ops/bench_elementwise_manifest.py`` and is
# intentionally out of scope for this file.


@pytest.mark.parametrize("shape, dtype", workloads_to_params(LogicalNotFwdOp))
def test_logical_not_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_bool(shape, dtype)
    op = LogicalNotFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.logical_not)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(BitwiseNotFwdOp))
def test_bitwise_not_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_int(shape, dtype)
    op = BitwiseNotFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.bitwise_not)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(IsnanFwdOp))
def test_isnan_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_special_floats(shape, dtype)
    op = IsnanFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.isnan)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(IsinfFwdOp))
def test_isinf_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_special_floats(shape, dtype)
    op = IsinfFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.isinf)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(IsfiniteFwdOp))
def test_isfinite_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_special_floats(shape, dtype)
    op = IsfiniteFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.isfinite)
