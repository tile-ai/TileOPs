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
    params: dict,
) -> None:
    """Profile op and torch baseline against the same inputs and record both.

    ``ManifestBenchmark`` is constructed at the call site of each per-op test,
    over the op that test builds. This helper only handles the profile + record
    pair.

    ``params`` is the workload metadata (shape / dtype / n_total) from the
    caller's scope; passing it explicitly keeps the report rows distinguishable
    instead of reflecting only this helper's locals.
    """
    functors = {
        "tileops": op,
        "torch": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }
    try:
        bm.compare(functors, *inputs, record_as=op, params=params)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# Per-op constants and tests — one block per manifest entry, so each op loads
# its own workloads and records a row of its own.

_EXP_OP = "ExpFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_EXP_OP))
def test_exp_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = ExpFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.exp, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_LOG_OP = "LogFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_LOG_OP))
def test_log_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_positive_away_from_zero(shape, dtype)
    n_total = inputs[0].numel()
    op = LogFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.log, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_SQRT_OP = "SqrtFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_SQRT_OP))
def test_sqrt_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_positive_away_from_zero(shape, dtype)
    n_total = inputs[0].numel()
    op = SqrtFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.sqrt, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_RSQRT_OP = "RsqrtFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_RSQRT_OP))
def test_rsqrt_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_positive_away_from_zero(shape, dtype)
    n_total = inputs[0].numel()
    op = RsqrtFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.rsqrt, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_ABS_OP = "AbsFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ABS_OP))
def test_abs_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = AbsFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.abs, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_NEG_OP = "NegFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_NEG_OP))
def test_neg_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = NegFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.neg, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_RECIPROCAL_OP = "ReciprocalFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_RECIPROCAL_OP))
def test_reciprocal_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_positive_away_from_zero(shape, dtype)
    n_total = inputs[0].numel()
    op = ReciprocalFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.reciprocal, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_SIGN_OP = "SignFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_SIGN_OP))
def test_sign_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = SignFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.sign, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_SIN_OP = "SinFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_SIN_OP))
def test_sin_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = SinFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.sin, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_COS_OP = "CosFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_COS_OP))
def test_cos_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = CosFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.cos, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_FLOOR_OP = "FloorFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_FLOOR_OP))
def test_floor_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = FloorFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.floor, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_CEIL_OP = "CeilFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_CEIL_OP))
def test_ceil_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = CeilFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.ceil, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_ROUND_OP = "RoundFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ROUND_OP))
def test_round_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = RoundFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.round, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_TRUNC_OP = "TruncFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_TRUNC_OP))
def test_trunc_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = TruncFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.trunc, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_ERF_OP = "ErfFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ERF_OP))
def test_erf_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = ErfFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.erf, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_LOG1P_OP = "Log1pFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_LOG1P_OP))
def test_log1p_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_positive_away_from_zero(shape, dtype)
    n_total = inputs[0].numel()
    op = Log1pFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.log1p, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_EXPM1_OP = "Expm1FwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_EXPM1_OP))
def test_expm1_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_normal(shape, dtype)
    n_total = inputs[0].numel()
    op = Expm1FwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.expm1, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


# SigmoidFwdOp / TanhFwdOp are activation ops; their manifest source.bench
# points to ``benchmarks/ops/bench_elementwise_manifest.py`` and is
# intentionally out of scope for this file.

_LOGICAL_NOT_OP = "LogicalNotFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_LOGICAL_NOT_OP))
def test_logical_not_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_bool(shape, dtype)
    n_total = inputs[0].numel()
    op = LogicalNotFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.logical_not, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_BITWISE_NOT_OP = "BitwiseNotFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_BITWISE_NOT_OP))
def test_bitwise_not_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_int(shape, dtype)
    n_total = inputs[0].numel()
    op = BitwiseNotFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.bitwise_not, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_ISNAN_OP = "IsnanFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ISNAN_OP))
def test_isnan_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_special_floats(shape, dtype)
    n_total = inputs[0].numel()
    op = IsnanFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.isnan, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_ISINF_OP = "IsinfFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ISINF_OP))
def test_isinf_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_special_floats(shape, dtype)
    n_total = inputs[0].numel()
    op = IsinfFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.isinf, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )


_ISFINITE_OP = "IsfiniteFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ISFINITE_OP))
def test_isfinite_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = draw_special_floats(shape, dtype)
    n_total = inputs[0].numel()
    op = IsfiniteFwdOp()
    bm = ManifestBenchmark(op, UnaryWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.isfinite, {"shape": shape, "dtype": dtype, "n_total": n_total}
    )
