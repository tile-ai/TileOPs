"""Benchmarks for the elementwise unary_math op family.

Measures latency, FLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes, dtypes, and roofline formulas are loaded from the ops
manifest (``tileops/manifest/elementwise_unary_math.yaml``).

Each op gets its own ``test_*_bench`` function so that the manifest
validator's per-op AST check (see ``scripts/validate_manifest.py`` →
``check_l4_benchmark``) can match ``load_workloads("<OpName>FwdOp")`` /
``ManifestBenchmark("<OpName>FwdOp", ...)`` calls one-to-one. A shared
``_profile_and_record`` helper handles the profile + record pair so the
per-op functions stay tiny and intentional.
"""

from typing import Callable

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark, workloads_to_params
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

# ---------------------------------------------------------------------------
# Workload + input generation
# ---------------------------------------------------------------------------


class _UnaryWorkload:
    """Minimal :class:`ShapeDtypeWorkload` for unary elementwise ops.

    Holds ``shape`` and ``dtype`` so that :class:`ManifestBenchmark` can call
    ``op.eval_roofline()`` after ``forward()`` has bound the dynamic vars.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


def _randn(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    return (torch.randn(shape, device="cuda", dtype=dtype),)


def _positive(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    # Domain restriction for log / sqrt / rsqrt / log1p / reciprocal.
    return (torch.rand(shape, device="cuda", dtype=dtype) + 0.5,)


def _bool_input(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, device="cuda", dtype=torch.bool)
    else:
        x = torch.randn(shape, device="cuda", dtype=dtype)
        mask = torch.rand(shape, device="cuda") > 0.5
        x[mask] = 0
    return (x,)


def _int_input(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    info = torch.iinfo(dtype)
    lo = max(info.min, -1024)
    hi = min(info.max, 1024)
    return (torch.randint(lo, hi, shape, device="cuda", dtype=dtype),)


def _special_floats(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    # Mix of normal floats, +/-inf, and NaN — exercises isnan/isinf/isfinite.
    x = torch.randn(shape, device="cuda", dtype=dtype)
    flat = x.view(-1)
    quarter = flat.numel() // 4
    flat[:quarter] = float("nan")
    flat[quarter:2 * quarter] = float("inf")
    flat[2 * quarter:3 * quarter] = float("-inf")
    return (x,)


# ---------------------------------------------------------------------------
# Shared bench harness
# ---------------------------------------------------------------------------


def _profile_and_record(
    op,
    bm: ManifestBenchmark,
    inputs: tuple,
    baseline_fn: Callable,
) -> None:
    """Profile op and torch baseline against the same inputs and record both.

    ``ManifestBenchmark`` must be constructed at the call site of each per-op
    test (with the literal op-name constant) so that the manifest validator's
    AST check can tie ``ManifestBenchmark("<OpName>FwdOp", ...)`` to the
    intended op. This helper only handles the profile + record pair.
    """
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Per-op constants and tests — one block per manifest entry so the
# validator AST check ties each ``load_workloads(<OpName>)`` /
# ``ManifestBenchmark(<OpName>, ...)`` call to its op.
# ===================================================================

_EXP_OP = "ExpFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_EXP_OP))
def test_exp_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = ExpFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_EXP_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.exp)


_LOG_OP = "LogFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_LOG_OP))
def test_log_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _positive(shape, dtype)
    n_total = inputs[0].numel()
    op = LogFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_LOG_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.log)


_SQRT_OP = "SqrtFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_SQRT_OP))
def test_sqrt_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _positive(shape, dtype)
    n_total = inputs[0].numel()
    op = SqrtFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_SQRT_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.sqrt)


_RSQRT_OP = "RsqrtFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_RSQRT_OP))
def test_rsqrt_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _positive(shape, dtype)
    n_total = inputs[0].numel()
    op = RsqrtFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_RSQRT_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.rsqrt)


_ABS_OP = "AbsFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ABS_OP))
def test_abs_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = AbsFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_ABS_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.abs)


_NEG_OP = "NegFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_NEG_OP))
def test_neg_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = NegFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_NEG_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.neg)


_RECIPROCAL_OP = "ReciprocalFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_RECIPROCAL_OP))
def test_reciprocal_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _positive(shape, dtype)
    n_total = inputs[0].numel()
    op = ReciprocalFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_RECIPROCAL_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.reciprocal)


_SIGN_OP = "SignFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_SIGN_OP))
def test_sign_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = SignFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_SIGN_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.sign)


_SIN_OP = "SinFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_SIN_OP))
def test_sin_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = SinFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_SIN_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.sin)


_COS_OP = "CosFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_COS_OP))
def test_cos_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = CosFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_COS_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.cos)


_FLOOR_OP = "FloorFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_FLOOR_OP))
def test_floor_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = FloorFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_FLOOR_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.floor)


_CEIL_OP = "CeilFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_CEIL_OP))
def test_ceil_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = CeilFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_CEIL_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.ceil)


_ROUND_OP = "RoundFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ROUND_OP))
def test_round_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = RoundFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_ROUND_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.round)


_TRUNC_OP = "TruncFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_TRUNC_OP))
def test_trunc_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = TruncFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_TRUNC_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.trunc)


_ERF_OP = "ErfFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ERF_OP))
def test_erf_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = ErfFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_ERF_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.erf)


_LOG1P_OP = "Log1pFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_LOG1P_OP))
def test_log1p_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _positive(shape, dtype)
    n_total = inputs[0].numel()
    op = Log1pFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_LOG1P_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.log1p)


_EXPM1_OP = "Expm1FwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_EXPM1_OP))
def test_expm1_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = Expm1FwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_EXPM1_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.expm1)


# SigmoidFwdOp / TanhFwdOp are activation ops; their manifest source.bench
# points to ``benchmarks/ops/bench_activation.py`` and is intentionally out
# of scope for this file.

_LOGICAL_NOT_OP = "LogicalNotFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_LOGICAL_NOT_OP))
def test_logical_not_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _bool_input(shape, dtype)
    n_total = inputs[0].numel()
    op = LogicalNotFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_LOGICAL_NOT_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.logical_not)


_BITWISE_NOT_OP = "BitwiseNotFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_BITWISE_NOT_OP))
def test_bitwise_not_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _int_input(shape, dtype)
    n_total = inputs[0].numel()
    op = BitwiseNotFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_BITWISE_NOT_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.bitwise_not)


_ISNAN_OP = "IsnanFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ISNAN_OP))
def test_isnan_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _special_floats(shape, dtype)
    n_total = inputs[0].numel()
    op = IsnanFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_ISNAN_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.isnan)


_ISINF_OP = "IsinfFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ISINF_OP))
def test_isinf_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _special_floats(shape, dtype)
    n_total = inputs[0].numel()
    op = IsinfFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_ISINF_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.isinf)


_ISFINITE_OP = "IsfiniteFwdOp"


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ISFINITE_OP))
def test_isfinite_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _special_floats(shape, dtype)
    n_total = inputs[0].numel()
    op = IsfiniteFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_ISFINITE_OP, op, _UnaryWorkload(shape, dtype))
    _profile_and_record(op, bm, inputs, torch.isfinite)

if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
