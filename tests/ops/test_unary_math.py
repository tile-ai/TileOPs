"""Tests for unary math elementwise ops (17 ops).

Covers L1 correctness across supported float dtypes and
L4 edge cases for numerically sensitive ops.
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops.elementwise import (
    AbsFwdOp,
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
    NegFwdOp,
    ReciprocalFwdOp,
    RoundFwdOp,
    RsqrtFwdOp,
    SignFwdOp,
    SinFwdOp,
    SqrtFwdOp,
    TruncFwdOp,
)
from workloads.device import run_device
from workloads.elementwise import ElementwiseWorkload, ErfRoundingWorkload, UnaryMathCase
from workloads.numerics import compare_outputs


class MathFixture(FixtureBase):
    """Parametrize over supported float dtypes for unary math ops."""

    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


class MathEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements."""

    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(4096, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


class UnaryMathTest(UnaryMathCase, TestBase):
    pass


def _randn(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(n, device=run_device(), dtype=dtype)


def _positive(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(n, device=run_device(), dtype=dtype).clamp(min=0.01) + 0.01


def _nonzero(n: int, dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(n, device=run_device(), dtype=dtype)
    return x + torch.sign(x) * 0.01


def _repeat_values(values: list[float], n: int, dtype: torch.dtype) -> torch.Tensor:
    base = torch.tensor(values, device=run_device(), dtype=dtype)
    repeats = (n + len(values) - 1) // len(values)
    return base.repeat(repeats)[:n]


def _make_math_test(n_total, dtype, gen_fn, op_cls):
    """Build test, instantiate op, and run check."""
    test = UnaryMathTest(n_total, dtype, op_cls.__name__, gen_fn=gen_fn)
    op = op_cls()
    test.check(op, *test.gen_inputs())


# L1 tests (17 ops)


@MathFixture
def test_exp(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, ExpFwdOp)


@MathFixture
def test_log(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _positive, LogFwdOp)


@MathFixture
def test_sqrt(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _positive, SqrtFwdOp)


@MathFixture
def test_rsqrt(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _positive, RsqrtFwdOp)


@MathFixture
def test_abs(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, AbsFwdOp)


@MathFixture
def test_neg(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, NegFwdOp)


@MathFixture
def test_reciprocal(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _nonzero, ReciprocalFwdOp)


@MathFixture
def test_sign(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, SignFwdOp)


@MathFixture
def test_sin(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, SinFwdOp)


@MathFixture
def test_cos(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, CosFwdOp)


@MathFixture
def test_floor(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        _randn,
        FloorFwdOp,
    )


@MathFixture
def test_ceil(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        _randn,
        CeilFwdOp,
    )


@MathFixture
def test_round(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        _randn,
        RoundFwdOp,
    )


@MathFixture
def test_trunc(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        _randn,
        TruncFwdOp,
    )


@MathFixture
def test_erf(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, ErfFwdOp)


@MathFixture
def test_log1p(n_total: int, dtype: torch.dtype) -> None:
    def _gen(n, gen_dtype):
        return torch.rand(n, device=run_device(), dtype=gen_dtype).clamp(min=0.01)

    _make_math_test(n_total, dtype, _gen, Log1pFwdOp)


@MathFixture
def test_expm1(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, Expm1FwdOp)


# Integer-dtype identity short-circuit for floor / ceil / round / trunc.
#
# The manifest declares these ops over both integer and float dtypes; the
# underlying kernels are float-only. ``torch.{floor,ceil,round,trunc}`` are
# no-ops on integer tensors, so the op layer short-circuits and returns a
# clone of the input unchanged.


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls",
    [FloorFwdOp, CeilFwdOp, RoundFwdOp, TruncFwdOp],
)
@pytest.mark.parametrize(
    "int_dtype",
    [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8],
)
def test_rounding_op_int_identity(op_cls, int_dtype: torch.dtype) -> None:
    n_total = 1024
    op = op_cls()
    if int_dtype == torch.uint8:
        x = torch.randint(0, 100, (n_total,), device=run_device(), dtype=int_dtype)
    else:
        x = torch.randint(-50, 50, (n_total,), device=run_device(), dtype=int_dtype)
    y = op(x)
    assert y.dtype == int_dtype
    assert y.shape == x.shape
    compare_outputs(y, x, ElementwiseWorkload(type(op).__name__, (x,)).verification(*(x,)))


@pytest.mark.smoke
def test_round_rejects_decimals_on_integer_input() -> None:
    """``torch.round`` rounds an integral input only to zero decimals."""
    op = RoundFwdOp(decimals=2)
    x = torch.randint(-100, 100, (256,), device=run_device(), dtype=torch.int32)
    with pytest.raises(ValueError, match="decimals"):
        op(x)


# Integer-dtype op-layer fallbacks for abs / neg / sign and the
# is{nan,inf,finite} predicates. Their manifest entries declare integer
# input dtypes alongside floats; the underlying kernels are float-only,
# so the op layer routes int input through a torch primitive (or the
# constant-bool result, for the predicates).


_INT_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, torch_fn",
    [
        (AbsFwdOp, torch.abs),
        (NegFwdOp, torch.neg),
        (SignFwdOp, torch.sign),
    ],
)
@pytest.mark.parametrize("int_dtype", _INT_DTYPES)
def test_unary_int_torch_fallback(op_cls, torch_fn, int_dtype) -> None:
    n_total = 1024
    op = op_cls()
    if int_dtype == torch.uint8:
        x = torch.randint(0, 100, (n_total,), device=run_device(), dtype=int_dtype)
    else:
        x = torch.randint(-50, 50, (n_total,), device=run_device(), dtype=int_dtype)
    y = op(x)
    assert y.dtype == int_dtype
    compare_outputs(
        y, torch_fn(x), ElementwiseWorkload(type(op).__name__, (x,)).verification(*(x,))
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, expected",
    [
        (IsnanFwdOp, False),
        (IsinfFwdOp, False),
        (IsfiniteFwdOp, True),
    ],
)
@pytest.mark.parametrize("non_float_dtype", _INT_DTYPES + [torch.bool])
def test_predicate_non_float_constant(op_cls, expected, non_float_dtype) -> None:
    """Predicate ops return constant bool on every non-float dtype the
    manifest declares (integer dtypes plus ``torch.bool``)."""
    n_total = 256
    op = op_cls()
    if non_float_dtype == torch.bool:
        x = torch.randint(0, 2, (n_total,), device=run_device(), dtype=torch.bool)
    elif non_float_dtype == torch.uint8:
        x = torch.randint(0, 100, (n_total,), device=run_device(), dtype=non_float_dtype)
    else:
        x = torch.randint(-50, 50, (n_total,), device=run_device(), dtype=non_float_dtype)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == x.shape
    assert (y == expected).all()


# L4 edge-case tests (fp32, 4K)


@MathEdgeFixture
def test_sqrt_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([-1.0, 0.0, 1e-38, 1.0], n, d),
        SqrtFwdOp,
    )


@MathEdgeFixture
def test_rsqrt_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([-1.0, 0.0, 1e-38, 1.0], n, d),
        RsqrtFwdOp,
    )


@MathEdgeFixture
def test_log_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([-1.0, 0.0, 1e-38, 1.0], n, d),
        LogFwdOp,
    )


@MathEdgeFixture
def test_log1p_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([-2.0, -1.0, 0.0, 1e-7], n, d),
        Log1pFwdOp,
    )


@MathEdgeFixture
def test_exp_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 88.8, -88.8, 200.0], n, d),
        ExpFwdOp,
    )


@MathEdgeFixture
def test_expm1_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 88.8, -88.8, 1e-7], n, d),
        Expm1FwdOp,
    )


@MathEdgeFixture
def test_erf_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 3.0, -3.0, 100.0], n, d),
        ErfFwdOp,
    )


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_erf_matches_rounded_erf_over_every_value(dtype: torch.dtype) -> None:
    """erf lands within one representable step of the rounded value, at every input.

    float16 and bfloat16 evaluate a polynomial, saturated past a clamp that normal
    inputs never reach, and the op tolerance is a whole ulp wider than the fit
    needs. Enumerating the dtype is cheap enough to leave nothing untested.
    """
    workload = ErfRoundingWorkload(dtype)
    TestBase.check(workload, ErfFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_erf_saturates_at_infinity_and_propagates_nan(dtype: torch.dtype) -> None:
    """Edge: erf reaches exactly +-1 at +-inf and answers NaN with NaN, as ``torch.erf`` does."""
    x = torch.tensor([float("inf"), -float("inf"), float("nan")], device=run_device(), dtype=dtype)
    out = ErfFwdOp()(x)
    workload = ErfRoundingWorkload(dtype)
    compare_outputs(out, workload.ref_program(x), workload.verification(x))


@MathEdgeFixture
def test_reciprocal_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 1.0, -1.0, 1e-38], n, d),
        ReciprocalFwdOp,
    )


@MathEdgeFixture
def test_sign_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([-5.0, 0.0, 3.0, float("nan")], n, d),
        SignFwdOp,
    )


@pytest.mark.smoke
@pytest.mark.parametrize("decimals", [0, 2, -1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_round_decimals(dtype: torch.dtype, decimals: int) -> None:
    """RoundFwdOp must honour the manifest 'decimals' parameter end-to-end.

    Uses ``torch.round(x, decimals=k)`` as the reference and the standard
    decomposition under the hood: ``round(x * 10**k) / 10**k``.
    """
    x = torch.randn(4096, device=run_device(), dtype=dtype) * 10.0
    op = RoundFwdOp(decimals=decimals)
    out = op(x)
    ref = torch.round(x.float(), decimals=decimals).to(dtype)
    # The decimals path runs entirely in fp32 internally and only down-casts
    # once at the end, so the standard per-dtype tolerances apply.
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(type(op).__name__, (x,), decimals=decimals).verification(*(x,)),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_round_decimals_no_overflow_low_precision(dtype: torch.dtype) -> None:
    """Decimals path must not overflow fp16/bf16 when ``|x| * 10**decimals`` exceeds dtype max.

    Regression: previously the op cast ``x.float() * 10**decimals`` back to
    ``self.dtype`` before rounding, so e.g. ``100 * 10**4 = 1e6`` overflowed
    fp16's ~65504 max and produced ``inf``. The reference is
    ``torch.round(x.float(), decimals=k).to(dtype)`` which is just ``100.0``.
    """
    x = torch.tensor([100.0], device=run_device(), dtype=dtype)
    op = RoundFwdOp(decimals=4)
    out = op(x)
    ref = torch.round(x.float(), decimals=4).to(dtype)
    assert torch.isfinite(out).all(), f"output contains non-finite values: {out}"
    compare_outputs(
        out, ref, ElementwiseWorkload(type(op).__name__, (x,), decimals=4).verification(*(x,))
    )


@pytest.mark.smoke
def test_round_decimals_default_is_zero() -> None:
    """Constructing RoundFwdOp without ``decimals`` must round to nearest integer."""
    x = torch.randn(1024, device=run_device(), dtype=torch.float32) * 5.0
    op = RoundFwdOp()
    out = op(x)
    ref = torch.round(x)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (x,)).verification(*(x,)))


@pytest.mark.smoke
def test_round_decimals_validates_input() -> None:
    """Non-zero decimals must enforce the same input contract as decimals=0.

    Regression: a wrong-dtype input would silently short-circuit through the fp32
    decomposition because the path bypassed the op's validation.
    """
    op = RoundFwdOp(decimals=2)
    # float16 is in the manifest union, so the same instance accepts it.
    assert op(torch.ones(2, device=run_device(), dtype=torch.float16)).dtype == torch.float16
    # A dtype outside the union must raise.
    with pytest.raises(ValueError, match="dtype"):
        op(torch.ones(2, device=run_device(), dtype=torch.float64))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8],
)
def test_reciprocal_int_promotes_to_float32(dtype: torch.dtype) -> None:
    """ReciprocalFwdOp must accept integral inputs and yield float32 output.

    Mirrors ``torch.reciprocal``'s int-input promotion: the manifest's
    ``promote_int_to_float(input)`` output dtype must round-trip against
    the PyTorch reference for every declared integer dtype.
    """
    n_total = 4096
    if dtype == torch.uint8:
        # uint8 range [1, 255] avoids zero (1/0 = inf disagrees with the
        # tolerance-based comparison) without saturating the reference.
        x = torch.randint(1, 256, (n_total,), device=run_device(), dtype=dtype)
    elif dtype == torch.int8:
        # int8 range [-127, 127] excluding zero.
        x = torch.randint(-127, 128, (n_total,), device=run_device(), dtype=dtype)
        x = torch.where(x == 0, torch.ones_like(x), x)
    else:
        x = torch.randint(-1000, 1001, (n_total,), device=run_device(), dtype=dtype)
        x = torch.where(x == 0, torch.ones_like(x), x)
    op = ReciprocalFwdOp()
    out = op(x)
    assert out.dtype == torch.float32, (
        f"output dtype must be float32 for int input, got {out.dtype}"
    )
    ref = torch.reciprocal(x)
    assert ref.dtype == torch.float32, (
        f"torch.reciprocal({dtype}) reference dtype changed: {ref.dtype}"
    )
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (x,)).verification(*(x,)))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8],
)
def test_reciprocal_int_roofline_prices_the_promoted_output(
    dtype: torch.dtype,
) -> None:
    """The roofline charges integer input bytes plus float32 output bytes."""
    n_total = 4
    op = ReciprocalFwdOp()
    op(torch.ones(n_total, device=run_device(), dtype=dtype))
    expected_bytes = n_total * (dtype.itemsize + torch.float32.itemsize)
    flops, bytes_ = op.eval_roofline()
    assert flops == n_total
    assert bytes_ == expected_bytes


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_reciprocal_int_input_validation() -> None:
    """One instance serves integer and float inputs, and rejects the rest.

    Promotion is per call, so float32 and int32 are both valid and land on
    different entries; a dtype outside the manifest union still raises.
    """
    op = ReciprocalFwdOp()
    assert op(torch.ones(4, device=run_device(), dtype=torch.float32)).dtype == torch.float32
    assert op(torch.ones(4, device=run_device(), dtype=torch.int32)).dtype == torch.float32
    with pytest.raises(ValueError, match="dtype"):
        op(torch.ones(4, device=run_device(), dtype=torch.float64))
