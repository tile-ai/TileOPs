"""Tests for the unary math elementwise ops."""

import pytest
import torch

from tests.launch_bounds import elementwise_launch_bounds, full_occupancy_bounds
from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.elementwise import (
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
from workloads.elementwise import (
    ElementwiseWorkload,
    ErfRoundingWorkload,
    Exhaustive16BitWorkload,
    Float32BinadeWorkload,
    Float32SpecialValuesWorkload,
    UnaryActivationCase,
)
from workloads.numerics import compare_outputs


class MathFixture(FixtureBase):
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


class UnaryMathTest(UnaryActivationCase, TestBase):
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


def _log1p_input(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(n, device=run_device(), dtype=dtype).clamp(min=0.01)


def _make_math_test(n_total, dtype, gen_fn, op_cls):
    test = UnaryMathTest(n_total, dtype, op_cls.__name__, gen_fn=gen_fn)
    op = op_cls()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "op_cls, gen_fn",
    [
        pytest.param(ExpFwdOp, _randn, id="exp"),
        pytest.param(LogFwdOp, _positive, id="log"),
        pytest.param(SqrtFwdOp, _positive, id="sqrt"),
        pytest.param(RsqrtFwdOp, _positive, id="rsqrt"),
        pytest.param(AbsFwdOp, _randn, id="abs"),
        pytest.param(NegFwdOp, _randn, id="neg"),
        pytest.param(ReciprocalFwdOp, _nonzero, id="reciprocal"),
        pytest.param(SignFwdOp, _randn, id="sign"),
        pytest.param(SinFwdOp, _randn, id="sin"),
        pytest.param(CosFwdOp, _randn, id="cos"),
        pytest.param(FloorFwdOp, _randn, id="floor"),
        pytest.param(CeilFwdOp, _randn, id="ceil"),
        pytest.param(RoundFwdOp, _randn, id="round"),
        pytest.param(TruncFwdOp, _randn, id="trunc"),
        pytest.param(ErfFwdOp, _randn, id="erf"),
        pytest.param(Log1pFwdOp, _log1p_input, id="log1p"),
        pytest.param(Expm1FwdOp, _randn, id="expm1"),
    ],
)
@MathFixture
def test_unary_math(op_cls, gen_fn, n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, gen_fn, op_cls)


def _non_float_input(n_total: int, dtype: torch.dtype) -> torch.Tensor:
    """*n_total* random values of an integer or bool *dtype*, signed where it is."""
    if dtype == torch.bool:
        low, high = 0, 2
    elif dtype == torch.uint8:
        low, high = 0, 100
    else:
        low, high = -50, 50
    return torch.randint(low, high, (n_total,), device=run_device(), dtype=dtype)


# An integral input reaches a torch fallback kernel, which does not depend on the pairing of
# op and dtype, so every op and every integral dtype appears once.


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, int_dtype",
    [
        (FloorFwdOp, torch.int8),
        (CeilFwdOp, torch.int16),
        (RoundFwdOp, torch.int32),
        (TruncFwdOp, torch.int64),
        (FloorFwdOp, torch.uint8),
    ],
)
def test_rounding_op_int_identity(op_cls, int_dtype: torch.dtype) -> None:
    op = op_cls()
    x = _non_float_input(1024, int_dtype)
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


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, torch_fn, int_dtype",
    [
        (AbsFwdOp, torch.abs, torch.int8),
        (NegFwdOp, torch.neg, torch.int16),
        (SignFwdOp, torch.sign, torch.int32),
        (AbsFwdOp, torch.abs, torch.int64),
        (SignFwdOp, torch.sign, torch.uint8),
    ],
)
def test_unary_int_torch_fallback(op_cls, torch_fn, int_dtype) -> None:
    op = op_cls()
    x = _non_float_input(1024, int_dtype)
    y = op(x)
    assert y.dtype == int_dtype
    compare_outputs(
        y, torch_fn(x), ElementwiseWorkload(type(op).__name__, (x,)).verification(*(x,))
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, expected, non_float_dtype",
    [
        (IsnanFwdOp, False, torch.int8),
        (IsinfFwdOp, False, torch.int16),
        (IsfiniteFwdOp, True, torch.int32),
        (IsnanFwdOp, False, torch.int64),
        (IsinfFwdOp, False, torch.uint8),
        (IsfiniteFwdOp, True, torch.bool),
    ],
)
def test_predicate_non_float_constant(op_cls, expected, non_float_dtype) -> None:
    """Predicate ops return a constant bool on every integer dtype and bool."""
    op = op_cls()
    x = _non_float_input(256, non_float_dtype)
    y = op(x)
    assert y.dtype == torch.bool
    assert y.shape == x.shape
    assert (y == expected).all()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, values",
    [
        pytest.param(SqrtFwdOp, [-1.0, 0.0, 1e-38, 1.0], id="sqrt"),
        pytest.param(RsqrtFwdOp, [-1.0, 0.0, 1e-38, 1.0], id="rsqrt"),
        pytest.param(LogFwdOp, [-1.0, 0.0, 1e-38, 1.0], id="log"),
        pytest.param(Log1pFwdOp, [-2.0, -1.0, 0.0, 1e-7], id="log1p"),
        pytest.param(ExpFwdOp, [0.0, 88.8, -88.8, 200.0], id="exp"),
        pytest.param(Expm1FwdOp, [0.0, 88.8, -88.8, 1e-7], id="expm1"),
        pytest.param(ErfFwdOp, [0.0, 3.0, -3.0, 100.0], id="erf"),
        pytest.param(SignFwdOp, [-5.0, 0.0, 3.0, float("nan")], id="sign"),
    ],
)
def test_unary_math_edge(op_cls, values: list[float]) -> None:
    """float32 domain edges, overflow and saturation."""
    _make_math_test(4096, torch.float32, lambda n, d: _repeat_values(values, n, d), op_cls)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_erf_matches_rounded_erf_over_every_value(dtype: torch.dtype) -> None:
    """erf lands within one representable step of the rounded value, at every input."""
    workload = ErfRoundingWorkload(dtype)
    TestBase.check(workload, ErfFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_erf_saturates_at_infinity_and_propagates_nan(dtype: torch.dtype) -> None:
    """erf reaches exactly +-1 at +-inf and answers NaN with NaN, as ``torch.erf`` does."""
    x = torch.tensor([float("inf"), -float("inf"), float("nan")], device=run_device(), dtype=dtype)
    out = ErfFwdOp()(x)
    workload = ErfRoundingWorkload(dtype)
    compare_outputs(out, workload.ref_program(x), workload.verification(x))


def _erf_launch_bounds(n_total: int, dtype: torch.dtype) -> tuple[int, list]:
    """Check erf on *n_total* elements of *dtype*; return its program's threads and launch bounds."""
    test = UnaryMathTest(n_total, dtype, "ErfFwdOp", gen_fn=_randn)
    op = ErfFwdOp()
    test.check(op, *test.gen_inputs())
    return elementwise_launch_bounds(op)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "n_total",
    [
        pytest.param(1_048_576, marks=pytest.mark.smoke),
        # A ragged last block, which ends in a partial vector.
        pytest.param(1_048_576 + 3, marks=pytest.mark.full),
    ],
)
def test_erf_float32_keeps_an_sm_of_threads_resident(n_total: int) -> None:
    """float32 erf is right, and asks for as many blocks an SM as the register file holds
    threads at 32 registers each."""
    threads, bounds = _erf_launch_bounds(n_total, torch.float32)
    assert bounds == full_occupancy_bounds(threads)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_erf_16bit_leaves_the_register_budget_to_nvcc() -> None:
    """float16 erf is right, and asks for no blocks an SM beyond one."""
    threads, bounds = _erf_launch_bounds(1_048_576, torch.float16)
    assert bounds == [(str(threads), "1")]


@pytest.mark.smoke
@pytest.mark.parametrize("repeats", [1, 16], ids=["scalar", "vector"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_16bit_reciprocal_matches_torch_bit_for_bit(dtype: torch.dtype, repeats: int) -> None:
    """The 16-bit reciprocal gives torch's bit for bit, at every input; tiled sixteen times,
    the input runs the vector path, where a thread holding a divisor out of the approximate
    reciprocal's range takes the full divide."""
    workload = Exhaustive16BitWorkload("ReciprocalFwdOp", dtype, repeats)
    TestBase.check(workload, ReciprocalFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
def test_float32_reciprocal_divides_every_value_class() -> None:
    """Every float32 value class, tiled into whole 1024-element blocks, gives torch's bits."""
    workload = Float32SpecialValuesWorkload("ReciprocalFwdOp", 13_312)
    TestBase.check(workload, ReciprocalFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
def test_reciprocal_tuned() -> None:
    """Tuned mode picks a launch whose result matches the reference."""
    test = UnaryMathTest(1_048_576, torch.float32, "ReciprocalFwdOp", gen_fn=_nonzero)
    op = ReciprocalFwdOp()
    op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_float32_reciprocal_matches_torch_bit_for_bit() -> None:
    """The float32 reciprocal's Newton step gives the IEEE quotient at every significand."""
    workload = Float32BinadeWorkload("ReciprocalFwdOp")
    TestBase.check(workload, ReciprocalFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("decimals", [2, -1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_round_decimals(dtype: torch.dtype, decimals: int) -> None:
    """RoundFwdOp honours a non-zero ``decimals``, as ``torch.round`` does."""
    x = torch.randn(4096, device=run_device(), dtype=dtype) * 10.0
    op = RoundFwdOp(decimals=decimals)
    out = op(x)
    ref = torch.round(x.float(), decimals=decimals).to(dtype)
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(type(op).__name__, (x,), decimals=decimals).verification(*(x,)),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_round_decimals_no_overflow_low_precision(dtype: torch.dtype) -> None:
    """Decimals path must not overflow fp16/bf16 when ``|x| * 10**decimals`` exceeds dtype max."""
    x = torch.tensor([100.0], device=run_device(), dtype=dtype)
    op = RoundFwdOp(decimals=4)
    out = op(x)
    ref = torch.round(x.float(), decimals=4).to(dtype)
    assert torch.isfinite(out).all(), f"output contains non-finite values: {out}"
    compare_outputs(
        out, ref, ElementwiseWorkload(type(op).__name__, (x,), decimals=4).verification(*(x,))
    )


@pytest.mark.smoke
def test_round_decimals_validates_input() -> None:
    """Non-zero decimals accept a manifest dtype and reject one outside the union."""
    op = RoundFwdOp(decimals=2)
    assert op(torch.ones(2, device=run_device(), dtype=torch.float16)).dtype == torch.float16
    with pytest.raises(ValueError, match="dtype"):
        op(torch.ones(2, device=run_device(), dtype=torch.float64))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8],
)
def test_reciprocal_int_promotes_to_float32(dtype: torch.dtype) -> None:
    """An integral input gives a float32 output, as ``torch.reciprocal`` promotes it."""
    n_total = 4096
    # Nonzero divisors within the dtype's range.
    if dtype == torch.uint8:
        x = torch.randint(1, 256, (n_total,), device=run_device(), dtype=dtype)
    elif dtype == torch.int8:
        x = torch.randint(-127, 128, (n_total,), device=run_device(), dtype=dtype)
        x = torch.where(x == 0, torch.ones_like(x), x)
    else:
        x = torch.randint(-1000, 1001, (n_total,), device=run_device(), dtype=dtype)
        x = torch.where(x == 0, torch.ones_like(x), x)
    op = ReciprocalFwdOp()
    out = op(x)
    assert out.dtype == torch.float32
    ref = torch.reciprocal(x)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (x,)).verification(*(x,)))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.int8, torch.int64])
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
    """One instance serves integer and float inputs, and rejects a dtype outside the union."""
    op = ReciprocalFwdOp()
    assert op(torch.ones(4, device=run_device(), dtype=torch.float32)).dtype == torch.float32
    assert op(torch.ones(4, device=run_device(), dtype=torch.int32)).dtype == torch.float32
    with pytest.raises(ValueError, match="dtype"):
        op(torch.ones(4, device=run_device(), dtype=torch.float64))
