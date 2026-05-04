"""Tests for unary math elementwise ops (17 ops).

Covers L1 correctness across supported float dtypes and
L4 edge cases for numerically sensitive ops.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import (
    AbsFwdOp,
    CeilFwdOp,
    CosFwdOp,
    ErfFwdOp,
    ExpFwdOp,
    Expm1FwdOp,
    FloorFwdOp,
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


class MathFixture(FixtureBase):
    """Parametrize over supported float dtypes for unary math ops."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


class MathEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


class UnaryMathTest(TestBase):
    """Generic test harness for a single-input, single-output unary op."""

    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None, ref_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._gen_fn = gen_fn
        self._ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        return (torch.randn(self.n_total, device="cuda", dtype=self.dtype),)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return self._ref_fn(x)


def _get_tolerances(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.float16:
        return {"atol": 1e-3, "rtol": 1e-3}
    if dtype == torch.bfloat16:
        return {"atol": 1.6e-2, "rtol": 1.6e-2}
    return {"atol": 1e-5, "rtol": 1e-5}


def _randn(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(n, device="cuda", dtype=dtype)


def _positive(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(n, device="cuda", dtype=dtype).clamp(min=0.01) + 0.01


def _nonzero(n: int, dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(n, device="cuda", dtype=dtype)
    return x + torch.sign(x) * 0.01


def _repeat_values(values: list[float], n: int, dtype: torch.dtype) -> torch.Tensor:
    base = torch.tensor(values, device="cuda", dtype=dtype)
    repeats = (n + len(values) - 1) // len(values)
    return base.repeat(repeats)[:n]


def _make_math_test(n_total, dtype, gen_fn, ref_fn, op_cls):
    """Build test, instantiate op, and run check."""
    test = UnaryMathTest(n_total, dtype, gen_fn=gen_fn, ref_fn=ref_fn)
    op = op_cls(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), **_get_tolerances(dtype))


# ---------------------------------------------------------------------------
# L1 tests (17 ops)
# ---------------------------------------------------------------------------


@MathFixture
def test_exp(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.exp, ExpFwdOp)


@MathFixture
def test_log(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _positive, torch.log, LogFwdOp)


@MathFixture
def test_sqrt(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _positive, torch.sqrt, SqrtFwdOp)


@MathFixture
def test_rsqrt(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _positive, torch.rsqrt, RsqrtFwdOp)


@MathFixture
def test_abs(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.abs, AbsFwdOp)


@MathFixture
def test_neg(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.neg, NegFwdOp)


@MathFixture
def test_reciprocal(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _nonzero, torch.reciprocal, ReciprocalFwdOp)


@MathFixture
def test_sign(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.sign, SignFwdOp)


@MathFixture
def test_sin(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.sin, SinFwdOp)


@MathFixture
def test_cos(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.cos, CosFwdOp)


@MathFixture
def test_floor(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        _randn,
        lambda x: torch.floor(x.float()).to(x.dtype),
        FloorFwdOp,
    )


@MathFixture
def test_ceil(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        _randn,
        lambda x: torch.ceil(x.float()).to(x.dtype),
        CeilFwdOp,
    )


@MathFixture
def test_round(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        _randn,
        lambda x: torch.round(x.float()).to(x.dtype),
        RoundFwdOp,
    )


@MathFixture
def test_trunc(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        _randn,
        lambda x: torch.trunc(x.float()).to(x.dtype),
        TruncFwdOp,
    )


@MathFixture
def test_erf(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.erf, ErfFwdOp)


@MathFixture
def test_log1p(n_total: int, dtype: torch.dtype) -> None:
    def _gen(n, gen_dtype):
        return torch.rand(n, device="cuda", dtype=gen_dtype).clamp(min=0.01)

    _make_math_test(n_total, dtype, _gen, torch.log1p, Log1pFwdOp)


@MathFixture
def test_expm1(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.expm1, Expm1FwdOp)


@pytest.mark.smoke
def test_math_ops_reject_non_float_dtype() -> None:
    from tileops.kernels.elementwise import ExpFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        ExpFwdKernel(N_total=16, dtype=torch.int32)


# ---------------------------------------------------------------------------
# L4 edge-case tests (fp32, 4K)
# ---------------------------------------------------------------------------


@MathEdgeFixture
def test_sqrt_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([-1.0, 0.0, 1e-38, 1.0], n, d),
        torch.sqrt,
        SqrtFwdOp,
    )


@MathEdgeFixture
def test_rsqrt_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([-1.0, 0.0, 1e-38, 1.0], n, d),
        torch.rsqrt,
        RsqrtFwdOp,
    )


@MathEdgeFixture
def test_log_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([-1.0, 0.0, 1e-38, 1.0], n, d),
        torch.log,
        LogFwdOp,
    )


@MathEdgeFixture
def test_log1p_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([-2.0, -1.0, 0.0, 1e-7], n, d),
        torch.log1p,
        Log1pFwdOp,
    )


@MathEdgeFixture
def test_exp_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 88.8, -88.8, 200.0], n, d),
        torch.exp,
        ExpFwdOp,
    )


@MathEdgeFixture
def test_expm1_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 88.8, -88.8, 1e-7], n, d),
        torch.expm1,
        Expm1FwdOp,
    )


@MathEdgeFixture
def test_erf_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 3.0, -3.0, 100.0], n, d),
        torch.erf,
        ErfFwdOp,
    )


@MathEdgeFixture
def test_reciprocal_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 1.0, -1.0, 1e-38], n, d),
        torch.reciprocal,
        ReciprocalFwdOp,
    )


@MathEdgeFixture
def test_sign_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([-5.0, 0.0, 3.0, float("nan")], n, d),
        torch.sign,
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
    n_total = 4096
    # Bound the inputs so |x * 10**decimals| stays well within fp16 range
    # (max ~65504) — relevant for decimals=2 with fp16/bf16.
    x = torch.randn(n_total, device="cuda", dtype=dtype) * 10.0
    op = RoundFwdOp(N_total=n_total, dtype=dtype)
    out = op(x, decimals=decimals)
    ref = torch.round(x.float(), decimals=decimals).to(dtype)
    # Non-zero decimals widens the working magnitude by 10**decimals before
    # rounding; the round-trip cast to the storage dtype between the scale
    # and unscale steps is unavoidable for a kernel that consumes self.dtype,
    # so we relax atol/rtol by 10**|decimals| for low-bit dtypes.
    tol = dict(_get_tolerances(dtype))
    if dtype in (torch.float16, torch.bfloat16) and decimals != 0:
        slack = 10.0 ** abs(decimals)
        tol = {"atol": tol["atol"] * slack, "rtol": tol["rtol"] * slack}
    torch.testing.assert_close(out, ref, **tol)


@pytest.mark.smoke
def test_round_decimals_default_is_zero() -> None:
    """Calling RoundFwdOp without ``decimals`` must round to nearest integer."""
    n_total = 1024
    x = torch.randn(n_total, device="cuda", dtype=torch.float32) * 5.0
    op = RoundFwdOp(N_total=n_total, dtype=torch.float32)
    out = op(x)
    ref = torch.round(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
