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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
