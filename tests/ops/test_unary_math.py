"""Tests for unary math elementwise ops (17 ops).

Covers L1 smoke correctness (fp16, 1M) and L4 edge cases (fp32, 4K).
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import (
    AbsOp,
    CeilOp,
    CosOp,
    ErfOp,
    Expm1Op,
    ExpOp,
    FloorOp,
    Log1pOp,
    LogOp,
    NegOp,
    ReciprocalOp,
    RoundOp,
    RsqrtOp,
    SignOp,
    SinOp,
    SqrtOp,
    TruncOp,
)


class MathFixture(FixtureBase):
    """Parametrize over shapes / dtypes for math ops."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.full),
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


def _randn(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(n, device="cuda", dtype=dtype)


def _positive(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(n, device="cuda", dtype=dtype).clamp(min=0.01) + 0.01


def _nonzero(n: int, dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(n, device="cuda", dtype=dtype)
    return x + torch.sign(x) * 0.01


def _make_math_test(n_total, dtype, gen_fn, ref_fn, op_cls):
    """Build test, instantiate op, and run check."""
    test = UnaryMathTest(n_total, dtype, gen_fn=gen_fn, ref_fn=ref_fn)
    op = op_cls(N_total=n_total, dtype=dtype)
    tol = {"atol": 1e-3, "rtol": 1e-3} if dtype == torch.float16 else {"atol": 1e-5, "rtol": 1e-5}
    test.check(op, *test.gen_inputs(), **tol)


# ---------------------------------------------------------------------------
# L1 tests (17 ops, fp16, 1M)
# ---------------------------------------------------------------------------


@MathFixture
def test_exp(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.exp, ExpOp)


@MathFixture
def test_log(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _positive, torch.log, LogOp)


@MathFixture
def test_sqrt(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _positive, torch.sqrt, SqrtOp)


@MathFixture
def test_rsqrt(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _positive, torch.rsqrt, RsqrtOp)


@MathFixture
def test_abs(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.abs, AbsOp)


@MathFixture
def test_neg(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.neg, NegOp)


@MathFixture
def test_reciprocal(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _nonzero, torch.reciprocal, ReciprocalOp)


@MathFixture
def test_sign(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.sign, SignOp)


@MathFixture
def test_sin(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.sin, SinOp)


@MathFixture
def test_cos(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.cos, CosOp)


@MathFixture
def test_floor(n_total: int, dtype: torch.dtype) -> None:
    # Kernel casts to fp32 before floor; reference must match
    _make_math_test(n_total, dtype, _randn,
                    lambda x: torch.floor(x.float()).to(x.dtype), FloorOp)


@MathFixture
def test_ceil(n_total: int, dtype: torch.dtype) -> None:
    # Kernel casts to fp32 before ceil; reference must match
    _make_math_test(n_total, dtype, _randn,
                    lambda x: torch.ceil(x.float()).to(x.dtype), CeilOp)


@MathFixture
def test_round(n_total: int, dtype: torch.dtype) -> None:
    # Kernel uses nearbyint(fp32(x)) for banker's rounding; reference must match
    _make_math_test(n_total, dtype, _randn,
                    lambda x: torch.round(x.float()).to(x.dtype), RoundOp)


@MathFixture
def test_trunc(n_total: int, dtype: torch.dtype) -> None:
    # Kernel casts to fp32 before trunc; reference must match
    _make_math_test(n_total, dtype, _randn,
                    lambda x: torch.trunc(x.float()).to(x.dtype), TruncOp)


@MathFixture
def test_erf(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.erf, ErfOp)


@MathFixture
def test_log1p(n_total: int, dtype: torch.dtype) -> None:
    def _gen(n, dtype):
        return torch.rand(n, device="cuda", dtype=dtype).clamp(min=0.01)

    _make_math_test(n_total, dtype, _gen, torch.log1p, Log1pOp)


@MathFixture
def test_expm1(n_total: int, dtype: torch.dtype) -> None:
    _make_math_test(n_total, dtype, _randn, torch.expm1, Expm1Op)


# ---------------------------------------------------------------------------
# L4 edge-case tests (fp32, 4K)
# ---------------------------------------------------------------------------


@MathEdgeFixture
def test_exp_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: exp of zeros should give all ones."""
    def _zeros(n, dtype):
        return torch.zeros(n, device="cuda", dtype=dtype)

    _make_math_test(n_total, dtype, _zeros, torch.exp, ExpOp)


@MathEdgeFixture
def test_log_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: log of ones should give all zeros."""
    def _ones(n, dtype):
        return torch.ones(n, device="cuda", dtype=dtype)

    _make_math_test(n_total, dtype, _ones, torch.log, LogOp)


@MathEdgeFixture
def test_abs_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: abs of negative inputs."""
    def _neg(n, dtype):
        return -torch.rand(n, device="cuda", dtype=dtype).clamp(min=0.01)

    _make_math_test(n_total, dtype, _neg, torch.abs, AbsOp)


@MathEdgeFixture
def test_sign_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: sign with mix of -inf, 0, +inf."""
    def _extreme(n, dtype):
        x = torch.zeros(n, device="cuda", dtype=dtype)
        x[:n // 3] = float("-inf")
        x[n // 3: 2 * n // 3] = 0.0
        x[2 * n // 3:] = float("inf")
        return x

    _make_math_test(n_total, dtype, _extreme, torch.sign, SignOp)


@MathEdgeFixture
def test_neg_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: neg of zeros should give zeros."""
    def _zeros(n, dtype):
        return torch.zeros(n, device="cuda", dtype=dtype)

    _make_math_test(n_total, dtype, _zeros, torch.neg, NegOp)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
