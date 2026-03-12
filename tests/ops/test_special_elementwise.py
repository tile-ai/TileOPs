"""Tests for special predicate elementwise ops (isnan, isinf, isfinite).

Covers L1 smoke correctness (fp16, 1M) and L4 edge cases (fp32, 4K).
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import IsfiniteOp, IsinfOp, IsnanOp


class SpecialFixture(FixtureBase):
    """Parametrize over shapes / dtypes for special predicate ops."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class SpecialEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


class IsnanTest(TestBase):
    """Test harness for isnan."""

    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._gen_fn = gen_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        # Inject ~10% NaNs
        mask = torch.rand(self.n_total, device="cuda") < 0.1
        x[mask] = float("nan")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isnan(x), torch.ones_like(x), torch.zeros_like(x))


class IsinfTest(TestBase):
    """Test harness for isinf."""

    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._gen_fn = gen_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        # Inject ~10% infs
        mask = torch.rand(self.n_total, device="cuda") < 0.1
        x[mask] = float("inf")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isinf(x), torch.ones_like(x), torch.zeros_like(x))


class IsfiniteTest(TestBase):
    """Test harness for isfinite."""

    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._gen_fn = gen_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        # Inject some NaN and inf
        mask_nan = torch.rand(self.n_total, device="cuda") < 0.05
        mask_inf = torch.rand(self.n_total, device="cuda") < 0.05
        x[mask_nan] = float("nan")
        x[mask_inf] = float("inf")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isfinite(x), torch.ones_like(x), torch.zeros_like(x))


@SpecialFixture
def test_isnan(n_total: int, dtype: torch.dtype) -> None:
    test = IsnanTest(n_total, dtype)
    op = IsnanOp(N_total=n_total, dtype=dtype)
    tol = {"atol": 1e-3, "rtol": 1e-3} if dtype == torch.float16 else {"atol": 1e-5, "rtol": 1e-5}
    test.check(op, *test.gen_inputs(), **tol)


@SpecialFixture
def test_isinf(n_total: int, dtype: torch.dtype) -> None:
    test = IsinfTest(n_total, dtype)
    op = IsinfOp(N_total=n_total, dtype=dtype)
    tol = {"atol": 1e-3, "rtol": 1e-3} if dtype == torch.float16 else {"atol": 1e-5, "rtol": 1e-5}
    test.check(op, *test.gen_inputs(), **tol)


@SpecialFixture
def test_isfinite(n_total: int, dtype: torch.dtype) -> None:
    test = IsfiniteTest(n_total, dtype)
    op = IsfiniteOp(N_total=n_total, dtype=dtype)
    tol = {"atol": 1e-3, "rtol": 1e-3} if dtype == torch.float16 else {"atol": 1e-5, "rtol": 1e-5}
    test.check(op, *test.gen_inputs(), **tol)


# ---------------------------------------------------------------------------
# L4 edge-case tests (fp32, 4K)
# ---------------------------------------------------------------------------


@SpecialEdgeFixture
def test_isnan_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: all NaN input."""
    def _all_nan(n, dtype):
        return torch.full((n,), float("nan"), device="cuda", dtype=dtype)

    test = IsnanTest(n_total, dtype, gen_fn=_all_nan)
    op = IsnanOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), atol=1e-5, rtol=1e-5)


@SpecialEdgeFixture
def test_isinf_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: mix of +inf and -inf."""
    def _all_inf(n, dtype):
        x = torch.full((n,), float("inf"), device="cuda", dtype=dtype)
        x[:n // 2] = float("-inf")
        return x

    test = IsinfTest(n_total, dtype, gen_fn=_all_inf)
    op = IsinfOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), atol=1e-5, rtol=1e-5)


@SpecialEdgeFixture
def test_isfinite_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: all finite input -- should return all 1.0."""
    def _all_finite(n, dtype):
        return torch.randn(n, device="cuda", dtype=dtype)

    test = IsfiniteTest(n_total, dtype, gen_fn=_all_finite)
    op = IsfiniteOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
