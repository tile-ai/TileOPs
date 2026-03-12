"""Tests for special predicate elementwise ops (isnan, isinf, isfinite).

Covers L1 smoke correctness (fp16, 1M) and L4 edge cases (fp32, 4K).
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, exact_compare
from tileops.ops.elementwise import IsfiniteOp, IsinfOp, IsnanOp


class SpecialFixture(FixtureBase):
    """Parametrize over shapes / dtypes for special predicate ops."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.float32, marks=pytest.mark.full),
        ]),
    ]


class SpecialEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


class SpecialTest(TestBase):
    """Generic test harness for special predicate ops."""

    def __init__(self, n_total: int, dtype: torch.dtype, ref_fn, gen_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._ref_fn = ref_fn
        self._gen_fn = gen_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        quarter = self.n_total // 4
        x[:quarter] = float("nan")
        x[quarter:2 * quarter] = float("inf")
        x[2 * quarter:3 * quarter] = float("-inf")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return self._ref_fn(x)


def _make_special_test(n_total, dtype, op_cls, ref_fn, gen_fn=None) -> None:
    test = SpecialTest(n_total, dtype, ref_fn=ref_fn, gen_fn=gen_fn)
    op = op_cls(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=exact_compare)


@SpecialFixture
def test_isnan(n_total: int, dtype: torch.dtype) -> None:
    _make_special_test(n_total, dtype, IsnanOp, torch.isnan)


@SpecialFixture
def test_isinf(n_total: int, dtype: torch.dtype) -> None:
    _make_special_test(n_total, dtype, IsinfOp, torch.isinf)


@SpecialFixture
def test_isfinite(n_total: int, dtype: torch.dtype) -> None:
    _make_special_test(n_total, dtype, IsfiniteOp, torch.isfinite)


# ---------------------------------------------------------------------------
# L4 edge-case tests (fp32, 4K)
# ---------------------------------------------------------------------------


@SpecialEdgeFixture
def test_isnan_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: all NaN input."""
    def _all_nan(n, dtype):
        return torch.full((n,), float("nan"), device="cuda", dtype=dtype)

    _make_special_test(n_total, dtype, IsnanOp, torch.isnan, gen_fn=_all_nan)


@SpecialEdgeFixture
def test_isinf_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: mix of +inf and -inf."""
    def _all_inf(n, dtype):
        x = torch.full((n,), float("inf"), device="cuda", dtype=dtype)
        x[:n // 2] = float("-inf")
        return x

    _make_special_test(n_total, dtype, IsinfOp, torch.isinf, gen_fn=_all_inf)


@SpecialEdgeFixture
def test_isfinite_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: all finite input."""
    def _all_finite(n, dtype):
        return torch.randn(n, device="cuda", dtype=dtype)

    _make_special_test(n_total, dtype, IsfiniteOp, torch.isfinite, gen_fn=_all_finite)


@pytest.mark.smoke
def test_special_predicates_reject_non_float_dtype() -> None:
    from tileops.kernels.elementwise import IsnanKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        IsnanKernel(N_total=16, dtype=torch.int32)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
