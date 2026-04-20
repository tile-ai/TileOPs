"""Tests for comparison elementwise ops (eq, ne, gt, lt, ge, le).

All comparison ops output torch.bool. Covers L1 smoke correctness
and L4 edge case for eq.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import EqFwdOp, GeFwdOp, GtFwdOp, LeFwdOp, LtFwdOp, NeFwdOp

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bool_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact comparison for boolean outputs."""
    assert output.dtype == torch.bool, f"Expected bool dtype, got {output.dtype}"
    assert torch.equal(output, output_ref), (
        f"Bool mismatch: {(output != output_ref).sum().item()} elements differ"
    )


class ComparisonTest(TestBase):
    """Reusable test body for comparison ops."""

    def __init__(self, n_total: int, dtype: torch.dtype, ref_fn):
        self.n_total = n_total
        self.dtype = dtype
        self.ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.ref_fn(a, b)


# ---------------------------------------------------------------------------
# Eq op
# ---------------------------------------------------------------------------


class EqFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@EqFixture
def test_eq_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.eq)
    shape = (n_total,)
    op = EqFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Ne op
# ---------------------------------------------------------------------------


class NeFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@NeFixture
def test_ne_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.ne)
    shape = (n_total,)
    op = NeFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Gt op
# ---------------------------------------------------------------------------


class GtFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@GtFixture
def test_gt_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.gt)
    shape = (n_total,)
    op = GtFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Lt op
# ---------------------------------------------------------------------------


class LtFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@LtFixture
def test_lt_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.lt)
    shape = (n_total,)
    op = LtFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Ge op
# ---------------------------------------------------------------------------


class GeFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@GeFixture
def test_ge_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.ge)
    shape = (n_total,)
    op = GeFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Le op
# ---------------------------------------------------------------------------


class LeFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@LeFixture
def test_le_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.le)
    shape = (n_total,)
    op = LeFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Dtype rejection tests
# ---------------------------------------------------------------------------


class ComparisonRejectFixture(FixtureBase):
    PARAMS = [
        ("op_cls, dtype", [
            pytest.param(EqFwdOp, torch.int32, marks=pytest.mark.smoke),
            pytest.param(EqFwdOp, torch.int64, marks=pytest.mark.smoke),
        ]),
    ]


@ComparisonRejectFixture
def test_comparison_rejects_integer_dtype(op_cls, dtype: torch.dtype) -> None:
    """Comparison ops only support float dtypes; integers must be rejected."""
    shape = (16,)
    with pytest.raises(ValueError, match="does not support dtype"):
        op_cls(a_shape=shape, b_shape=shape, dtype=dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
