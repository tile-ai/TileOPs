"""Tests for logical elementwise ops (logical_and, logical_or, logical_not).

Logical ops take boolean inputs and produce boolean outputs.
Covers L1 smoke correctness for binary logical ops, and all supported
dtypes for logical_not.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, exact_compare
from tileops.ops.elementwise import LogicalAndOp, LogicalNotOp, LogicalOrOp

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bool_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact comparison for boolean outputs."""
    assert output.dtype == torch.bool, f"Expected bool dtype, got {output.dtype}"
    assert torch.equal(output, output_ref), (
        f"Bool mismatch: {(output != output_ref).sum().item()} elements differ"
    )


class LogicalTest(TestBase):
    """Reusable test body for logical ops."""

    def __init__(self, n_total: int, dtype: torch.dtype, ref_fn):
        self.n_total = n_total
        self.dtype = dtype
        self.ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Comparison ops produce bool inputs for logical ops
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda") > 0
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda") > 0
        # Cast to the input dtype (bool stored as float for kernel compatibility)
        a = a.to(self.dtype)
        b = b.to(self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.ref_fn(a.bool(), b.bool())


# ---------------------------------------------------------------------------
# LogicalAnd op
# ---------------------------------------------------------------------------


class LogicalAndFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_000_000, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_000_000, torch.float32, marks=pytest.mark.full),
        ]),
    ]


@LogicalAndFixture
def test_logical_and_op(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalTest(n_total, dtype, torch.logical_and)
    shape = (n_total,)
    op = LogicalAndOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# LogicalOr op
# ---------------------------------------------------------------------------


class LogicalOrFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_000_000, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_000_000, torch.float32, marks=pytest.mark.full),
        ]),
    ]


@LogicalOrFixture
def test_logical_or_op(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalTest(n_total, dtype, torch.logical_or)
    shape = (n_total,)
    op = LogicalOrOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# LogicalNot op
# ---------------------------------------------------------------------------


class LogicalFixture(FixtureBase):
    """Parametrize over supported dtypes for logical_not."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.float32, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.bool, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.uint8, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int8, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int16, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int32, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int64, marks=pytest.mark.full),
        ]),
    ]


class LogicalNotTest(TestBase):
    """Test harness for logical_not."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device="cuda") > 0.5
            return (x,)

        if self.dtype == torch.uint8:
            x = torch.randint(0, 8, (self.n_total,), device="cuda", dtype=self.dtype)
        elif self.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            x = torch.randint(-4, 4, (self.n_total,), device="cuda", dtype=self.dtype)
        else:
            x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)

        mask = torch.rand(self.n_total, device="cuda") > 0.5
        x[mask] = 0
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_not(x)


@LogicalFixture
def test_logical_not(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalNotTest(n_total, dtype)
    op = LogicalNotOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=exact_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
