"""Tests for logical elementwise ops (logical_and, logical_or, logical_not).

Logical ops under test accept numeric tensors, interpret non-zero values
as True, and produce boolean outputs. Covers L1 smoke correctness for
binary logical ops, and all supported dtypes for logical_not.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, exact_compare
from tileops.ops.elementwise import LogicalAndFwdOp, LogicalNotFwdOp, LogicalOrFwdOp

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
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda") > 0
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda") > 0
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
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@LogicalAndFixture
def test_logical_and_op(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalTest(n_total, dtype, torch.logical_and)
    shape = (n_total,)
    op = LogicalAndFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# LogicalOr op
# ---------------------------------------------------------------------------


class LogicalOrFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@LogicalOrFixture
def test_logical_or_op(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalTest(n_total, dtype, torch.logical_or)
    shape = (n_total,)
    op = LogicalOrFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Broadcast pattern tests for binary logical ops (L3)
# ---------------------------------------------------------------------------

_BROADCAST_PATTERNS = [
    ((2, 64, 128), (1, 1, 128)),   # bias-add
    ((2, 64, 128), (2, 64, 1)),    # row broadcast
    ((64, 128), (1, 1)),           # scalar broadcast
]

_LOGICAL_OPS = [
    ("logical_and", LogicalAndFwdOp, torch.logical_and),
    ("logical_or", LogicalOrFwdOp, torch.logical_or),
]


class LogicalBroadcastFixture(FixtureBase):
    PARAMS = [
        ("op_name, op_cls, ref_fn, a_shape, b_shape", [
            pytest.param(name, cls, ref, a_s, b_s,
                         marks=pytest.mark.full)
            for j, (name, cls, ref) in enumerate(_LOGICAL_OPS)
            for i, (a_s, b_s) in enumerate(_BROADCAST_PATTERNS)
        ]),
    ]


@LogicalBroadcastFixture
def test_logical_broadcast(
    op_name, op_cls, ref_fn, a_shape, b_shape,
) -> None:
    dtype = torch.float16
    a = (torch.randn(*a_shape, dtype=dtype, device="cuda") > 0).to(dtype)
    b = (torch.randn(*b_shape, dtype=dtype, device="cuda") > 0).to(dtype)
    op = op_cls(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    ref = ref_fn(a.bool(), b.bool())
    with torch.no_grad():
        out = op(a, b)
    _bool_compare(out, ref)


# ---------------------------------------------------------------------------
# LogicalNot op
# ---------------------------------------------------------------------------


class LogicalFixture(FixtureBase):
    """Parametrize over supported dtypes for logical_not."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bool, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.uint8, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int8, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int32, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int64, marks=pytest.mark.smoke),
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
    op = LogicalNotFwdOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=exact_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
