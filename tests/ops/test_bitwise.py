"""Tests for bitwise elementwise ops (bitwise_and, bitwise_or, bitwise_xor).

Bitwise ops operate on integer inputs. We use int32 tensors for testing.
Covers L1 smoke correctness.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import BitwiseAndOp, BitwiseOrOp, BitwiseXorOp

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _exact_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact comparison for integer outputs."""
    assert torch.equal(output, output_ref), (
        f"Mismatch: {(output != output_ref).sum().item()} elements differ"
    )


class BitwiseTest(TestBase):
    """Reusable test body for bitwise ops."""

    def __init__(self, n_total: int, ref_fn):
        self.n_total = n_total
        self.dtype = torch.int32
        self.ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(-1000, 1000, (self.n_total,), dtype=torch.int32, device="cuda")
        b = torch.randint(-1000, 1000, (self.n_total,), dtype=torch.int32, device="cuda")
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.ref_fn(a, b)


# ---------------------------------------------------------------------------
# BitwiseAnd op
# ---------------------------------------------------------------------------


class BitwiseAndFixture(FixtureBase):
    PARAMS = [
        ("n_total", [
            pytest.param(1_000_000, marks=pytest.mark.smoke),
            pytest.param(4_000_000, marks=pytest.mark.full),
        ]),
    ]


@BitwiseAndFixture
def test_bitwise_and_op(n_total: int) -> None:
    test = BitwiseTest(n_total, torch.bitwise_and)
    shape = (n_total,)
    op = BitwiseAndOp(a_shape=shape, b_shape=shape, dtype=torch.int32)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


# ---------------------------------------------------------------------------
# BitwiseOr op
# ---------------------------------------------------------------------------


class BitwiseOrFixture(FixtureBase):
    PARAMS = [
        ("n_total", [
            pytest.param(1_000_000, marks=pytest.mark.smoke),
            pytest.param(4_000_000, marks=pytest.mark.full),
        ]),
    ]


@BitwiseOrFixture
def test_bitwise_or_op(n_total: int) -> None:
    test = BitwiseTest(n_total, torch.bitwise_or)
    shape = (n_total,)
    op = BitwiseOrOp(a_shape=shape, b_shape=shape, dtype=torch.int32)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


# ---------------------------------------------------------------------------
# BitwiseXor op
# ---------------------------------------------------------------------------


class BitwiseXorFixture(FixtureBase):
    PARAMS = [
        ("n_total", [
            pytest.param(1_000_000, marks=pytest.mark.smoke),
            pytest.param(4_000_000, marks=pytest.mark.full),
        ]),
    ]


@BitwiseXorFixture
def test_bitwise_xor_op(n_total: int) -> None:
    test = BitwiseTest(n_total, torch.bitwise_xor)
    shape = (n_total,)
    op = BitwiseXorOp(a_shape=shape, b_shape=shape, dtype=torch.int32)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
