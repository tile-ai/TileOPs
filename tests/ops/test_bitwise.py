"""Tests for bitwise elementwise ops (bitwise_and, bitwise_or, bitwise_xor, bitwise_not).

Bitwise ops operate on integer inputs. We use int32 tensors for testing
binary bitwise ops, and all bool/integer dtypes for bitwise_not.
Covers L1 smoke correctness.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, exact_compare
from tileops.ops.elementwise import BitwiseAndOp, BitwiseNotOp, BitwiseOrOp, BitwiseXorOp

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


# ---------------------------------------------------------------------------
# BitwiseNot op
# ---------------------------------------------------------------------------


class BitwiseFixture(FixtureBase):
    """Parametrize over torch-supported bitwise_not dtypes."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.bool, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.uint8, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int8, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int16, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int32, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.int64, marks=pytest.mark.full),
        ]),
    ]


class BitwiseNotTest(TestBase):
    """Test harness for bitwise_not."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device="cuda") > 0.5
        elif self.dtype == torch.uint8:
            x = torch.randint(0, 256, (self.n_total,), device="cuda", dtype=self.dtype)
        else:
            x = torch.randint(-128, 128, (self.n_total,), device="cuda", dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_not(x)


@BitwiseFixture
def test_bitwise_not(n_total: int, dtype: torch.dtype) -> None:
    test = BitwiseNotTest(n_total, dtype)
    op = BitwiseNotOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=exact_compare)


@pytest.mark.parametrize("dtype", [
    pytest.param(torch.float16, marks=pytest.mark.smoke),
    pytest.param(torch.bfloat16, marks=pytest.mark.full),
    pytest.param(torch.float32, marks=pytest.mark.full),
])
def test_bitwise_not_rejects_float_dtype(dtype: torch.dtype) -> None:
    from tileops.kernels.elementwise import BitwiseNotKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseNotKernel(N_total=16, dtype=dtype)


# ---------------------------------------------------------------------------
# Dtype rejection tests for binary bitwise ops
# ---------------------------------------------------------------------------


class BitwiseBinaryRejectFixture(FixtureBase):
    PARAMS = [
        ("op_cls, dtype", [
            pytest.param(BitwiseAndOp, torch.float16, marks=pytest.mark.smoke),
            pytest.param(BitwiseOrOp, torch.float16, marks=pytest.mark.full),
            pytest.param(BitwiseXorOp, torch.float16, marks=pytest.mark.full),
            pytest.param(BitwiseAndOp, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(BitwiseAndOp, torch.float32, marks=pytest.mark.full),
        ]),
    ]


@BitwiseBinaryRejectFixture
def test_bitwise_binary_rejects_float_dtype(op_cls, dtype: torch.dtype) -> None:
    """Binary bitwise ops only support integer dtypes; floats must be rejected."""
    shape = (16,)
    with pytest.raises(ValueError, match="does not support dtype"):
        op_cls(a_shape=shape, b_shape=shape, dtype=dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
