"""Tests for binary arithmetic elementwise ops with broadcast.

Covers L1 smoke correctness for sub, mul, div, remainder, pow,
floor_divide, lerp, maximum, minimum (plus existing add).
Also includes L4 edge case tests for div, remainder, floor_divide, pow.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import (
    AddOp,
    DivOp,
    FloorDivideOp,
    LerpOp,
    MaximumOp,
    MinimumOp,
    MulOp,
    PowOp,
    RemainderOp,
    SubOp,
    coalesce_broadcast_dims,
)

# ---------------------------------------------------------------------------
# coalesce_broadcast_dims unit tests
# ---------------------------------------------------------------------------


class CoalesceFixture(FixtureBase):
    PARAMS = [
        ("a_shape, b_shape, expected_ndim", [
            # same-shape: coalesces to 1D
            pytest.param((1024, 1024), (1024, 1024), 1, marks=pytest.mark.smoke),
            # bias-add: (B,S,D) + (1,1,D) -> 2 groups
            pytest.param((2, 512, 768), (1, 1, 768), 2, marks=pytest.mark.full),
            # row broadcast: (B,S,D) + (B,S,1) -> 2 groups
            pytest.param((2, 512, 768), (2, 512, 1), 2, marks=pytest.mark.full),
            # scalar: (M,N) + (1,1) -> 2 groups (M*N collapsed, 1 broadcast)
            pytest.param((1024, 1024), (1, 1), 1, marks=pytest.mark.full),
            # interleaved: (A,1,C) + (1,B,1) -> 3 groups
            pytest.param((4, 1, 8), (1, 8, 1), 3, marks=pytest.mark.full),
            # outer product: (M,1) + (1,N) -> 2 groups
            pytest.param((64, 1), (1, 128), 2, marks=pytest.mark.full),
            # non-broadcast size-1: (2,1,3) + (2,1,3) -> 1 (all contiguous)
            pytest.param((2, 1, 3), (2, 1, 3), 1, marks=pytest.mark.full),
            # scalar (0-dim) input: () + (4,) -> 1
            pytest.param((), (4,), 1, marks=pytest.mark.full),
        ]),
    ]


@CoalesceFixture
def test_coalesce_broadcast_dims(a_shape, b_shape, expected_ndim) -> None:
    """Verify coalesce output shape count matches expected coalesced ndim."""
    out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(
        a_shape, b_shape,
    )
    # Verify output shape matches torch broadcast
    assert out_shape == torch.broadcast_shapes(a_shape, b_shape)
    # Verify coalesced ndim
    assert len(coalesced_shape) == expected_ndim, (
        f"Expected {expected_ndim} coalesced dims, got {len(coalesced_shape)}: "
        f"{coalesced_shape}"
    )
    # Verify strides have correct length
    assert len(a_strides) == len(coalesced_shape)
    assert len(b_strides) == len(coalesced_shape)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


# ---------------------------------------------------------------------------
# Add op correctness tests
# ---------------------------------------------------------------------------


class AddSameShapeFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(16_384, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
            pytest.param(16_384, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class AddSameShapeTest(TestBase):

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() + b.float()).to(a.dtype)


@AddSameShapeFixture
def test_add_same_shape(n_total: int, dtype: torch.dtype) -> None:
    test = AddSameShapeTest(n_total, dtype)
    shape = (n_total,)
    op = AddOp(a_shape=shape, b_shape=shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Broadcast pattern tests (L3)
# ---------------------------------------------------------------------------


class AddBroadcastFixture(FixtureBase):
    PARAMS = [
        ("a_shape, b_shape, dtype", [
            pytest.param(
                (2, 512, 768), (1, 1, 768), torch.float16, marks=pytest.mark.smoke,
            ),
            pytest.param(
                (2, 512, 768), (2, 512, 1), torch.float16, marks=pytest.mark.full,
            ),
            pytest.param(
                (1024, 1024), (1, 1), torch.float16, marks=pytest.mark.full,
            ),
            pytest.param(
                (4, 1, 8), (1, 8, 1), torch.float16, marks=pytest.mark.full,
            ),
        ]),
    ]


class AddBroadcastTest(TestBase):

    def __init__(self, a_shape: tuple, b_shape: tuple, dtype: torch.dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.a_shape, dtype=self.dtype, device="cuda")
        b = torch.randn(self.b_shape, dtype=self.dtype, device="cuda")
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() + b.float()).to(a.dtype)


@AddBroadcastFixture
def test_add_broadcast(a_shape, b_shape, dtype: torch.dtype) -> None:
    test = AddBroadcastTest(a_shape, b_shape, dtype)
    op = AddOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class AddStrategyFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype, strategy", [
            pytest.param(4_096, torch.float16, "direct", marks=pytest.mark.smoke),
            pytest.param(16_384, torch.float16, "explicit_parallel", marks=pytest.mark.full),
        ]),
    ]


@AddStrategyFixture
def test_add_strategies(n_total: int, dtype: torch.dtype, strategy: str) -> None:
    """Verify both binary strategies produce correct results."""
    test = AddSameShapeTest(n_total, dtype)
    shape = (n_total,)
    op = AddOp(a_shape=shape, b_shape=shape, dtype=dtype, strategy=strategy)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Generic binary test helper
# ---------------------------------------------------------------------------


class BinarySameShapeTest(TestBase):
    """Reusable test body for binary same-shape ops."""

    def __init__(self, n_total: int, dtype: torch.dtype, ref_fn):
        self.n_total = n_total
        self.dtype = dtype
        self.ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.ref_fn(a.float(), b.float()).to(a.dtype)


class BinaryPositiveTest(TestBase):
    """Test body for ops that need positive inputs (div, remainder, pow, etc.)."""

    def __init__(self, n_total: int, dtype: torch.dtype, ref_fn):
        self.n_total = n_total
        self.dtype = dtype
        self.ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        b = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.ref_fn(a.float(), b.float()).to(a.dtype)


# ---------------------------------------------------------------------------
# Sub op
# ---------------------------------------------------------------------------


class SubFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


@SubFixture
def test_sub_op(n_total: int, dtype: torch.dtype) -> None:
    test = BinarySameShapeTest(n_total, dtype, lambda a, b: a - b)
    shape = (n_total,)
    op = SubOp(a_shape=shape, b_shape=shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Mul op
# ---------------------------------------------------------------------------


class MulFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


@MulFixture
def test_mul_op(n_total: int, dtype: torch.dtype) -> None:
    test = BinarySameShapeTest(n_total, dtype, lambda a, b: a * b)
    shape = (n_total,)
    op = MulOp(a_shape=shape, b_shape=shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Div op
# ---------------------------------------------------------------------------


class DivFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


@DivFixture
def test_div_op(n_total: int, dtype: torch.dtype) -> None:
    test = BinaryPositiveTest(n_total, dtype, lambda a, b: a / b)
    shape = (n_total,)
    op = DivOp(a_shape=shape, b_shape=shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Remainder op
# ---------------------------------------------------------------------------


class RemainderFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


class RemainderTest(TestBase):
    """Remainder is computed in native dtype; reference must match."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        b = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Compute reference in same dtype as the kernel
        return a - torch.floor(a / b) * b


@RemainderFixture
def test_remainder_op(n_total: int, dtype: torch.dtype) -> None:
    test = RemainderTest(n_total, dtype)
    shape = (n_total,)
    op = RemainderOp(a_shape=shape, b_shape=shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Pow op
# ---------------------------------------------------------------------------


class PowFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


class PowPositiveTest(TestBase):
    """Pow needs positive base and small exponent to avoid overflow in fp16."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.5
        b = torch.rand(self.n_total, dtype=self.dtype, device="cuda") * 2.0
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.pow(a.float(), b.float()).to(a.dtype)


@PowFixture
def test_pow_op(n_total: int, dtype: torch.dtype) -> None:
    test = PowPositiveTest(n_total, dtype)
    shape = (n_total,)
    op = PowOp(a_shape=shape, b_shape=shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# FloorDivide op
# ---------------------------------------------------------------------------


class FloorDivideFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


class FloorDivideTest(TestBase):
    """Floor divide is computed in native dtype; allow +-1 rounding differences."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        b = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Compute reference in the same dtype as the kernel to match floor behaviour
        return torch.floor(a / b)


@FloorDivideFixture
def test_floor_divide_op(n_total: int, dtype: torch.dtype) -> None:
    test = FloorDivideTest(n_total, dtype)
    shape = (n_total,)
    op = FloorDivideOp(a_shape=shape, b_shape=shape, dtype=dtype)
    # Floor divide in reduced precision can differ by 1; use atol=1.0
    atol = 1.0 if dtype != torch.float32 else 1e-5
    test.check(op, *test.gen_inputs(), atol=atol, rtol=0.0)


# ---------------------------------------------------------------------------
# Lerp op (ternary in PyTorch; compile-time weight=0.5)
# ---------------------------------------------------------------------------


class LerpFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


class LerpTest(TestBase):

    def __init__(self, n_total: int, dtype: torch.dtype, weight: float = 0.5):
        self.n_total = n_total
        self.dtype = dtype
        self.weight = weight

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.lerp(a.float(), b.float(), self.weight).to(a.dtype)


@LerpFixture
def test_lerp_op(n_total: int, dtype: torch.dtype) -> None:
    """Validate lerp across multiple construction-time weight values."""
    # Lerp computes a + w*(b-a) in native dtype; the intermediate multiply
    # adds rounding error proportional to weight magnitude in fp16.
    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 5e-3, 5e-3
    else:  # bfloat16
        atol, rtol = 1.6e-2, 1.6e-2
    for weight in [0.0, 0.3, 0.5, 0.7, 1.0]:
        test = LerpTest(n_total, dtype, weight=weight)
        shape = (n_total,)
        op = LerpOp(a_shape=shape, b_shape=shape, dtype=dtype, weight=weight)
        test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Maximum op
# ---------------------------------------------------------------------------


class MaximumFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


@MaximumFixture
def test_maximum_op(n_total: int, dtype: torch.dtype) -> None:
    test = BinarySameShapeTest(n_total, dtype, lambda a, b: torch.maximum(a, b))
    shape = (n_total,)
    op = MaximumOp(a_shape=shape, b_shape=shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Minimum op
# ---------------------------------------------------------------------------


class MinimumFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


@MinimumFixture
def test_minimum_op(n_total: int, dtype: torch.dtype) -> None:
    test = BinarySameShapeTest(n_total, dtype, lambda a, b: torch.minimum(a, b))
    shape = (n_total,)
    op = MinimumOp(a_shape=shape, b_shape=shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Maximum/Minimum NaN propagation tests
# ---------------------------------------------------------------------------


class MaxMinNanFixture(FixtureBase):
    PARAMS = [
        ("dtype", [
            pytest.param(torch.float32, marks=pytest.mark.smoke),
            pytest.param(torch.float16, marks=pytest.mark.full),
        ]),
    ]


@MaxMinNanFixture
def test_maximum_nan_propagation(dtype: torch.dtype) -> None:
    """Verify maximum propagates NaN when either operand is NaN."""
    nan = float("nan")
    a = torch.tensor([nan, 1.0, nan, 2.0], dtype=dtype, device="cuda")
    b = torch.tensor([3.0, nan, nan, 1.0], dtype=dtype, device="cuda")
    shape = (4,)
    op = MaximumOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.maximum(a, b)
    with torch.no_grad():
        out = op(a, b)
    # NaN positions must match: both output and ref should be NaN at same indices
    assert torch.equal(torch.isnan(out), torch.isnan(ref)), (
        f"NaN positions differ: out={out}, ref={ref}"
    )
    # Non-NaN values must match exactly
    mask = ~torch.isnan(ref)
    assert torch.equal(out[mask], ref[mask]), (
        f"Non-NaN values differ: out={out[mask]}, ref={ref[mask]}"
    )


@MaxMinNanFixture
def test_minimum_nan_propagation(dtype: torch.dtype) -> None:
    """Verify minimum propagates NaN when either operand is NaN."""
    nan = float("nan")
    a = torch.tensor([nan, 1.0, nan, 2.0], dtype=dtype, device="cuda")
    b = torch.tensor([3.0, nan, nan, 1.0], dtype=dtype, device="cuda")
    shape = (4,)
    op = MinimumOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.minimum(a, b)
    with torch.no_grad():
        out = op(a, b)
    # NaN positions must match
    assert torch.equal(torch.isnan(out), torch.isnan(ref)), (
        f"NaN positions differ: out={out}, ref={ref}"
    )
    # Non-NaN values must match exactly
    mask = ~torch.isnan(ref)
    assert torch.equal(out[mask], ref[mask]), (
        f"Non-NaN values differ: out={out[mask]}, ref={ref[mask]}"
    )


# ---------------------------------------------------------------------------
# L4 edge case tests (fp32, 4K)
# ---------------------------------------------------------------------------


class EdgeCaseFixture(FixtureBase):
    PARAMS = [
        ("op_cls, ref_fn, gen_fn", [
            # div: avoid div-by-zero
            pytest.param(
                DivOp,
                lambda a, b: a / b,
                lambda n, d: (
                    torch.randn(n, dtype=d, device="cuda"),
                    torch.rand(n, dtype=d, device="cuda") + 0.1,
                ),
                marks=pytest.mark.smoke,
            ),
            # remainder: positive inputs
            pytest.param(
                RemainderOp,
                lambda a, b: a % b,
                lambda n, d: (
                    torch.rand(n, dtype=d, device="cuda") + 0.1,
                    torch.rand(n, dtype=d, device="cuda") + 0.1,
                ),
                marks=pytest.mark.full,
            ),
            # floor_divide: positive inputs
            pytest.param(
                FloorDivideOp,
                lambda a, b: torch.floor(a / b),
                lambda n, d: (
                    torch.rand(n, dtype=d, device="cuda") + 0.1,
                    torch.rand(n, dtype=d, device="cuda") + 0.1,
                ),
                marks=pytest.mark.full,
            ),
            # pow: positive base, small exponent
            pytest.param(
                PowOp,
                lambda a, b: torch.pow(a, b),
                lambda n, d: (
                    torch.rand(n, dtype=d, device="cuda") + 0.5,
                    torch.rand(n, dtype=d, device="cuda") * 2.0,
                ),
                marks=pytest.mark.full,
            ),
            # maximum: mixed sign
            pytest.param(
                MaximumOp,
                lambda a, b: torch.maximum(a, b),
                lambda n, d: (
                    torch.randn(n, dtype=d, device="cuda"),
                    torch.randn(n, dtype=d, device="cuda"),
                ),
                marks=pytest.mark.full,
            ),
        ]),
    ]


@EdgeCaseFixture
def test_binary_arith_edge_cases(op_cls, ref_fn, gen_fn) -> None:
    """L4 edge case tests: fp32, 4K elements."""
    n = 4096
    dtype = torch.float32
    shape = (n,)
    a, b = gen_fn(n, dtype)
    op = op_cls(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
