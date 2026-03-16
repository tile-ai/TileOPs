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


# ---------------------------------------------------------------------------
# Broadcast pattern tests for all binary arith ops (L3)
# ---------------------------------------------------------------------------

# Broadcast patterns: (a_shape, b_shape)
_BROADCAST_PATTERNS = [
    # bias-add: (B,S,D) + (1,1,D)
    ((2, 64, 128), (1, 1, 128)),
    # row broadcast: (B,S,D) + (B,S,1)
    ((2, 64, 128), (2, 64, 1)),
    # scalar broadcast: (M,N) + (1,1)
    ((64, 128), (1, 1)),
]

# (op_name, op_cls, ref_fn, gen_a, gen_b)
_ARITH_BROADCAST_OPS = [
    ("sub", SubOp, lambda a, b: (a.float() - b.float()).to(a.dtype),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda")),
    ("mul", MulOp, lambda a, b: (a.float() * b.float()).to(a.dtype),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda")),
    ("div", DivOp, lambda a, b: (a.float() / b.float()).to(a.dtype),
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1),
    ("remainder", RemainderOp,
     lambda a, b: a - torch.floor(a.float() / b.float()).to(a.dtype) * b,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1),
    ("pow", PowOp, lambda a, b: torch.pow(a.float(), b.float()).to(a.dtype),
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.5,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") * 2.0),
    ("floor_divide", FloorDivideOp,
     lambda a, b: torch.floor(a.float() / b.float()).to(a.dtype),
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1),
    ("lerp", LerpOp, lambda a, b: torch.lerp(a.float(), b.float(), 0.5).to(a.dtype),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda")),
    ("maximum", MaximumOp, lambda a, b: torch.maximum(a.float(), b.float()).to(a.dtype),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda")),
    ("minimum", MinimumOp, lambda a, b: torch.minimum(a.float(), b.float()).to(a.dtype),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda")),
]


class ArithBroadcastFixture(FixtureBase):
    PARAMS = [
        ("op_name, op_cls, ref_fn, gen_a, gen_b, a_shape, b_shape", [
            pytest.param(name, cls, ref, ga, gb, a_s, b_s,
                         marks=pytest.mark.smoke if i == 0 and j == 0
                         else pytest.mark.full)
            for j, (name, cls, ref, ga, gb) in enumerate(_ARITH_BROADCAST_OPS)
            for i, (a_s, b_s) in enumerate(_BROADCAST_PATTERNS)
        ]),
    ]


@ArithBroadcastFixture
def test_binary_arith_broadcast(
    op_name, op_cls, ref_fn, gen_a, gen_b, a_shape, b_shape,
) -> None:
    dtype = torch.float16
    a = gen_a(a_shape, dtype)
    b = gen_b(b_shape, dtype)
    op = op_cls(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    atol, rtol = _get_tolerances(dtype)
    if op_name == "floor_divide":
        atol = 1.0  # floor rounding tolerance
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


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
            pytest.param(16_384, torch.bfloat16, marks=pytest.mark.full),
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
            pytest.param(16_384, torch.bfloat16, marks=pytest.mark.full),
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
            pytest.param(16_384, torch.bfloat16, marks=pytest.mark.full),
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
            pytest.param(16_384, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


class RemainderTest(TestBase):
    """Remainder reference matches the kernel: fp32 division+floor, native multiply-subtract."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        b = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # fp32 division+floor, cast back, native multiply-subtract
        floored = torch.floor(a.float() / b.float()).to(a.dtype)
        return a - floored * b


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
            pytest.param(16_384, torch.bfloat16, marks=pytest.mark.full),
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
            pytest.param(16_384, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(16_384, torch.float32, marks=pytest.mark.full),
        ]),
    ]


class FloorDivideTest(TestBase):
    """Floor divide reference matches the kernel: fp32 division+floor, cast back."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        b = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # fp32 division+floor, cast back to native dtype
        return torch.floor(a.float() / b.float()).to(a.dtype)


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
            pytest.param(16_384, torch.bfloat16, marks=pytest.mark.full),
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
            pytest.param(16_384, torch.bfloat16, marks=pytest.mark.full),
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
            pytest.param(16_384, torch.bfloat16, marks=pytest.mark.full),
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
# Maximum/Minimum signed-zero regression tests
# ---------------------------------------------------------------------------


class SignedZeroFixture(FixtureBase):
    PARAMS = [
        ("dtype", [
            pytest.param(torch.float16, marks=pytest.mark.smoke),
            pytest.param(torch.bfloat16, marks=pytest.mark.full),
            pytest.param(torch.float32, marks=pytest.mark.full),
        ]),
    ]


@SignedZeroFixture
def test_maximum_signed_zero(dtype: torch.dtype) -> None:
    """maximum(+0.0, -0.0) must return +0.0 (IEEE / PyTorch semantics)."""
    pos_zero = torch.tensor(0.0, dtype=dtype, device="cuda")
    neg_zero = torch.tensor(-0.0, dtype=dtype, device="cuda")

    # Both orderings: (+0, -0) and (-0, +0)
    a = torch.stack([pos_zero, neg_zero, pos_zero, neg_zero])
    b = torch.stack([neg_zero, pos_zero, pos_zero, neg_zero])
    shape = (4,)
    op = MaximumOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.maximum(a, b)
    with torch.no_grad():
        out = op(a, b)

    # Value equality
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    # Sign-bit equality: +0 and -0 compare equal but have different sign bits
    out_signbits = torch.signbit(out)
    ref_signbits = torch.signbit(ref)
    assert torch.equal(out_signbits, ref_signbits), (
        f"Signed-zero mismatch: out signs={out_signbits}, ref signs={ref_signbits}"
    )


@SignedZeroFixture
def test_minimum_signed_zero(dtype: torch.dtype) -> None:
    """minimum(-0.0, +0.0) must return -0.0 (IEEE / PyTorch semantics)."""
    pos_zero = torch.tensor(0.0, dtype=dtype, device="cuda")
    neg_zero = torch.tensor(-0.0, dtype=dtype, device="cuda")

    # Both orderings: (-0, +0) and (+0, -0)
    a = torch.stack([neg_zero, pos_zero, neg_zero, pos_zero])
    b = torch.stack([pos_zero, neg_zero, neg_zero, pos_zero])
    shape = (4,)
    op = MinimumOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.minimum(a, b)
    with torch.no_grad():
        out = op(a, b)

    # Value equality
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    # Sign-bit equality
    out_signbits = torch.signbit(out)
    ref_signbits = torch.signbit(ref)
    assert torch.equal(out_signbits, ref_signbits), (
        f"Signed-zero mismatch: out signs={out_signbits}, ref signs={ref_signbits}"
    )


@SignedZeroFixture
def test_maximum_signed_zero_with_nan(dtype: torch.dtype) -> None:
    """Signed-zero fix must not regress NaN propagation."""
    nan = float("nan")
    # Mix of NaN pairs and non-NaN signed-zero pairs so both code paths execute
    a = torch.tensor([nan, 1.0, -0.0, 0.0, nan, 3.0], dtype=dtype, device="cuda")
    b = torch.tensor([1.0, nan, 0.0, -0.0, -0.0, 2.0], dtype=dtype, device="cuda")
    shape = (6,)
    op = MaximumOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.maximum(a, b)
    with torch.no_grad():
        out = op(a, b)
    # NaN positions must match
    assert torch.equal(torch.isnan(out), torch.isnan(ref)), (
        f"NaN positions differ: out={out}, ref={ref}"
    )
    # Non-NaN values must exist and match (including sign bits for zeros)
    mask = ~torch.isnan(ref)
    assert mask.any(), "Test bug: expected some non-NaN reference values"
    torch.testing.assert_close(out[mask], ref[mask], atol=0, rtol=0)
    assert torch.equal(torch.signbit(out[mask]), torch.signbit(ref[mask])), (
        f"Signed-zero mismatch in non-NaN values: "
        f"out signs={torch.signbit(out[mask])}, ref signs={torch.signbit(ref[mask])}"
    )


@SignedZeroFixture
def test_minimum_signed_zero_with_nan(dtype: torch.dtype) -> None:
    """Signed-zero fix must not regress NaN propagation."""
    nan = float("nan")
    # Mix of NaN pairs and non-NaN signed-zero pairs so both code paths execute
    a = torch.tensor([nan, -0.0, 0.0, 1.0, nan, 2.0], dtype=dtype, device="cuda")
    b = torch.tensor([1.0, nan, -0.0, 0.0, 0.0, 3.0], dtype=dtype, device="cuda")
    shape = (6,)
    op = MinimumOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.minimum(a, b)
    with torch.no_grad():
        out = op(a, b)
    # NaN positions must match
    assert torch.equal(torch.isnan(out), torch.isnan(ref)), (
        f"NaN positions differ: out={out}, ref={ref}"
    )
    # Non-NaN values must exist and match (including sign bits for zeros)
    mask = ~torch.isnan(ref)
    assert mask.any(), "Test bug: expected some non-NaN reference values"
    torch.testing.assert_close(out[mask], ref[mask], atol=0, rtol=0)
    assert torch.equal(torch.signbit(out[mask]), torch.signbit(ref[mask])), (
        f"Signed-zero mismatch in non-NaN values: "
        f"out signs={torch.signbit(out[mask])}, ref signs={torch.signbit(ref[mask])}"
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


# ---------------------------------------------------------------------------
# Dtype contract tests
# ---------------------------------------------------------------------------


class FloatOnlyBinaryRejectFixture(FixtureBase):
    PARAMS = [
        ("op_cls, dtype", [
            pytest.param(DivOp, torch.int32, marks=pytest.mark.smoke),
            pytest.param(RemainderOp, torch.int32, marks=pytest.mark.full),
            pytest.param(PowOp, torch.int32, marks=pytest.mark.full),
            pytest.param(FloorDivideOp, torch.int64, marks=pytest.mark.full),
            pytest.param(LerpOp, torch.int32, marks=pytest.mark.full),
            pytest.param(MaximumOp, torch.int32, marks=pytest.mark.full),
            pytest.param(MinimumOp, torch.int64, marks=pytest.mark.full),
        ]),
    ]


@FloatOnlyBinaryRejectFixture
def test_float_only_binary_ops_reject_integer_dtype(op_cls, dtype: torch.dtype) -> None:
    """Float-only binary ops must reject integer dtypes at construction time."""
    shape = (16,)
    with pytest.raises(ValueError, match="does not support dtype"):
        op_cls(a_shape=shape, b_shape=shape, dtype=dtype)


@pytest.mark.smoke
def test_binary_op_rejects_runtime_dtype_mismatch() -> None:
    """Runtime inputs should fail fast instead of reaching backend lowering."""
    op = SubOp(a_shape=(16,), b_shape=(16,), dtype=torch.float16)
    a = torch.randn(16, device="cuda", dtype=torch.float32)
    b = torch.randn(16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="Expected a.dtype"):
        op(a, b)


# ---------------------------------------------------------------------------
# BinaryKernel autotune_configs tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_binary_kernel_has_autotune_configs() -> None:
    """BinaryKernel subclasses must expose autotune_configs with >= 3 entries."""

    shape = (4096,)
    for op_cls in (MaximumOp, MinimumOp, AddOp, SubOp, MulOp):
        op = op_cls(a_shape=shape, b_shape=shape, dtype=torch.float16)
        # Access autotune_configs from the underlying kernel object
        kernel = op.kernel
        configs = kernel.autotune_configs
        assert configs is not None, (
            f"{kernel.__class__.__name__} must define autotune_configs"
        )
        assert len(configs) >= 3, (
            f"{kernel.__class__.__name__}.autotune_configs has {len(configs)} entries, need >= 3"
        )
        # Each config must have "threads" and "num_per_thread" keys
        for cfg in configs:
            assert "threads" in cfg, f"Config missing 'threads': {cfg}"
            assert "num_per_thread" in cfg, f"Config missing 'num_per_thread': {cfg}"


@pytest.mark.smoke
def test_binary_kernel_autotune_configs_distinct() -> None:
    """autotune_configs entries must be distinct (no duplicates)."""
    shape = (4096,)
    op = AddOp(a_shape=shape, b_shape=shape, dtype=torch.float16)
    configs = op.kernel.autotune_configs
    config_tuples = [(c["threads"], c["num_per_thread"]) for c in configs]
    assert len(config_tuples) == len(set(config_tuples)), (
        f"Duplicate configs found: {config_tuples}"
    )


# ---------------------------------------------------------------------------
# Optimized maximum/minimum correctness on larger shapes
# ---------------------------------------------------------------------------


class OptimizedMaxMinFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1024 * 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024 * 10240, torch.float16, marks=pytest.mark.full),
            pytest.param(1024 * 4096, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


@OptimizedMaxMinFixture
def test_maximum_optimized_large(n_total: int, dtype: torch.dtype) -> None:
    """Optimized maximum matches torch.maximum on large DNN-realistic shapes."""
    shape = (n_total,)
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn(*shape, device="cuda", dtype=dtype)
    op = MaximumOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.maximum(a, b)
    with torch.no_grad():
        out = op(a, b)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@OptimizedMaxMinFixture
def test_minimum_optimized_large(n_total: int, dtype: torch.dtype) -> None:
    """Optimized minimum matches torch.minimum on large DNN-realistic shapes."""
    shape = (n_total,)
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn(*shape, device="cuda", dtype=dtype)
    op = MinimumOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.minimum(a, b)
    with torch.no_grad():
        out = op(a, b)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
