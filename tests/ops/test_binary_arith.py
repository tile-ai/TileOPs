"""Tests for binary arithmetic elementwise ops with broadcast.

Covers L1 smoke correctness for sub, mul, div, remainder, pow,
floor_divide, lerp, maximum, minimum (plus existing add).
Also includes L4 edge case tests for div, remainder, floor_divide, pow.
"""

import functools

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.ops.elementwise import (
    AddFwdOp,
    DivFwdOp,
    FloorDivideFwdOp,
    LerpScalarFwdOp,
    LerpTensorFwdOp,
    MaximumFwdOp,
    MinimumFwdOp,
    MulFwdOp,
    PowFwdOp,
    RemainderFwdOp,
    SubFwdOp,
)
from workloads.device import run_device, run_device_available
from workloads.elementwise import (
    AddBroadcastWorkload,
    AddSameShapeCase,
    BinaryPositiveCase,
    BinarySameShapeCase,
    ElementwiseWorkload,
    FloorDivideCase,
    LerpCancellationWorkload,
    LerpCase,
    PowPositiveWorkload,
    RemainderCase,
)
from workloads.numerics import compare_outputs


class AddSameShapeTest(AddSameShapeCase, TestBase):
    pass


# coalesce_broadcast_dims unit tests


class CoalesceFixture(FixtureBase):
    PARAMS = [
        (
            "a_shape, b_shape, expected_ndim",
            [
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
            ],
        ),
    ]


# Add op correctness tests


class AddSameShapeFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(16_384, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@AddSameShapeFixture
def test_add_same_shape(n_total: int, dtype: torch.dtype) -> None:
    test = AddSameShapeTest(n_total, dtype)
    op = AddFwdOp()
    test.check(op, *test.gen_inputs())


# Broadcast pattern tests (L3)


class AddBroadcastFixture(FixtureBase):
    PARAMS = [
        (
            "a_shape, b_shape, dtype",
            [
                pytest.param(
                    (2, 512, 768),
                    (1, 1, 768),
                    torch.float16,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (2, 512, 768),
                    (2, 512, 1),
                    torch.float16,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    (1024, 1024),
                    (1, 1),
                    torch.float16,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    (4, 1, 8),
                    (1, 8, 1),
                    torch.float16,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


class AddBroadcastTest(AddBroadcastWorkload, TestBase):
    pass


@AddBroadcastFixture
def test_add_broadcast(a_shape, b_shape, dtype: torch.dtype) -> None:
    test = AddBroadcastTest(a_shape, b_shape, dtype)
    op = AddFwdOp()
    test.check(op, *test.gen_inputs())


# Broadcast pattern tests for all binary arith ops (L3)

# (op_name, op_cls, ref_fn, gen_a, gen_b)
_ARITH_BROADCAST_OPS = [
    (
        "sub",
        SubFwdOp,
        lambda a, b: (a.float() - b.float()).to(a.dtype),
        lambda s, d: torch.randn(*s, dtype=d, device=run_device()),
        lambda s, d: torch.randn(*s, dtype=d, device=run_device()),
    ),
    (
        "mul",
        MulFwdOp,
        lambda a, b: (a.float() * b.float()).to(a.dtype),
        lambda s, d: torch.randn(*s, dtype=d, device=run_device()),
        lambda s, d: torch.randn(*s, dtype=d, device=run_device()),
    ),
    (
        "div",
        DivFwdOp,
        lambda a, b: (a.float() / b.float()).to(a.dtype),
        lambda s, d: torch.rand(*s, dtype=d, device=run_device()) + 0.1,
        lambda s, d: torch.rand(*s, dtype=d, device=run_device()) + 0.1,
    ),
    (
        "remainder",
        RemainderFwdOp,
        torch.remainder,
        lambda s, d: torch.rand(*s, dtype=d, device=run_device()) + 0.1,
        lambda s, d: torch.rand(*s, dtype=d, device=run_device()) + 0.1,
    ),
    (
        "pow",
        PowFwdOp,
        lambda a, b: torch.pow(a.float(), b.float()).to(a.dtype),
        lambda s, d: torch.rand(*s, dtype=d, device=run_device()) + 0.5,
        lambda s, d: torch.rand(*s, dtype=d, device=run_device()) * 2.0,
    ),
    (
        "floor_divide",
        FloorDivideFwdOp,
        torch.floor_divide,
        lambda s, d: torch.rand(*s, dtype=d, device=run_device()) + 0.1,
        lambda s, d: torch.rand(*s, dtype=d, device=run_device()) + 0.1,
    ),
    (
        "lerp",
        LerpScalarFwdOp,
        lambda a, b: torch.lerp(a.float(), b.float(), 0.5).to(a.dtype),
        lambda s, d: torch.randn(*s, dtype=d, device=run_device()),
        lambda s, d: torch.randn(*s, dtype=d, device=run_device()),
    ),
    (
        "maximum",
        MaximumFwdOp,
        lambda a, b: torch.maximum(a.float(), b.float()).to(a.dtype),
        lambda s, d: torch.randn(*s, dtype=d, device=run_device()),
        lambda s, d: torch.randn(*s, dtype=d, device=run_device()),
    ),
    (
        "minimum",
        MinimumFwdOp,
        lambda a, b: torch.minimum(a.float(), b.float()).to(a.dtype),
        lambda s, d: torch.randn(*s, dtype=d, device=run_device()),
        lambda s, d: torch.randn(*s, dtype=d, device=run_device()),
    ),
]


class ArithBroadcastFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        patterns = [
            # bias-add: (B,S,D) + (1,1,D)
            ((2, 64, 128), (1, 1, 128)),
            # row broadcast: (B,S,D) + (B,S,1)
            ((2, 64, 128), (2, 64, 1)),
            # scalar broadcast: (M,N) + (1,1)
            ((64, 128), (1, 1)),
        ]
        return [
            (
                "op_name, op_cls, ref_fn, gen_a, gen_b, a_shape, b_shape",
                [
                    pytest.param(
                        name,
                        cls,
                        ref,
                        ga,
                        gb,
                        a_s,
                        b_s,
                        marks=pytest.mark.smoke if i == 0 and j == 0 else pytest.mark.full,
                    )
                    for j, (name, cls, ref, ga, gb) in enumerate(_ARITH_BROADCAST_OPS)
                    for i, (a_s, b_s) in enumerate(patterns)
                ],
            ),
        ]


@ArithBroadcastFixture
def test_binary_arith_broadcast(
    op_name,
    op_cls,
    ref_fn,
    gen_a,
    gen_b,
    a_shape,
    b_shape,
) -> None:
    dtype = torch.float16
    a = gen_a(a_shape, dtype)
    b = gen_b(b_shape, dtype)
    op = op_cls()
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


# Generic binary test helper


class BinarySameShapeTest(BinarySameShapeCase, TestBase):
    pass


class BinaryPositiveTest(BinaryPositiveCase, TestBase):
    pass


# Same-shape correctness for simple binary arith ops


class RemainderTest(RemainderCase, TestBase):
    pass


class PowPositiveTest(PowPositiveWorkload, TestBase):
    """Pow needs positive base and small exponent to avoid overflow in fp16."""


class BinaryArithOpFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, make_test",
            [
                pytest.param(
                    SubFwdOp, lambda n, d: BinarySameShapeTest(n, d, "SubFwdOp"), id="sub"
                ),
                pytest.param(
                    MulFwdOp, lambda n, d: BinarySameShapeTest(n, d, "MulFwdOp"), id="mul"
                ),
                pytest.param(DivFwdOp, lambda n, d: BinaryPositiveTest(n, d, "DivFwdOp"), id="div"),
                pytest.param(RemainderFwdOp, RemainderTest, id="remainder"),
                pytest.param(PowFwdOp, PowPositiveTest, id="pow"),
                pytest.param(
                    MaximumFwdOp,
                    lambda n, d: BinarySameShapeTest(n, d, "MaximumFwdOp"),
                    id="maximum",
                ),
                pytest.param(
                    MinimumFwdOp,
                    lambda n, d: BinarySameShapeTest(n, d, "MinimumFwdOp"),
                    id="minimum",
                ),
            ],
        ),
        (
            "n_total, dtype",
            [
                pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@BinaryArithOpFixture
def test_binary_arith_op(op_cls, make_test, n_total: int, dtype: torch.dtype) -> None:
    test = make_test(n_total, dtype)
    op = op_cls()
    test.check(op, *test.gen_inputs())


class FloorDivideFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


class FloorDivideTest(FloorDivideCase, TestBase):
    pass


@FloorDivideFixture
def test_floor_divide_op(n_total: int, dtype: torch.dtype) -> None:
    test = FloorDivideTest(n_total, dtype)
    op = FloorDivideFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        # Wide enough that float32 keeps two vectors a thread.
        pytest.param((1 << 18,), (1 << 18,), id="same"),
        pytest.param((4, 2048), (1, 2048), id="bias"),
        pytest.param((2, 16, 56, 56), (16, 1, 1), id="channel"),
    ],
)
def test_floor_ops_match_torch_on_special_values(a_shape, b_shape, dtype) -> None:
    """``1.0 // 0.1`` is 9, an infinite divisor floors a mixed-sign quotient to -1,
    and a zero result keeps its sign.

    The special values sit among ordinary ones, so a thread whose cheap form fails
    for one element redoes it next to threads that keep theirs, on every staged
    layout: same shape, a broadcast row and a broadcast channel.
    """
    values = [0.0, -0.0, 1.0, -1.0, 0.1, -2.5, 7.0, 3e4, float("inf"), float("-inf"), float("nan")]
    grid = torch.tensor(values, device=run_device())
    pairs = torch.cartesian_prod(grid, grid)
    a = torch.rand(a_shape, device=run_device()) + 0.5
    b = torch.rand(b_shape, device=run_device()) + 0.5
    a.view(-1)[: len(pairs)] = pairs[:, 0]
    if a_shape == b_shape:
        b.view(-1)[: len(pairs)] = pairs[:, 1]
        # Again one block-wide chunk on, where a thread holds its second vector.
        a.view(-1)[512 : 512 + len(pairs)] = pairs[:, 0]
        b.view(-1)[512 : 512 + len(pairs)] = pairs[:, 1]
    else:
        b.view(-1)[: len(values)] = grid
    a, b = a.to(dtype), b.to(dtype)
    cases = [
        (FloorDivideFwdOp(), torch.floor_divide),
        (DivFwdOp(rounding_mode="floor"), lambda x, y: torch.div(x, y, rounding_mode="floor")),
        (RemainderFwdOp(), torch.remainder),
    ]
    for op, ref_fn in cases:
        out, ref = op(a, b), ref_fn(a, b)
        compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, ()).verification(a, b))
        number = ~ref.isnan()
        assert torch.equal(torch.signbit(out[number]), torch.signbit(ref[number]))


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    "dtype, make_op, ref_fn",
    [
        pytest.param(torch.bfloat16, DivFwdOp, torch.div, id="bfloat16-div"),
        pytest.param(
            torch.bfloat16,
            functools.partial(DivFwdOp, rounding_mode="trunc"),
            functools.partial(torch.div, rounding_mode="trunc"),
            id="bfloat16-div-trunc",
        ),
        pytest.param(
            torch.float16,
            functools.partial(DivFwdOp, rounding_mode="trunc"),
            functools.partial(torch.div, rounding_mode="trunc"),
            id="float16-div-trunc",
        ),
        pytest.param(torch.bfloat16, RemainderFwdOp, torch.remainder, id="bfloat16-remainder"),
        pytest.param(
            torch.bfloat16, FloorDivideFwdOp, torch.floor_divide, id="bfloat16-floor-divide"
        ),
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        pytest.param((1 << 16,), (1 << 16,), id="same"),
        pytest.param((64, 1024), (1, 1024), id="bias"),
    ],
)
def test_16bit_div_matches_torch_bit_for_bit(a_shape, b_shape, dtype, make_op, ref_fn) -> None:
    """The fast 16-bit divide gives torch's result bit for bit.

    Random bit patterns put divisors past ``2**126`` and below ``2**-126`` among
    ordinary ones. Those are the divisors the floored ops' cheap tier declines,
    because the reciprocal it takes answers a zero or an infinity for them.
    """
    gen = torch.Generator(device=run_device()).manual_seed(0)

    def bits(shape):
        raw = torch.randint(-(2**15), 2**15, shape, generator=gen, device=run_device())
        return raw.to(torch.int16).view(dtype)

    a, b = bits(a_shape), bits(b_shape)
    out, ref = make_op()(a, b), ref_fn(a, b)
    number = ~ref.isnan()
    assert torch.equal(out.isnan(), ~number)
    assert torch.equal(out[number].view(torch.int16), ref[number].view(torch.int16))


# Lerp op (ternary in PyTorch; compile-time weight=0.5)


class LerpFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


class LerpTest(LerpCase, TestBase):
    pass


@LerpFixture
def test_lerp_op(n_total: int, dtype: torch.dtype) -> None:
    """Validate lerp across multiple construction-time weight values."""
    for weight in [0.0, 0.3, 0.5, 0.7, 1.0]:
        test = LerpTest(n_total, dtype, weight=weight)
        op = LerpScalarFwdOp(weight=weight)
        test.check(op, *test.gen_inputs())


# Maximum/Minimum NaN propagation tests


class MaxMinNanFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, torch_ref",
            [
                pytest.param(MaximumFwdOp, torch.maximum, id="maximum"),
                pytest.param(MinimumFwdOp, torch.minimum, id="minimum"),
            ],
        ),
        (
            "dtype",
            [
                pytest.param(torch.float16, marks=pytest.mark.smoke),
                pytest.param(torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


def _nan_against_one(op_cls) -> torch.Tensor:
    """``op_cls`` applied to a negative NaN and 1.0, which torch answers with the NaN."""
    negative_nan = torch.tensor([0xFE00], dtype=torch.uint16, device=run_device()).view(
        torch.float16
    )
    other = torch.tensor([1.0], dtype=torch.float16, device=run_device())
    return op_cls()(negative_nan, other)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls", [MaximumFwdOp, MinimumFwdOp])
def test_max_min_propagate_nan(op_cls) -> None:
    """A NaN operand makes the result NaN, whichever backend serves the op."""
    assert torch.isnan(_nan_against_one(op_cls)).all()


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls", [MaximumFwdOp, MinimumFwdOp])
def test_max_min_canonicalize_the_nan_payload(op_cls) -> None:
    """The NaN in the result does not carry the operand's payload: two NaN operands with
    different bits give the same result bits."""
    other = torch.tensor([1.0], dtype=torch.float16, device=run_device())
    results = [
        op_cls()(
            torch.tensor([bits], dtype=torch.uint16, device=run_device()).view(torch.float16), other
        )
        for bits in (0xFE00, 0x7E01)
    ]
    assert torch.equal(results[0].view(torch.uint16), results[1].view(torch.uint16))


@MaxMinNanFixture
def test_max_min_nan_propagation(op_cls, torch_ref, dtype: torch.dtype) -> None:
    """Verify maximum/minimum propagate NaN when either operand is NaN."""
    nan = float("nan")
    a = torch.tensor([nan, 1.0, nan, 2.0], dtype=dtype, device=run_device())
    b = torch.tensor([3.0, nan, nan, 1.0], dtype=dtype, device=run_device())
    op = op_cls()
    ref = torch_ref(a, b)
    with torch.no_grad():
        out = op(a, b)
    # NaN positions must match: both output and ref should be NaN at same indices
    assert torch.equal(torch.isnan(out), torch.isnan(ref)), (
        f"NaN positions differ: out={out}, ref={ref}"
    )
    # Non-NaN values must match exactly
    mask = ~torch.isnan(ref)
    (
        compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, ()).verification(a, b)),
        (f"Non-NaN values differ: out={out[mask]}, ref={ref[mask]}"),
    )


# Maximum/Minimum signed-zero regression tests


class SignedZeroFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, torch_ref",
            [
                pytest.param(MaximumFwdOp, torch.maximum, id="maximum"),
                pytest.param(MinimumFwdOp, torch.minimum, id="minimum"),
            ],
        ),
        (
            "dtype",
            [
                pytest.param(torch.float16, marks=pytest.mark.smoke),
                pytest.param(torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@SignedZeroFixture
def test_max_min_signed_zero(op_cls, torch_ref, dtype: torch.dtype) -> None:
    """maximum(+0,-0)=+0 / minimum(-0,+0)=-0 (IEEE / PyTorch semantics)."""
    pos_zero = torch.tensor(0.0, dtype=dtype, device=run_device())
    neg_zero = torch.tensor(-0.0, dtype=dtype, device=run_device())

    # All four orderings: (+0,-0), (-0,+0), (+0,+0), (-0,-0)
    a = torch.stack([pos_zero, neg_zero, pos_zero, neg_zero])
    b = torch.stack([neg_zero, pos_zero, pos_zero, neg_zero])
    op = op_cls()
    ref = torch_ref(a, b)
    with torch.no_grad():
        out = op(a, b)

    # Value equality
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))
    # Sign-bit equality: +0 and -0 compare equal but have different sign bits
    out_signbits = torch.signbit(out)
    ref_signbits = torch.signbit(ref)
    assert torch.equal(out_signbits, ref_signbits), (
        f"Signed-zero mismatch: out signs={out_signbits}, ref signs={ref_signbits}"
    )


class SignedZeroNanFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, torch_ref, a_vals, b_vals",
            [
                pytest.param(
                    MaximumFwdOp,
                    torch.maximum,
                    [float("nan"), 1.0, -0.0, 0.0, float("nan"), 3.0],
                    [1.0, float("nan"), 0.0, -0.0, -0.0, 2.0],
                    id="maximum",
                ),
                pytest.param(
                    MinimumFwdOp,
                    torch.minimum,
                    [float("nan"), -0.0, 0.0, 1.0, float("nan"), 2.0],
                    [1.0, float("nan"), -0.0, 0.0, 0.0, 3.0],
                    id="minimum",
                ),
            ],
        ),
        (
            "dtype",
            [
                pytest.param(torch.float16, marks=pytest.mark.smoke),
                pytest.param(torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@SignedZeroNanFixture
def test_max_min_signed_zero_with_nan(
    op_cls, torch_ref, a_vals, b_vals, dtype: torch.dtype
) -> None:
    """Signed-zero fix must not regress NaN propagation."""
    # Mix of NaN pairs and non-NaN signed-zero pairs so both code paths execute
    a = torch.tensor(a_vals, dtype=dtype, device=run_device())
    b = torch.tensor(b_vals, dtype=dtype, device=run_device())
    op = op_cls()
    ref = torch_ref(a, b)
    with torch.no_grad():
        out = op(a, b)
    # NaN positions must match
    assert torch.equal(torch.isnan(out), torch.isnan(ref)), (
        f"NaN positions differ: out={out}, ref={ref}"
    )
    # Non-NaN values must exist and match (including sign bits for zeros)
    mask = ~torch.isnan(ref)
    assert mask.any(), "Test bug: expected some non-NaN reference values"
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, ()).verification(a, b))
    assert torch.equal(torch.signbit(out[mask]), torch.signbit(ref[mask])), (
        f"Signed-zero mismatch in non-NaN values: "
        f"out signs={torch.signbit(out[mask])}, ref signs={torch.signbit(ref[mask])}"
    )


# L4 edge case tests (fp32, 4K)


class EdgeCaseFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, ref_fn, gen_fn",
            [
                # div: avoid div-by-zero
                pytest.param(
                    DivFwdOp,
                    lambda a, b: a / b,
                    lambda n, d: (
                        torch.randn(n, dtype=d, device=run_device()),
                        torch.rand(n, dtype=d, device=run_device()) + 0.1,
                    ),
                    marks=pytest.mark.smoke,
                ),
                # remainder: positive inputs
                pytest.param(
                    RemainderFwdOp,
                    lambda a, b: a % b,
                    lambda n, d: (
                        torch.rand(n, dtype=d, device=run_device()) + 0.1,
                        torch.rand(n, dtype=d, device=run_device()) + 0.1,
                    ),
                    marks=pytest.mark.full,
                ),
                # floor_divide: positive inputs
                pytest.param(
                    FloorDivideFwdOp,
                    torch.floor_divide,
                    lambda n, d: (
                        torch.rand(n, dtype=d, device=run_device()) + 0.1,
                        torch.rand(n, dtype=d, device=run_device()) + 0.1,
                    ),
                    marks=pytest.mark.full,
                ),
                # pow: positive base, small exponent
                pytest.param(
                    PowFwdOp,
                    lambda a, b: torch.pow(a, b),
                    lambda n, d: (
                        torch.rand(n, dtype=d, device=run_device()) + 0.5,
                        torch.rand(n, dtype=d, device=run_device()) * 2.0,
                    ),
                    marks=pytest.mark.full,
                ),
                # maximum: mixed sign
                pytest.param(
                    MaximumFwdOp,
                    lambda a, b: torch.maximum(a, b),
                    lambda n, d: (
                        torch.randn(n, dtype=d, device=run_device()),
                        torch.randn(n, dtype=d, device=run_device()),
                    ),
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@EdgeCaseFixture
def test_binary_arith_edge_cases(op_cls, ref_fn, gen_fn) -> None:
    """L4 edge case tests: fp32, 4K elements."""
    n = 4096
    dtype = torch.float32
    a, b = gen_fn(n, dtype)
    op = op_cls()
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


# Dtype contract tests


class FloatOnlyBinaryRejectFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, dtype",
            [
                pytest.param(DivFwdOp, torch.int32, marks=pytest.mark.smoke),
                pytest.param(RemainderFwdOp, torch.int32, marks=pytest.mark.smoke),
                pytest.param(PowFwdOp, torch.int32, marks=pytest.mark.smoke),
                pytest.param(FloorDivideFwdOp, torch.int64, marks=pytest.mark.smoke),
                pytest.param(LerpScalarFwdOp, torch.int32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FloatOnlyBinaryRejectFixture
def test_float_only_binary_ops_reject_integer_dtype(op_cls, dtype: torch.dtype) -> None:
    """Float-only binary ops must reject integer dtypes.

    The element type arrives with the tensors, so the rejection does too, and
    the manifest dtype gate now fires before the kernel's own check.
    """
    shape = (16,)
    op = op_cls()
    a = torch.ones(shape, device=run_device(), dtype=dtype)
    with pytest.raises(ValueError, match="dtype is outside"):
        op(a, a)


@pytest.mark.smoke
def test_binary_op_rejects_runtime_dtype_mismatch() -> None:
    """Runtime inputs should fail fast instead of reaching backend lowering."""
    op = SubFwdOp()
    a = torch.randn(16, device=run_device(), dtype=torch.float32)
    b = torch.randn(16, device=run_device(), dtype=torch.float16)
    # The manifest types ``input`` and ``other`` with one index ``T``; the generated
    # gate names the operand that disagrees.
    with pytest.raises(ValueError, match="differs from T"):
        op(a, b)


@pytest.mark.smoke
def test_binary_op_does_not_keep_its_inputs() -> None:
    """A call leaves nothing holding its tensors once the caller drops them."""
    import gc
    import weakref

    op = AddFwdOp()
    a = torch.randn(16, device=run_device(), dtype=torch.float16)
    b = torch.randn(16, device=run_device(), dtype=torch.float16)
    op(a, b)
    alive = [weakref.ref(a), weakref.ref(b)]
    del a, b
    gc.collect()
    assert all(ref() is None for ref in alive)


# BinaryKernel autotune_configs tests


# Optimized maximum/minimum correctness on larger shapes


class OptimizedMaxMinFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, torch_ref",
            [
                pytest.param(MaximumFwdOp, torch.maximum, id="maximum"),
                pytest.param(MinimumFwdOp, torch.minimum, id="minimum"),
            ],
        ),
        (
            "n_total, dtype",
            [
                pytest.param(1024 * 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024 * 4096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1024 * 10240, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@OptimizedMaxMinFixture
def test_max_min_optimized_large(op_cls, torch_ref, n_total: int, dtype: torch.dtype) -> None:
    """Optimized maximum/minimum match torch on large DNN-realistic shapes."""
    shape = (n_total,)
    a = torch.randn(*shape, device=run_device(), dtype=dtype)
    b = torch.randn(*shape, device=run_device(), dtype=dtype)
    op = op_cls()
    ref = torch_ref(a, b)
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


# register_copy broadcast downgrade regression test


# tune=True reaches the autotuner


@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_binary_ops_under_tuning(tune: bool) -> None:
    for op_cls, ref_fn in (
        (AddFwdOp, torch.add),
        (MaximumFwdOp, torch.maximum),
        (MinimumFwdOp, torch.minimum),
    ):
        a = torch.randn(4096, device=run_device(), dtype=torch.float16)
        b = torch.randn(4096, device=run_device(), dtype=torch.float16)
        compare_outputs(
            op_cls(tune=tune)(a, b),
            ref_fn(a, b),
            ElementwiseWorkload(op_cls.__name__, (a, b)).verification(a, b),
        )


# LerpTensorFwdOp — Tensor-weight torch.lerp overload (manifest:
# elementwise_multi_input). Covers same-shape, 3-way broadcast, dtype
# rejection, and dtype-mismatch rejection at forward().


_LERP_TENSOR_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("dtype", _LERP_TENSOR_DTYPES)
def test_lerp_tensor_same_shape(dtype: torch.dtype) -> None:
    """LerpTensorFwdOp matches torch.lerp on same-shape inputs."""
    shape = (4, 8)
    a = torch.randn(shape, device=run_device(), dtype=dtype)
    b = torch.randn(shape, device=run_device(), dtype=dtype)
    w = torch.rand(shape, device=run_device(), dtype=dtype)
    op = LerpTensorFwdOp()
    out = op(a, b, w)
    ref = torch.lerp(a, b, w)
    compare_outputs(
        out, ref, ElementwiseWorkload(type(op).__name__, (a, b, w)).verification(*(a, b, w))
    )


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_lerp_tensor_broadcast() -> None:
    """LerpTensorFwdOp supports the manifest's 3-way broadcast rule."""
    a_shape, b_shape, w_shape = (3, 1), (1, 4), (3, 4)
    dtype = torch.float32
    a = torch.randn(a_shape, device=run_device(), dtype=dtype)
    b = torch.randn(b_shape, device=run_device(), dtype=dtype)
    w = torch.rand(w_shape, device=run_device(), dtype=dtype)
    op = LerpTensorFwdOp()
    out = op(a, b, w)
    ref = torch.lerp(a, b, w)
    compare_outputs(
        out, ref, ElementwiseWorkload(type(op).__name__, (a, b, w)).verification(*(a, b, w))
    )
    assert tuple(out.shape) == (3, 4)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_lerp_tensor_rejects_fp8_dtype(bad_dtype: torch.dtype) -> None:
    """LerpTensorFwdOp must reject fp8 dtypes (manifest declares no fp8)."""
    shape = (4, 8)
    op = LerpTensorFwdOp()
    x = torch.zeros(shape, device=run_device()).to(bad_dtype)
    with pytest.raises((ValueError, TypeError)):
        op(x, x, x)


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_lerp_tensor_dtype_mismatch_rejected() -> None:
    """forward() must reject operands that disagree with each other."""
    shape = (4, 8)
    op = LerpTensorFwdOp()
    a = torch.randn(shape, device=run_device(), dtype=torch.float32)
    b = torch.randn(shape, device=run_device(), dtype=torch.float32)
    w_bad = torch.rand(shape, device=run_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="weight.dtype"):
        op(a, b, w_bad)


# DivFwdOp rounding_mode trunc/floor coverage


_DIV_ROUNDING_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
_DIV_ROUNDING_MODES = ["trunc", "floor"]


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("rounding_mode", _DIV_ROUNDING_MODES)
@pytest.mark.parametrize("dtype", _DIV_ROUNDING_DTYPES)
def test_div_rounding_mode_eager(rounding_mode: str, dtype: torch.dtype) -> None:
    """DivFwdOp(rounding_mode=...) matches torch.div for trunc and floor."""
    shape = (64, 256)
    # Both positive and negative quotients naturally arise from randn inputs;
    # clamp ``b`` away from zero so division is well-defined.
    a = torch.randn(*shape, dtype=dtype, device=run_device()) * 5.0
    b = torch.randn(*shape, dtype=dtype, device=run_device()) * 2.0 + 1.0
    b = torch.where(b.abs() < 0.5, torch.full_like(b, 1.0), b)
    op = DivFwdOp(rounding_mode=rounding_mode)
    with torch.no_grad():
        out = op(a, b)
    # torch rounds the quotient to ``dtype`` before rounding it to a whole number.
    ref = torch.div(a, b, rounding_mode=rounding_mode)
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(type(op).__name__, (a, b), rounding_mode=rounding_mode).verification(
            *(a, b)
        ),
    )


@pytest.mark.smoke
def test_div_rejects_an_unknown_rounding_mode() -> None:
    """DivFwdOp rejects an unknown rounding mode."""
    with pytest.raises(ValueError, match="rounding_mode"):
        DivFwdOp(rounding_mode="invalid")


# Per-dtype int / bool correctness for arithmetic ops with manifest int union

_BINARY_INT_DTYPES = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]
# Add / Mul / Maximum / Minimum accept the full union including bool;
# Sub mirrors PyTorch and excludes bool (bool subtraction is undefined).
_FULL_UNION_OPS = [
    (AddFwdOp, torch.add),
    (MulFwdOp, torch.mul),
    (MaximumFwdOp, torch.maximum),
    (MinimumFwdOp, torch.minimum),
]


def _gen_int_pair(n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype == torch.uint8:
        lo, hi = 0, 32
    elif dtype == torch.int8:
        lo, hi = -16, 16
    else:
        lo, hi = -64, 64
    a = torch.randint(lo, hi, (n,), dtype=dtype, device=run_device())
    b = torch.randint(lo, hi, (n,), dtype=dtype, device=run_device())
    return a, b


# Dtype-coverage axis: exercise every manifest-declared int dtype on a
# single representative op (AddFwdOp). The op-coverage axis below fixes
# dtype = int32 and varies op_cls. Decoupling the axes avoids the
# dtype x op cross product.
class BinaryArithIntDtypeFixture(FixtureBase):
    PARAMS = [
        ("dtype", [pytest.param(dt, marks=pytest.mark.smoke) for dt in _BINARY_INT_DTYPES]),
    ]


@BinaryArithIntDtypeFixture
def test_binary_arith_integer_dtype_add(dtype: torch.dtype) -> None:
    """AddFwdOp matches torch.add on every manifest-declared int dtype."""
    n = 4_096
    a, b = _gen_int_pair(n, dtype)
    op = AddFwdOp()
    ref = torch.add(a, b)
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


# Op-coverage axis: at fixed dtype = int32, every full-union arithmetic
# op (plus SubFwdOp) matches its torch reference.
_INT_OP_CASES = _FULL_UNION_OPS + [(SubFwdOp, torch.sub)]


class BinaryArithOpIntFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, ref_fn",
            [
                pytest.param(op_cls, ref_fn, marks=pytest.mark.smoke)
                for op_cls, ref_fn in _INT_OP_CASES
            ],
        ),
    ]


@BinaryArithOpIntFixture
def test_binary_arith_op_int32(op_cls, ref_fn) -> None:
    """Each arithmetic op matches its torch reference on int32 inputs."""
    n = 4_096
    a, b = _gen_int_pair(n, torch.int32)
    op = op_cls()
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


# Bool-axis reference mapping. The kernel implements:
#   AddFwdOp(bool, bool) := a | b   (logical OR — kernel uses ``a + b``,
#                                    which lowers to OR for bool operands;
#                                    must NOT be XOR)
#   MulFwdOp(bool, bool) := a & b   (logical AND — kernel uses ``a * b``)
#   MaximumFwdOp(bool, bool) := a | b   (T.max on bool == OR)
#   MinimumFwdOp(bool, bool) := a & b   (T.min on bool == AND)
# torch.add / torch.mul on bool tensors happen to coincide with OR/AND,
# but we use torch.logical_or / torch.logical_and as the explicit
# reference so the contract — and any future divergence in PyTorch's
# bool arithmetic semantics — is documented at the call site.
_FULL_UNION_BOOL_REFS = [
    (AddFwdOp, torch.logical_or),
    (MulFwdOp, torch.logical_and),
    (MaximumFwdOp, torch.logical_or),
    (MinimumFwdOp, torch.logical_and),
]


class BinaryArithBoolDtypeFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, ref_fn",
            [
                pytest.param(op_cls, ref_fn, marks=pytest.mark.smoke)
                for op_cls, ref_fn in _FULL_UNION_BOOL_REFS
            ],
        ),
    ]


@BinaryArithBoolDtypeFixture
def test_binary_arith_bool_dtype(op_cls, ref_fn) -> None:
    """Add/Mul/Maximum/Minimum match logical OR/AND on torch.bool inputs.

    SubFwdOp is excluded because torch.sub raises on bool inputs.
    """
    n = 4_096
    a = torch.randint(0, 2, (n,), device=run_device()).to(torch.bool)
    b = torch.randint(0, 2, (n,), device=run_device()).to(torch.bool)
    op = op_cls()
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


@pytest.mark.smoke
def test_add_bool_is_or_not_xor() -> None:
    """AddFwdOp(bool) must lower to OR (True+True=True), not XOR.

    Sentinel guard: if a future TileLang change lowers ``+`` on bool as
    XOR, this test fails on the (True, True) lane (XOR would give False,
    OR gives True). Random-bool tests cover both lanes statistically;
    this test pins the contract on a deterministic input.
    """
    a = torch.tensor([True, True, False, False], device=run_device())
    b = torch.tensor([True, False, True, False], device=run_device())
    op = AddFwdOp()
    expected = torch.tensor([True, True, True, False], device=run_device())
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(
        out, expected, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b))
    )


@pytest.mark.smoke
def test_add_rejects_a_float_alpha_for_integer_input() -> None:
    """``torch.add`` refuses a floating-point ``alpha`` for integral inputs."""
    x = torch.ones(8, device=run_device(), dtype=torch.int32)
    with pytest.raises(ValueError, match="alpha"):
        AddFwdOp(alpha=0.5)(x, x)


@pytest.mark.smoke
def test_sub_rejects_bool_dtype() -> None:
    """torch.sub raises on bool; SubFwdOp must reject it at construction time."""
    shape = (16,)
    op = SubFwdOp()
    x = torch.zeros(shape, device=run_device(), dtype=torch.bool)
    with pytest.raises(ValueError, match="dtype is outside"):
        op(x, x)


class FullUnionFP8RejectFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, dtype",
            [
                pytest.param(AddFwdOp, torch.float8_e4m3fn, marks=pytest.mark.smoke),
                pytest.param(SubFwdOp, torch.float8_e5m2, marks=pytest.mark.smoke),
                pytest.param(MulFwdOp, torch.float8_e4m3fn, marks=pytest.mark.smoke),
                pytest.param(MaximumFwdOp, torch.float8_e5m2, marks=pytest.mark.smoke),
                pytest.param(MinimumFwdOp, torch.float8_e4m3fn, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FullUnionFP8RejectFixture
def test_full_union_binary_ops_reject_fp8_dtype(
    op_cls,
    dtype: torch.dtype,
) -> None:
    """Add/Sub/Mul/Maximum/Minimum reject fp8 at the public op layer.

    Pins the manifest dtype union: even though the kernel templates can
    compile for fp8, the elementwise_binary manifest stops at float32,
    so the public ops must refuse fp8 at construction time.
    """
    shape = (16,)
    op = op_cls()
    x = torch.zeros(shape, device=run_device()).to(dtype)
    with pytest.raises(ValueError, match="dtype is outside"):
        op(x, x)


@pytest.mark.smoke
def test_add_bool_broadcast() -> None:
    """AddFwdOp(bool) with broadcast inputs lowers via the forced 'direct'
    strategy and still matches torch.logical_or semantics."""
    a_shape = (8, 16)
    b_shape = (1, 16)
    a = torch.randint(0, 2, a_shape, device=run_device()).to(torch.bool)
    b = torch.randint(0, 2, b_shape, device=run_device()).to(torch.bool)
    op = AddFwdOp()
    ref = torch.logical_or(a, b)
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lerp_tensor_cancellation_uses_float_intermediates(dtype: torch.dtype) -> None:
    """Rounding end-start to the storage dtype can erase a nonzero midpoint."""
    workload = LerpCancellationWorkload(dtype)
    TestBase.check(workload, LerpTensorFwdOp(), *workload.gen_inputs())
