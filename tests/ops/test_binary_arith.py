"""Tests for the binary arithmetic elementwise ops with broadcast."""

import functools

import pytest
import torch

from tests.cuda_graph_contract import check_replay
from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.elementwise import (
    AddFwdOp,
    DivFwdOp,
    FloorDivideFwdOp,
    LerpFwdOp,
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
    BinaryPositiveCase,
    BinarySameShapeCase,
    ElementwiseCall,
    ElementwiseWorkload,
    Float16DivisorClassesWorkload,
    LerpCancellationWorkload,
    LerpCase,
    PowPositiveWorkload,
)
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call


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


# One op per broadcast grid configuration: unstaged (sub), staged with a guarded quotient
# and no bias-row reuse (remainder), staged (maximum).
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
        "remainder",
        RemainderFwdOp,
        torch.remainder,
        lambda s, d: torch.rand(*s, dtype=d, device=run_device()) + 0.1,
        lambda s, d: torch.rand(*s, dtype=d, device=run_device()) + 0.1,
    ),
    (
        "maximum",
        MaximumFwdOp,
        lambda a, b: torch.maximum(a.float(), b.float()).to(a.dtype),
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


class BinarySameShapeTest(BinarySameShapeCase, TestBase):
    pass


class BinaryPositiveTest(BinaryPositiveCase, TestBase):
    pass


class PowPositiveTest(PowPositiveWorkload, TestBase):
    pass


class BinaryArithOpFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, make_test",
            [
                pytest.param(
                    AddFwdOp, lambda n, d: BinarySameShapeTest(n, d, "AddFwdOp"), id="add"
                ),
                pytest.param(
                    SubFwdOp, lambda n, d: BinarySameShapeTest(n, d, "SubFwdOp"), id="sub"
                ),
                pytest.param(
                    MulFwdOp, lambda n, d: BinarySameShapeTest(n, d, "MulFwdOp"), id="mul"
                ),
                pytest.param(DivFwdOp, lambda n, d: BinaryPositiveTest(n, d, "DivFwdOp"), id="div"),
                pytest.param(
                    RemainderFwdOp,
                    lambda n, d: BinaryPositiveTest(n, d, "RemainderFwdOp"),
                    id="remainder",
                ),
                pytest.param(PowFwdOp, PowPositiveTest, id="pow"),
                pytest.param(
                    FloorDivideFwdOp,
                    lambda n, d: BinaryPositiveTest(n, d, "FloorDivideFwdOp"),
                    id="floor_divide",
                ),
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


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="MulFwdOp")
@pytest.mark.parametrize(
    "T, A, B",
    [
        pytest.param(
            "float32", [1, 64], [1], marks=pytest.mark.smoke, id="deepseek-v4-flash-indexer"
        ),
        pytest.param(
            "bfloat16", [1, 24 * 256], [1, 24 * 256], marks=pytest.mark.full,
            id="qwen3.8-27b-attention-gate",
        ),
    ],
)  # fmt: skip
def test_mul_graph_replay(T: str, A: list[int], B: list[int]) -> None:
    """A decode step's product of two operands."""
    check_replay(ElementwiseCall(manifest_call("MulFwdOp", {"T": T}, A=A, B=B)), MulFwdOp())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        # Wide enough that float32 keeps two vectors a thread.
        pytest.param((1 << 18,), (1 << 18,), id="same", marks=pytest.mark.smoke),
        pytest.param((4, 2048), (1, 2048), id="bias", marks=pytest.mark.full),
        pytest.param((2, 16, 56, 56), (16, 1, 1), id="channel", marks=pytest.mark.full),
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
        pytest.param((1 << 16,), (1 << 16,), id="same", marks=pytest.mark.smoke),
        pytest.param((64, 1024), (1, 1024), id="bias", marks=pytest.mark.full),
        pytest.param((4, 64, 35, 35), (64, 1, 1), id="channel-ragged", marks=pytest.mark.full),
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


# Every DivFwdOp grid that runs a guarded quotient: same-shape blocks, the bias grid with
# either operand as the bias row, staged rows holding a stride-0 operand, the flat channel
# grid and ragged channel rows.
_DIV_LAYOUTS = [
    pytest.param((1 << 16,), (1 << 16,), id="same", marks=pytest.mark.smoke),
    pytest.param((256, 1024), (1, 1024), id="bias", marks=pytest.mark.full),
    pytest.param((1, 1024), (256, 1024), id="bias-a", marks=pytest.mark.full),
    pytest.param((256, 2048), (256, 1), id="column", marks=pytest.mark.full),
    pytest.param((4, 64, 16, 16), (64, 1, 1), id="channel", marks=pytest.mark.full),
    pytest.param((4, 64, 35, 35), (64, 1, 1), id="channel-ragged", marks=pytest.mark.full),
]


@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("a_shape, b_shape", _DIV_LAYOUTS)
def test_div_matches_torch_bit_for_bit(a_shape, b_shape, dtype) -> None:
    """bfloat16 and float32 division give torch's quotient on the fast path and in the fallback.

    Ordinary operands keep most threads on the fast path. Random bit patterns, zeros
    and float32 operands just inside and outside ``[2**-31, 2**33)`` send neighbouring
    threads to the fallback. The pinned float32 pairs are quotients the fast path rounds
    correctly only with its reciprocal's Newton step.
    """
    gen = torch.Generator(device=run_device()).manual_seed(0)
    bits = torch.int16 if dtype == torch.bfloat16 else torch.int32

    def operand(shape):
        x = torch.randn(shape, generator=gen, device=run_device())
        raw = torch.randint(-(2**31), 2**31, shape, generator=gen, device=run_device())
        pick = torch.randint(0, 64, shape, generator=gen, device=run_device())
        ends = torch.tensor([2.0**-31, 2.0**33], device=run_device())
        zero = torch.zeros(1, device=run_device())
        edges = torch.cat([ends, torch.nextafter(ends, torch.zeros_like(ends)), zero])
        x = torch.where(pick == 0, raw.to(torch.int32).view(torch.float32), x)
        return torch.where(pick == 1, edges[raw.remainder(5)], x).to(dtype)

    a, b = operand(a_shape), operand(b_shape)
    if a_shape == b_shape and dtype == torch.float32:
        pinned = torch.tensor(
            [[0x3F8CCAF1, 0x3F90DDD7], [0x3F8E1F31, 0x3F90DEBB], [0x3F92A0A1, 0x3F9FF717]],
            dtype=torch.int32,
            device=run_device(),
        ).view(torch.float32)
        # The first 2048 elements hold only ordinary operands, so the pinned pairs' thread
        # stays on the fast path.
        a[:2048], b[:2048] = torch.randn(2, 2048, generator=gen, device=run_device()).add(3)
        a[:3], b[:3] = pinned[:, 0], pinned[:, 1]
    out, ref = DivFwdOp()(a, b), torch.div(a, b)
    number = ~ref.isnan()
    assert torch.equal(out.isnan(), ~number)
    assert torch.equal(out[number].view(bits), ref[number].view(bits))


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("broadcast", [False, True], ids=["same", "column"])
def test_float16_div_rounds_subnormal_ties_as_torch(broadcast) -> None:
    """float16 division gives torch's quotient where a two-ulp quotient rounds a subnormal
    tie the wrong way, and keeps the special results of zero, infinite and extreme divisors."""
    workload = Float16DivisorClassesWorkload("DivFwdOp", broadcast)
    TestBase.check(workload, DivFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        pytest.param((1 << 16,), (1 << 16,), id="same"),
        pytest.param((2, 64, 56, 56), (64, 1, 1), id="channel"),
    ],
)
def test_float32_floor_ops_match_torch_bit_for_bit(a_shape, b_shape) -> None:
    """The float32 floored ops' approximate-quotient tier gives torch's result bit for bit.

    Multiples of b and multiples one or two ulp off put the quotient on and beside a
    whole number. Quotients beside ``2**20``, divisors on and beside ``2**-126`` and
    ``2**126``, and a divisor past ``2**108`` fall on both sides of the tier's guard,
    in neighbouring threads.
    """
    gen = torch.Generator(device=run_device()).manual_seed(0)

    def uniform(shape, lo, hi):
        return torch.rand(shape, generator=gen, device=run_device()) * (hi - lo) + lo

    def randint(shape, lo, hi):
        return torch.randint(lo, hi, shape, generator=gen, device=run_device())

    b = torch.exp2(uniform(b_shape, -30.0, 30.0)) * randint(b_shape, 0, 2).mul(2).sub(1)
    edge = randint(b_shape, 0, 8)
    b = torch.where(edge == 0, b.sign() * torch.exp2(uniform(b_shape, -127.0, -125.0)), b)
    b = torch.where(edge == 1, b.sign() * torch.exp2(uniform(b_shape, 125.0, 127.0)), b)
    # The range ends, the floats just outside them, and a b whose 2**20 * |b| overflows.
    ends = torch.tensor([2.0**-126, 2.0**126], device=run_device())
    outside = torch.nextafter(ends, torch.tensor([0.0, float("inf")], device=run_device()))
    pinned = torch.cat([ends, outside, torch.tensor([2.0**108], device=run_device())])
    b.view(-1)[: 2 * len(pinned)] = torch.cat([pinned, -pinned])
    k = randint(a_shape, -(1 << 21), 1 << 21).float()
    k = torch.where(randint(a_shape, 0, 4) == 0, k.sign() * (2**20 + randint(a_shape, -2, 3)), k)
    # Beside 2**126 a whole quotient of at most one keeps a finite.
    k = torch.where(b.abs().expand(a_shape) > 2.0**120, randint(a_shape, -1, 2).float(), k)
    a = (k * b).view(torch.int32).add(randint(a_shape, -2, 3).int()).view(torch.float32)
    cases = [
        (FloorDivideFwdOp(), torch.floor_divide),
        (DivFwdOp(rounding_mode="floor"), lambda x, y: torch.div(x, y, rounding_mode="floor")),
        (RemainderFwdOp(), torch.remainder),
    ]
    for op, ref_fn in cases:
        out, ref = op(a, b), ref_fn(a, b)
        number = ~ref.isnan()
        assert torch.equal(out.isnan(), ~number)
        assert torch.equal(out[number].view(torch.int32), ref[number].view(torch.int32))


# Lerp with a Number weight


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
    """Validate lerp across Number weights, one instance serving every value."""
    op = LerpFwdOp()
    for weight in [0.0, 0.3, 0.5, 0.7, 1.0]:
        test = LerpTest(n_total, dtype, weight=weight)
        test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [
        pytest.param((2, 64, 128), (2, 64, 128), marks=pytest.mark.smoke),
        pytest.param((3, 1), (1, 4), marks=pytest.mark.full),
    ],
)
def test_lerp_number_weight_broadcast(a_shape, b_shape) -> None:
    a = torch.randn(a_shape, dtype=torch.float16, device=run_device())
    b = torch.randn(b_shape, dtype=torch.float16, device=run_device())
    workload = ElementwiseWorkload("LerpFwdOp", (a, b, 0.5))
    TestBase.check(workload, LerpFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
def test_lerp_rejects_integer_dtype() -> None:
    a = torch.ones(16, device=run_device(), dtype=torch.int32)
    with pytest.raises(ValueError, match="dtype is outside"):
        LerpFwdOp()(a, a, 0.5)


# Maximum/Minimum NaN and signed zero


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls", [MaximumFwdOp, MinimumFwdOp])
def test_max_min_return_a_canonical_nan(op_cls) -> None:
    """A NaN operand gives NaN, without the operand's payload: two NaN operands with
    different bits give the same result bits."""
    other = torch.tensor([1.0], dtype=torch.float16, device=run_device())
    results = [
        op_cls()(
            torch.tensor([bits], dtype=torch.uint16, device=run_device()).view(torch.float16), other
        )
        for bits in (0xFE00, 0x7E01)
    ]
    assert all(torch.isnan(r).all() for r in results)
    assert torch.equal(results[0].view(torch.uint16), results[1].view(torch.uint16))


class MaxMinSpecialValuesFixture(FixtureBase):
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


@MaxMinSpecialValuesFixture
def test_max_min_nan_and_signed_zero(op_cls, torch_ref, dtype: torch.dtype) -> None:
    """NaN in either or both operands gives NaN; maximum(+0, -0) is +0 and minimum(-0, +0)
    is -0 in every operand order."""
    nan = float("nan")
    a = torch.tensor([nan, 1.0, nan, 0.0, -0.0, 0.0, -0.0, 3.0], dtype=dtype, device=run_device())
    b = torch.tensor([1.0, nan, nan, -0.0, 0.0, 0.0, -0.0, 2.0], dtype=dtype, device=run_device())
    op = op_cls()
    ref = torch_ref(a, b)
    with torch.no_grad():
        out = op(a, b)
    assert torch.equal(torch.isnan(out), torch.isnan(ref)), f"out={out}, ref={ref}"
    mask = ~torch.isnan(ref)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, ()).verification(a, b))
    assert torch.equal(torch.signbit(out[mask]), torch.signbit(ref[mask])), f"out={out}"


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
            ],
        ),
    ]


@FloatOnlyBinaryRejectFixture
def test_float_only_binary_ops_reject_integer_dtype(op_cls, dtype: torch.dtype) -> None:
    """Float-only binary ops reject integer dtypes at the manifest dtype gate."""
    shape = (16,)
    op = op_cls()
    a = torch.ones(shape, device=run_device(), dtype=dtype)
    with pytest.raises(ValueError, match="dtype is outside"):
        op(a, a)


@pytest.mark.smoke
def test_binary_op_rejects_runtime_dtype_mismatch() -> None:
    """Operands of different dtypes are rejected before any kernel is built."""
    op = SubFwdOp()
    a = torch.randn(16, device=run_device(), dtype=torch.float32)
    b = torch.randn(16, device=run_device(), dtype=torch.float16)
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


# Maximum/minimum on launches long enough to keep every element a thread


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
            ],
        ),
    ]


@OptimizedMaxMinFixture
def test_max_min_optimized_large(op_cls, torch_ref, n_total: int, dtype: torch.dtype) -> None:
    shape = (n_total,)
    a = torch.randn(*shape, device=run_device(), dtype=dtype)
    b = torch.randn(*shape, device=run_device(), dtype=dtype)
    op = op_cls()
    ref = torch_ref(a, b)
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


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
        op = op_cls()
        if tune:
            op.request_tune()
        compare_outputs(
            op(a, b),
            ref_fn(a, b),
            ElementwiseWorkload(op_cls.__name__, (a, b)).verification(a, b),
        )


# LerpFwdOp with a Tensor weight


_LERP_TENSOR_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("dtype", _LERP_TENSOR_DTYPES)
def test_lerp_tensor_same_shape(dtype: torch.dtype) -> None:
    """LerpFwdOp matches torch.lerp on same-shape inputs."""
    shape = (4, 8)
    a = torch.randn(shape, device=run_device(), dtype=dtype)
    b = torch.randn(shape, device=run_device(), dtype=dtype)
    w = torch.rand(shape, device=run_device(), dtype=dtype)
    op = LerpFwdOp()
    out = op(a, b, w)
    ref = torch.lerp(a, b, w)
    compare_outputs(
        out, ref, ElementwiseWorkload(type(op).__name__, (a, b, w)).verification(*(a, b, w))
    )


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    ("a_shape", "b_shape", "w_shape", "dtype"),
    [
        # Per-channel weight: whole vectors of input and end, one weight element a vector.
        pytest.param((2, 16, 7, 8), (2, 16, 7, 8), (16, 1, 1), torch.float16, id="channel-weight"),
        # A 0-dim weight: one coalesced row, the weight read at stride 0 throughout.
        pytest.param((64, 96), (64, 96), (), torch.bfloat16, id="scalar-weight"),
        # An odd row length leaves one-element vectors; end and weight broadcast differently.
        pytest.param((5, 33), (33,), (5, 1), torch.float32, id="odd-row"),
    ],
)
def test_lerp_tensor_reads_broadcast_operands_in_place(a_shape, b_shape, w_shape, dtype) -> None:
    """Operands that broadcast are read at stride 0 and still match torch.lerp."""
    a = torch.randn(a_shape, device=run_device(), dtype=dtype)
    b = torch.randn(b_shape, device=run_device(), dtype=dtype)
    w = torch.rand(w_shape, device=run_device(), dtype=dtype) * 1.5 - 0.25
    TestBase.check(ElementwiseWorkload("LerpFwdOp", (a, b, w)), LerpFwdOp(), a, b, w)


@pytest.mark.smoke
def test_lerp_tensor_rejects_fp8_dtype() -> None:
    """LerpFwdOp rejects fp8, which its manifest does not declare."""
    shape = (4, 8)
    op = LerpFwdOp()
    x = torch.zeros(shape, device=run_device()).to(torch.float8_e4m3fn)
    with pytest.raises((ValueError, TypeError)):
        op(x, x, x)


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
def test_lerp_tensor_dtype_mismatch_rejected() -> None:
    """LerpFwdOp rejects operands whose dtypes disagree."""
    shape = (4, 8)
    op = LerpFwdOp()
    a = torch.randn(shape, device=run_device(), dtype=torch.float32)
    b = torch.randn(shape, device=run_device(), dtype=torch.float32)
    w_bad = torch.rand(shape, device=run_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="weight.dtype"):
        op(a, b, w_bad)


# DivFwdOp rounding modes


_DIV_ROUNDING_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
_DIV_ROUNDING_MODES = ["trunc", "floor"]


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("rounding_mode", _DIV_ROUNDING_MODES)
@pytest.mark.parametrize("dtype", _DIV_ROUNDING_DTYPES)
def test_div_rounding_mode_eager(rounding_mode: str, dtype: torch.dtype) -> None:
    """DivFwdOp(rounding_mode=...) matches torch.div for trunc and floor."""
    shape = (64, 256)
    # Quotients of both signs, with b kept away from zero.
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


# Integer and bool operands

_BINARY_INT_DTYPES = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]
# Add / Mul / Maximum / Minimum accept bool too; Sub, as torch.sub, does not.
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


# Every integer dtype on one op; every op on int32 below.
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


# On bool, add and maximum are logical OR, mul and minimum logical AND.
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
    """Add/Mul/Maximum/Minimum match logical OR/AND on torch.bool inputs."""
    n = 4_096
    a = torch.randint(0, 2, (n,), device=run_device()).to(torch.bool)
    b = torch.randint(0, 2, (n,), device=run_device()).to(torch.bool)
    op = op_cls()
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


@pytest.mark.smoke
def test_add_rejects_a_float_alpha_for_integer_input() -> None:
    """``torch.add`` refuses a floating-point ``alpha`` for integral inputs."""
    x = torch.ones(8, device=run_device(), dtype=torch.int32)
    with pytest.raises(ValueError, match="alpha"):
        AddFwdOp(alpha=0.5)(x, x)


@pytest.mark.smoke
def test_sub_rejects_bool_dtype() -> None:
    """SubFwdOp rejects bool, as torch.sub does."""
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
    """Add/Sub/Mul/Maximum/Minimum reject fp8, which their manifest dtype union omits."""
    shape = (16,)
    op = op_cls()
    x = torch.zeros(shape, device=run_device()).to(dtype)
    with pytest.raises(ValueError, match="dtype is outside"):
        op(x, x)


@pytest.mark.smoke
def test_add_bool_broadcast() -> None:
    """AddFwdOp on broadcast bool operands (the direct strategy) is logical OR."""
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
    TestBase.check(workload, LerpFwdOp(), *workload.gen_inputs())
