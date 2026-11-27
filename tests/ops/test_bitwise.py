"""Tests for the bitwise elementwise ops (bitwise_and, bitwise_or, bitwise_xor, bitwise_not)."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.elementwise import (
    BitwiseAndFwdOp,
    BitwiseNotFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
)
from workloads.device import run_device
from workloads.elementwise import (
    BitwiseCase,
    BitwiseNotWorkload,
    ElementwiseWorkload,
    FullRangeCase,
)
from workloads.numerics import compare_outputs


class BitwiseTest(BitwiseCase, TestBase):
    pass


class FullRangeTest(FullRangeCase, TestBase):
    pass


_BITWISE_OPS = [
    ("bitwise_and", BitwiseAndFwdOp, torch.bitwise_and),
    ("bitwise_or", BitwiseOrFwdOp, torch.bitwise_or),
    ("bitwise_xor", BitwiseXorFwdOp, torch.bitwise_xor),
]


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls", [BitwiseAndFwdOp, BitwiseOrFwdOp, BitwiseXorFwdOp])
def test_bitwise_binary_op(op_cls) -> None:
    test = BitwiseTest(4_096, op_cls.__name__)
    test.check(op_cls(), *test.gen_inputs())


# Every op on the bias broadcast; the other broadcast patterns on one op.


class BitwiseBroadcastFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        bias = [
            pytest.param(
                name,
                cls,
                ref,
                (2, 64, 128),
                (1, 1, 128),
                marks=pytest.mark.smoke if j == 0 else pytest.mark.full,
            )
            for j, (name, cls, ref) in enumerate(_BITWISE_OPS)
        ]
        name, cls, ref = _BITWISE_OPS[0]
        others = [
            pytest.param(name, cls, ref, a_s, b_s, marks=pytest.mark.full)
            for a_s, b_s in (((2, 64, 128), (2, 64, 1)), ((64, 128), (1, 1)))
        ]
        return [("op_name, op_cls, ref_fn, a_shape, b_shape", bias + others)]


@BitwiseBroadcastFixture
def test_bitwise_broadcast(
    op_name,
    op_cls,
    ref_fn,
    a_shape,
    b_shape,
) -> None:
    a = torch.randint(-1000, 1000, a_shape, dtype=torch.int32, device=run_device())
    b = torch.randint(-1000, 1000, b_shape, dtype=torch.int32, device=run_device())
    op = op_cls()
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


class LongInt64BitwiseFixture(FixtureBase):
    """int64 operands long enough to fill the device several times over, in each layout."""

    PARAMS = [
        (
            "a_shape, b_shape",
            [
                pytest.param((4096, 4096), (4096, 4096), marks=pytest.mark.smoke),
                pytest.param((16_777_139,), (16_777_139,), marks=pytest.mark.full),
                pytest.param((16, 256, 56, 56), (256, 1, 1), marks=pytest.mark.full),
                pytest.param((4096, 4096), (1, 4096), marks=pytest.mark.full),
                pytest.param((4096, 4096), (4096, 1), marks=pytest.mark.full),
                pytest.param((4096, 4097), (1, 4097), marks=pytest.mark.full),
                pytest.param((131_072, 128), (131_072, 1), marks=pytest.mark.full),
            ],
        ),
    ]


@LongInt64BitwiseFixture
def test_bitwise_or_long_int64(a_shape, b_shape) -> None:
    test = FullRangeTest(((a_shape, torch.int64), (b_shape, torch.int64)), "BitwiseOrFwdOp")
    test.check(BitwiseOrFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
def test_bitwise_or_int64_under_tuning() -> None:
    test = FullRangeTest((((4096,), torch.int64), ((4096,), torch.int64)), "BitwiseOrFwdOp")
    op = BitwiseOrFwdOp()
    op.request_tune()
    test.check(op, *test.gen_inputs())


class BoolBitwiseFixture(FixtureBase):
    PARAMS = [
        (
            "op_name, op_cls, ref_fn, a_shape, b_shape",
            [
                pytest.param(
                    name,
                    cls,
                    ref,
                    a_s,
                    b_s,
                    marks=pytest.mark.smoke if a_s == b_s else pytest.mark.full,
                )
                for a_s, b_s in [
                    ((2048, 4096), (2048, 4096)),
                    ((2, 512, 768), (1, 1, 768)),
                ]
                for name, cls, ref in _BITWISE_OPS
            ],
        ),
    ]


@BoolBitwiseFixture
def test_bool_bitwise_fast_path(
    op_name,
    op_cls,
    ref_fn,
    a_shape,
    b_shape,
) -> None:
    a = torch.randint(0, 2, a_shape, device=run_device()).bool()
    b = torch.randint(0, 2, b_shape, device=run_device()).bool()
    op = op_cls()
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    assert out.dtype == torch.bool
    compare_outputs(out, ref, ElementwiseWorkload(type(op).__name__, (a, b)).verification(*(a, b)))


class BitwiseFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(1_048_576, torch.bool, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.uint8, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.int8, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.int16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.int32, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.int64, marks=pytest.mark.smoke),
            ],
        ),
    ]


class BitwiseNotTest(BitwiseNotWorkload, TestBase):
    pass


@BitwiseFixture
def test_bitwise_not(n_total: int, dtype: torch.dtype) -> None:
    test = BitwiseNotTest(n_total, dtype)
    op = BitwiseNotFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "n_total",
    [
        pytest.param(16_777_216, marks=pytest.mark.smoke),
        pytest.param(16_777_139, marks=pytest.mark.full),
    ],
)
def test_bitwise_not_long_int64(n_total: int) -> None:
    """int64 input long enough to fill the device several times over."""
    test = FullRangeTest((((n_total,), torch.int64),), "BitwiseNotFwdOp")
    test.check(BitwiseNotFwdOp(), *test.gen_inputs())


# Float dtypes, outside the binary bitwise ops' manifest


class BitwiseBinaryRejectFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, dtype",
            [
                pytest.param(BitwiseAndFwdOp, torch.float16, marks=pytest.mark.smoke),
                pytest.param(BitwiseOrFwdOp, torch.float16, marks=pytest.mark.full),
                pytest.param(BitwiseXorFwdOp, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@BitwiseBinaryRejectFixture
def test_bitwise_binary_rejects_float_dtype(op_cls, dtype: torch.dtype) -> None:
    """Binary bitwise ops only support integer dtypes; floats must be rejected."""
    shape = (16,)
    op = op_cls()
    x = torch.zeros(shape, device=run_device(), dtype=dtype)
    with pytest.raises(ValueError, match="dtype is outside"):
        op(x, x)
