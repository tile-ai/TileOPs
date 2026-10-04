"""Tests for bitwise elementwise ops (bitwise_and, bitwise_or, bitwise_xor, bitwise_not).

Bitwise ops operate on integer inputs. We use int32 tensors for testing
binary bitwise ops, and all bool/integer dtypes for bitwise_not.
Covers L1 smoke correctness.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import (
    BitwiseAndFwdOp,
    BitwiseNotFwdOp,
    BitwiseOrFwdOp,
    BitwiseXorFwdOp,
)
from workloads.device import run_device
from workloads.elementwise import BitwiseCase, BitwiseNotWorkload, ElementwiseWorkload
from workloads.numerics import compare_outputs


class BitwiseTest(BitwiseCase, TestBase):
    pass


class BitwiseAndFixture(FixtureBase):
    PARAMS = [
        (
            "n_total",
            [
                pytest.param(4_096, marks=pytest.mark.smoke),
                pytest.param(16_384, marks=pytest.mark.full),
            ],
        ),
    ]


@BitwiseAndFixture
def test_bitwise_and_op(n_total: int) -> None:
    test = BitwiseTest(n_total, "BitwiseAndFwdOp")
    op = BitwiseAndFwdOp()
    test.check(op, *test.gen_inputs())


class BitwiseOrFixture(FixtureBase):
    PARAMS = [
        (
            "n_total",
            [
                pytest.param(4_096, marks=pytest.mark.smoke),
                pytest.param(16_384, marks=pytest.mark.full),
            ],
        ),
    ]


@BitwiseOrFixture
def test_bitwise_or_op(n_total: int) -> None:
    test = BitwiseTest(n_total, "BitwiseOrFwdOp")
    op = BitwiseOrFwdOp()
    test.check(op, *test.gen_inputs())


class BitwiseXorFixture(FixtureBase):
    PARAMS = [
        (
            "n_total",
            [
                pytest.param(4_096, marks=pytest.mark.smoke),
                pytest.param(16_384, marks=pytest.mark.full),
            ],
        ),
    ]


@BitwiseXorFixture
def test_bitwise_xor_op(n_total: int) -> None:
    test = BitwiseTest(n_total, "BitwiseXorFwdOp")
    op = BitwiseXorFwdOp()
    test.check(op, *test.gen_inputs())


# Broadcast pattern tests for binary bitwise ops (L3)


_BITWISE_OPS = [
    ("bitwise_and", BitwiseAndFwdOp, torch.bitwise_and),
    ("bitwise_or", BitwiseOrFwdOp, torch.bitwise_or),
    ("bitwise_xor", BitwiseXorFwdOp, torch.bitwise_xor),
]


class BitwiseBroadcastFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        patterns = [
            ((2, 64, 128), (1, 1, 128)),  # bias-add
            ((2, 64, 128), (2, 64, 1)),  # row broadcast
            ((64, 128), (1, 1)),  # scalar broadcast
        ]
        return [
            (
                "op_name, op_cls, ref_fn, a_shape, b_shape",
                [
                    pytest.param(
                        name,
                        cls,
                        ref,
                        a_s,
                        b_s,
                        marks=pytest.mark.smoke if i == 0 and j == 0 else pytest.mark.full,
                    )
                    for j, (name, cls, ref) in enumerate(_BITWISE_OPS)
                    for i, (a_s, b_s) in enumerate(patterns)
                ],
            ),
        ]


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
    """Parametrize over torch-supported bitwise_not dtypes."""

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
    """Test fixture for bitwise_not."""


@BitwiseFixture
def test_bitwise_not(n_total: int, dtype: torch.dtype) -> None:
    test = BitwiseNotTest(n_total, dtype)
    op = BitwiseNotFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, marks=pytest.mark.smoke),
        pytest.param(torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param(torch.float32, marks=pytest.mark.smoke),
    ],
)
def test_bitwise_not_rejects_float_dtype(dtype: torch.dtype) -> None:
    from tileops.kernels.elementwise import BitwiseNotFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseNotFwdKernel(N_total=16, dtype=dtype)


# Dtype rejection tests for binary bitwise ops


class BitwiseBinaryRejectFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, dtype",
            [
                pytest.param(BitwiseAndFwdOp, torch.float16, marks=pytest.mark.smoke),
                pytest.param(BitwiseAndFwdOp, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(BitwiseAndFwdOp, torch.float32, marks=pytest.mark.smoke),
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
