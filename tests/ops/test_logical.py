"""Tests for logical elementwise ops (logical_and, logical_or, logical_not).

Logical ops under test accept numeric tensors, interpret non-zero values
as True, and produce boolean outputs. Covers L1 smoke correctness for
binary logical ops, and all supported dtypes for logical_not.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import LogicalAndFwdOp, LogicalNotFwdOp, LogicalOrFwdOp
from workloads.device import run_device
from workloads.elementwise import LogicalCase, LogicalNotWorkload


class LogicalTest(LogicalCase, TestBase):
    pass


class LogicalAndFixture(FixtureBase):
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


@LogicalAndFixture
def test_logical_and_op(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalTest(n_total, dtype, "LogicalAndFwdOp")
    op = LogicalAndFwdOp()
    test.check(op, *test.gen_inputs())


class LogicalOrFixture(FixtureBase):
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


@LogicalOrFixture
def test_logical_or_op(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalTest(n_total, dtype, "LogicalOrFwdOp")
    op = LogicalOrFwdOp()
    test.check(op, *test.gen_inputs())


# Broadcast pattern tests for binary logical ops (L3)


_LOGICAL_OPS = [
    ("logical_and", LogicalAndFwdOp, torch.logical_and),
    ("logical_or", LogicalOrFwdOp, torch.logical_or),
]


class LogicalBroadcastFixture(FixtureBase):
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
                    for j, (name, cls, ref) in enumerate(_LOGICAL_OPS)
                    for i, (a_s, b_s) in enumerate(patterns)
                ],
            ),
        ]


@LogicalBroadcastFixture
def test_logical_broadcast(
    op_name,
    op_cls,
    ref_fn,
    a_shape,
    b_shape,
) -> None:
    dtype = torch.float16
    a = (torch.randn(*a_shape, dtype=dtype, device=run_device()) > 0).to(dtype)
    b = (torch.randn(*b_shape, dtype=dtype, device=run_device()) > 0).to(dtype)
    op = op_cls()
    test = LogicalTest(a.numel(), a.dtype, op_cls.__name__)
    test.check(op, a, b)


@pytest.mark.smoke
def test_logical_and_bool_broadcast() -> None:
    """Bool-input binary broadcast path uses uint8 storage internally."""
    a_shape = (2, 512, 768)
    b_shape = (1, 1, 768)
    a = torch.randint(0, 2, a_shape, device=run_device()).to(torch.bool)
    b = torch.randint(0, 2, b_shape, device=run_device()).to(torch.bool)
    op = LogicalAndFwdOp()
    test = LogicalTest(a.numel(), a.dtype, "LogicalAndFwdOp")
    test.check(op, a, b)


class LogicalFixture(FixtureBase):
    """Parametrize over supported dtypes for logical_not."""

    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.bool, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.uint8, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.int8, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.int16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.int32, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.int64, marks=pytest.mark.smoke),
            ],
        ),
    ]


class LogicalNotTest(LogicalNotWorkload, TestBase):
    """Test fixture for logical_not."""


@LogicalFixture
def test_logical_not(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalNotTest(n_total, dtype)
    op = LogicalNotFwdOp()
    test.check(op, *test.gen_inputs())


# Per-dtype correctness across the manifest dtype union for binary logical
# ops. The manifest declares
# ``bool | uint8 | int8 | int16 | int32 | int64 | float16 | bfloat16 | float32``
# for both LogicalAndFwdOp and LogicalOrFwdOp; the float path is covered
# above. The int / bool cells exercise the kernel's non-zero truthiness
# path on every manifest-declared integral dtype.

_INT_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
_LOGICAL_OP_CASES = [
    (LogicalAndFwdOp, torch.logical_and),
    (LogicalOrFwdOp, torch.logical_or),
]


def _gen_int_logical_inputs(
    n: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate int inputs sprinkled with zeros to exercise both truthy
    and falsy lanes of the non-zero truthiness path.
    """
    if dtype == torch.uint8:
        lo, hi = 0, 8
    elif dtype == torch.int8:
        lo, hi = -8, 8
    else:
        lo, hi = -32, 32
    a = torch.randint(lo, hi, (n,), dtype=dtype, device=run_device())
    b = torch.randint(lo, hi, (n,), dtype=dtype, device=run_device())
    # Force a mix of zeros so non-zero truthiness is non-trivial.
    a[::3] = 0
    b[::5] = 0
    return a, b


# Full (op_cls, dtype) product: every binary logical op must match its
# torch reference on every manifest-declared integral dtype and on bool.
class LogicalIntBoolMatrixFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, ref_fn, dtype",
            [
                pytest.param(op_cls, ref_fn, dt, marks=pytest.mark.smoke)
                for op_cls, ref_fn in _LOGICAL_OP_CASES
                for dt in (*_INT_DTYPES, torch.bool)
            ],
        ),
    ]


@LogicalIntBoolMatrixFixture
def test_logical_int_bool_matrix(
    op_cls,
    ref_fn,
    dtype: torch.dtype,
) -> None:
    """Each binary logical op matches torch on every int / bool dtype."""
    n = 4_096
    if dtype == torch.bool:
        a = torch.randint(0, 2, (n,), device=run_device()).to(torch.bool)
        b = torch.randint(0, 2, (n,), device=run_device()).to(torch.bool)
    else:
        a, b = _gen_int_logical_inputs(n, dtype)
    op = op_cls()
    test = LogicalTest(a.numel(), a.dtype, op_cls.__name__)
    test.check(op, a, b)
