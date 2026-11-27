"""Tests for the logical elementwise ops (logical_and, logical_or, logical_not).

Nonzero reads as true; the result is bool.
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.elementwise import UnaryKernel
from tileops._ops.elementwise import LogicalAndFwdOp, LogicalNotFwdOp, LogicalOrFwdOp
from workloads.device import run_device
from workloads.elementwise import LogicalCase, LogicalNotWorkload


class LogicalTest(LogicalCase, TestBase):
    pass


class LogicalOpFixture(FixtureBase):
    PARAMS = [
        ("op_cls", [LogicalAndFwdOp, LogicalOrFwdOp]),
        (
            "dtype",
            [
                pytest.param(torch.float16, marks=pytest.mark.smoke),
                pytest.param(torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@LogicalOpFixture
def test_logical_binary_op(op_cls, dtype: torch.dtype) -> None:
    test = LogicalTest(4_096, dtype, op_cls.__name__)
    test.check(op_cls(), *test.gen_inputs())


# Both ops on the bias broadcast; the other broadcast patterns on logical_and.


_LOGICAL_OPS = [
    ("logical_and", LogicalAndFwdOp, torch.logical_and),
    ("logical_or", LogicalOrFwdOp, torch.logical_or),
]


class LogicalBroadcastFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        patterns = [
            ((2, 64, 128), (1, 1, 128)),
            ((2, 64, 128), (2, 64, 1)),
            ((64, 128), (1, 1)),
        ]
        cases = [(op, *patterns[0]) for op in _LOGICAL_OPS] + [
            (_LOGICAL_OPS[0], *pattern) for pattern in patterns[1:]
        ]
        return [
            (
                "op_name, op_cls, ref_fn, a_shape, b_shape",
                [
                    pytest.param(
                        *op, a_s, b_s, marks=pytest.mark.smoke if i == 0 else pytest.mark.full
                    )
                    for i, (op, a_s, b_s) in enumerate(cases)
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


@pytest.mark.parametrize(
    "op_cls, dtype",
    [
        pytest.param(LogicalOrFwdOp, torch.float16, id="or-float16", marks=pytest.mark.smoke),
        pytest.param(LogicalOrFwdOp, torch.bfloat16, id="or-bfloat16", marks=pytest.mark.smoke),
        pytest.param(LogicalOrFwdOp, torch.float32, id="or-float32", marks=pytest.mark.smoke),
        pytest.param(LogicalAndFwdOp, torch.float16, id="and-float16", marks=pytest.mark.full),
    ],
)
def test_logical_nan_is_true(op_cls, dtype: torch.dtype) -> None:
    """A NaN operand is nonzero and reads as true; -0.0 reads as false.

    Equal or negated nonzero pairs, infinities and the smallest subnormal read as true.
    """
    nan, inf = float("nan"), float("inf")
    tiny = torch.finfo(dtype).smallest_normal * torch.finfo(dtype).eps
    a = [nan, nan, nan, 0.0, -0.0, 1.0, 0.0, nan, 1.0, -1.0, inf, -inf, tiny, -tiny, 0.0]
    b = [nan, 0.0, 1.0, nan, 0.0, 0.0, -0.0, -0.0, 1.0, 1.0, inf, inf, -tiny, 0.0, tiny]
    a = torch.tensor(a, device=run_device(), dtype=dtype)
    b = torch.tensor(b, device=run_device(), dtype=dtype)
    test = LogicalTest(a.numel(), a.dtype, op_cls.__name__)
    test.check(op_cls(), a, b)


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
    pass


@LogicalFixture
def test_logical_not(n_total: int, dtype: torch.dtype) -> None:
    test = LogicalNotTest(n_total, dtype)
    op = LogicalNotFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("dtype,tail", [(torch.bool, 0), (torch.bool, 3), (torch.int8, 0)])
def test_logical_not_near_one_wave(tail: int, dtype: torch.dtype) -> None:
    """Bool vector loads, a partial block and the int8 fallback match torch."""
    props = torch.cuda.get_device_properties(run_device())
    resident = props.multi_processor_count * props.max_threads_per_multi_processor
    run = 256 * 64
    n_total = 20 * resident // run * run
    if not UnaryKernel.fits_l2_budget(n_total, 2, props.L2_cache_size):
        pytest.skip("twenty elements per resident thread exceed three quarters of this L2")
    test = LogicalNotTest(n_total + tail, dtype)
    test.check(LogicalNotFwdOp(), *test.gen_inputs())


# The integral and bool dtypes of the binary logical ops; the float ones are covered above.

_INT_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def _gen_int_logical_inputs(
    n: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer operands with zeros among them, so both truth values occur."""
    if dtype == torch.uint8:
        lo, hi = 0, 8
    elif dtype == torch.int8:
        lo, hi = -8, 8
    else:
        lo, hi = -32, 32
    a = torch.randint(lo, hi, (n,), dtype=dtype, device=run_device())
    b = torch.randint(lo, hi, (n,), dtype=dtype, device=run_device())
    a[::3] = 0
    b[::5] = 0
    return a, b


# Every integral dtype and bool on logical_and; logical_or on one dtype of each strategy:
# int8 (direct), int32 (register_copy) and bool (uint8 storage).
class LogicalIntBoolMatrixFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, ref_fn, dtype",
            [
                pytest.param(LogicalAndFwdOp, torch.logical_and, dt, marks=pytest.mark.smoke)
                for dt in (*_INT_DTYPES, torch.bool)
            ]
            + [
                pytest.param(LogicalOrFwdOp, torch.logical_or, dt, marks=pytest.mark.smoke)
                for dt in (torch.int8, torch.int32, torch.bool)
            ],
        ),
    ]


@LogicalIntBoolMatrixFixture
def test_logical_int_bool_matrix(
    op_cls,
    ref_fn,
    dtype: torch.dtype,
) -> None:
    n = 4_096
    if dtype == torch.bool:
        a = torch.randint(0, 2, (n,), device=run_device()).to(torch.bool)
        b = torch.randint(0, 2, (n,), device=run_device()).to(torch.bool)
    else:
        a, b = _gen_int_logical_inputs(n, dtype)
    op = op_cls()
    test = LogicalTest(a.numel(), a.dtype, op_cls.__name__)
    test.check(op, a, b)
