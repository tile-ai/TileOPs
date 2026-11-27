"""Tests for the comparison elementwise ops (eq, ne, gt, lt, ge, le), which return bool."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.elementwise import EqFwdOp, GeFwdOp, GtFwdOp, LeFwdOp, LtFwdOp, NeFwdOp
from tileops.backend import BUILTIN
from workloads.device import run_device
from workloads.elementwise import BinarySameShapeCase, ElementwiseWorkload
from workloads.numerics import compare_outputs


class ComparisonTest(BinarySameShapeCase, TestBase):
    pass


_CMP_OP_CASES = [
    (EqFwdOp, torch.eq),
    (NeFwdOp, torch.ne),
    (GtFwdOp, torch.gt),
    (LtFwdOp, torch.lt),
    (GeFwdOp, torch.ge),
    (LeFwdOp, torch.le),
]


class ComparisonFixture(FixtureBase):
    PARAMS = [
        ("op_cls", [pytest.param(op_cls, id=op_cls.__name__) for op_cls, _ in _CMP_OP_CASES]),
        (
            "dtype",
            [
                pytest.param(torch.float16, marks=pytest.mark.smoke),
                pytest.param(torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@ComparisonFixture
def test_comparison_op(op_cls, dtype: torch.dtype) -> None:
    test = ComparisonTest(4_096, dtype, op_cls.__name__)
    test.check(op_cls(), *test.gen_inputs())


# Every op on the bias broadcast, whose float16 run builds its bool bytes from the op's
# two-lane compare; the other broadcast patterns on one op.
_BROADCAST_CASES = [
    pytest.param(
        op_cls,
        (2, 64, 128),
        (1, 1, 128),
        marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
        id=f"{op_cls.__name__}-bias",
    )
    for i, (op_cls, _) in enumerate(_CMP_OP_CASES)
] + [
    pytest.param(EqFwdOp, (2, 64, 128), (2, 64, 1), marks=pytest.mark.full, id="column"),
    pytest.param(EqFwdOp, (64, 128), (1, 1), marks=pytest.mark.full, id="scalar"),
]


@pytest.mark.parametrize("op_cls, a_shape, b_shape", _BROADCAST_CASES)
def test_comparison_broadcast(op_cls, a_shape, b_shape) -> None:
    a = torch.randn(*a_shape, dtype=torch.float16, device=run_device())
    b = torch.randn(*b_shape, dtype=torch.float16, device=run_device())
    ComparisonTest(a.numel(), a.dtype, op_cls.__name__).check(op_cls(), a, b)


# Integer and bool operands

_INT_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def _gen_int_inputs(n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype == torch.uint8:
        lo, hi = 0, 16
    elif dtype == torch.int8:
        lo, hi = -16, 16
    else:
        lo, hi = -64, 64
    a = torch.randint(lo, hi, (n,), dtype=dtype, device=run_device())
    b = torch.randint(lo, hi, (n,), dtype=dtype, device=run_device())
    # Inject some equal positions so eq/ge/le exercise the True branch.
    b[: n // 4] = a[: n // 4]
    return a, b


# Every integer dtype on one op; every op on int32 below.
class ComparisonIntDtypeFixture(FixtureBase):
    PARAMS = [
        ("dtype", [pytest.param(dt, marks=pytest.mark.smoke) for dt in _INT_DTYPES]),
    ]


@ComparisonIntDtypeFixture
def test_comparison_integer_dtype_eq(dtype: torch.dtype) -> None:
    """EqFwdOp matches torch.eq on every manifest-declared int dtype."""
    n = 4_096
    a, b = _gen_int_inputs(n, dtype)
    op = EqFwdOp()
    test = ComparisonTest(a.numel(), a.dtype, "EqFwdOp")
    test.check(op, a, b)


class ComparisonOpIntFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, ref_fn",
            [
                pytest.param(op_cls, ref_fn, marks=pytest.mark.smoke)
                for op_cls, ref_fn in _CMP_OP_CASES
            ],
        ),
    ]


@ComparisonOpIntFixture
def test_comparison_op_int32(op_cls, ref_fn) -> None:
    """Each comparison op matches its torch reference on int32 inputs."""
    n = 4_096
    a, b = _gen_int_inputs(n, torch.int32)
    op = op_cls()
    test = ComparisonTest(a.numel(), a.dtype, op_cls.__name__)
    test.check(op, a, b)


class ComparisonBoolDtypeFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, ref_fn",
            [
                pytest.param(op_cls, ref_fn, marks=pytest.mark.smoke)
                for op_cls, ref_fn in _CMP_OP_CASES
            ],
        ),
    ]


@ComparisonBoolDtypeFixture
def test_comparison_bool_dtype(op_cls, ref_fn) -> None:
    """Comparison ops match torch reference on torch.bool inputs."""
    n = 4_096
    a = torch.randint(0, 2, (n,), device=run_device()).to(torch.bool)
    b = torch.randint(0, 2, (n,), device=run_device()).to(torch.bool)
    op = op_cls()
    test = ComparisonTest(a.numel(), a.dtype, op_cls.__name__)
    test.check(op, a, b)


# Dtypes outside the manifest dtype union


class ComparisonRejectFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, dtype",
            [
                pytest.param(EqFwdOp, torch.complex64, marks=pytest.mark.smoke),
                pytest.param(NeFwdOp, torch.float8_e4m3fn, marks=pytest.mark.smoke),
                pytest.param(GtFwdOp, torch.float8_e5m2, marks=pytest.mark.smoke),
                pytest.param(LtFwdOp, torch.complex64, marks=pytest.mark.full),
                pytest.param(GeFwdOp, torch.float8_e4m3fn, marks=pytest.mark.full),
                pytest.param(LeFwdOp, torch.float8_e5m2, marks=pytest.mark.full),
            ],
        ),
    ]


@ComparisonRejectFixture
def test_comparison_rejects_unsupported_dtype(
    op_cls,
    dtype: torch.dtype,
) -> None:
    """Comparison ops reject dtypes outside the supported set (e.g. complex)."""
    shape = (16,)
    op = op_cls()
    x = torch.zeros(shape, device=run_device(), dtype=dtype)
    with pytest.raises(ValueError, match="dtype is outside"):
        op(x, x)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_comparison_bool_result_from_int8_storage() -> None:
    """A broadcast comparison writes an int8 buffer the op returns as a bool tensor."""
    a = torch.randn(2, 64, 128, device=run_device(), dtype=torch.float16)
    b = torch.randn(1, 1, 128, device=run_device(), dtype=torch.float16)
    op = GtFwdOp(target=BUILTIN)
    ComparisonTest(a.numel(), a.dtype, "GtFwdOp").check(op, a, b)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_comparison_nan_ordering(dtype: torch.dtype) -> None:
    """Against a NaN, ``ne`` answers true and the ordered comparisons false.

    Why: CUDA's half comparison intrinsics are all ordered, so ``ne`` needs its own form.
    """
    nan = float("nan")
    a = torch.tensor([nan, nan, 1.0, 1.0, 2.0], device=run_device(), dtype=dtype)
    b = torch.tensor([nan, 1.0, nan, 1.0, 3.0], device=run_device(), dtype=dtype)
    for op_cls, ref_fn in (
        (EqFwdOp, torch.eq),
        (NeFwdOp, torch.ne),
        (LtFwdOp, torch.lt),
        (LeFwdOp, torch.le),
        (GtFwdOp, torch.gt),
        (GeFwdOp, torch.ge),
    ):
        compare_outputs(
            op_cls()(a, b),
            ref_fn(a, b),
            ElementwiseWorkload(op_cls.__name__, (a, b)).verification(a, b),
        )


_RUN_BROADCAST_CASES = [
    pytest.param(
        (8, 1024), (1, 1024), torch.float32, id="full-block-a-streams", marks=pytest.mark.smoke
    ),
    pytest.param(
        (1, 1024), (8, 1024), torch.float32, id="full-block-b-streams", marks=pytest.mark.smoke
    ),
    pytest.param(
        (8, 1024), (1, 1024), torch.float16, id="full-block-plain", marks=pytest.mark.smoke
    ),
    pytest.param(
        (2, 8, 384), (8, 1), torch.float32, id="vector-grid-streams", marks=pytest.mark.smoke
    ),
    pytest.param((2, 517), (1, 517), torch.float32, id="ragged-row", marks=pytest.mark.smoke),
    # Past 256 blocks of 512 threads, and rows that end mid-block: the wide vector grid.
    pytest.param(
        (4, 256, 56, 56), (256, 1, 1), torch.float16, id="wide-vector-grid", marks=pytest.mark.full
    ),
]


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("a_shape, b_shape, dtype", _RUN_BROADCAST_CASES)
def test_comparison_run_broadcast(a_shape, b_shape, dtype) -> None:
    """A row broadcast whose bool result each thread writes from its own contiguous run."""
    a = torch.randn(*a_shape, dtype=dtype, device=run_device())
    b = torch.randn(*b_shape, dtype=dtype, device=run_device())
    ComparisonTest(a.numel(), a.dtype, "LtFwdOp").check(LtFwdOp(target=BUILTIN), a, b)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_comparison_offset_view() -> None:
    """Operands starting off a 16-byte boundary are read in vectors only after a copy."""
    a = torch.randn(4097, dtype=torch.float32, device=run_device())[1:]
    b = torch.randn(4097, dtype=torch.float32, device=run_device())[1:]
    ComparisonTest(a.numel(), a.dtype, "LtFwdOp").check(LtFwdOp(target=BUILTIN), a, b)
