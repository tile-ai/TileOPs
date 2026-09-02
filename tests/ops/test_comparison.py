"""Tests for comparison elementwise ops (eq, ne, gt, lt, ge, le).

All comparison ops output torch.bool. Covers L1 smoke correctness
and L4 edge case for eq.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import EqFwdOp, GeFwdOp, GtFwdOp, LeFwdOp, LtFwdOp, NeFwdOp
from workloads.elementwise import RandnPairWorkload


def _bool_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact comparison for boolean outputs."""
    assert output.dtype == torch.bool, f"Expected bool dtype, got {output.dtype}"
    assert torch.equal(output, output_ref), (
        f"Bool mismatch: {(output != output_ref).sum().item()} elements differ"
    )


class ComparisonTest(RandnPairWorkload, TestBase):
    """Reusable test body for comparison ops."""

    def __init__(self, n_total: int, dtype: torch.dtype, ref_fn):
        super().__init__(n_total, dtype)
        self.ref_fn = ref_fn

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.ref_fn(a, b)


class EqFixture(FixtureBase):
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


@EqFixture
def test_eq_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.eq)
    op = EqFwdOp()
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


class NeFixture(FixtureBase):
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


@NeFixture
def test_ne_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.ne)
    op = NeFwdOp()
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


class GtFixture(FixtureBase):
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


@GtFixture
def test_gt_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.gt)
    op = GtFwdOp()
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


class LtFixture(FixtureBase):
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


@LtFixture
def test_lt_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.lt)
    op = LtFwdOp()
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


class GeFixture(FixtureBase):
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


@GeFixture
def test_ge_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.ge)
    op = GeFwdOp()
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


class LeFixture(FixtureBase):
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


@LeFixture
def test_le_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.le)
    op = LeFwdOp()
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# Broadcast pattern tests for all comparison ops (L3)

_BROADCAST_PATTERNS = [
    ((2, 64, 128), (1, 1, 128)),  # bias-add
    ((2, 64, 128), (2, 64, 1)),  # row broadcast
    ((64, 128), (1, 1)),  # scalar broadcast
]

_CMP_OPS = [
    ("eq", EqFwdOp, torch.eq),
    ("ne", NeFwdOp, torch.ne),
    ("gt", GtFwdOp, torch.gt),
    ("lt", LtFwdOp, torch.lt),
    ("ge", GeFwdOp, torch.ge),
    ("le", LeFwdOp, torch.le),
]


class ComparisonBroadcastFixture(FixtureBase):
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
                    marks=pytest.mark.smoke if i == 0 and j == 0 else pytest.mark.full,
                )
                for j, (name, cls, ref) in enumerate(_CMP_OPS)
                for i, (a_s, b_s) in enumerate(_BROADCAST_PATTERNS)
            ],
        ),
    ]


@ComparisonBroadcastFixture
def test_comparison_broadcast(
    op_name,
    op_cls,
    ref_fn,
    a_shape,
    b_shape,
) -> None:
    dtype = torch.float16
    a = torch.randn(*a_shape, dtype=dtype, device="cuda")
    b = torch.randn(*b_shape, dtype=dtype, device="cuda")
    op = op_cls()
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    _bool_compare(out, ref)


# L4 edge case: eq with some equal elements


class EqEdgeCaseFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(4096, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@EqEdgeCaseFixture
def test_eq_edge_case(n_total: int, dtype: torch.dtype) -> None:
    """L4: eq with known-equal elements at specific positions."""
    a = torch.randn(n_total, dtype=dtype, device="cuda")
    b = a.clone()
    # Make some elements differ
    b[::2] = torch.randn(n_total // 2, dtype=dtype, device="cuda")
    op = EqFwdOp()
    ref = torch.eq(a, b)
    with torch.no_grad():
        out = op(a, b)
    assert out.dtype == torch.bool
    assert torch.equal(out, ref)


# Per-dtype correctness across the manifest dtype union

_INT_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
_CMP_OP_CASES = [
    (EqFwdOp, torch.eq),
    (NeFwdOp, torch.ne),
    (GtFwdOp, torch.gt),
    (LtFwdOp, torch.lt),
    (GeFwdOp, torch.ge),
    (LeFwdOp, torch.le),
]


def _gen_int_inputs(n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype == torch.uint8:
        lo, hi = 0, 16
    elif dtype == torch.int8:
        lo, hi = -16, 16
    else:
        lo, hi = -64, 64
    a = torch.randint(lo, hi, (n,), dtype=dtype, device="cuda")
    b = torch.randint(lo, hi, (n,), dtype=dtype, device="cuda")
    # Inject some equal positions so eq/ge/le exercise the True branch.
    b[: n // 4] = a[: n // 4]
    return a, b


# Dtype-coverage axis: exercise every manifest-declared int dtype on a
# single representative op (EqFwdOp). The op-coverage axis below uses a
# fixed dtype and varies op_cls; this avoids the dtype x op cross product.
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
    ref = torch.eq(a, b)
    with torch.no_grad():
        out = op(a, b)
    _bool_compare(out, ref)


# Op-coverage axis: at a fixed integer dtype, every comparison op must
# match its torch reference. Combined with the dtype-coverage axis above,
# this gives full per-op-per-dtype coverage in N + M cases instead of
# N x M.
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
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    _bool_compare(out, ref)


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
    a = torch.randint(0, 2, (n,), device="cuda").to(torch.bool)
    b = torch.randint(0, 2, (n,), device="cuda").to(torch.bool)
    op = op_cls()
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    _bool_compare(out, ref)


# Dtype rejection tests (dtypes outside the manifest dtype union: fp8 and
# complex must raise at construction time).


class ComparisonRejectFixture(FixtureBase):
    # Smoke cases (first three) cover the three rejected dtypes on three
    # distinct op_cls so the tier validator's "each dtype needs a smoke"
    # rule is satisfied while keeping "full cases must not differ from a
    # smoke case only by dtype" — each smoke op_cls is unique.
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
    x = torch.zeros(shape, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match="has dtype|does not support dtype"):
        op(x, x)


@pytest.mark.smoke
@pytest.mark.parametrize("strategy", ["explicit_parallel", "direct"])
def test_comparison_bool_result_per_strategy(strategy: str) -> None:
    """Every strategy returns the same bool tensor, whatever it stores underneath.

    Every strategy but ``direct`` now writes an int8 buffer the op views back as
    bool; only ``register_copy`` used to.
    """
    from tileops.kernels.elementwise import GtFwdKernel

    shape = (4096,)
    a = torch.randn(shape, device="cuda", dtype=torch.float16)
    b = torch.randn(shape, device="cuda", dtype=torch.float16)
    out = GtFwdKernel(shape, shape, torch.float16, config={"strategy": strategy}).forward(a, b)
    assert out.dtype == torch.bool
    assert torch.equal(out, torch.gt(a, b))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_comparison_nan_ordering(dtype: torch.dtype) -> None:
    """``ne`` is the one comparison IEEE 754 reads as true against a NaN.

    ``<``, ``<=``, ``>``, ``>=`` and ``==`` are ordered and answer false when
    either operand is NaN; ``!=`` is unordered and answers true, NaN against
    itself included. CUDA's half comparison intrinsics are all ordered, so the
    half formats are where the two can disagree.
    """
    nan = float("nan")
    a = torch.tensor([nan, nan, 1.0, 1.0, 2.0], device="cuda", dtype=dtype)
    b = torch.tensor([nan, 1.0, nan, 1.0, 3.0], device="cuda", dtype=dtype)
    for op_cls, ref_fn in (
        (EqFwdOp, torch.eq),
        (NeFwdOp, torch.ne),
        (LtFwdOp, torch.lt),
        (LeFwdOp, torch.le),
        (GtFwdOp, torch.gt),
        (GeFwdOp, torch.ge),
    ):
        assert torch.equal(op_cls()(a, b), ref_fn(a, b)), op_cls.__name__
