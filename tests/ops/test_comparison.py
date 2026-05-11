"""Tests for comparison elementwise ops (eq, ne, gt, lt, ge, le).

All comparison ops output torch.bool. Covers L1 smoke correctness
and L4 edge case for eq.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import EqFwdOp, GeFwdOp, GtFwdOp, LeFwdOp, LtFwdOp, NeFwdOp

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bool_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact comparison for boolean outputs."""
    assert output.dtype == torch.bool, f"Expected bool dtype, got {output.dtype}"
    assert torch.equal(output, output_ref), (
        f"Bool mismatch: {(output != output_ref).sum().item()} elements differ"
    )


class ComparisonTest(TestBase):
    """Reusable test body for comparison ops."""

    def __init__(self, n_total: int, dtype: torch.dtype, ref_fn):
        self.n_total = n_total
        self.dtype = dtype
        self.ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.ref_fn(a, b)


# ---------------------------------------------------------------------------
# Eq op
# ---------------------------------------------------------------------------


class EqFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@EqFixture
def test_eq_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.eq)
    shape = (n_total,)
    op = EqFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Ne op
# ---------------------------------------------------------------------------


class NeFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@NeFixture
def test_ne_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.ne)
    shape = (n_total,)
    op = NeFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Gt op
# ---------------------------------------------------------------------------


class GtFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@GtFixture
def test_gt_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.gt)
    shape = (n_total,)
    op = GtFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Lt op
# ---------------------------------------------------------------------------


class LtFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@LtFixture
def test_lt_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.lt)
    shape = (n_total,)
    op = LtFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Ge op
# ---------------------------------------------------------------------------


class GeFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@GeFixture
def test_ge_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.ge)
    shape = (n_total,)
    op = GeFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Le op
# ---------------------------------------------------------------------------


class LeFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4_096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4_096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@LeFixture
def test_le_op(n_total: int, dtype: torch.dtype) -> None:
    test = ComparisonTest(n_total, dtype, torch.le)
    shape = (n_total,)
    op = LeFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_bool_compare)


# ---------------------------------------------------------------------------
# Broadcast pattern tests for all comparison ops (L3)
# ---------------------------------------------------------------------------

_BROADCAST_PATTERNS = [
    ((2, 64, 128), (1, 1, 128)),   # bias-add
    ((2, 64, 128), (2, 64, 1)),    # row broadcast
    ((64, 128), (1, 1)),           # scalar broadcast
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
        ("op_name, op_cls, ref_fn, a_shape, b_shape", [
            pytest.param(name, cls, ref, a_s, b_s,
                         marks=pytest.mark.smoke if i == 0 and j == 0
                         else pytest.mark.full)
            for j, (name, cls, ref) in enumerate(_CMP_OPS)
            for i, (a_s, b_s) in enumerate(_BROADCAST_PATTERNS)
        ]),
    ]


@ComparisonBroadcastFixture
def test_comparison_broadcast(
    op_name, op_cls, ref_fn, a_shape, b_shape,
) -> None:
    dtype = torch.float16
    a = torch.randn(*a_shape, dtype=dtype, device="cuda")
    b = torch.randn(*b_shape, dtype=dtype, device="cuda")
    op = op_cls(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    _bool_compare(out, ref)


# ---------------------------------------------------------------------------
# L4 edge case: eq with some equal elements
# ---------------------------------------------------------------------------


class EqEdgeCaseFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@EqEdgeCaseFixture
def test_eq_edge_case(n_total: int, dtype: torch.dtype) -> None:
    """L4: eq with known-equal elements at specific positions."""
    a = torch.randn(n_total, dtype=dtype, device="cuda")
    b = a.clone()
    # Make some elements differ
    b[::2] = torch.randn(n_total // 2, dtype=dtype, device="cuda")
    shape = (n_total,)
    op = EqFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
    ref = torch.eq(a, b)
    with torch.no_grad():
        out = op(a, b)
    assert out.dtype == torch.bool
    assert torch.equal(out, ref)


# ---------------------------------------------------------------------------
# Per-dtype correctness across the manifest dtype union
# ---------------------------------------------------------------------------

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
        ("dtype", [
            pytest.param(dt, marks=pytest.mark.smoke)
            for dt in _INT_DTYPES
        ]),
    ]


@ComparisonIntDtypeFixture
def test_comparison_integer_dtype_eq(dtype: torch.dtype) -> None:
    """EqFwdOp matches torch.eq on every manifest-declared int dtype."""
    n = 4_096
    shape = (n,)
    a, b = _gen_int_inputs(n, dtype)
    op = EqFwdOp(a_shape=shape, b_shape=shape, dtype=dtype)
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
        ("op_cls, ref_fn", [
            pytest.param(op_cls, ref_fn, marks=pytest.mark.smoke)
            for op_cls, ref_fn in _CMP_OP_CASES
        ]),
    ]


@ComparisonOpIntFixture
def test_comparison_op_int32(op_cls, ref_fn) -> None:
    """Each comparison op matches its torch reference on int32 inputs."""
    n = 4_096
    shape = (n,)
    a, b = _gen_int_inputs(n, torch.int32)
    op = op_cls(a_shape=shape, b_shape=shape, dtype=torch.int32)
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    _bool_compare(out, ref)


class ComparisonBoolDtypeFixture(FixtureBase):
    PARAMS = [
        ("op_cls, ref_fn", [
            pytest.param(op_cls, ref_fn, marks=pytest.mark.smoke)
            for op_cls, ref_fn in _CMP_OP_CASES
        ]),
    ]


@ComparisonBoolDtypeFixture
def test_comparison_bool_dtype(op_cls, ref_fn) -> None:
    """Comparison ops match torch reference on torch.bool inputs."""
    n = 4_096
    shape = (n,)
    a = torch.randint(0, 2, (n,), device="cuda").to(torch.bool)
    b = torch.randint(0, 2, (n,), device="cuda").to(torch.bool)
    op = op_cls(a_shape=shape, b_shape=shape, dtype=torch.bool)
    ref = ref_fn(a, b)
    with torch.no_grad():
        out = op(a, b)
    _bool_compare(out, ref)


# ---------------------------------------------------------------------------
# Dtype rejection tests (dtypes outside the manifest dtype union: fp8 and
# complex must raise at construction time).
# ---------------------------------------------------------------------------


class ComparisonRejectFixture(FixtureBase):
    PARAMS = [
        ("op_cls, dtype", [
            pytest.param(EqFwdOp, torch.complex64, marks=pytest.mark.smoke),
            pytest.param(NeFwdOp, torch.complex64, marks=pytest.mark.full),
            pytest.param(GtFwdOp, torch.complex64, marks=pytest.mark.full),
            pytest.param(LtFwdOp, torch.complex64, marks=pytest.mark.full),
            pytest.param(GeFwdOp, torch.complex64, marks=pytest.mark.full),
            pytest.param(LeFwdOp, torch.complex64, marks=pytest.mark.full),
            pytest.param(EqFwdOp, torch.float8_e4m3fn, marks=pytest.mark.smoke),
            pytest.param(NeFwdOp, torch.float8_e5m2, marks=pytest.mark.full),
            pytest.param(GtFwdOp, torch.float8_e4m3fn, marks=pytest.mark.full),
            pytest.param(LtFwdOp, torch.float8_e5m2, marks=pytest.mark.full),
            pytest.param(GeFwdOp, torch.float8_e4m3fn, marks=pytest.mark.full),
            pytest.param(LeFwdOp, torch.float8_e5m2, marks=pytest.mark.full),
        ]),
    ]


@ComparisonRejectFixture
def test_comparison_rejects_unsupported_dtype(
    op_cls, dtype: torch.dtype,
) -> None:
    """Comparison ops reject dtypes outside the supported set (e.g. complex)."""
    shape = (16,)
    with pytest.raises(ValueError, match="does not support dtype"):
        op_cls(a_shape=shape, b_shape=shape, dtype=dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
