"""Correctness tests for vector norm ops (l1_norm, l2_norm, inf_norm).

Covers: L1NormOp, L2NormOp, InfNormOp.
All norms reduce along dim=-1 and return the same dtype as input.
Uses torch.linalg.vector_norm as the reference implementation.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, allclose_compare

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class VectorNormBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.full),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-pow2 last dim
                pytest.param(128, 300, torch.float32, marks=pytest.mark.full),
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class VectorNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


class VectorNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class VectorNorm4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class VectorNorm1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.full),
                pytest.param(512, torch.float32, marks=pytest.mark.full),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# TestBase helpers
# ---------------------------------------------------------------------------


# Map op_kind to the ord parameter for torch.linalg.vector_norm
_ORD_MAP = {"l1": 1, "l2": 2, "inf": float("inf")}


class VectorNormTest(TestBase):
    """Parameterized test helper for vector norm ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for reference, then cast back to input dtype
        ord_val = _ORD_MAP[self.op_kind]
        ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=-1)
        return ref.to(self.dtype)


def _get_tolerances(dtype: torch.dtype):
    """Return (atol, rtol) for the given dtype."""
    if dtype == torch.float32:
        return 1e-5, 1e-5
    # fp16/bf16 have larger rounding errors
    return 1e-2, 1e-2


def _norm_compare(output: torch.Tensor, output_ref: torch.Tensor, atol: float, rtol: float):
    """Comparison with configurable tolerance."""
    allclose_compare(output, output_ref, atol=atol, rtol=rtol)


def _make_noncontig_input(m: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a non-contiguous 2D tensor of shape (m, n*2) for slicing tests."""
    return torch.randn(m, n * 2, dtype=dtype, device="cuda")


def _make_1d_input(n: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a 1D tensor of shape (n,) for 1D tests."""
    return torch.randn(n, dtype=dtype, device="cuda")


def _make_op(m: int, n: int, dtype: torch.dtype, op_kind: str):
    """Create the appropriate Op for the given op_kind."""
    from tileops.ops.reduction.inf_norm import InfNormOp
    from tileops.ops.reduction.l1_norm import L1NormOp
    from tileops.ops.reduction.l2_norm import L2NormOp

    op_map = {
        "l1": L1NormOp,
        "l2": L2NormOp,
        "inf": InfNormOp,
    }
    cls = op_map[op_kind]
    return cls(M=m, N=n, dtype=dtype)


# ---------------------------------------------------------------------------
# L1NormOp tests
# ---------------------------------------------------------------------------


@VectorNormBasicFixture
def test_l1_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op(m, n, dtype, "l1")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@VectorNormNonContigFixture
def test_l1_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = _make_op(m, n, dtype, "l1")
    ref = torch.linalg.vector_norm(x.float().contiguous(), ord=1, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm3DFixture
def test_l1_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = _make_op(M, hidden, dtype, "l1")
    ref = torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm4DFixture
def test_l1_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    M = b0 * b1 * b2
    op = _make_op(M, n, dtype, "l1")
    ref = torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm1DFixture
def test_l1_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = _make_op(1, n, dtype, "l1")
    ref = torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y.view_as(ref), ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# L2NormOp tests
# ---------------------------------------------------------------------------


@VectorNormBasicFixture
def test_l2_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op(m, n, dtype, "l2")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@VectorNormNonContigFixture
def test_l2_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = _make_op(m, n, dtype, "l2")
    ref = torch.linalg.vector_norm(x.float().contiguous(), ord=2, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm3DFixture
def test_l2_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = _make_op(M, hidden, dtype, "l2")
    ref = torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm4DFixture
def test_l2_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    M = b0 * b1 * b2
    op = _make_op(M, n, dtype, "l2")
    ref = torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm1DFixture
def test_l2_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = _make_op(1, n, dtype, "l2")
    ref = torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y.view_as(ref), ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# InfNormOp tests
# ---------------------------------------------------------------------------


@VectorNormBasicFixture
def test_inf_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op(m, n, dtype, "inf")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@VectorNormNonContigFixture
def test_inf_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = _make_op(m, n, dtype, "inf")
    ref = torch.linalg.vector_norm(x.float().contiguous(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm3DFixture
def test_inf_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = _make_op(M, hidden, dtype, "inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm4DFixture
def test_inf_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    M = b0 * b1 * b2
    op = _make_op(M, n, dtype, "inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm1DFixture
def test_inf_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = _make_op(1, n, dtype, "inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y.view_as(ref), ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# NaN propagation regression tests (inf norm)
# ---------------------------------------------------------------------------


class VectorNormNaNFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(4, 512, torch.float32, marks=pytest.mark.full),
                pytest.param(4, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(4, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(4, 300, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


@VectorNormNaNFixture
def test_inf_nan_propagation(m: int, n: int, dtype: torch.dtype) -> None:
    """InfNormOp must return NaN for rows containing NaN, matching PyTorch."""
    x = torch.randn(m, n, dtype=dtype, device="cuda")
    # Inject NaN into the first row
    x[0, 0] = float("nan")
    # Inject NaN into the last position of the second row
    x[1, -1] = float("nan")
    # Rows 2+ remain finite

    op = _make_op(m, n, dtype, "inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)

    # Rows with NaN should produce NaN
    assert y[0].isnan().item(), f"Row 0 should be NaN, got {y[0]}"
    assert y[1].isnan().item(), f"Row 1 should be NaN, got {y[1]}"
    # Finite rows should match reference
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y[2:], ref[2:], atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Dtype smoke tests
# ---------------------------------------------------------------------------

_DTYPE_SMOKE_M, _DTYPE_SMOKE_N = 64, 512


def _make_dtype_smoke_fixture(dt: torch.dtype) -> type:
    """Create a single-param smoke fixture for the given dtype."""
    dt_name = str(dt).split(".")[-1]

    class _Fixture(FixtureBase):
        PARAMS = [
            (
                "m, n, dtype",
                [pytest.param(_DTYPE_SMOKE_M, _DTYPE_SMOKE_N, dt, marks=pytest.mark.full)],
            )
        ]

    _Fixture.__name__ = f"_DtypeSmoke_{dt_name}"
    _Fixture.__qualname__ = _Fixture.__name__
    return _Fixture


_DtypeSmoke_float16 = _make_dtype_smoke_fixture(torch.float16)
_DtypeSmoke_bfloat16 = _make_dtype_smoke_fixture(torch.bfloat16)
_DtypeSmoke_float32 = _make_dtype_smoke_fixture(torch.float32)


@_DtypeSmoke_float16
def test_l1_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op(m, n, dtype, "l1")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_bfloat16
def test_l1_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op(m, n, dtype, "l1")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_float32
def test_l1_smoke_float32(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op(m, n, dtype, "l1")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_float16
def test_l2_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op(m, n, dtype, "l2")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_bfloat16
def test_l2_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op(m, n, dtype, "l2")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_float32
def test_l2_smoke_float32(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op(m, n, dtype, "l2")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_float16
def test_inf_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op(m, n, dtype, "inf")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_bfloat16
def test_inf_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op(m, n, dtype, "inf")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_float32
def test_inf_smoke_float32(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op(m, n, dtype, "inf")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
