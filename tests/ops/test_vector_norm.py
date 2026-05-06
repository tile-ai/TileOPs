"""Correctness tests for vector norm ops (l1_norm, l2_norm, inf_norm).

Covers: L1NormFwdOp, L2NormFwdOp, InfNormFwdOp.
All norms reduce along a configurable dim and return the same dtype as input.
Uses torch.linalg.vector_norm as the reference implementation.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, allclose_compare
from tileops.kernels.reduction.vector_norm import VectorNormKernel
from workloads.vector_norm import L1NormTest as _L1NormWorkload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class VectorNormBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
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
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


class VectorNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class VectorNorm4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class VectorNorm1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# TestBase helpers — inherit gen_inputs() from workload classes
# ---------------------------------------------------------------------------


# Map op_kind to the ord parameter for torch.linalg.vector_norm
_ORD_MAP = {"l1": 1, "l2": 2, "inf": float("inf")}


class VectorNormTest(_L1NormWorkload, TestBase):
    """Parameterized test helper for vector norm ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for reference, then cast back to input dtype
        ord_val = _ORD_MAP[self.op_kind]
        ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=-1)
        return ref.to(self.dtype)


class _TailBlockVectorNormKernel(VectorNormKernel):
    """Force tiled tests to cover tail-M masking with block_m > M."""

    _TAIL_BLOCK_M = 4
    _TAIL_TILE_N = 8192

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self._needs_tiling, "tail-M regression test must use the tiled kernel"
        self.config = {
            "block_m": self._TAIL_BLOCK_M,
            "threads": 128,
            "tile_n": self._TAIL_TILE_N,
        }


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


def _make_op(
    dtype: torch.dtype,
    op_kind: str,
    dim: int = -1,
    keepdim: bool = False,
    kernel_map=None,
):
    """Create the appropriate Op for the given op_kind."""
    from tileops.ops.reduction.inf_norm import InfNormFwdOp
    from tileops.ops.reduction.l1_norm import L1NormFwdOp
    from tileops.ops.reduction.l2_norm import L2NormFwdOp

    op_map = {
        "l1": L1NormFwdOp,
        "l2": L2NormFwdOp,
        "inf": InfNormFwdOp,
    }
    cls = op_map[op_kind]
    return cls(dtype=dtype, dim=dim, keepdim=keepdim, kernel_map=kernel_map)


# ---------------------------------------------------------------------------
# L1NormFwdOp tests
# ---------------------------------------------------------------------------


@VectorNormBasicFixture
def test_l1_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op(dtype, "l1")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@VectorNormNonContigFixture
def test_l1_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = _make_op(dtype, "l1")
    ref = torch.linalg.vector_norm(x.float().contiguous(), ord=1, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm3DFixture
def test_l1_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = _make_op(dtype, "l1")
    ref = torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm4DFixture
def test_l1_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = _make_op(dtype, "l1")
    ref = torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm1DFixture
def test_l1_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = _make_op(dtype, "l1")
    ref = torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y.view_as(ref), ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# L2NormFwdOp tests
# ---------------------------------------------------------------------------


@VectorNormBasicFixture
def test_l2_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op(dtype, "l2")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@VectorNormNonContigFixture
def test_l2_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = _make_op(dtype, "l2")
    ref = torch.linalg.vector_norm(x.float().contiguous(), ord=2, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm3DFixture
def test_l2_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = _make_op(dtype, "l2")
    ref = torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm4DFixture
def test_l2_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = _make_op(dtype, "l2")
    ref = torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm1DFixture
def test_l2_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = _make_op(dtype, "l2")
    ref = torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y.view_as(ref), ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# InfNormFwdOp tests
# ---------------------------------------------------------------------------


@VectorNormBasicFixture
def test_inf_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op(dtype, "inf")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@VectorNormNonContigFixture
def test_inf_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = _make_op(dtype, "inf")
    ref = torch.linalg.vector_norm(x.float().contiguous(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm3DFixture
def test_inf_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = _make_op(dtype, "inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm4DFixture
def test_inf_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = _make_op(dtype, "inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNorm1DFixture
def test_inf_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = _make_op(dtype, "inf")
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
                pytest.param(4, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(4, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(4, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4, 300, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


@VectorNormNaNFixture
def test_inf_nan_propagation(m: int, n: int, dtype: torch.dtype) -> None:
    """InfNormFwdOp must return NaN for rows containing NaN, matching PyTorch."""
    x = torch.randn(m, n, dtype=dtype, device="cuda")
    # Inject NaN into the first row
    x[0, 0] = float("nan")
    # Inject NaN into the last position of the second row
    x[1, -1] = float("nan")
    # Rows 2+ remain finite

    op = _make_op(dtype, "inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)

    # Rows with NaN should produce NaN
    assert y[0].isnan().item(), f"Row 0 should be NaN, got {y[0]}"
    assert y[1].isnan().item(), f"Row 1 should be NaN, got {y[1]}"
    # Finite rows should match reference
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y[2:], ref[2:], atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Spec tests: dim=0, dim=1, keepdim=True
# ---------------------------------------------------------------------------


class VectorNormSpecFixture(FixtureBase):
    PARAMS = [
        (
            "op_kind, dtype",
            [
                pytest.param("l1", torch.float16, marks=pytest.mark.smoke),
                pytest.param("l2", torch.float16, marks=pytest.mark.smoke),
                pytest.param("inf", torch.float16, marks=pytest.mark.smoke),
                pytest.param("l1", torch.float32, marks=pytest.mark.smoke),
                pytest.param("l2", torch.float32, marks=pytest.mark.smoke),
                pytest.param("inf", torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@VectorNormSpecFixture
def test_spec_dim0(op_kind: str, dtype: torch.dtype) -> None:
    """Reduce along dim=0."""
    x = torch.randn(64, 512, dtype=dtype, device="cuda")
    op = _make_op(dtype, op_kind, dim=0)
    ord_val = _ORD_MAP[op_kind]
    ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=0).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNormSpecFixture
def test_spec_dim1_3d(op_kind: str, dtype: torch.dtype) -> None:
    """Reduce along dim=1 of a 3D tensor."""
    x = torch.randn(4, 64, 512, dtype=dtype, device="cuda")
    op = _make_op(dtype, op_kind, dim=1)
    ord_val = _ORD_MAP[op_kind]
    ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=1).to(dtype)
    y = op(x)
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNormSpecFixture
def test_spec_keepdim(op_kind: str, dtype: torch.dtype) -> None:
    """keepdim=True preserves the reduced dimension as size 1."""
    x = torch.randn(32, 512, dtype=dtype, device="cuda")
    op = _make_op(dtype, op_kind, keepdim=True)
    ord_val = _ORD_MAP[op_kind]
    ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=-1, keepdim=True).to(dtype)
    y = op(x)
    assert y.shape == ref.shape, f"Expected shape {ref.shape}, got {y.shape}"
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@VectorNormSpecFixture
def test_spec_dim0_keepdim(op_kind: str, dtype: torch.dtype) -> None:
    """dim=0 + keepdim=True."""
    x = torch.randn(64, 512, dtype=dtype, device="cuda")
    op = _make_op(dtype, op_kind, dim=0, keepdim=True)
    ord_val = _ORD_MAP[op_kind]
    ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=0, keepdim=True).to(dtype)
    y = op(x)
    assert y.shape == ref.shape, f"Expected shape {ref.shape}, got {y.shape}"
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


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
                [pytest.param(_DTYPE_SMOKE_M, _DTYPE_SMOKE_N, dt, marks=pytest.mark.smoke)],
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
    op = _make_op(dtype, "l1")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_bfloat16
def test_l1_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op(dtype, "l1")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_float32
def test_l1_smoke_float32(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op(dtype, "l1")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_float16
def test_l2_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op(dtype, "l2")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_bfloat16
def test_l2_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op(dtype, "l2")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_float32
def test_l2_smoke_float32(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op(dtype, "l2")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_float16
def test_inf_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op(dtype, "inf")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_bfloat16
def test_inf_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op(dtype, "inf")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@_DtypeSmoke_float32
def test_inf_smoke_float32(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op(dtype, "inf")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Empty-dim full-reduction (dim=[])
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["l1", "l2", "inf"])
@pytest.mark.parametrize("keepdim", [False, True])
def test_empty_dim_full_reduction_keepdim(op_kind: str, keepdim: bool) -> None:
    dtype = torch.float16
    x = torch.randn(32, 256, dtype=dtype, device="cuda")
    op = _make_op(dtype, op_kind, dim=[], keepdim=keepdim)
    ref = torch.linalg.vector_norm(
        x.float(), ord=_ORD_MAP[op_kind], dim=[], keepdim=keepdim,
    ).to(dtype)
    y = op(x)
    assert y.shape == ref.shape
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["l1", "l2", "inf"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32],
)
def test_empty_dim_full_reduction_3d_dtypes(
    op_kind: str, dtype: torch.dtype,
) -> None:
    x = torch.randn(2, 16, 128, dtype=dtype, device="cuda")
    op = _make_op(dtype, op_kind, dim=[], keepdim=False)
    ref = torch.linalg.vector_norm(
        x.float(), ord=_ORD_MAP[op_kind], dim=[], keepdim=False,
    ).to(dtype)
    y = op(x)
    assert y.shape == ref.shape
    atol, rtol = _get_tolerances(dtype)
    allclose_compare(y, ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["l1", "l2", "inf"])
def test_vector_norm_long_sequence_tiled(op_kind: str) -> None:
    """Exercise the N-tiled path with a tail-M block."""
    dtype = torch.bfloat16
    test = VectorNormTest(3, 33024, dtype, op_kind)
    op = _make_op(
        dtype,
        op_kind,
        kernel_map={"vector_norm": _TailBlockVectorNormKernel},
    )
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)
    kernel = op._kernel_cache[(3, 33024)]
    assert kernel.config["block_m"] > test.shape[0]
    assert kernel.config["tile_n"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
