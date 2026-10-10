"""Correctness tests for vector norm ops (l1_norm, l2_norm, inf_norm).

Covers VectorNormFwdOp at ord 1, 2 and inf.
All norms reduce along a configurable dim and return the same dtype as input.
Uses torch.linalg.vector_norm as the reference implementation.
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import VectorNormCase, vector_norm_verification


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


# Map op_kind to the ord parameter for torch.linalg.vector_norm
_ORD_MAP = {"l1": 1, "l2": 2, "inf": float("inf")}


class VectorNormTest(VectorNormCase, TestBase):
    pass


def _make_noncontig_input(m: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a non-contiguous 2D tensor of shape (m, n*2) for slicing tests."""
    return torch.randn(m, n * 2, dtype=dtype, device=run_device())


def _make_1d_input(n: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a 1D tensor of shape (n,) for 1D tests."""
    return torch.randn(n, dtype=dtype, device=run_device())


def _make_op(
    op_kind: str,
    dim: int = -1,
    keepdim: bool = False,
    tune: bool = False,
    target=None,
):
    """Build VectorNormFwdOp at the ``ord`` this op_kind names."""
    from math import inf

    from tileops.ops.reduction.vector_norm import VectorNormFwdOp

    order = {"l1": 1, "l2": 2, "inf": inf}[op_kind]
    op = VectorNormFwdOp(order, dim=dim, keepdim=keepdim, target=target)
    if tune:
        op.autotune()
    return op


@VectorNormBasicFixture
def test_l1_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op("l1")
    test.check(op, *test.gen_inputs())


@VectorNormNonContigFixture
def test_l1_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = _make_op("l1")
    ref = torch.linalg.vector_norm(x.float().contiguous(), ord=1, dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNorm3DFixture
def test_l1_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = _make_op("l1")
    ref = torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNorm4DFixture
def test_l1_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = _make_op("l1")
    ref = torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNorm1DFixture
def test_l1_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = _make_op("l1")
    ref = torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y.view_as(ref), ref, vector_norm_verification(dtype))


@VectorNormBasicFixture
def test_l2_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op("l2")
    test.check(op, *test.gen_inputs())


@VectorNormNonContigFixture
def test_l2_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = _make_op("l2")
    ref = torch.linalg.vector_norm(x.float().contiguous(), ord=2, dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNorm3DFixture
def test_l2_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = _make_op("l2")
    ref = torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNorm4DFixture
def test_l2_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = _make_op("l2")
    ref = torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNorm1DFixture
def test_l2_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = _make_op("l2")
    ref = torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y.view_as(ref), ref, vector_norm_verification(dtype))


@VectorNormBasicFixture
def test_inf_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op("inf")
    test.check(op, *test.gen_inputs())


@VectorNormNonContigFixture
def test_inf_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    x_full = _make_noncontig_input(m, n, dtype)
    x = x_full[:, :n]
    op = _make_op("inf")
    ref = torch.linalg.vector_norm(x.float().contiguous(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNorm3DFixture
def test_inf_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    op = _make_op("inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNorm4DFixture
def test_inf_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    x = torch.randn(b0, b1, b2, n, dtype=dtype, device=run_device())
    op = _make_op("inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNorm1DFixture
def test_inf_1d(n: int, dtype: torch.dtype) -> None:
    x = _make_1d_input(n, dtype)
    op = _make_op("inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)
    compare_outputs(y.view_as(ref), ref, vector_norm_verification(dtype))


# NaN propagation regression tests (inf norm)


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
    """``ord=inf`` must return NaN for rows containing NaN, matching PyTorch."""
    x = torch.randn(m, n, dtype=dtype, device=run_device())
    # Inject NaN into the first row
    x[0, 0] = float("nan")
    # Inject NaN into the last position of the second row
    x[1, -1] = float("nan")
    # Rows 2+ remain finite

    op = _make_op("inf")
    ref = torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(dtype)
    y = op(x)

    # Rows with NaN should produce NaN
    assert y[0].isnan().item(), f"Row 0 should be NaN, got {y[0]}"
    assert y[1].isnan().item(), f"Row 1 should be NaN, got {y[1]}"
    # Finite rows should match reference
    compare_outputs(y[2:], ref[2:], vector_norm_verification(dtype))


# Spec tests: dim=0, dim=1, keepdim=True


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
    x = torch.randn(64, 512, dtype=dtype, device=run_device())
    op = _make_op(op_kind, dim=0)
    ord_val = _ORD_MAP[op_kind]
    ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=0).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNormSpecFixture
def test_spec_dim1_3d(op_kind: str, dtype: torch.dtype) -> None:
    """Reduce along dim=1 of a 3D tensor."""
    x = torch.randn(4, 64, 512, dtype=dtype, device=run_device())
    op = _make_op(op_kind, dim=1)
    ord_val = _ORD_MAP[op_kind]
    ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=1).to(dtype)
    y = op(x)
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNormSpecFixture
def test_spec_keepdim(op_kind: str, dtype: torch.dtype) -> None:
    """keepdim=True preserves the reduced dimension as size 1."""
    x = torch.randn(32, 512, dtype=dtype, device=run_device())
    op = _make_op(op_kind, keepdim=True)
    ord_val = _ORD_MAP[op_kind]
    ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=-1, keepdim=True).to(dtype)
    y = op(x)
    assert y.shape == ref.shape, f"Expected shape {ref.shape}, got {y.shape}"
    compare_outputs(y, ref, vector_norm_verification(dtype))


@VectorNormSpecFixture
def test_spec_dim0_keepdim(op_kind: str, dtype: torch.dtype) -> None:
    """dim=0 + keepdim=True."""
    x = torch.randn(64, 512, dtype=dtype, device=run_device())
    op = _make_op(op_kind, dim=0, keepdim=True)
    ord_val = _ORD_MAP[op_kind]
    ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=0, keepdim=True).to(dtype)
    y = op(x)
    assert y.shape == ref.shape, f"Expected shape {ref.shape}, got {y.shape}"
    compare_outputs(y, ref, vector_norm_verification(dtype))


def _make_dtype_smoke_fixture(dt: torch.dtype) -> type:
    """Create a single-param smoke fixture for the given dtype."""
    m = 64
    n = 512
    dt_name = str(dt).split(".")[-1]

    class _Fixture(FixtureBase):
        PARAMS = [
            (
                "m, n, dtype",
                [pytest.param(m, n, dt, marks=pytest.mark.smoke)],
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
    op = _make_op("l1")
    test.check(op, *test.gen_inputs())


@_DtypeSmoke_bfloat16
def test_l1_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op("l1")
    test.check(op, *test.gen_inputs())


@_DtypeSmoke_float32
def test_l1_smoke_float32(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op("l1")
    test.check(op, *test.gen_inputs())


@_DtypeSmoke_float16
def test_l2_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op("l2")
    test.check(op, *test.gen_inputs())


@_DtypeSmoke_bfloat16
def test_l2_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op("l2")
    test.check(op, *test.gen_inputs())


@_DtypeSmoke_float32
def test_l2_smoke_float32(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op("l2")
    test.check(op, *test.gen_inputs())


@_DtypeSmoke_float16
def test_inf_smoke_float16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op("inf")
    test.check(op, *test.gen_inputs())


@_DtypeSmoke_bfloat16
def test_inf_smoke_bfloat16(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op("inf")
    test.check(op, *test.gen_inputs())


@_DtypeSmoke_float32
def test_inf_smoke_float32(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op("inf")
    test.check(op, *test.gen_inputs())


# Empty-dim full-reduction (dim=[])


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["l1", "l2", "inf"])
@pytest.mark.parametrize("keepdim", [False, True])
def test_empty_dim_full_reduction_keepdim(op_kind: str, keepdim: bool) -> None:
    dtype = torch.float16
    x = torch.randn(32, 256, dtype=dtype, device=run_device())
    op = _make_op(op_kind, dim=[], keepdim=keepdim)
    ref = torch.linalg.vector_norm(
        x.float(),
        ord=_ORD_MAP[op_kind],
        dim=[],
        keepdim=keepdim,
    ).to(dtype)
    y = op(x)
    assert y.shape == ref.shape
    compare_outputs(y, ref, vector_norm_verification(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["l1", "l2", "inf"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_empty_dim_full_reduction_3d_dtypes(
    op_kind: str,
    dtype: torch.dtype,
) -> None:
    x = torch.randn(2, 16, 128, dtype=dtype, device=run_device())
    op = _make_op(op_kind, dim=[], keepdim=False)
    ref = torch.linalg.vector_norm(
        x.float(),
        ord=_ORD_MAP[op_kind],
        dim=[],
        keepdim=False,
    ).to(dtype)
    y = op(x)
    assert y.shape == ref.shape
    compare_outputs(y, ref, vector_norm_verification(dtype))


@pytest.mark.smoke
def test_vector_norm_tiled_autotune() -> None:
    """``tune=True`` must build and time every tiled candidate.

    N is not a power of two: a power-of-two N_padded lets ``compute_tile_n``
    fall back on an exact divisor, which hides a mis-derived tile_n.
    """
    m, n, dtype = 4, 39999, torch.float16
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op("l2")
    op.autotune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("op_kind", ["l1", "l2", "inf"])
def test_vector_norm_edge_axes_in_own_layout(op_kind: str) -> None:
    """``dim=[0, 2]`` reduces without a permute: a rows pass, then a fold.

    ``l2`` takes its square root only at the fold; a sqrt applied per partial
    would compound instead of finishing.
    """
    dtype = torch.float16
    x = torch.randn(4, 24, 4096, dtype=dtype, device=run_device())
    op = _make_op(op_kind, dim=[0, 2])
    ords = {"l1": 1, "l2": 2, "inf": torch.inf}
    ref = torch.linalg.vector_norm(x.float(), ords[op_kind], (0, 2)).to(dtype)
    compare_outputs(op(x), ref, vector_norm_verification(dtype))


@pytest.mark.smoke
def test_inf_norm_edge_axes_carries_nan() -> None:
    """A NaN survives both passes: the fold compares bit patterns, not floats."""
    x = torch.randn(4, 24, 4096, dtype=torch.float16, device=run_device())
    x[3, 7, 4095] = float("nan")
    y = _make_op("inf", dim=[0, 2])(x)
    ref = torch.linalg.vector_norm(x, torch.inf, (0, 2))
    assert torch.equal(y.isnan(), ref.isnan())
