"""Correctness tests for dim=None reduction (reduce over all dimensions).

Covers: SumOp, MeanOp, AmaxOp, AminOp, ProdOp, VarOp, StdOp, VarMeanOp
with dim=None. Also covers LogSumExpOp, AllOp, AnyOp, CountNonzeroOp,
L1NormOp, L2NormOp, InfNormOp.

Each test verifies that reducing with dim=None matches the corresponding
PyTorch reference (full reduction over all dimensions).
"""

import pytest
import torch

from tests.test_base import FixtureBase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class DimNoneFixture(FixtureBase):
    PARAMS = [
        (
            "shape, keepdim, dtype",
            [
                # 2D: basic case
                pytest.param(
                    (4, 256), False, torch.float16,
                    marks=pytest.mark.smoke,
                ),
                # 3D: keepdim=False (keep total elements moderate for kernel)
                pytest.param(
                    (4, 8, 256), False, torch.float16,
                    marks=pytest.mark.smoke,
                ),
                # 3D: keepdim=True
                pytest.param(
                    (4, 8, 256), True, torch.float16,
                    marks=pytest.mark.full,
                ),
                # bfloat16
                pytest.param(
                    (4, 8, 256), False, torch.bfloat16,
                    marks=pytest.mark.full,
                ),
                # float32
                pytest.param(
                    (4, 8, 256), False, torch.float32,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-2, "rtol": 1e-2}


def _all_dims(shape: tuple) -> list[int]:
    """Return list of all dim indices for a given shape."""
    return list(range(len(shape)))


# ---------------------------------------------------------------------------
# Unit test: normalize_dim(None, ndim) -> list(range(ndim))
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_normalize_dim_none() -> None:
    """normalize_dim(None, ndim) must return list(range(ndim))."""
    from tileops.ops.reduction._multidim import normalize_dim

    assert normalize_dim(None, 3) == [0, 1, 2]
    assert normalize_dim(None, 1) == [0]
    assert normalize_dim(None, 5) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Simple reduce ops: sum, mean, amax, amin, prod
# ---------------------------------------------------------------------------


@DimNoneFixture
def test_sum_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import SumOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = SumOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.sum(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@DimNoneFixture
def test_mean_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import MeanOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = MeanOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.mean(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@DimNoneFixture
def test_amax_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import AmaxOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AmaxOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.amax(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@DimNoneFixture
def test_amin_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import AminOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AminOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.amin(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@DimNoneFixture
def test_prod_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import ProdOp

    # Use small uniform values to avoid overflow/underflow in prod
    x = torch.rand(*shape, dtype=dtype, device="cuda") * 0.5 + 0.75
    op = ProdOp(dtype=dtype, dim=None, keepdim=keepdim)
    # PyTorch prod: reduce all dims manually (prod doesn't accept list[int])
    ref = x.float()
    for d in sorted(_all_dims(shape), reverse=True):
        ref = torch.prod(ref, dim=d, keepdim=keepdim)
    ref = ref.to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Welford ops: var, std, var_mean
# ---------------------------------------------------------------------------


@DimNoneFixture
def test_var_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import VarOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.var(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@DimNoneFixture
def test_std_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import StdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = StdOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.std(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@DimNoneFixture
def test_var_mean_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import VarMeanOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarMeanOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref_var = torch.var(
        x.float(), dim=dims, keepdim=keepdim, correction=1,
    ).to(dtype)
    ref_mean = torch.mean(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    var_out, mean_out = op(x)
    tol = _tol(dtype)
    assert var_out.shape == ref_var.shape, f"var shape: {var_out.shape} vs {ref_var.shape}"
    assert mean_out.shape == ref_mean.shape, f"mean shape: {mean_out.shape} vs {ref_mean.shape}"
    assert torch.allclose(var_out, ref_var, **tol), f"var err: {(var_out - ref_var).abs().max()}"
    assert torch.allclose(mean_out, ref_mean, **tol), f"mean err: {(mean_out - ref_mean).abs().max()}"


# ---------------------------------------------------------------------------
# LogSumExp
# ---------------------------------------------------------------------------


@DimNoneFixture
def test_logsumexp_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.logsumexp import LogSumExpOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = LogSumExpOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.logsumexp(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Logical reduce ops: all, any, count_nonzero
# ---------------------------------------------------------------------------


class DimNoneLogicalFixture(FixtureBase):
    PARAMS = [
        (
            "shape, keepdim, dtype",
            [
                pytest.param(
                    (4, 8, 256), False, torch.float32,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 8, 256), True, torch.float32,
                    marks=pytest.mark.full,
                ),
                # bool: canonical dtype for all/any
                pytest.param(
                    (4, 8, 256), False, torch.bool,
                    marks=pytest.mark.full,
                ),
                # fp16: supported dtype for count_nonzero
                pytest.param(
                    (4, 8, 256), False, torch.float16,
                    marks=pytest.mark.full,
                ),
                # bf16: supported dtype for count_nonzero
                pytest.param(
                    (4, 8, 256), False, torch.bfloat16,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


def _make_logical_input(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Create test input appropriate for the dtype."""
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    return torch.randn(*shape, dtype=dtype, device="cuda")


@DimNoneLogicalFixture
def test_all_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.all_op import AllOp

    x = _make_logical_input(shape, dtype)
    op = AllOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.all(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), "all dim=None mismatch"


@DimNoneLogicalFixture
def test_any_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.any_op import AnyOp

    x = _make_logical_input(shape, dtype)
    op = AnyOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.any(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), "any dim=None mismatch"


@pytest.mark.smoke
def test_count_nonzero_dim_none() -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroOp

    shape = (4, 8, 256)
    x = torch.randn(*shape, dtype=torch.float32, device="cuda")
    x[x < 0] = 0.0
    op = CountNonzeroOp(dtype=torch.float32, dim=None)
    dims = _all_dims(shape)
    ref = torch.count_nonzero(x, dim=dims)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), "count_nonzero dim=None mismatch"


@pytest.mark.parametrize("dtype", [
    pytest.param(torch.float16, marks=pytest.mark.smoke),
    pytest.param(torch.bfloat16, marks=pytest.mark.full),
])
def test_count_nonzero_dim_none_dtypes(dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroOp

    shape = (4, 8, 256)
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    x[x < 0] = 0.0
    op = CountNonzeroOp(dtype=dtype, dim=None)
    dims = _all_dims(shape)
    ref = torch.count_nonzero(x, dim=dims)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), f"count_nonzero dim=None mismatch for {dtype}"


# ---------------------------------------------------------------------------
# Vector norm ops: l1, l2, inf
# ---------------------------------------------------------------------------


@DimNoneFixture
def test_l1_norm_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.l1_norm import L1NormOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = L1NormOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.linalg.vector_norm(
        x.float(), ord=1, dim=dims, keepdim=keepdim,
    ).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@DimNoneFixture
def test_l2_norm_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.l2_norm import L2NormOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = L2NormOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.linalg.vector_norm(
        x.float(), ord=2, dim=dims, keepdim=keepdim,
    ).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@DimNoneFixture
def test_inf_norm_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.inf_norm import InfNormOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = InfNormOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.linalg.vector_norm(
        x.float(), ord=float("inf"), dim=dims, keepdim=keepdim,
    ).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
