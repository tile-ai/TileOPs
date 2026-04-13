"""Correctness tests for dim=None reduction (reduce over all dimensions).

Covers: SumFwdOp, MeanFwdOp, AmaxFwdOp, AminFwdOp, ProdFwdOp, VarFwdOp, StdFwdOp, VarMeanFwdOp
with dim=None. Also covers LogSumExpFwdOp, AllFwdOp, AnyFwdOp, CountNonzeroFwdOp,
L1NormFwdOp, L2NormFwdOp, InfNormFwdOp.

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
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = SumFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = MeanFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
    from tileops.ops.reduction.reduce import AmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AmaxFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
    from tileops.ops.reduction.reduce import AminFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AminFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
    from tileops.ops.reduction.reduce import ProdFwdOp

    # Use small uniform values to avoid overflow/underflow in prod
    x = torch.rand(*shape, dtype=dtype, device="cuda") * 0.5 + 0.75
    op = ProdFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
    from tileops.ops.reduction.reduce import VarFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
    from tileops.ops.reduction.reduce import StdFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = StdFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarMeanFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
    from tileops.ops.reduction.all_op import AllFwdOp

    x = _make_logical_input(shape, dtype)
    op = AllFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.all(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), "all dim=None mismatch"


@DimNoneLogicalFixture
def test_any_dim_none(
    shape: tuple, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x = _make_logical_input(shape, dtype)
    op = AnyFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.any(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), "any dim=None mismatch"


@pytest.mark.smoke
def test_count_nonzero_dim_none() -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    shape = (4, 8, 256)
    x = torch.randn(*shape, dtype=torch.float32, device="cuda")
    x[x < 0] = 0.0
    op = CountNonzeroFwdOp(dtype=torch.float32, dim=None)
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
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    shape = (4, 8, 256)
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    x[x < 0] = 0.0
    op = CountNonzeroFwdOp(dtype=dtype, dim=None)
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
    from tileops.ops.reduction.l1_norm import L1NormFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = L1NormFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
    from tileops.ops.reduction.l2_norm import L2NormFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = L2NormFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
    from tileops.ops.reduction.inf_norm import InfNormFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = InfNormFwdOp(dtype=dtype, dim=None, keepdim=keepdim)
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
