"""Correctness tests for dim=None reduction (reduce over all dimensions).

Covers: SumFwdOp, MeanFwdOp, AmaxFwdOp, AminFwdOp, ProdFwdOp, VarFwdOp, StdFwdOp, VarMeanFwdOp
with dim=None. Also covers LogSumExpFwdOp, AllFwdOp, AnyFwdOp, CountNonzeroFwdOp,
VectorNormFwdOp.

Each test verifies that reducing with dim=None matches the corresponding
PyTorch reference (full reduction over all dimensions).
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import (
    reduction_verification,
    vector_norm_verification,
)


class DimNoneFixture(FixtureBase):
    PARAMS = [
        (
            "shape, keepdim, dtype",
            [
                # 3D: keepdim=False (keep total elements moderate for kernel)
                pytest.param(
                    (4, 8, 256),
                    False,
                    torch.float16,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 8, 256),
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 8, 256),
                    False,
                    torch.float32,
                    marks=pytest.mark.smoke,
                ),
                # 2D: basic case
                pytest.param(
                    (4, 256),
                    False,
                    torch.float16,
                    marks=pytest.mark.full,
                ),
                # 3D: keepdim=True
                pytest.param(
                    (4, 8, 256),
                    True,
                    torch.float16,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


def _all_dims(shape: tuple) -> list[int]:
    """Return list of all dim indices for a given shape."""
    return list(range(len(shape)))


# Simple reduce ops: sum, mean, amax, amin, prod


@DimNoneFixture
def test_sum_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = SumFwdOp(dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.sum(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@DimNoneFixture
def test_mean_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = MeanFwdOp(dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.mean(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@DimNoneFixture
def test_amax_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import AmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = AmaxFwdOp(dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.amax(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@DimNoneFixture
def test_amin_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import AminFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = AminFwdOp(dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.amin(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


# Welford ops: var, std, var_mean


@DimNoneFixture
def test_var_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VarFwdOp(dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.var(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@DimNoneFixture
def test_std_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = StdFwdOp(dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.std(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@DimNoneFixture
def test_var_mean_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VarMeanFwdOp(dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref_var = torch.var(
        x.float(),
        dim=dims,
        keepdim=keepdim,
        correction=1,
    ).to(dtype)
    ref_mean = torch.mean(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    var_out, mean_out = op(x)

    assert var_out.shape == ref_var.shape, f"var shape: {var_out.shape} vs {ref_var.shape}"
    assert mean_out.shape == ref_mean.shape, f"mean shape: {mean_out.shape} vs {ref_mean.shape}"
    compare_outputs(var_out, ref_var, reduction_verification((ref_var).dtype))
    compare_outputs(mean_out, ref_mean, reduction_verification((ref_mean).dtype))


# Logical reduce ops: all, any, count_nonzero


class DimNoneLogicalFixture(FixtureBase):
    PARAMS = [
        (
            "shape, keepdim, dtype",
            [
                pytest.param(
                    (4, 8, 256),
                    False,
                    torch.float32,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 8, 256),
                    False,
                    torch.bool,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 8, 256),
                    False,
                    torch.float16,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 8, 256),
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 8, 256),
                    True,
                    torch.float32,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


def _make_logical_input(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Create test input appropriate for the dtype."""
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=run_device())
    return torch.randn(*shape, dtype=dtype, device=run_device())


@DimNoneLogicalFixture
def test_all_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x = _make_logical_input(shape, dtype)
    op = AllFwdOp(dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.all(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@DimNoneLogicalFixture
def test_any_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    x = _make_logical_input(shape, dtype)
    op = AnyFwdOp(dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.any(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@pytest.mark.smoke
def test_count_nonzero_dim_none() -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    shape = (4, 8, 256)
    x = torch.randn(*shape, dtype=torch.float32, device=run_device())
    x[x < 0] = 0.0
    op = CountNonzeroFwdOp(dim=None)
    dims = _all_dims(shape)
    ref = torch.count_nonzero(x, dim=dims)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, marks=pytest.mark.smoke),
        pytest.param(torch.bfloat16, marks=pytest.mark.smoke),
    ],
)
def test_count_nonzero_dim_none_dtypes(dtype: torch.dtype) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    shape = (4, 8, 256)
    x = torch.randn(*shape, dtype=dtype, device=run_device())
    x[x < 0] = 0.0
    op = CountNonzeroFwdOp(dim=None)
    dims = _all_dims(shape)
    ref = torch.count_nonzero(x, dim=dims)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


# Vector norm ops: l1, l2, inf


@DimNoneFixture
def test_l1_norm_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.vector_norm import VectorNormFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VectorNormFwdOp(1, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.linalg.vector_norm(
        x.float(),
        ord=1,
        dim=dims,
        keepdim=keepdim,
    ).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, vector_norm_verification(x.dtype))


@DimNoneFixture
def test_l2_norm_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.vector_norm import VectorNormFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VectorNormFwdOp(2, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.linalg.vector_norm(
        x.float(),
        ord=2,
        dim=dims,
        keepdim=keepdim,
    ).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, vector_norm_verification(x.dtype))


@DimNoneFixture
def test_inf_norm_dim_none(
    shape: tuple,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from math import inf

    from tileops.ops.reduction.vector_norm import VectorNormFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VectorNormFwdOp(inf, dim=None, keepdim=keepdim)
    dims = _all_dims(shape)
    ref = torch.linalg.vector_norm(
        x.float(),
        ord=float("inf"),
        dim=dims,
        keepdim=keepdim,
    ).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, vector_norm_verification(x.dtype))
