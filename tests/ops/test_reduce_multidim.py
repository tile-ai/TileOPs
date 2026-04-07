"""Correctness tests for multi-dim reduction (dim=list[int]).

Covers: SumOp, MeanOp, AmaxOp, AminOp, VarOp, StdOp, VarMeanOp with
list[int] dim. Also covers multi-dim for LogSumExpOp, AllOp, AnyOp,
CountNonzeroOp, L1NormOp, L2NormOp, InfNormOp.

Each test verifies that reducing over multiple dims at once matches
the corresponding PyTorch reference.
"""

import pytest
import torch

from tests.test_base import FixtureBase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MultiDimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dims, keepdim, dtype",
            [
                # 3D: reduce two dims
                pytest.param(
                    (4, 32, 256), [0, 1], False, torch.float16,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256), [0, 1], True, torch.float16,
                    marks=pytest.mark.full,
                ),
                # 4D: reduce middle two dims
                pytest.param(
                    (2, 4, 8, 256), [1, 2], False, torch.float16,
                    marks=pytest.mark.full,
                ),
                # 4D: reduce first and last
                pytest.param(
                    (2, 4, 8, 256), [0, 3], False, torch.float16,
                    marks=pytest.mark.full,
                ),
                # bfloat16
                pytest.param(
                    (4, 32, 256), [0, 1], False, torch.bfloat16,
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


# ---------------------------------------------------------------------------
# Simple reduce ops: sum, mean, amax, amin
# ---------------------------------------------------------------------------


@MultiDimFixture
def test_sum_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import SumOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = SumOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.sum(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_mean_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import MeanOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = MeanOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.mean(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_amax_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import AmaxOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AmaxOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.amax(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_prod_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import ProdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = ProdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    # PyTorch doesn't support list[int] for prod, so iterate dims manually.
    ref = x.float()
    for d in sorted(dims, reverse=True):
        ref = torch.prod(ref, dim=d, keepdim=keepdim)
    ref = ref.to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_amin_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import AminOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AminOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.amin(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Welford ops: var, std, var_mean
# ---------------------------------------------------------------------------


@MultiDimFixture
def test_var_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import VarOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.var(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_std_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import StdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = StdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.std(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_var_mean_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import VarMeanOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarMeanOp(dtype=dtype, dim=dims, keepdim=keepdim)
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


@MultiDimFixture
def test_logsumexp_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.logsumexp import LogSumExpOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = LogSumExpOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.logsumexp(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Logical reduce ops: all, any, count_nonzero
# ---------------------------------------------------------------------------


class MultiDimLogicalFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dims, keepdim, dtype",
            [
                pytest.param(
                    (4, 32, 256), [0, 1], False, torch.float32,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256), [0, 1], True, torch.float32,
                    marks=pytest.mark.full,
                ),
                # bool: exercises storage-dtype conversion (bool -> float32)
                pytest.param(
                    (4, 32, 256), [0, 1], False, torch.bool,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@MultiDimLogicalFixture
def test_all_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.all_op import AllOp

    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    else:
        x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AllOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.all(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), "all multi-dim mismatch"


@MultiDimLogicalFixture
def test_any_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.any_op import AnyOp

    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    else:
        x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AnyOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.any(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), "any multi-dim mismatch"


class MultiDimCountFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dims, dtype",
            [
                pytest.param(
                    (4, 32, 256), [0, 1], torch.float32,
                    marks=pytest.mark.smoke,
                ),
                # bool: exercises storage-dtype conversion (bool -> float32)
                pytest.param(
                    (4, 32, 256), [0, 1], torch.bool,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@MultiDimCountFixture
def test_count_nonzero_multidim(
    shape: tuple, dims: list, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroOp

    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    else:
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        # Zero out some elements to make it interesting
        x[x < 0] = 0.0
    op = CountNonzeroOp(dtype=dtype, dim=dims)
    ref = torch.count_nonzero(x, dim=dims)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), "count_nonzero multi-dim mismatch"


# ---------------------------------------------------------------------------
# Vector norm ops: l1, l2, inf
# ---------------------------------------------------------------------------


@MultiDimFixture
def test_l1_norm_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.l1_norm import L1NormOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = L1NormOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.linalg.vector_norm(
        x.float(), ord=1, dim=dims, keepdim=keepdim,
    ).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_l2_norm_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.l2_norm import L2NormOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = L2NormOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.linalg.vector_norm(
        x.float(), ord=2, dim=dims, keepdim=keepdim,
    ).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_inf_norm_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.inf_norm import InfNormOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = InfNormOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.linalg.vector_norm(
        x.float(), ord=float("inf"), dim=dims, keepdim=keepdim,
    ).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Regression: empty dim list must be rejected before reaching the helper
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_empty_dim_list_raises() -> None:
    """dim=[] must raise ValueError; the helper cannot produce correct semantics."""
    from tileops.ops.reduction._multidim import normalize_dim

    with pytest.raises(ValueError, match="dim=\\[\\] is not supported"):
        normalize_dim([], ndim=3)


@pytest.mark.smoke
def test_sum_empty_dim_raises() -> None:
    """SumOp(dim=[]) must raise before dispatching to the multi-dim helper."""
    from tileops.ops.reduction.reduce import SumOp

    x = torch.randn(2, 3, 4, dtype=torch.float16, device="cuda")
    op = SumOp(dtype=torch.float16, dim=[], keepdim=False)
    with pytest.raises(ValueError, match="dim=\\[\\] is not supported"):
        op(x)


@pytest.mark.smoke
def test_mean_empty_dim_raises() -> None:
    """MeanOp(dim=[]) must raise before dispatching to the multi-dim helper."""
    from tileops.ops.reduction.reduce import MeanOp

    x = torch.randn(2, 3, 4, dtype=torch.float16, device="cuda")
    op = MeanOp(dtype=torch.float16, dim=[], keepdim=False)
    with pytest.raises(ValueError, match="dim=\\[\\] is not supported"):
        op(x)


@pytest.mark.smoke
def test_negative_dims_accepted() -> None:
    """Negative dims should be normalized and produce correct results."""
    from tileops.ops.reduction.reduce import SumOp

    x = torch.randn(4, 8, 256, dtype=torch.float16, device="cuda")
    op = SumOp(dtype=torch.float16, dim=[-1, 0], keepdim=False)
    ref = torch.sum(x.float(), dim=[0, 2], keepdim=False).to(torch.float16)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **_tol(torch.float16))


@pytest.mark.smoke
def test_duplicate_dims_raises() -> None:
    """Duplicate dims (after normalization) must raise ValueError at op level."""
    from tileops.ops.reduction.reduce import SumOp

    x = torch.randn(4, 8, 256, dtype=torch.float16, device="cuda")
    op = SumOp(dtype=torch.float16, dim=[1, 1], keepdim=False)
    with pytest.raises(ValueError, match="Duplicate dims"):
        op(x)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
