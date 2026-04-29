"""Correctness tests for multi-dim reduction (dim=list[int]).

Covers: SumFwdOp, MeanFwdOp, AmaxFwdOp, AminFwdOp, VarFwdOp, StdFwdOp, VarMeanFwdOp with
list[int] dim. Also covers multi-dim for LogSumExpFwdOp, AllFwdOp, AnyFwdOp,
CountNonzeroFwdOp, L1NormFwdOp, L2NormFwdOp, InfNormFwdOp.

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
                    (4, 32, 256), [0, 1], False, torch.bfloat16,
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
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = SumFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.sum(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_mean_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = MeanFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.mean(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_amax_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import AmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AmaxFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.amax(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_prod_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import ProdFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = ProdFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
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
    from tileops.ops.reduction.reduce import AminFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AminFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
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
    from tileops.ops.reduction.reduce import VarFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.var(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_std_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = StdFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.std(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@MultiDimFixture
def test_var_mean_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarMeanFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
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
    from tileops.ops.reduction.logsumexp import LogSumExpFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = LogSumExpFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
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
                    (4, 32, 256), [0, 1], False, torch.bool,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256), [0, 1], False, torch.complex64,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256), [0, 1], True, torch.float32,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


def _make_logical_input(
    shape: tuple, dtype: torch.dtype,
) -> torch.Tensor:
    """Generate input tensor for logical reduce ops."""
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    if dtype.is_complex:
        return torch.randn(*shape, dtype=dtype, device="cuda")
    return torch.randn(*shape, dtype=dtype, device="cuda")


@MultiDimLogicalFixture
def test_all_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    x = _make_logical_input(shape, dtype)
    op = AllFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.all(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.equal(y, ref), "all multi-dim mismatch"


@MultiDimLogicalFixture
def test_any_multidim(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x = _make_logical_input(shape, dtype)
    op = AnyFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
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
                pytest.param(
                    (4, 32, 256), [0, 1], torch.bool,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256), [0, 1], torch.complex64,
                    marks=pytest.mark.smoke,
                ),
            ],
        ),
    ]


@MultiDimCountFixture
def test_count_nonzero_multidim(
    shape: tuple, dims: list, dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    elif dtype.is_complex:
        x = torch.randn(*shape, dtype=dtype, device="cuda")
    else:
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        # Zero out some elements to make it interesting
        x[x < 0] = 0.0
    op = CountNonzeroFwdOp(dtype=dtype, dim=dims)
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
    from tileops.ops.reduction.l1_norm import L1NormFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = L1NormFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
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
    from tileops.ops.reduction.l2_norm import L2NormFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = L2NormFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
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
    from tileops.ops.reduction.inf_norm import InfNormFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = InfNormFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.linalg.vector_norm(
        x.float(), ord=float("inf"), dim=dims, keepdim=keepdim,
    ).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Empty dim list / tuple is full-reduction (matches PyTorch semantics)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_normalize_dim_empty_default_rejects() -> None:
    """normalize_dim is policy-neutral by default; empty-dim must raise."""
    from tileops.ops.reduction._multidim import normalize_dim

    with pytest.raises(ValueError, match="dim=\\[\\] is not supported"):
        normalize_dim([], ndim=3)
    with pytest.raises(ValueError, match="dim=\\[\\] is not supported"):
        normalize_dim((), ndim=3)


@pytest.mark.smoke
def test_normalize_dim_empty_full_opt_in() -> None:
    """Callers whose contract is full-reduction opt in via empty_dim_policy."""
    from tileops.ops.reduction._multidim import normalize_dim

    assert normalize_dim([], ndim=3, empty_dim_policy="full") == [0, 1, 2]
    assert normalize_dim((), ndim=3, empty_dim_policy="full") == [0, 1, 2]
    assert normalize_dim([], ndim=1, empty_dim_policy="full") == [0]


@pytest.mark.smoke
def test_sum_empty_dim_full_reduction() -> None:
    """SumFwdOp(dim=[]) reduces over all dims (matches dim=None path)."""
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(2, 3, 4, dtype=torch.float16, device="cuda")
    op = SumFwdOp(dtype=torch.float16, dim=[], keepdim=False)
    op_none = SumFwdOp(dtype=torch.float16, dim=None, keepdim=False)
    y = op(x)
    y_none = op_none(x)
    assert y.shape == y_none.shape
    assert torch.allclose(y, y_none, **_tol(torch.float16))


@pytest.mark.smoke
def test_mean_empty_dim_full_reduction() -> None:
    """MeanFwdOp(dim=()) reduces over all dims (matches dim=None path)."""
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = torch.randn(2, 3, 4, dtype=torch.float16, device="cuda")
    op = MeanFwdOp(dtype=torch.float16, dim=(), keepdim=True)
    op_none = MeanFwdOp(dtype=torch.float16, dim=None, keepdim=True)
    y = op(x)
    y_none = op_none(x)
    assert y.shape == y_none.shape
    assert torch.allclose(y, y_none, **_tol(torch.float16))


@pytest.mark.smoke
def test_logsumexp_empty_dim_rejects() -> None:
    """LogSumExpFwdOp keeps reject policy (per manifest); shared helper change must not leak."""
    from tileops.ops.reduction.logsumexp import LogSumExpFwdOp

    x = torch.randn(2, 3, 4, dtype=torch.float16, device="cuda")
    op = LogSumExpFwdOp(dtype=torch.float16, dim=[], keepdim=False)
    with pytest.raises(ValueError, match="dim=\\[\\] is not supported"):
        op(x)


@pytest.mark.smoke
def test_all_empty_dim_rejects() -> None:
    """AllFwdOp keeps reject policy; shared helper change must not silently full-reduce."""
    from tileops.ops.reduction.all_op import AllFwdOp

    x = (torch.randn(2, 3, 4, device="cuda") > 0).to(torch.float16)
    op = AllFwdOp(dtype=torch.float16, dim=[], keepdim=False)
    with pytest.raises(ValueError, match="dim=\\[\\] is not supported"):
        op(x)


@pytest.mark.smoke
def test_negative_dims_accepted() -> None:
    """Negative dims should be normalized and produce correct results."""
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(4, 8, 256, dtype=torch.float16, device="cuda")
    op = SumFwdOp(dtype=torch.float16, dim=[-1, 0], keepdim=False)
    ref = torch.sum(x.float(), dim=[0, 2], keepdim=False).to(torch.float16)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **_tol(torch.float16))


@pytest.mark.smoke
def test_duplicate_dims_raises() -> None:
    """Duplicate dims (after normalization) must raise ValueError at op level."""
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(4, 8, 256, dtype=torch.float16, device="cuda")
    op = SumFwdOp(dtype=torch.float16, dim=[1, 1], keepdim=False)
    with pytest.raises(ValueError, match="Duplicate dims"):
        op(x)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
