"""Correctness tests for multi-dim reduction (dim=list[int]).

Covers: SumFwdOp, MeanFwdOp, AmaxFwdOp, AminFwdOp, VarFwdOp, StdFwdOp, VarMeanFwdOp with
list[int] dim. Also covers multi-dim for LogSumExpFwdOp, AllFwdOp, AnyFwdOp,
CountNonzeroFwdOp, VectorNormFwdOp.

Each test verifies that reducing over multiple dims at once matches
the corresponding PyTorch reference.
"""

import pytest
import torch

from tests.test_base import FixtureBase
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import (
    LogSumExpWorkload,
    reduction_tolerance,
    reduction_verification,
    vector_norm_verification,
)


class MultiDimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dims, keepdim, dtype",
            [
                # 3D: reduce two dims
                pytest.param(
                    (4, 32, 256),
                    [0, 1],
                    False,
                    torch.float16,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256),
                    [0, 1],
                    False,
                    torch.bfloat16,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256),
                    [0, 1],
                    True,
                    torch.float16,
                    marks=pytest.mark.full,
                ),
                # 4D: reduce middle two dims
                pytest.param(
                    (2, 4, 8, 256),
                    [1, 2],
                    False,
                    torch.float16,
                    marks=pytest.mark.full,
                ),
                # 4D: reduce first and last
                pytest.param(
                    (2, 4, 8, 256),
                    [0, 3],
                    False,
                    torch.float16,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


# Simple reduce ops: sum, mean, amax, amin


@MultiDimFixture
def test_sum_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = SumFwdOp(dim=dims, keepdim=keepdim)
    ref = torch.sum(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@MultiDimFixture
def test_mean_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = MeanFwdOp(dim=dims, keepdim=keepdim)
    ref = torch.mean(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@pytest.mark.smoke
def test_mean_edge_axes_fp16_keeps_fp32_intermediates() -> None:
    """A mean over edge axes must not narrow its partial sums to the storage dtype.

    fp16 rows of 100.0 sum to 409600 per row — past fp16's max — so any pass
    that casts an undivided sum back to fp16 answers inf instead of 100.
    """
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = torch.full((4, 8, 1024), 100.0, dtype=torch.float16, device=run_device())
    y = MeanFwdOp(dim=[0, 2])(x)
    assert torch.isfinite(y).all(), "edge-axes mean overflowed an intermediate"
    compare_outputs(
        y, torch.full_like(y, 100.0), reduction_verification((torch.full_like(y, 100.0)).dtype)
    )


@MultiDimFixture
def test_amax_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import AmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = AmaxFwdOp(dim=dims, keepdim=keepdim)
    ref = torch.amax(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@pytest.mark.smoke
def test_prod_multidim_rejected() -> None:
    """ProdFwdOp narrows ``dim`` to ``int`` per its manifest signature, so
    the multi-dim (``list[int]`` / ``tuple[int, ...]``) overload is rejected
    at construction time; so is an empty sequence."""
    from tileops.ops.reduction.reduce import ProdFwdOp

    for dim in ([0, 1], (0, 1), [], None):
        with pytest.raises(ValueError, match="ProdFwdOp: dim = "):
            ProdFwdOp(dim=dim)


@MultiDimFixture
def test_amin_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import AminFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = AminFwdOp(dim=dims, keepdim=keepdim)
    ref = torch.amin(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


# Welford ops: var, std, var_mean


@MultiDimFixture
def test_var_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VarFwdOp(dim=dims, keepdim=keepdim)
    ref = torch.var(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@MultiDimFixture
def test_std_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = StdFwdOp(dim=dims, keepdim=keepdim)
    ref = torch.std(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@MultiDimFixture
def test_var_mean_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VarMeanFwdOp(dim=dims, keepdim=keepdim)
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


# LogSumExp


@MultiDimFixture
def test_logsumexp_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.softmax import LogSumExpFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = LogSumExpFwdOp(dim=dims, keepdim=keepdim)
    ref = torch.logsumexp(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, LogSumExpWorkload(tuple(x.shape), x.dtype).verification(x))


@pytest.mark.smoke
def test_logsumexp_edge_axes_special_values() -> None:
    """Own-layout edge-axis logsumexp preserves -inf and NaN row semantics."""
    from tileops.ops.reduction.softmax import LogSumExpFwdOp

    x = torch.randn(4, 32, 256, dtype=torch.float16, device=run_device())
    x[:, 0, :] = float("-inf")
    x[2, 1, 7] = float("nan")
    y = LogSumExpFwdOp(dim=[0, 2])(x)
    ref = torch.logsumexp(x.float(), dim=[0, 2]).to(x.dtype)
    assert y[0].item() == float("-inf")
    assert torch.isnan(y[1])
    finite = torch.isfinite(ref)
    compare_outputs(
        y[finite], ref[finite], LogSumExpWorkload(tuple(x.shape), x.dtype).verification(x)
    )


# Logical reduce ops: all, any, count_nonzero


class MultiDimLogicalFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dims, keepdim, dtype",
            [
                pytest.param(
                    (4, 32, 256),
                    [0, 1],
                    False,
                    torch.float32,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256),
                    [0, 1],
                    False,
                    torch.bool,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256),
                    [0, 1],
                    False,
                    torch.complex64,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256),
                    [0, 1],
                    True,
                    torch.float32,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


def _make_logical_input(
    shape: tuple,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate input tensor for logical reduce ops."""
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=run_device())
    if dtype.is_complex:
        return torch.randn(*shape, dtype=dtype, device=run_device())
    return torch.randn(*shape, dtype=dtype, device=run_device())


@MultiDimLogicalFixture
def test_all_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x = _make_logical_input(shape, dtype)
    op = AllFwdOp(dim=dims, keepdim=keepdim)
    ref = torch.all(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@MultiDimLogicalFixture
def test_any_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.logical_reduce import AnyFwdOp

    x = _make_logical_input(shape, dtype)
    op = AnyFwdOp(dim=dims, keepdim=keepdim)
    ref = torch.any(x.bool(), dim=dims, keepdim=keepdim)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


class MultiDimCountFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dims, dtype",
            [
                pytest.param(
                    (4, 32, 256),
                    [0, 1],
                    torch.float32,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256),
                    [0, 1],
                    torch.bool,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    (4, 32, 256),
                    [0, 1],
                    torch.complex64,
                    marks=pytest.mark.smoke,
                ),
            ],
        ),
    ]


@MultiDimCountFixture
def test_count_nonzero_multidim(
    shape: tuple,
    dims: list,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp

    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=torch.bool, device=run_device())
    elif dtype.is_complex:
        x = torch.randn(*shape, dtype=dtype, device=run_device())
    else:
        x = torch.randn(*shape, dtype=dtype, device=run_device())
        # Zero out some elements to make it interesting
        x[x < 0] = 0.0
    op = CountNonzeroFwdOp(dim=dims)
    ref = torch.count_nonzero(x, dim=dims)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


# Vector norm ops: l1, l2, inf


@MultiDimFixture
def test_l1_norm_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.vector_norm import VectorNormFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VectorNormFwdOp(1, dim=dims, keepdim=keepdim)
    ref = torch.linalg.vector_norm(
        x.float(),
        ord=1,
        dim=dims,
        keepdim=keepdim,
    ).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, vector_norm_verification(x.dtype))


@MultiDimFixture
def test_l2_norm_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from tileops.ops.reduction.vector_norm import VectorNormFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VectorNormFwdOp(2, dim=dims, keepdim=keepdim)
    ref = torch.linalg.vector_norm(
        x.float(),
        ord=2,
        dim=dims,
        keepdim=keepdim,
    ).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, vector_norm_verification(x.dtype))


@MultiDimFixture
def test_inf_norm_multidim(
    shape: tuple,
    dims: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    from math import inf

    from tileops.ops.reduction.vector_norm import VectorNormFwdOp

    x = torch.randn(*shape, dtype=dtype, device=run_device())
    op = VectorNormFwdOp(inf, dim=dims, keepdim=keepdim)
    ref = torch.linalg.vector_norm(
        x.float(),
        ord=float("inf"),
        dim=dims,
        keepdim=keepdim,
    ).to(dtype)
    y = op(x)

    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, vector_norm_verification(x.dtype))


# Empty dim list / tuple is full-reduction (matches PyTorch semantics)


@pytest.mark.smoke
def test_sum_empty_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(2, 3, 4, dtype=torch.float16, device=run_device())
    op = SumFwdOp(dim=[], keepdim=False)
    op_none = SumFwdOp(dim=None, keepdim=False)
    assert torch.allclose(op(x), op_none(x), **reduction_tolerance(torch.float16))


@pytest.mark.smoke
def test_mean_empty_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = torch.randn(2, 3, 4, dtype=torch.float16, device=run_device())
    op = MeanFwdOp(dim=(), keepdim=True)
    op_none = MeanFwdOp(dim=None, keepdim=True)
    assert torch.allclose(op(x), op_none(x), **reduction_tolerance(torch.float16))


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", ["amin", "amax", "count_nonzero"])
def test_simple_op_empty_dim_full_reduction(op_name: str) -> None:
    from tileops.ops.reduction.logical_reduce import CountNonzeroFwdOp
    from tileops.ops.reduction.reduce import AmaxFwdOp, AminFwdOp

    op_cls = {"amin": AminFwdOp, "amax": AmaxFwdOp, "count_nonzero": CountNonzeroFwdOp}[op_name]
    x = torch.randn(2, 3, 4, dtype=torch.float16, device=run_device())
    y_empty = op_cls(dim=[])(x)
    y_none = op_cls(dim=None)(x)
    assert y_empty.shape == y_none.shape
    if op_name == "count_nonzero":
        assert (y_empty == y_none).all()
    else:
        assert torch.allclose(y_empty, y_none, **reduction_tolerance(torch.float16))


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", ["std", "var"])
def test_welford_op_empty_dim_full_reduction(op_name: str) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp, VarFwdOp

    op_cls = {"std": StdFwdOp, "var": VarFwdOp}[op_name]
    x = torch.randn(2, 3, 4, dtype=torch.float16, device=run_device())
    y_empty = op_cls(dim=[], keepdim=False)(x)
    y_none = op_cls(dim=None, keepdim=False)(x)
    assert torch.allclose(y_empty, y_none, **reduction_tolerance(torch.float16))


@pytest.mark.smoke
def test_var_mean_empty_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = torch.randn(2, 3, 4, dtype=torch.float16, device=run_device())
    var_e, mean_e = VarMeanFwdOp(dim=[], keepdim=False)(x)
    var_n, mean_n = VarMeanFwdOp(dim=None, keepdim=False)(x)
    assert torch.allclose(var_e, var_n, **reduction_tolerance(torch.float16))
    assert torch.allclose(mean_e, mean_n, **reduction_tolerance(torch.float16))


@pytest.mark.smoke
def test_logsumexp_empty_dim_rejects() -> None:
    from tileops.ops.reduction.softmax import LogSumExpFwdOp

    x = torch.randn(2, 3, 4, dtype=torch.float16, device=run_device())
    op = LogSumExpFwdOp(dim=[], keepdim=False)
    with pytest.raises(ValueError, match="an empty dim is rejected"):
        op(x)


@pytest.mark.smoke
def test_all_empty_dim_is_noop() -> None:
    """AllFwdOp honors the spec's ``dim=[]`` no-op contract: output equals
    ``x.bool()`` with the input shape."""
    from tileops.ops.reduction.logical_reduce import AllFwdOp

    x = (torch.randn(2, 3, 4, device=run_device()) > 0).to(torch.float16)
    op = AllFwdOp(dim=[], keepdim=False)
    y = op(x)
    assert y.shape == x.shape
    assert y.dtype == torch.bool
    compare_outputs(y, x.bool(), reduction_verification((x.bool()).dtype))


@pytest.mark.smoke
def test_negative_dims_accepted() -> None:
    """Negative dims should be normalized and produce correct results."""
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(4, 8, 256, dtype=torch.float16, device=run_device())
    op = SumFwdOp(dim=[-1, 0], keepdim=False)
    ref = torch.sum(x.float(), dim=[0, 2], keepdim=False).to(torch.float16)
    y = op(x)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@pytest.mark.smoke
def test_duplicate_dims_raises() -> None:
    """Duplicate dims (after normalization) must raise ValueError at op level."""
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(4, 8, 256, dtype=torch.float16, device=run_device())
    op = SumFwdOp(dim=[1, 1], keepdim=False)
    with pytest.raises(ValueError, match="unique_axes"):
        op(x)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_edge_axis_reduce_returns_the_storage_dtype(dtype: torch.dtype) -> None:
    """The edge-axis columns pass writes the result dtype itself, with no cast after.

    Nothing downstream converts, so a pass left writing fp32 reaches the caller
    as fp32.
    """
    from tileops.ops.reduction import CountNonzeroFwdOp
    from tileops.ops.reduction.reduce import AmaxFwdOp, MeanFwdOp, SumFwdOp

    x = torch.randn(4, 32, 512, dtype=dtype, device=run_device())
    for op_cls, ref in (
        (SumFwdOp, lambda z: torch.sum(z.float(), dim=[0, 2])),
        (MeanFwdOp, lambda z: torch.mean(z.float(), dim=[0, 2])),
        (AmaxFwdOp, lambda z: torch.amax(z.float(), dim=[0, 2])),
    ):
        out = op_cls(dim=[0, 2])(x)
        assert out.dtype == dtype, f"{op_cls.__name__} returned {out.dtype}"
        compare_outputs(out, ref(x).to(dtype), reduction_verification(dtype))

    counted = CountNonzeroFwdOp(dim=[0, 2])(x)
    assert counted.dtype == torch.int64
    compare_outputs(
        counted,
        torch.count_nonzero(x, dim=[0, 2]),
        reduction_verification((torch.count_nonzero(x, dim=[0, 2])).dtype),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("shape", [(768, 512), (1536, 512)], ids=["ragged-split", "exact-split"])
def test_leading_axis_split_reduces_every_row(shape):
    """A split that does not divide the reduced extent still sums every row."""
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(shape, dtype=torch.float32, device=run_device())
    compare_outputs(SumFwdOp(dim=0)(x), x.sum(dim=0), reduction_verification((x.sum(dim=0)).dtype))
