"""Regression tests for reduction-op constructor defaults and empty-dim semantics.

Pins two manifest-conformance invariants for the reduction op family:

1. For the ten ops whose manifest declares ``default: null`` on ``dim``
   (Sum/Mean/Amax/Amin/Var/Std/VarMean/All/Any/CountNonzero), constructing
   the op with only ``dtype=`` performs a full reduction (output shape
   equals ``torch.<op>(x).shape``). ``ProdFwdOp`` keeps its documented
   ``dim=-1`` default.

2. ``AllFwdOp`` / ``AnyFwdOp`` honor the spec's ``dim=[]`` / ``dim=()``
   no-op contract: output shape equals the input shape, output dtype is
   ``bool``, and values equal ``x.bool()``.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


_FLOAT_SHAPE = (2, 4, 8)
_LOGICAL_SHAPE = (2, 4, 8)


def _make_float(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype, device="cuda")


def _make_logical(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    # values in {-1, 0, 1} so .bool() has both T and F.
    return (torch.randint(-1, 2, shape, device="cuda")).to(dtype)


# ---------------------------------------------------------------------------
# AC-3: default dim=None for the ten ops -> full reduction on 3-D input
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_sum_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = SumFwdOp(dtype=torch.float16)
    y = op(x)
    assert y.shape == torch.sum(x).shape


@pytest.mark.smoke
def test_mean_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = MeanFwdOp(dtype=torch.float16)
    y = op(x)
    assert y.shape == torch.mean(x).shape


@pytest.mark.smoke
def test_amax_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import AmaxFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = AmaxFwdOp(dtype=torch.float16)
    y = op(x)
    assert y.shape == torch.amax(x).shape


@pytest.mark.smoke
def test_amin_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import AminFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = AminFwdOp(dtype=torch.float16)
    y = op(x)
    assert y.shape == torch.amin(x).shape


@pytest.mark.smoke
def test_var_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = VarFwdOp(dtype=torch.float16)
    y = op(x)
    assert y.shape == torch.var(x).shape


@pytest.mark.smoke
def test_std_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import StdFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = StdFwdOp(dtype=torch.float16)
    y = op(x)
    assert y.shape == torch.std(x).shape


@pytest.mark.smoke
def test_var_mean_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = _make_float(_FLOAT_SHAPE, torch.float16)
    op = VarMeanFwdOp(dtype=torch.float16)
    var_out, mean_out = op(x)
    ref_var, ref_mean = torch.var_mean(x)
    assert var_out.shape == ref_var.shape
    assert mean_out.shape == ref_mean.shape


@pytest.mark.smoke
def test_all_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    x = _make_logical(_LOGICAL_SHAPE, torch.float16)
    op = AllFwdOp(dtype=torch.float16)
    y = op(x)
    assert y.shape == torch.all(x.bool()).shape
    assert y.dtype == torch.bool


@pytest.mark.smoke
def test_any_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x = _make_logical(_LOGICAL_SHAPE, torch.float16)
    op = AnyFwdOp(dtype=torch.float16)
    y = op(x)
    assert y.shape == torch.any(x.bool()).shape
    assert y.dtype == torch.bool


@pytest.mark.smoke
def test_count_nonzero_default_dim_full_reduction() -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    x = _make_logical(_LOGICAL_SHAPE, torch.float16)
    op = CountNonzeroFwdOp(dtype=torch.float16)
    y = op(x)
    assert y.shape == torch.count_nonzero(x).shape
    assert y.dtype == torch.int64


# ---------------------------------------------------------------------------
# AC-4: ProdFwdOp keeps documented dim=-1 default
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_prod_default_dim_last_axis() -> None:
    from tileops.ops.reduction.reduce import ProdFwdOp

    # use a narrow value range so fp16 prod is numerically stable
    x = torch.rand(*_FLOAT_SHAPE, dtype=torch.float16, device="cuda") * 0.01 + 0.99
    op = ProdFwdOp(dtype=torch.float16)
    y = op(x)
    assert y.shape == torch.prod(x, dim=-1).shape


# ---------------------------------------------------------------------------
# AC-5: AllFwdOp/AnyFwdOp dim=[] / dim=() noop contract
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("empty_dim", [[], ()])
def test_all_empty_dim_noop(empty_dim) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    x = _make_logical(_LOGICAL_SHAPE, torch.float16)
    op = AllFwdOp(dtype=torch.float16, dim=empty_dim)
    y = op(x)
    assert y.shape == x.shape
    assert y.dtype == torch.bool
    assert torch.equal(y, x.bool())


@pytest.mark.smoke
@pytest.mark.parametrize("empty_dim", [[], ()])
def test_any_empty_dim_noop(empty_dim) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x = _make_logical(_LOGICAL_SHAPE, torch.float16)
    op = AnyFwdOp(dtype=torch.float16, dim=empty_dim)
    y = op(x)
    assert y.shape == x.shape
    assert y.dtype == torch.bool
    assert torch.equal(y, x.bool())


# ---------------------------------------------------------------------------
# AC-6: normalize_dim noop policy returns []
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_normalize_dim_noop_returns_empty() -> None:
    from tileops.ops.reduction._multidim import normalize_dim

    assert normalize_dim([], ndim=3, empty_dim_policy="noop") == []
    assert normalize_dim((), ndim=3, empty_dim_policy="noop") == []


@pytest.mark.smoke
def test_normalize_dim_reject_raises_on_empty() -> None:
    from tileops.ops.reduction._multidim import normalize_dim

    with pytest.raises(ValueError):
        normalize_dim([], ndim=3, empty_dim_policy="reject")


@pytest.mark.smoke
def test_normalize_dim_full_returns_all() -> None:
    from tileops.ops.reduction._multidim import normalize_dim

    assert normalize_dim([], ndim=3, empty_dim_policy="full") == [0, 1, 2]


@pytest.mark.smoke
def test_empty_dim_policy_class_attrs() -> None:
    """AC-6: per-op empty_dim_policy bindings."""
    from tileops.ops.reduction.all_op import AllFwdOp
    from tileops.ops.reduction.any_op import AnyFwdOp
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp
    from tileops.ops.reduction.reduce import (
        AmaxFwdOp,
        AminFwdOp,
        MeanFwdOp,
        ProdFwdOp,
        StdFwdOp,
        SumFwdOp,
        VarFwdOp,
        VarMeanFwdOp,
        _ReduceOpBase,
    )

    assert _ReduceOpBase._empty_dim_policy == "reject"
    assert AllFwdOp._empty_dim_policy == "noop"
    assert AnyFwdOp._empty_dim_policy == "noop"
    for cls in (
        SumFwdOp, MeanFwdOp, AmaxFwdOp, AminFwdOp,
        StdFwdOp, VarFwdOp, VarMeanFwdOp, CountNonzeroFwdOp,
    ):
        assert cls._empty_dim_policy == "full", cls.__name__
    # ProdFwdOp inherits default (reject); empty dim is not in its contract
    assert ProdFwdOp._empty_dim_policy == "reject"
