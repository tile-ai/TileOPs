"""Scalar-input conformance tests for status-implemented reduction ops.

Asserts that each of ``Sum/Mean/Amax/Amin/Var/Std/VarMean/All/Any/CountNonzero``
accepts a 0-D input tensor paired with each ``dim`` form PyTorch accepts
(``None``, ``0``, ``-1``, ``()``, ``[]``) and returns a result that matches
the corresponding ``torch.<op>`` reference in both value and shape.

For the Welford family (``VarFwdOp``, ``StdFwdOp``, ``VarMeanFwdOp``) the
scalar input with default ``correction=1`` produces ``nan`` and emits a
``UserWarning`` matching PyTorch's behavior.
"""

from __future__ import annotations

import warnings

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


_DIM_FORMS = [None, 0, -1, (), []]


def _ids(prefix: str):
    return [f"{prefix}-dim={d!r}" for d in _DIM_FORMS]


# ---------------------------------------------------------------------------
# Arithmetic + logical reductions on scalar input
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=_ids("sum"))
def test_sum_scalar_input(dim) -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.tensor(3.5, dtype=torch.float32, device="cuda")
    op = SumFwdOp(dtype=torch.float32, dim=dim)
    y = op(x)
    ref = torch.sum(x, dim=dim) if dim is not None else torch.sum(x)
    assert y.shape == ref.shape
    torch.testing.assert_close(y, ref)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=_ids("mean"))
def test_mean_scalar_input(dim) -> None:
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = torch.tensor(2.0, dtype=torch.float32, device="cuda")
    op = MeanFwdOp(dtype=torch.float32, dim=dim)
    y = op(x)
    ref = torch.mean(x, dim=dim) if dim is not None else torch.mean(x)
    assert y.shape == ref.shape
    torch.testing.assert_close(y, ref)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=_ids("amax"))
def test_amax_scalar_input(dim) -> None:
    from tileops.ops.reduction.reduce import AmaxFwdOp

    x = torch.tensor(-1.5, dtype=torch.float32, device="cuda")
    op = AmaxFwdOp(dtype=torch.float32, dim=dim)
    y = op(x)
    ref = torch.amax(x, dim=dim) if dim is not None else torch.amax(x)
    assert y.shape == ref.shape
    torch.testing.assert_close(y, ref)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=_ids("amin"))
def test_amin_scalar_input(dim) -> None:
    from tileops.ops.reduction.reduce import AminFwdOp

    x = torch.tensor(4.25, dtype=torch.float32, device="cuda")
    op = AminFwdOp(dtype=torch.float32, dim=dim)
    y = op(x)
    ref = torch.amin(x, dim=dim) if dim is not None else torch.amin(x)
    assert y.shape == ref.shape
    torch.testing.assert_close(y, ref)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", [0, -1], ids=["prod-dim=0", "prod-dim=-1"])
def test_prod_scalar_input(dim) -> None:
    from tileops.ops.reduction.reduce import ProdFwdOp

    x = torch.tensor(3.0, dtype=torch.float32, device="cuda")
    op = ProdFwdOp(dtype=torch.float32, dim=dim)
    y = op(x)
    ref = torch.prod(x, dim=dim)
    assert y.shape == ref.shape
    torch.testing.assert_close(y, ref)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=_ids("all"))
def test_all_scalar_input(dim) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    x = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    op = AllFwdOp(dtype=torch.float32, dim=dim)
    y = op(x)
    ref = torch.all(x, dim=dim) if dim is not None else torch.all(x)
    assert y.shape == ref.shape
    assert y.dtype == torch.bool
    assert torch.equal(y, ref)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=_ids("any"))
def test_any_scalar_input(dim) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    x = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    op = AnyFwdOp(dtype=torch.float32, dim=dim)
    y = op(x)
    ref = torch.any(x, dim=dim) if dim is not None else torch.any(x)
    assert y.shape == ref.shape
    assert y.dtype == torch.bool
    assert torch.equal(y, ref)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=_ids("count_nonzero"))
def test_count_nonzero_scalar_input(dim) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    x = torch.tensor(2.5, dtype=torch.float32, device="cuda")
    op = CountNonzeroFwdOp(dtype=torch.float32, dim=dim)
    y = op(x)
    ref = torch.count_nonzero(x, dim=dim) if dim is not None else torch.count_nonzero(x)
    assert y.shape == ref.shape
    assert y.dtype == ref.dtype
    assert torch.equal(y, ref)


# ---------------------------------------------------------------------------
# Welford-family reductions on scalar input -> nan + UserWarning
# ---------------------------------------------------------------------------


def _expect_var_warning() -> bool:
    """Return whether PyTorch emits a UserWarning for var on scalar input.

    PyTorch raises a "degrees of freedom is <= 0" UserWarning when
    ``correction >= N``. We probe the reference path so the test stays
    aligned with the local PyTorch build's exact warning emission.
    """
    x = torch.tensor(1.0, dtype=torch.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.var(x)
    return any(issubclass(w.category, UserWarning) for w in caught)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=_ids("var"))
def test_var_scalar_input(dim) -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    x = torch.tensor(1.5, dtype=torch.float32, device="cuda")
    op = VarFwdOp(dtype=torch.float32, dim=dim)
    expect_warn = _expect_var_warning()
    with warnings.catch_warnings(record=True) as op_caught:
        warnings.simplefilter("always")
        y = op(x)
    ref = torch.var(x, dim=dim) if dim is not None else torch.var(x)
    assert y.shape == ref.shape == ()
    assert torch.isnan(y).item()
    assert torch.isnan(ref).item()
    if expect_warn:
        assert any(issubclass(w.category, UserWarning) for w in op_caught)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=_ids("std"))
def test_std_scalar_input(dim) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp

    x = torch.tensor(-0.75, dtype=torch.float32, device="cuda")
    op = StdFwdOp(dtype=torch.float32, dim=dim)
    expect_warn = _expect_var_warning()
    with warnings.catch_warnings(record=True) as op_caught:
        warnings.simplefilter("always")
        y = op(x)
    ref = torch.std(x, dim=dim) if dim is not None else torch.std(x)
    assert y.shape == ref.shape == ()
    assert torch.isnan(y).item()
    assert torch.isnan(ref).item()
    if expect_warn:
        assert any(issubclass(w.category, UserWarning) for w in op_caught)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", _DIM_FORMS, ids=_ids("var_mean"))
def test_var_mean_scalar_input(dim) -> None:
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = torch.tensor(2.25, dtype=torch.float32, device="cuda")
    op = VarMeanFwdOp(dtype=torch.float32, dim=dim)
    expect_warn = _expect_var_warning()
    with warnings.catch_warnings(record=True) as op_caught:
        warnings.simplefilter("always")
        var_out, mean_out = op(x)
    var_ref, mean_ref = (
        torch.var_mean(x, dim=dim) if dim is not None else torch.var_mean(x)
    )
    assert var_out.shape == var_ref.shape == ()
    assert mean_out.shape == mean_ref.shape == ()
    assert torch.isnan(var_out).item()
    assert torch.isnan(var_ref).item()
    torch.testing.assert_close(mean_out, mean_ref)
    if expect_warn:
        assert any(issubclass(w.category, UserWarning) for w in op_caught)
