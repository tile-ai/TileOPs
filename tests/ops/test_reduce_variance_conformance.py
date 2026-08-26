"""Spec-conformance tests for variance reductions.

``VarFwdOp``, ``StdFwdOp`` and ``VarMeanFwdOp`` against ``torch.var`` / ``torch.std`` /
``torch.var_mean``. The three share a forward, so they share a case table.

The axes are kept apart rather than crossed, because each answers a different question:
the ``dim`` shape picks a ``normalize_dim`` branch and ``keepdim`` an output-shape branch,
so those two are crossed; ``correction`` is a constant the kernel bakes in, where only
"zero" and "nonzero" differ; and the element type has to be swept but changes no branch.
"""

from __future__ import annotations

import pytest
import torch

from tileops.ops.reduction.reduce import StdFwdOp, VarFwdOp, VarMeanFwdOp

_SHAPE = (4, 8, 256)
_UNALIGNED_SHAPE = (4, 8, 255)  # innermost off a tile multiple: the masked-load boundary

_DIMS = [
    pytest.param(-1, id="dim=int"),
    pytest.param((0, 2), id="dim=tuple"),
    pytest.param(None, id="dim=None"),
]


def _tol(dtype: torch.dtype) -> dict:
    # The reduction-test convention: 1e-4 for fp32, 1e-2 for half precision. Welford
    # accumulates in fp32, but the narrowing cast at the boundary still rounds.
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-2, "rtol": 1e-2}


def _ref_var(x, dim, keepdim, correction):
    return (torch.var(x.float(), dim=dim, keepdim=keepdim, correction=correction).to(x.dtype),)


def _ref_std(x, dim, keepdim, correction):
    return (torch.std(x.float(), dim=dim, keepdim=keepdim, correction=correction).to(x.dtype),)


def _ref_var_mean(x, dim, keepdim, correction):
    var, mean = torch.var_mean(x.float(), dim=dim, keepdim=keepdim, correction=correction)
    return var.to(x.dtype), mean.to(x.dtype)


#: Each op with the reference it must match. Every reference returns a tuple, so one
#: comparison serves the two single-output ops and the one that returns a pair.
_OPS = [
    pytest.param(VarFwdOp, _ref_var, id="var"),
    pytest.param(StdFwdOp, _ref_std, id="std"),
    pytest.param(VarMeanFwdOp, _ref_var_mean, id="var-mean"),
]


def _check(op_cls, ref_fn, x, dim, keepdim, correction) -> None:
    """Run *op_cls* against its reference and compare every output it returns."""
    out = op_cls(dim=dim, correction=correction, keepdim=keepdim)(x)
    got = out if isinstance(out, tuple) else (out,)
    want = ref_fn(x, dim, keepdim, correction)
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape, f"shape {g.shape} vs ref {w.shape}"
        assert g.dtype == w.dtype, f"dtype {g.dtype} vs ref {w.dtype}"
        torch.testing.assert_close(g, w, **_tol(x.dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, ref_fn", _OPS)
@pytest.mark.parametrize("dim", _DIMS)
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
def test_the_output_shape_matches_torch(op_cls, ref_fn, dim, keepdim) -> None:
    """The two axes that pick branches, crossed: ``dim=None, keepdim=False`` is 0-D."""
    torch.manual_seed(0)
    x = torch.randn(*_SHAPE, dtype=torch.float16, device="cuda")

    _check(op_cls, ref_fn, x, dim, keepdim, correction=1)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, ref_fn", _OPS)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"]
)
def test_every_declared_dtype_matches_torch(op_cls, ref_fn, dtype) -> None:
    """Swept, not crossed: the element type reaches no branch the shape axes do not."""
    torch.manual_seed(0)
    x = torch.randn(*_SHAPE, dtype=dtype, device="cuda")

    _check(op_cls, ref_fn, x, dim=-1, keepdim=False, correction=1)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, ref_fn", _OPS)
@pytest.mark.parametrize(
    "dim", [pytest.param(-1, id="dim=int"), pytest.param((0, 2), id="dim=tuple")]
)
def test_a_zero_correction_matches_torch(op_cls, ref_fn, dim) -> None:
    """The one ``correction`` that differs in kind: the denominator is ``N``, not ``N - c``.

    ``dim=(0, 2)`` bakes the correction into the edge-axis merge instead of the
    rows kernel, so both denominators are exercised.
    """
    torch.manual_seed(0)
    x = torch.randn(*_SHAPE, dtype=torch.float16, device="cuda")

    _check(op_cls, ref_fn, x, dim=dim, keepdim=False, correction=0)


@pytest.mark.smoke
def test_edge_axis_variance_keeps_a_large_mean_fp16() -> None:
    """The edge-axis path merges Welford partials; a naive sum of squares would cancel."""
    torch.manual_seed(0)
    x = (torch.randn(4, 8, 256, dtype=torch.float16, device="cuda") + 60.0).half()

    _check(VarFwdOp, _ref_var, x, dim=(0, 2), keepdim=False, correction=1)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, ref_fn", _OPS)
@pytest.mark.parametrize("dim", _DIMS)
def test_an_unaligned_innermost_dim_matches_torch(op_cls, ref_fn, dim) -> None:
    """255 flushes the masked-load boundary that a tile-multiple extent skips."""
    torch.manual_seed(0)
    x = torch.randn(*_UNALIGNED_SHAPE, dtype=torch.float16, device="cuda")

    _check(op_cls, ref_fn, x, dim, keepdim=False, correction=1)


@pytest.mark.smoke
def test_var_mean_returns_the_pair_in_torch_s_order() -> None:
    """The only shape-of-return difference in the family, so the only test that needs it."""
    torch.manual_seed(0)
    x = torch.randn(*_SHAPE, dtype=torch.float16, device="cuda")

    out = VarMeanFwdOp(dim=-1)(x)

    assert isinstance(out, tuple) and len(out) == 2, out
    ref_var, ref_mean = _ref_var_mean(x, -1, False, 1)
    torch.testing.assert_close(out[0], ref_var, **_tol(x.dtype))
    torch.testing.assert_close(out[1], ref_mean, **_tol(x.dtype))
