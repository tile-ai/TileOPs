"""Spec-conformance tests for variance reductions.

``VarFwdOp``, ``StdFwdOp`` and ``VarMeanFwdOp`` against ``torch.var`` / ``torch.std`` /
``torch.var_mean``. The three share a forward, so they share a case table.

The axes are kept apart rather than crossed, because each answers a different question:
the ``dim`` shape picks which axes reduce and ``keepdim`` an output-shape branch,
so those two are crossed; ``correction`` is a constant the kernel bakes in, where only
"zero" and "nonzero" differ; and the element type has to be swept but changes no branch.
"""

from __future__ import annotations

import pytest
import torch

from tileops.ops.reduction.reduce import StdFwdOp, VarFwdOp, VarMeanFwdOp
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import reduction_verification

_DIMS = [
    pytest.param(-1, id="dim=int"),
    pytest.param((0, 2), id="dim=tuple"),
    pytest.param(None, id="dim=None"),
]


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
        compare_outputs(g, w, reduction_verification((w).dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, ref_fn", _OPS)
@pytest.mark.parametrize("dim", _DIMS)
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
def test_the_output_shape_matches_torch(op_cls, ref_fn, dim, keepdim) -> None:
    """The two axes that pick branches, crossed: ``dim=None, keepdim=False`` is 0-D."""
    shape = (4, 8, 256)
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float16, device=run_device())

    _check(op_cls, ref_fn, x, dim, keepdim, correction=1)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, ref_fn", _OPS)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"]
)
def test_every_declared_dtype_matches_torch(op_cls, ref_fn, dtype) -> None:
    """Swept, not crossed: the element type reaches no branch the shape axes do not."""
    shape = (4, 8, 256)
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=run_device())

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
    shape = (4, 8, 256)
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float16, device=run_device())

    _check(op_cls, ref_fn, x, dim=dim, keepdim=False, correction=0)


@pytest.mark.parametrize(
    "op_cls, ref_fn, shape, dim",
    [
        # The edge-axis path merges Welford partials.
        pytest.param(VarFwdOp, _ref_var, (4, 8, 256), (0, 2), marks=pytest.mark.smoke, id="edge"),
        # Padded rows, one tile and several: the pad is masked out of the centered sum.
        pytest.param(VarFwdOp, _ref_var, (4, 8, 255), -1, marks=pytest.mark.full, id="row"),
        pytest.param(
            VarMeanFwdOp, _ref_var_mean, (4, 8, 255), -1, marks=pytest.mark.full, id="row-pair"
        ),
        pytest.param(StdFwdOp, _ref_std, (4, 33000), -1, marks=pytest.mark.full, id="tiled-row"),
        pytest.param(
            VarMeanFwdOp,
            _ref_var_mean,
            (4, 33000),
            -1,
            marks=pytest.mark.full,
            id="tiled-row-pair",
        ),
    ],
)
def test_variance_keeps_a_large_mean_fp16(op_cls, ref_fn, shape, dim) -> None:
    """A mean far from zero, where a naive sum of squares would cancel."""
    torch.manual_seed(0)
    x = (torch.randn(*shape, dtype=torch.float16, device=run_device()) + 60.0).half()

    _check(op_cls, ref_fn, x, dim=dim, keepdim=False, correction=1)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, ref_fn", _OPS)
@pytest.mark.parametrize("dim", _DIMS)
def test_an_unaligned_innermost_dim_matches_torch(op_cls, ref_fn, dim) -> None:
    """255 flushes the masked-load boundary that a tile-multiple extent skips."""
    unaligned_shape = (4, 8, 255)
    torch.manual_seed(0)
    x = torch.randn(*unaligned_shape, dtype=torch.float16, device=run_device())

    _check(op_cls, ref_fn, x, dim, keepdim=False, correction=1)


@pytest.mark.smoke
def test_var_mean_returns_the_pair_in_torch_s_order() -> None:
    """The only shape-of-return difference in the family, so the only test that needs it."""
    shape = (4, 8, 256)
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float16, device=run_device())

    out = VarMeanFwdOp(dim=-1)(x)

    assert isinstance(out, tuple) and len(out) == 2, out
    ref_var, ref_mean = _ref_var_mean(x, -1, False, 1)
    compare_outputs(out[0], ref_var, reduction_verification((ref_var).dtype))
    compare_outputs(out[1], ref_mean, reduction_verification((ref_mean).dtype))


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:.*degrees of freedom:UserWarning")
@pytest.mark.parametrize("correction", [0.5, 2, 3])
@pytest.mark.parametrize("op_cls, ref_fn", [(VarFwdOp, torch.var), (StdFwdOp, torch.std)])
def test_a_fractional_or_excess_correction_matches_torch(op_cls, ref_fn, correction) -> None:
    """torch divides by ``max(0, n - correction)``: a spread over no degrees of freedom is
    ``inf``, including one too small for the storage dtype, and no spread is NaN."""
    x = torch.tensor([[0.0, 1e-4], [1.0, 1.0]], dtype=torch.float16, device=run_device())
    got = op_cls(dim=1, correction=correction)(x)
    compare_outputs(
        got,
        ref_fn(x, dim=1, correction=correction),
        reduction_verification((ref_fn(x, dim=1, correction=correction)).dtype),
    )
