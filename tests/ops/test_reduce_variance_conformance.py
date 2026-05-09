"""Spec-conformance tests for variance reductions.

Covers ``VarFwdOp``, ``StdFwdOp``, ``VarMeanFwdOp`` against the PyTorch
references (``torch.var`` / ``torch.std`` / ``torch.var_mean``) across the
three ``dim`` shapes the manifest signature declares — ``int``,
``tuple[int, ...]``, ``None`` — every ``correction`` value in ``{0, 1, 2}``,
and both ``keepdim`` values.

These tests assert the output shape matches PyTorch (in particular, a 0-D
tensor for ``dim=None, keepdim=False``) and that the numerics match within
standard reduction tolerances. ``VarMeanFwdOp`` additionally asserts the
returned tuple shape/dtype matches ``torch.var_mean`` element-by-element.
Once green, the corresponding manifest entries can flip from
``status: spec-only`` to ``status: implemented`` in a separate
manifest-only PR per the trust model.
"""

from __future__ import annotations

import pytest
import torch

from tileops.ops.reduction.reduce import (
    StdFwdOp,
    VarFwdOp,
    VarMeanFwdOp,
)

_SHAPE = (4, 8, 256)
_UNALIGNED_SHAPE = (4, 8, 255)


def _tol(dtype: torch.dtype) -> dict:
    # Match the reduction-test convention shared by test_reduce_dim_none.py /
    # test_reduce_multidim.py / test_reduce_arithmetic_conformance.py: 1e-4
    # for fp32, 1e-2 for half precision (Welford accumulates in fp32 but the
    # narrowing cast at the boundary still produces ULP-scale rounding that
    # tighter tolerances catch noisily on real GPUs).
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-2, "rtol": 1e-2}


def _ref_var(x: torch.Tensor, dim, keepdim: bool, correction: int) -> torch.Tensor:
    return torch.var(
        x.float(), dim=dim, keepdim=keepdim, correction=correction,
    ).to(x.dtype)


def _ref_std(x: torch.Tensor, dim, keepdim: bool, correction: int) -> torch.Tensor:
    return torch.std(
        x.float(), dim=dim, keepdim=keepdim, correction=correction,
    ).to(x.dtype)


def _ref_var_mean(
    x: torch.Tensor, dim, keepdim: bool, correction: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    var, mean = torch.var_mean(
        x.float(), dim=dim, keepdim=keepdim, correction=correction,
    )
    return var.to(x.dtype), mean.to(x.dtype)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
@pytest.mark.parametrize("correction", [0, 1, 2], ids=["c=0", "c=1", "c=2"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def test_var_conformance(
    dim, correction: int, keepdim: bool, dtype: torch.dtype,
) -> None:
    """Each (dim-shape, correction, keepdim, dtype) cell must match torch.var."""
    torch.manual_seed(0)
    x = torch.randn(*_SHAPE, dtype=dtype, device="cuda")
    op = VarFwdOp(dtype=dtype, dim=dim, correction=correction, keepdim=keepdim)
    y = op(x)
    ref = _ref_var(x, dim, keepdim, correction)
    assert y.shape == ref.shape, (
        f"VarFwdOp dim={dim} correction={correction} keepdim={keepdim} "
        f"dtype={dtype}: shape {y.shape} vs ref {ref.shape}"
    )
    torch.testing.assert_close(y, ref, **_tol(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
@pytest.mark.parametrize("correction", [0, 1, 2], ids=["c=0", "c=1", "c=2"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def test_std_conformance(
    dim, correction: int, keepdim: bool, dtype: torch.dtype,
) -> None:
    """Each (dim-shape, correction, keepdim, dtype) cell must match torch.std."""
    torch.manual_seed(0)
    x = torch.randn(*_SHAPE, dtype=dtype, device="cuda")
    op = StdFwdOp(dtype=dtype, dim=dim, correction=correction, keepdim=keepdim)
    y = op(x)
    ref = _ref_std(x, dim, keepdim, correction)
    assert y.shape == ref.shape, (
        f"StdFwdOp dim={dim} correction={correction} keepdim={keepdim} "
        f"dtype={dtype}: shape {y.shape} vs ref {ref.shape}"
    )
    torch.testing.assert_close(y, ref, **_tol(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
@pytest.mark.parametrize("correction", [0, 1, 2], ids=["c=0", "c=1", "c=2"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def test_var_mean_conformance(
    dim, correction: int, keepdim: bool, dtype: torch.dtype,
) -> None:
    """Each (dim-shape, correction, keepdim, dtype) cell must match torch.var_mean.

    Asserts both elements of the returned 2-tuple match PyTorch in shape,
    dtype, and numerics.
    """
    torch.manual_seed(0)
    x = torch.randn(*_SHAPE, dtype=dtype, device="cuda")
    op = VarMeanFwdOp(dtype=dtype, dim=dim, correction=correction, keepdim=keepdim)
    out = op(x)
    assert isinstance(out, tuple) and len(out) == 2, (
        f"VarMeanFwdOp must return a 2-tuple, got {type(out).__name__}"
    )
    var_y, mean_y = out
    ref_var, ref_mean = _ref_var_mean(x, dim, keepdim, correction)
    # Tuple field order: (var, mean) — same as torch.var_mean.
    assert var_y.shape == ref_var.shape, (
        f"var shape {var_y.shape} vs ref {ref_var.shape}"
    )
    assert mean_y.shape == ref_mean.shape, (
        f"mean shape {mean_y.shape} vs ref {ref_mean.shape}"
    )
    assert var_y.dtype == ref_var.dtype, (
        f"var dtype {var_y.dtype} vs ref {ref_var.dtype}"
    )
    assert mean_y.dtype == ref_mean.dtype, (
        f"mean dtype {mean_y.dtype} vs ref {ref_mean.dtype}"
    )
    torch.testing.assert_close(var_y, ref_var, **_tol(dtype))
    torch.testing.assert_close(mean_y, ref_mean, **_tol(dtype))




@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
def test_var_unaligned_innermost(dim) -> None:
    """Unaligned innermost dim must still match torch.var.

    The aligned ``_SHAPE`` (innermost = 256, a kernel-tile multiple) bypasses
    the Welford kernel's masked-load boundary path. Use 255 to flush the pad
    branch on every dim-mode cell.
    """
    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.randn(*_UNALIGNED_SHAPE, dtype=dtype, device="cuda")
    op = VarFwdOp(dtype=dtype, dim=dim, correction=1, keepdim=False)
    y = op(x)
    ref = _ref_var(x, dim, False, 1)
    assert y.shape == ref.shape
    torch.testing.assert_close(y, ref, **_tol(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
def test_var_mean_unaligned_innermost(dim) -> None:
    """Unaligned innermost dim must still match torch.var_mean on both outputs."""
    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.randn(*_UNALIGNED_SHAPE, dtype=dtype, device="cuda")
    op = VarMeanFwdOp(dtype=dtype, dim=dim, correction=1, keepdim=False)
    var_y, mean_y = op(x)
    ref_var, ref_mean = _ref_var_mean(x, dim, False, 1)
    assert var_y.shape == ref_var.shape
    assert mean_y.shape == ref_mean.shape
    torch.testing.assert_close(var_y, ref_var, **_tol(dtype))
    torch.testing.assert_close(mean_y, ref_mean, **_tol(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
def test_std_unaligned_innermost(dim) -> None:
    """Unaligned innermost dim must still match torch.std."""
    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.randn(*_UNALIGNED_SHAPE, dtype=dtype, device="cuda")
    op = StdFwdOp(dtype=dtype, dim=dim, correction=1, keepdim=False)
    y = op(x)
    ref = _ref_std(x, dim, False, 1)
    assert y.shape == ref.shape
    torch.testing.assert_close(y, ref, **_tol(dtype))


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
