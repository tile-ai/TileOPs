"""Spec-conformance tests for arithmetic reductions.

Covers ``SumFwdOp``, ``MeanFwdOp``, ``AmaxFwdOp``, ``AminFwdOp`` against the
PyTorch reference (``torch.sum`` / ``torch.mean`` / ``torch.amax`` /
``torch.amin``) across the three ``dim`` shapes the manifest signature
declares — ``int``, ``tuple[int, ...]``, ``None`` — and both ``keepdim``
values.

These tests assert the output shape matches PyTorch (in particular, a 0-D
tensor for ``dim=None, keepdim=False``) and that the numerics match within
standard reduction tolerances. Once green, the corresponding manifest
entries can flip from ``status: spec-only`` to ``status: implemented``
in a separate manifest-only PR per the trust model.
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch

from tileops.ops.reduction.reduce import (
    AmaxFwdOp,
    AminFwdOp,
    MeanFwdOp,
    SumFwdOp,
)

# (op_cls, torch_fn) pairs.
_OP_CASES: list[tuple[type, Callable]] = [
    (SumFwdOp, torch.sum),
    (MeanFwdOp, torch.mean),
    (AmaxFwdOp, torch.amax),
    (AminFwdOp, torch.amin),
]

_SHAPE = (4, 8, 256)


def _tol(dtype: torch.dtype) -> dict:
    # Reduce kernels accumulate in fp32 and only narrow at the boundary, so
    # half-precision tolerances can stay close to the unit in the last place
    # of the storage dtype rather than the looser 1e-2 default.
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-3, "rtol": 1e-3}


def _ref(torch_fn: Callable, x: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """PyTorch ground-truth reference. Normalizes ``dim=None`` to all dims
    so the same call shape works for sum/mean (which accept None directly)
    and amax/amin (which require an explicit dim list)."""
    if dim is None:
        dims = list(range(x.ndim))
    elif isinstance(dim, tuple):
        dims = list(dim)
    else:
        dims = dim
    return torch_fn(x.float(), dim=dims, keepdim=keepdim).to(x.dtype)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, torch_fn", _OP_CASES, ids=[c[0].__name__ for c in _OP_CASES])
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True], ids=["keepdim=False", "keepdim=True"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def test_arithmetic_reduce_conformance(
    op_cls: type,
    torch_fn: Callable,
    dim,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    """Each (op, dim-shape, keepdim, dtype) cell must match PyTorch."""
    torch.manual_seed(0)
    x = torch.randn(*_SHAPE, dtype=dtype, device="cuda")
    op = op_cls(dtype=dtype, dim=dim, keepdim=keepdim)
    y = op(x)
    ref = _ref(torch_fn, x, dim, keepdim)
    assert y.shape == ref.shape, (
        f"{op_cls.__name__} dim={dim} keepdim={keepdim} dtype={dtype}: "
        f"shape {y.shape} vs ref {ref.shape}"
    )
    torch.testing.assert_close(y, ref, **_tol(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, torch_fn", _OP_CASES, ids=[c[0].__name__ for c in _OP_CASES])
def test_dim_none_keepdim_false_returns_0d(op_cls: type, torch_fn: Callable) -> None:
    """``dim=None, keepdim=False`` must return a 0-D tensor matching PyTorch."""
    x = torch.randn(*_SHAPE, dtype=torch.float32, device="cuda")
    op = op_cls(dtype=torch.float32, dim=None, keepdim=False)
    y = op(x)
    ref = _ref(torch_fn, x, None, False)
    assert y.ndim == 0, f"{op_cls.__name__}: expected 0-D, got shape {y.shape}"
    assert ref.ndim == 0
    torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
