"""``SumFwdOp``, ``MeanFwdOp``, ``AmaxFwdOp`` and ``AminFwdOp`` over an innermost extent no
tile divides, after a tuple or ``None`` ``dim`` moves the reduced axes to the end.

``dim=int``, the dtypes and ``keepdim`` are covered in ``test_reduce.py``; tuple dims in
``test_reduce_multidim.py``; ``dim=None`` in ``test_reduce_dim_none.py``.
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
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import reduction_verification

# (op_cls, torch_fn) pairs.
_OP_CASES: list[tuple[type, Callable]] = [
    (SumFwdOp, torch.sum),
    (MeanFwdOp, torch.mean),
    (AmaxFwdOp, torch.amax),
    (AminFwdOp, torch.amin),
]


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
@pytest.mark.parametrize(
    "op_cls, torch_fn",
    _OP_CASES,
    ids=[c[0].__name__ for c in _OP_CASES],
)
@pytest.mark.parametrize(
    "dim", [pytest.param((0, 2), id="dim=tuple"), pytest.param(None, id="dim=None")]
)
def test_arithmetic_reduce_unaligned_innermost(
    op_cls: type,
    torch_fn: Callable,
    dim,
) -> None:
    """Unaligned innermost dim must still match PyTorch.

    An aligned innermost extent of 256 (a kernel-tile multiple) bypasses
    the simple-reduce kernel's masked-load boundary path. Use 255 to flush
    the pad branch, whose fill value is each op's identity. ``dim=-1`` at an
    unaligned N is covered by ``test_reduce.py``'s ``(128, 300)`` cases.
    """
    torch.manual_seed(0)
    unaligned_shape = (4, 8, 255)
    dtype = torch.float16
    x = torch.randn(*unaligned_shape, dtype=dtype, device=run_device())
    op = op_cls(dim=dim, keepdim=False)
    y = op(x)
    ref = _ref(torch_fn, x, dim, False)
    assert y.shape == ref.shape, (
        f"{op_cls.__name__} dim={dim} unaligned: shape {y.shape} vs ref {ref.shape}"
    )
    compare_outputs(y, ref, reduction_verification((ref).dtype))
