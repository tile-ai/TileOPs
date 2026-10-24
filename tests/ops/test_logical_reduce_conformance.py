"""PyTorch conformance for logical reductions and count-nonzero.

The cases cover manifest-supported ``dim`` forms, shape, dtype, and values;
``All`` and ``Any`` also cover both ``keepdim`` forms.
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch

from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import reduction_verification

# (op_cls, torch_fn) pairs for ops sharing the (dim, keepdim) signature.
_OP_CASES: list[tuple[type, Callable]] = [
    (AllFwdOp, torch.all),
    (AnyFwdOp, torch.any),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, torch_fn",
    _OP_CASES,
    ids=[c[0].__name__ for c in _OP_CASES],
)
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
def test_logical_reduce_conformance(
    op_cls: type,
    torch_fn: Callable,
    dim,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    """Each supported logical-reduction cell must match PyTorch."""
    shape = (4, 8, 256)
    torch.manual_seed(0)
    # Mix in exact zeros so All/Any actually see a False contribution.
    raw = torch.randn(*shape, dtype=dtype, device=run_device())
    zero_mask = torch.rand(shape, device=run_device()) < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = op_cls(dim=dim, keepdim=keepdim)
    y = op(x)
    if dim is None:
        ref = torch_fn(x)
        if keepdim:
            ref = ref.reshape([1] * x.ndim)
    else:
        ref = torch_fn(x, dim=dim, keepdim=keepdim)

    assert y.dtype == torch.bool, f"{op_cls.__name__} output dtype {y.dtype}, expected torch.bool"
    assert ref.dtype == torch.bool
    assert y.shape == ref.shape, (
        f"{op_cls.__name__} dim={dim} keepdim={keepdim} dtype={dtype}: "
        f"shape {y.shape} vs ref {ref.shape}"
    )
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, torch_fn",
    _OP_CASES,
    ids=[c[0].__name__ for c in _OP_CASES],
)
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
def test_logical_reduce_unaligned_innermost(
    op_cls: type,
    torch_fn: Callable,
    dim,
) -> None:
    """Unaligned innermost dim must still match PyTorch.

    An aligned innermost extent of 256 (a kernel-tile multiple) bypasses
    the logical-reduce kernel's masked-load boundary path. Use 255 to flush
    the pad branch on every (op, dim-mode) cell.
    """
    unaligned_shape = (4, 8, 255)
    torch.manual_seed(0)
    dtype = torch.float16
    raw = torch.randn(*unaligned_shape, dtype=dtype, device=run_device())
    zero_mask = torch.rand(unaligned_shape, device=run_device()) < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = op_cls(dim=dim, keepdim=False)
    y = op(x)
    ref = torch_fn(x) if dim is None else torch_fn(x, dim=dim, keepdim=False)

    assert y.dtype == torch.bool
    assert y.shape == ref.shape, (
        f"{op_cls.__name__} dim={dim} unaligned: shape {y.shape} vs ref {ref.shape}"
    )
    compare_outputs(y, ref, reduction_verification((ref).dtype))


# CountNonzero: separate matrix because the op does not accept ``keepdim``.


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def test_count_nonzero_conformance(dim, dtype: torch.dtype) -> None:
    """``CountNonzeroFwdOp`` must match ``torch.count_nonzero`` and emit int64.

    All input dtypes here differ from the int64 output dtype, satisfying the
    spec-conformance requirement that at least one input dtype differ from
    the output dtype.
    """
    shape = (4, 8, 256)
    torch.manual_seed(0)
    raw = torch.randn(*shape, dtype=dtype, device=run_device())
    zero_mask = torch.rand(shape, device=run_device()) < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = CountNonzeroFwdOp(dim=dim)
    y = op(x)
    ref = torch.count_nonzero(x, dim=dim)

    assert y.dtype == torch.int64, f"CountNonzeroFwdOp output dtype {y.dtype}, expected torch.int64"
    assert ref.dtype == torch.int64
    assert y.shape == ref.shape, (
        f"CountNonzeroFwdOp dim={dim} dtype={dtype}: shape {y.shape} vs ref {ref.shape}"
    )
    compare_outputs(y, ref, reduction_verification((ref).dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(-1, id="dim=int"),
        pytest.param((0, 2), id="dim=tuple"),
        pytest.param(None, id="dim=None"),
    ],
)
def test_count_nonzero_unaligned_innermost(dim) -> None:
    """Unaligned innermost dim must still match ``torch.count_nonzero``."""
    unaligned_shape = (4, 8, 255)
    torch.manual_seed(0)
    dtype = torch.float16
    raw = torch.randn(*unaligned_shape, dtype=dtype, device=run_device())
    zero_mask = torch.rand(unaligned_shape, device=run_device()) < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = CountNonzeroFwdOp(dim=dim)
    y = op(x)
    ref = torch.count_nonzero(x, dim=dim)

    assert y.dtype == torch.int64
    assert y.shape == ref.shape, (
        f"CountNonzeroFwdOp dim={dim} unaligned: shape {y.shape} vs ref {ref.shape}"
    )
    compare_outputs(y, ref, reduction_verification((ref).dtype))
