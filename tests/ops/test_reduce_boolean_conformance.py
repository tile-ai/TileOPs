"""Spec-conformance tests for boolean / count reductions.

Covers ``AllFwdOp``, ``AnyFwdOp``, ``CountNonzeroFwdOp`` against the PyTorch
references (``torch.all`` / ``torch.any`` / ``torch.count_nonzero``) across
the three ``dim`` shapes the manifest signature declares -- ``int``,
``tuple[int, ...]``, ``None``. ``All`` / ``Any`` exercise both ``keepdim``
values; ``CountNonzero`` does not accept ``keepdim`` (matching
``torch.count_nonzero``).

Output dtype is part of the contract: ``All`` / ``Any`` must return
``torch.bool`` and ``CountNonzero`` must return ``torch.int64``. Each test
asserts both the output shape (in particular, a 0-D tensor for
``dim=None, keepdim=False``) and the exact output dtype, plus numeric
equality with PyTorch. Once green, the corresponding manifest entries can
flip from ``status: spec-only`` to ``status: implemented`` in a separate
manifest-only PR per the trust model.
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch

from tileops.ops.reduction.all_op import AllFwdOp
from tileops.ops.reduction.any_op import AnyFwdOp
from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

# (op_cls, torch_fn) pairs for ops sharing the (dim, keepdim) signature.
_OP_CASES: list[tuple[type, Callable]] = [
    (AllFwdOp, torch.all),
    (AnyFwdOp, torch.any),
]

_SHAPE = (4, 8, 256)
_UNALIGNED_SHAPE = (4, 8, 255)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, torch_fn", _OP_CASES, ids=[c[0].__name__ for c in _OP_CASES],
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
    """Each (op, dim-shape, keepdim, dtype) cell must match PyTorch.

    Input dtypes ``{fp16, bf16, fp32}`` differ from the bool output dtype,
    satisfying the spec-conformance requirement that at least one input
    dtype differ from the output dtype.
    """
    torch.manual_seed(0)
    # Mix in exact zeros so All/Any actually see a False contribution.
    raw = torch.randn(*_SHAPE, dtype=dtype, device="cuda")
    zero_mask = torch.rand(_SHAPE, device="cuda") < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = op_cls(dtype=dtype, dim=dim, keepdim=keepdim)
    y = op(x)
    if dim is None:
        ref = torch_fn(x)
        if keepdim:
            ref = ref.reshape([1] * x.ndim)
    else:
        ref = torch_fn(x, dim=dim, keepdim=keepdim)

    assert y.dtype == torch.bool, (
        f"{op_cls.__name__} output dtype {y.dtype}, expected torch.bool"
    )
    assert ref.dtype == torch.bool
    assert y.shape == ref.shape, (
        f"{op_cls.__name__} dim={dim} keepdim={keepdim} dtype={dtype}: "
        f"shape {y.shape} vs ref {ref.shape}"
    )
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, torch_fn", _OP_CASES, ids=[c[0].__name__ for c in _OP_CASES],
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
    op_cls: type, torch_fn: Callable, dim,
) -> None:
    """Unaligned innermost dim must still match PyTorch.

    The aligned ``_SHAPE`` (innermost = 256, a kernel-tile multiple) bypasses
    the logical-reduce kernel's masked-load boundary path. Use 255 to flush
    the pad branch on every (op, dim-mode) cell.
    """
    torch.manual_seed(0)
    dtype = torch.float16
    raw = torch.randn(*_UNALIGNED_SHAPE, dtype=dtype, device="cuda")
    zero_mask = torch.rand(_UNALIGNED_SHAPE, device="cuda") < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = op_cls(dtype=dtype, dim=dim, keepdim=False)
    y = op(x)
    ref = torch_fn(x) if dim is None else torch_fn(x, dim=dim, keepdim=False)

    assert y.dtype == torch.bool
    assert y.shape == ref.shape, (
        f"{op_cls.__name__} dim={dim} unaligned: shape {y.shape} vs ref {ref.shape}"
    )
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# CountNonzero: separate matrix because the op does not accept ``keepdim``.
# ---------------------------------------------------------------------------


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
    torch.manual_seed(0)
    raw = torch.randn(*_SHAPE, dtype=dtype, device="cuda")
    zero_mask = torch.rand(_SHAPE, device="cuda") < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = CountNonzeroFwdOp(dtype=dtype, dim=dim)
    y = op(x)
    ref = torch.count_nonzero(x, dim=dim)

    assert y.dtype == torch.int64, (
        f"CountNonzeroFwdOp output dtype {y.dtype}, expected torch.int64"
    )
    assert ref.dtype == torch.int64
    assert y.shape == ref.shape, (
        f"CountNonzeroFwdOp dim={dim} dtype={dtype}: "
        f"shape {y.shape} vs ref {ref.shape}"
    )
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


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
    torch.manual_seed(0)
    dtype = torch.float16
    raw = torch.randn(*_UNALIGNED_SHAPE, dtype=dtype, device="cuda")
    zero_mask = torch.rand(_UNALIGNED_SHAPE, device="cuda") < 0.1
    x = raw.masked_fill(zero_mask, 0)

    op = CountNonzeroFwdOp(dtype=dtype, dim=dim)
    y = op(x)
    ref = torch.count_nonzero(x, dim=dim)

    assert y.dtype == torch.int64
    assert y.shape == ref.shape, (
        f"CountNonzeroFwdOp dim={dim} unaligned: shape {y.shape} vs ref {ref.shape}"
    )
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
