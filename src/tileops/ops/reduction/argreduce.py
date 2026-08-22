"""Arg-reduction operators (argmax, argmin)."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.argreduce import ArgreduceKernel

from ._boundary import register_reduction_op
from .reduce import _ReduceOpBase

__all__ = ["ArgmaxFwdOp", "ArgminFwdOp"]


class _ArgreduceOpBase(_ReduceOpBase):
    """Tell the kernel the reduced axis's stride, and let it pick the layout.

    Reducing a non-last axis can be done two ways: transpose so the axis is last, which
    copies the whole tensor, or give a thread each output element and stride along the
    axis, which reads the original buffer coalesced. Which one pays off follows from the
    row count, the axis length and its stride — all three facts the kernel already holds,
    so the choice is the kernel's (:class:`~tileops.kernels.reduction.argreduce.ArgreduceKernel`).
    This op's part is the stride, which the shape and the reduced axis decide.
    """

    def _build_kernel_kwargs(self, x: torch.Tensor, axes: "tuple[int, ...]") -> dict:
        """Elements between two neighbours along the reduced axis, on top of the shared set.

        One for the last axis and for a full reduction, which is the flattened buffer.
        """
        return {
            **super()._build_kernel_kwargs(x, axes),
            "inner_stride": prod(x.shape[axes[-1] + 1 :]) if len(axes) == 1 else 1,
        }


class ArgmaxFwdOp(_ArgreduceOpBase):
    """Argmax reduction along an arbitrary dim, returning int64 indices.

    Construction: ``ArgmaxFwdOp(dim=None, keepdim=False)``.

    Args:
        dim: Reduction dimension. ``None`` (the default) matches
            ``torch.argmax(x)`` semantics: the input is treated as a
            contiguous flattened 1D buffer and the returned index is into
            that flattened tensor.
        keepdim: Whether to retain the reduced dimension as size 1.
        target: Which set of kernels serves this op — a target name, ``BUILTIN``
            for the in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "argmax"
    _kernel_key = "argreduce"
    _kernel_cls = ArgreduceKernel

    def __init__(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            dim=dim,
            keepdim=keepdim,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    def _validate_dim(self) -> None:
        """Argmax accepts a scalar ``int`` dim or ``None`` (full-tensor reduction).

        ``dim=None`` matches ``torch.argmax(x)`` semantics: the input is
        treated as a contiguous flattened 1D buffer and the returned index
        is into that flattened tensor.
        """
        if self.dim is None or isinstance(self.dim, int):
            return
        raise ValueError(
            f"ArgmaxFwdOp only supports scalar dim (int) or None, "
            f"got {type(self.dim).__name__}: {self.dim!r}"
        )


class ArgminFwdOp(_ArgreduceOpBase):
    """Argmin reduction along an arbitrary dim, returning int64 indices.

    Construction: ``ArgminFwdOp(dim=None, keepdim=False)``.

    Args:
        dim: Reduction dimension. ``None`` (the default) matches
            ``torch.argmin(x)`` semantics: the input is treated as a
            contiguous flattened 1D buffer and the returned index is into
            that flattened tensor.
        keepdim: Whether to retain the reduced dimension as size 1.
        target: Which set of kernels serves this op — a target name, ``BUILTIN``
            for the in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "argmin"
    _kernel_key = "argreduce"
    _kernel_cls = ArgreduceKernel

    def __init__(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            dim=dim,
            keepdim=keepdim,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    def _validate_dim(self) -> None:
        """Argmin accepts a scalar ``int`` dim or ``None`` (full-tensor reduction).

        ``dim=None`` matches ``torch.argmin(x)`` semantics: the input is
        treated as a contiguous flattened 1D buffer and the returned index
        is into that flattened tensor.
        """
        if self.dim is None or isinstance(self.dim, int):
            return
        raise ValueError(
            f"ArgminFwdOp only supports scalar dim (int) or None, "
            f"got {type(self.dim).__name__}: {self.dim!r}"
        )


for _op_cls in (
    ArgmaxFwdOp,
    ArgminFwdOp,
):
    register_reduction_op(_op_cls)
