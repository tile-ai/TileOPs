"""Arg-reduction operators (argmax, argmin)."""

from math import prod
from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction.argreduce import ArgreduceKernel

from .reduce import _ReduceOpBase

__all__ = ["ArgmaxFwdOp", "ArgminFwdOp"]


class _ArgreduceOpBase(_ReduceOpBase):
    """Tell the kernel the reduced axis's stride, and let it pick the layout.

    Reducing a non-last axis can be done two ways: transpose so the axis is last, which
    copies the whole tensor, or give a thread each output element and stride along the
    axis, which reads the original buffer coalesced. Which one pays off follows from the
    row count, the axis length and its stride — all three facts the kernel already holds,
    so the choice is the kernel's (`tileops.kernels.reduction.argreduce.ArgreduceKernel`).
    This op's part is the stride, which the shape and the reduced axis decide.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"argreduce": ArgreduceKernel}
    _kernel_key = "argreduce"

    def __init__(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Reduction axis. ``None`` (the default) returns the index into the
                flattened input, as ``torch.argmax(x)`` does.
            keepdim: Whether to retain the reduced dimension as size 1.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel map.
            tune: Whether to autotune the kernel.
        """
        super().__init__(dim, keepdim, target=target, kernel_map=kernel_map, tune=tune)

    def _output_dtype(self, x: torch.Tensor) -> torch.dtype:
        return torch.int64

    def _scalar_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The one element of a 0-d input is at index 0."""
        return torch.zeros((), dtype=torch.int64, device=x.device)

    def _build_kernel_kwargs(self, shape, axes, device_index) -> dict:
        """Elements between two neighbours along the reduced axis, on top of the shared set.

        One for the last axis and for a full reduction, which is the flattened buffer.
        """
        return {
            **super()._build_kernel_kwargs(shape, axes, device_index),
            "inner_stride": prod(shape[axes[-1] + 1 :]) if len(axes) == 1 else 1,
        }


class ArgmaxFwdOp(_ArgreduceOpBase):
    """Index of the maximum along ``dim``, following ``torch.argmax``; returns int64."""

    _op_kind = "argmax"


class ArgminFwdOp(_ArgreduceOpBase):
    """Index of the minimum along ``dim``, following ``torch.argmin``; returns int64."""

    _op_kind = "argmin"
