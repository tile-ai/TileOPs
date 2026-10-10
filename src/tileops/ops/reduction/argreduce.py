"""Arg-reduction operators (argmax, argmin)."""

from typing import ClassVar, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.reduction.argreduce import (
    ArgreduceKernel,
    ArgreduceSplitKernel,
    ArgreduceStridedKernel,
)
from tileops.kernels.reduction.call_spec import ArgreduceCall, ArgreduceFwdInterface
from tileops.ops.reduction.reduce import _ReduceOpBase

__all__ = ["ArgmaxFwdOp", "ArgminFwdOp"]


class _ArgreduceOpBase(_ReduceOpBase):
    """Argmax and argmin: the index of the extremum along one axis, or of the flattened input."""

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "argreduce": ArgreduceKernel,
        "argreduce_split": ArgreduceSplitKernel,
        "argreduce_strided": ArgreduceStridedKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"reduce": ArgreduceFwdInterface}

    def __init__(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Reduction axis. ``None`` (the default) returns the index into the
                flattened input, as ``torch.argmax(x)`` does.
            keepdim: Whether to retain the reduced dimension as size 1.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(dim, keepdim, target=target)

    def _output_dtype(self, x: torch.Tensor) -> torch.dtype:
        return torch.int64

    def _scalar_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The one element of a 0-d input is at index 0."""
        return torch.zeros((), dtype=torch.int64, device=x.device)

    def _call(self, x: torch.Tensor, axes: "tuple[int, ...]", n: int) -> ArgreduceCall:
        return ArgreduceCall(
            device=x.device,
            shape=tuple(x.shape),
            axes=axes,
            keepdim=self.keepdim,
            op_kind=self._op_kind,
            dtype=x.dtype,
        )


class ArgmaxFwdOp(_ArgreduceOpBase):
    """Index of the maximum along ``dim``, following ``torch.argmax``; returns int64."""

    _op_kind = "argmax"


class ArgminFwdOp(_ArgreduceOpBase):
    """Index of the minimum along ``dim``, following ``torch.argmin``; returns int64."""

    _op_kind = "argmin"
