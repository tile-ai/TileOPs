"""Logical reduction operators (all, any, count_nonzero)."""

from typing import ClassVar, List, Mapping, Tuple, Union

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.reduction.call_spec import (
    CountNonzeroFwdInterface,
    LogicalReduceCall,
    LogicalReduceFwdInterface,
)
from tileops.kernels.reduction.logical_reduce import (
    CountNonzeroEdgeTwoPassKernel,
    LogicalReduceEdgeFusedKernel,
    LogicalReduceEdgeTwoPassKernel,
    LogicalReduceKernel,
)
from tileops.ops.reduction.reduce import _ReduceOpBase

__all__ = ["AllFwdOp", "AnyFwdOp", "CountNonzeroFwdOp"]


class _LogicalReduceOpBase(_ReduceOpBase):
    """Shared dispatch for logical reductions.

    Every numeric dtype is accepted, bool, int32, int64 and complex included; the op passes
    the tensor its manifest declares and the kernel reads it at its own bytes.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "logical_reduce_edge_fused": LogicalReduceEdgeFusedKernel,
        "logical_reduce_edge_two_pass": LogicalReduceEdgeTwoPassKernel,
        "logical_reduce": LogicalReduceKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "reduce": LogicalReduceFwdInterface
    }
    _output: ClassVar[torch.dtype] = torch.bool

    def _output_dtype(self, x: torch.Tensor) -> torch.dtype:
        return self._output

    def _scalar_forward(self, x: torch.Tensor) -> torch.Tensor:
        """One element: whether it is nonzero."""
        return (x != 0).to(self._output)

    def _call(self, x: torch.Tensor, axes: "tuple[int, ...]", n: int) -> LogicalReduceCall:
        """The facts that pick a logical reduction implementation and build it."""
        return LogicalReduceCall(
            device=x.device,
            shape=tuple(x.shape),
            axes=axes,
            op_kind=self._op_kind,
            dtype=x.dtype,
            keepdim=self.keepdim,
        )


class AllFwdOp(_LogicalReduceOpBase):
    """Whether every element along ``dim`` is nonzero, following ``torch.all``; returns bool.

    An empty ``dim`` reduces nothing: the output is ``x != 0`` with the input's shape.
    """

    _op_kind = "all"
    _empty = "noop"
    _identity = True


class AnyFwdOp(_LogicalReduceOpBase):
    """Whether any element along ``dim`` is nonzero, following ``torch.any``; returns bool.

    An empty ``dim`` reduces nothing: the output is ``x != 0`` with the input's shape.
    """

    _op_kind = "any"
    _empty = "noop"
    _identity = False


class CountNonzeroFwdOp(_LogicalReduceOpBase):
    """Count of nonzero elements along ``dim``, following ``torch.count_nonzero``; returns int64.

    There is no ``keepdim``: a reduced axis always goes, as in torch.
    """

    _op_kind = "count_nonzero"
    _output = torch.int64
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "logical_reduce_edge_fused": LogicalReduceEdgeFusedKernel,
        "logical_reduce_edge_two_pass": CountNonzeroEdgeTwoPassKernel,
        "logical_reduce": LogicalReduceKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"reduce": CountNonzeroFwdInterface}

    def __init__(
        self,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(dim, False, target=target)
