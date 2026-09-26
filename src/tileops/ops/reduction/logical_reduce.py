"""Logical reduction operators (all, any, count_nonzero)."""

from typing import ClassVar, Dict, List, Mapping, Optional, Tuple, Union

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.reduction._primitives import (
    device_smem_budget,
    edge_axis_plan,
    edge_axis_split,
)
from tileops.kernels.reduction.call_spec import LogicalReduceCall
from tileops.kernels.reduction.logical_reduce import (
    LogicalReduceEdgeFusedKernel,
    LogicalReduceKernel,
    storage_dtype_for,
)

from ..op_base import Op
from .reduce import _ReduceOpBase

__all__ = ["AllFwdOp", "AnyFwdOp", "CountNonzeroFwdOp"]


class _LogicalReduceOpBase(_ReduceOpBase):
    """Shared dispatch for logical reductions.

    Every numeric dtype is accepted, bool, int32, int64 and complex included. A dtype
    TileLang cannot store as shared memory is converted inside the kernel, so the op hands
    over the tensor its manifest declares.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "logical_reduce_edge_fused": LogicalReduceEdgeFusedKernel,
        "logical_reduce": LogicalReduceKernel,
    }
    _output: ClassVar[torch.dtype] = torch.bool

    def _output_dtype(self, x: torch.Tensor) -> torch.dtype:
        return self._output

    def _scalar_forward(self, x: torch.Tensor) -> torch.Tensor:
        """One element: whether it is nonzero."""
        return (x != 0).to(self._output)

    def _call(self, x: torch.Tensor, axes: "tuple[int, ...]", m: int, n: int) -> LogicalReduceCall:
        """The facts that pick a logical reduction implementation and build it."""
        device_index = x.device.index
        k, j = edge_axis_split(x.ndim, axes)
        kept = 0
        trail_needs_tiling = False
        if k:
            kernel_dtype = storage_dtype_for(x.dtype)
            elem_bytes = torch.tensor([], dtype=kernel_dtype).element_size()
            smem_budget = device_smem_budget(device_index)
            _, kept, _, planner, _ = edge_axis_plan(tuple(x.shape), k, j, elem_bytes, smem_budget)
            trail_needs_tiling = planner.needs_tiling
        return LogicalReduceCall(
            device=x.device,
            shape=tuple(x.shape),
            axes=axes,
            op_kind=self._op_kind,
            dtype=x.dtype,
            keepdim=self.keepdim,
            edge_axes=bool(k),
            kept=kept,
            trail_needs_tiling=trail_needs_tiling,
            reduced_count=n,
            m=m,
            tune=self.tune,
        )

    def entry_for(self, role: str, call: LogicalReduceCall) -> Entry:
        """Two implementations, so the one that serves the call says how it is built."""
        return Op.entry_for(self, role, call)


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

    def __init__(
        self,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel map.
            tune: Whether to autotune the kernel.
        """
        super().__init__(dim, False, target=target, kernel_map=kernel_map, tune=tune)
