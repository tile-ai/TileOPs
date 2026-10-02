"""The vector-norm reduction operator, one ``ord`` of ``torch.linalg.vector_norm`` per call."""

from math import inf
from typing import ClassVar, Dict, List, Mapping, Optional, Tuple, Union

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.reduction.call_spec import VectorNormFwdInterface
from tileops.kernels.reduction.reduce import ReduceFoldKernel
from tileops.kernels.reduction.vector_norm import (
    VectorNormEdgeKernel,
    VectorNormKernel,
)
from tileops.ops.reduction.reduce import ReduceCallOp

__all__ = ["VectorNormFwdOp"]


class VectorNormFwdOp(ReduceCallOp):
    """``torch.linalg.vector_norm(x, ord, dim, keepdim, dtype=dtype)``.

    ``ord`` selects the fold: ``1`` sums magnitudes, ``2`` takes the root of the sum of
    squares, and ``inf`` takes the largest magnitude. A row holding a NaN yields NaN at
    ``ord=inf``, as in torch.
    """

    _ORD_KINDS = {1: "l1", 2: "l2", inf: "inf"}
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "vector_norm_fold": ReduceFoldKernel,
        "vector_norm": VectorNormKernel,
        "vector_norm_edge": VectorNormEdgeKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"reduce": VectorNormFwdInterface}

    def __init__(
        self,
        ord: Union[int, float] = 2,
        dim: Union[int, List[int], Tuple[int, ...], None] = None,
        keepdim: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            ord: Norm order; 1, 2 or ``inf``.
            dim: Axes to reduce: an ``int``, a sequence of them, or ``None`` for all.
            keepdim: Whether a reduced axis stays as a length-1 axis.
            dtype: The dtype the input is cast to before the reduction, and the output's;
                it may not narrow the input's. ``None`` keeps the input's.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional custom kernel map.
            tune: Whether to autotune the kernel.
        """
        self.ord = ord
        self.dtype = dtype
        super().__init__(dim, keepdim, target=target, kernel_map=kernel_map, tune=tune)

    @property
    def _op_kind(self) -> str:
        """The fold the kernels are built for, named by ``ord``."""
        return self._ORD_KINDS[self.ord]

    def _scalar_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The norm of one element is its magnitude."""
        return x.abs()
