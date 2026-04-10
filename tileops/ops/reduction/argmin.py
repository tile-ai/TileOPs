"""Argmin op: returns int64 indices of the minimum along a given dim.

The Op layer validates inputs, reshapes to 2D (M, N), pads to alignment,
calls the kernel, and reshapes the output back. Output dtype is always int64.
Kernels are cached by ``(M, N)`` so the same op instance handles varying shapes.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction.argreduce import ArgreduceKernel

from .reduce import _ReduceOpBase

__all__ = ["ArgminFwdOp"]


class ArgminFwdOp(_ReduceOpBase):
    """Argmin reduction along an arbitrary dim, returning int64 indices.

    Construction: ``ArgminFwdOp(dtype=..., dim=-1, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dtype: Input data type.
        dim: Reduction dimension (default -1).
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    _op_kind = "argmin"
    _kernel_key = "argreduce"
    _kernel_cls = ArgreduceKernel

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: int = -1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        super().__init__(
            dtype=dtype, dim=dim, keepdim=keepdim,
            kernel_map=kernel_map, tune=tune,
        )

    def _validate_dim(self) -> None:
        """Argmin only supports single-dim reduction."""
        if not isinstance(self.dim, int):
            raise ValueError(
                f"ArgminFwdOp only supports scalar dim (int), "
                f"got {type(self.dim).__name__}: {self.dim!r}"
            )

    def _pad_value(self) -> float:
        """Pad with +inf so padded positions never win argmin."""
        return float("inf")
