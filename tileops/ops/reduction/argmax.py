"""Argmax op: returns int64 indices of the maximum along a given dim.

The Op layer validates inputs, reshapes to 2D (M, N), pads to alignment,
calls the kernel, and reshapes the output back. Output dtype is always int64.
Kernels are cached by ``(M, N)`` so the same op instance handles varying shapes.
"""

from math import prod
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.argreduce import ArgreduceKernel

from ..op import Op

__all__ = ["ArgmaxFwdOp"]


class ArgmaxFwdOp(Op):
    """Argmax reduction along an arbitrary dim, returning int64 indices.

    Construction: ``ArgmaxFwdOp(dtype=..., dim=-1, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Args:
        dtype: Input data type.
        dim: Reduction dimension (default -1).
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: int = -1,
        keepdim: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.dtype = dtype
        self.dim = dim
        self.keepdim = keepdim
        self._tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"argreduce": ArgreduceKernel}

    # ------------------------------------------------------------------
    # Kernel cache
    # ------------------------------------------------------------------

    def _get_or_create_kernel(self, M: int, N: int) -> object:
        """Return a cached kernel for (M, N), creating one if needed."""
        key = (M, N)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map["argreduce"]
            self._kernel_cache[key] = kernel_cls(
                M, N, "argmax", self.dtype, tune=self._tune
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute argmax along the configured dim.

        Args:
            x: Input CUDA tensor of the configured dtype.

        Returns:
            Int64 tensor of indices.
        """
        # --- validation ---
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

        orig_shape = x.shape

        # Validate and normalize dim.
        if self.dim < -x.ndim or self.dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {self.dim})"
            )
        dim = self.dim % x.ndim

        # N = size along reduction dim, M = product of all other dims.
        N = x.shape[dim]
        M = prod(s for i, s in enumerate(x.shape) if i != dim)

        # If reduction dim is not the last, move it to the end.
        if dim != x.ndim - 1:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)

        # Get or create cached kernel for this (M, N).
        kernel = self._get_or_create_kernel(M, N)

        # Pad to alignment with -inf so padded positions never win argmax.
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        if N_padded != N:
            x = F.pad(x, (0, N_padded - N), value=float("-inf"))

        y = kernel(x)

        # --- reshape output ---
        if self.keepdim:
            kept_shape = list(orig_shape)
            kept_shape[dim] = 1
            y = y.reshape(kept_shape)
        else:
            reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
            y = y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)

        return y
