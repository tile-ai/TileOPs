"""AllOp: returns bool indicating if all elements are non-zero along ``dim``.

The Op layer validates inputs, normalizes ``dim``, reshapes to 2D (M, N),
pads to alignment (with 1, which is neutral for AND/all), calls the kernel,
and reshapes the output back.  Output dtype is always bool.

Supports any numeric dtype as input including torch.bool, int32, int64, and
complex types. Inputs with unsupported TileLang storage dtypes (bool, int32,
int64, complex64, complex128) are pre-converted to float32 before the kernel
call.

Kernels are cached by ``(M, N)`` so that the same op instance can handle
varying shapes.
"""

from math import prod
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.logical_reduce import LogicalReduceKernel
from tileops.kernels.reduction.logical_reduce.fwd import (
    _UNSUPPORTED_STORAGE_DTYPES,
    to_logical_float32,
)

from ..op import Op

__all__ = ["AllOp"]


class AllOp(Op):
    """All reduction along ``dim``, returning bool.

    Construction: ``AllOp(dtype=..., dim=-1, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Padded positions use 1 (True), which is neutral for AND/all.

    Supports any numeric dtype including torch.bool, int32, int64, and complex
    types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
    complex64, complex128) are pre-converted to float32 in forward().

    Args:
        dtype: Input data type (float16, bfloat16, float32, int32, int64,
               bool, complex64, complex128).
        dim: Reduction dimension (default -1).  Only a single ``int`` is
            supported; passing ``list[int]`` raises ``NotImplementedError``.
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
        if isinstance(dim, (list, tuple)):
            raise NotImplementedError("Multi-dim reduction not yet supported")
        self.dtype = dtype
        self.dim = dim
        self.keepdim = keepdim
        self._tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"logical_reduce": LogicalReduceKernel}

    def _get_or_create_kernel(self, M: int, N: int) -> object:
        """Return a cached kernel for (M, N), creating one if needed."""
        key = (M, N)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map["logical_reduce"]
            self._kernel_cache[key] = kernel_cls(
                M, N, "all", self.dtype, tune=self._tune,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all along the configured dim."""
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

        # Pre-convert unsupported storage dtypes (bool, int32, int64, complex)
        # to float32. TileLang cannot handle these as shared-memory storage
        # dtypes; the kernel is compiled for float32 in those cases.
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_float32(x)

        # Get or create cached kernel for this (M, N).
        kernel = self._get_or_create_kernel(M, N)

        # Pad to alignment with 1.0 (True is neutral for AND/all).
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        if N_padded != N:
            x = F.pad(x, (0, N_padded - N), value=1.0)

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
