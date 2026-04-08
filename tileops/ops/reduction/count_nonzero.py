"""CountNonzeroOp: counts non-zero elements along ``dim``, returning int64.

The Op layer validates inputs, normalizes ``dim``, reshapes to 2D (M, N),
pads to alignment (with 0, which is neutral for sum/count), calls the kernel,
and reshapes the output back.  Output dtype is always int64.

Supports any numeric dtype as input including torch.bool, int32, int64, and
complex types. Inputs with unsupported TileLang storage dtypes (bool, int32,
int64, complex64, complex128) are pre-converted to float32 before the kernel
call.

Kernels are cached by ``(M, N)`` so that the same op instance can handle
varying shapes.

Note: Unlike AllOp/AnyOp, CountNonzeroOp does NOT accept ``keepdim``.
The reduction dimension is always removed, matching ``torch.count_nonzero``.
"""

from math import prod
from typing import Dict, List, Optional, Union

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
from ._multidim import flatten_for_multidim, normalize_dim, restore_multidim_shape

__all__ = ["CountNonzeroOp"]


class CountNonzeroOp(Op):
    """Count nonzero reduction along ``dim``, returning int64.

    Construction: ``CountNonzeroOp(dtype=..., dim=-1)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    Padded positions use 0, which is neutral for sum/count.

    Note: No ``keepdim`` parameter -- the reduction dimension is always
    removed, matching ``torch.count_nonzero`` semantics.

    Supports any numeric dtype including torch.bool, int32, int64, and complex
    types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
    complex64, complex128) are pre-converted to float32 in forward().

    Args:
        dtype: Input data type (float16, bfloat16, float32, int32, int64,
               bool, complex64, complex128).
        dim: Reduction dimension (default -1).  Accepts ``int`` or
            ``list[int]`` for multi-dim reduction.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], None] = -1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.dtype = dtype
        self.dim = dim
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
                M, N, "count_nonzero", self.dtype, tune=self._tune,
            )
        return self._kernel_cache[key]

    def _reduce_2d(self, x: torch.Tensor, M: int, N: int) -> torch.Tensor:
        """Run kernel on 2D (M, N) tensor: dtype-convert, pad, call kernel."""
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_float32(x)
        kernel = self._get_or_create_kernel(M, N)
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        if N_padded != N:
            x = F.pad(x, (0, N_padded - N), value=0.0)
        return kernel(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute count_nonzero along the configured dim."""
        # --- validation ---
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

        orig_shape = x.shape

        # --- multi-dim path (includes dim=None for full reduction) ---
        if isinstance(self.dim, (list, tuple)) or self.dim is None:
            dims = normalize_dim(self.dim, x.ndim)
            x, orig_shape, _kept = flatten_for_multidim(x, dims)
            N = x.shape[-1]
            M = prod(x.shape[:-1])
            x = x.reshape(M, N)
            y = self._reduce_2d(x, M, N)
            # count_nonzero has no keepdim
            return restore_multidim_shape(y, orig_shape, dims, keepdim=False)

        # --- single-dim path ---
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
        y = self._reduce_2d(x, M, N)

        # --- reshape output (no keepdim for count_nonzero) ---
        reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
        y = y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)

        return y
