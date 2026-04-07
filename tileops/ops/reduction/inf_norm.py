"""InfNormOp: computes infinity norm (max absolute value) along a given dim.

The Op layer validates inputs, reshapes to 2D (M, N), pads to alignment
(with 0.0, which is neutral for max of absolute values), calls the kernel,
and reshapes the output back. Output dtype matches input dtype; internal
computation in fp32.

NaN propagation: T.reduce_max in TileLang does not propagate NaN (it drops
NaN values). To match torch.linalg.vector_norm(ord=inf) semantics, the Op
layer detects rows containing NaN before the kernel call and patches the
output to NaN for those rows.
"""

from math import prod
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.vector_norm import VectorNormKernel

from ..op import Op
from ._multidim import flatten_for_multidim, normalize_dim, restore_multidim_shape

__all__ = ["InfNormOp"]


class InfNormOp(Op):
    """Infinity norm reduction along a configurable dim.

    Construction: ``InfNormOp(dtype=..., dim=-1, keepdim=False)``.  M and N are
    derived from the input tensor at forward time, and kernels are cached
    by ``(M, N)`` to avoid rebuilds.

    NaN handling: rows containing any NaN produce NaN output, matching
    torch.linalg.vector_norm(ord=inf) semantics.

    Args:
        dtype: Input data type (float16, bfloat16, float32).
        dim: Reduction dimension (default -1).  Accepts ``int`` or
            ``list[int]`` for multi-dim reduction.
        keepdim: Whether to retain the reduced dimension as size 1.
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], None] = -1,
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
        return {"vector_norm": VectorNormKernel}

    def _get_or_create_kernel(self, M: int, N: int) -> object:
        """Return a cached kernel for (M, N), creating one if needed."""
        key = (M, N)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map["vector_norm"]
            self._kernel_cache[key] = kernel_cls(
                M, N, "inf", self.dtype, tune=self._tune
            )
        return self._kernel_cache[key]

    def _reduce_2d(self, x: torch.Tensor, M: int, N: int) -> torch.Tensor:
        """Run kernel on 2D (M, N) tensor: NaN detect, pad, call kernel, NaN patch."""
        nan_mask = x.isnan().any(dim=-1)  # shape (M,)
        kernel = self._get_or_create_kernel(M, N)
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        if N_padded != N:
            x = F.pad(x, (0, N_padded - N))
        y = kernel(x)
        if nan_mask.any():
            y[nan_mask] = float("nan")
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute infinity norm along the configured dim."""
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
            return restore_multidim_shape(y, orig_shape, dims, self.keepdim)

        # --- single-dim path ---
        if self.dim < -x.ndim or self.dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {self.dim})"
            )
        dim = self.dim % x.ndim

        N = x.shape[dim]
        M = prod(s for i, s in enumerate(x.shape) if i != dim)

        if dim != x.ndim - 1:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)
        y = self._reduce_2d(x, M, N)

        if self.keepdim:
            kept_shape = list(orig_shape)
            kept_shape[dim] = 1
            y = y.reshape(kept_shape)
        else:
            reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
            y = y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)

        return y
