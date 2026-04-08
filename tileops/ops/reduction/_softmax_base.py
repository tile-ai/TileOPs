"""Base class for softmax-family operators (L2 Op layer).

Provides the shared validate -> reshape -> pad -> kernel -> trim -> reshape
pattern for softmax, log_softmax, and logsumexp ops.

Construction: ``op(dtype=..., dim=-1)``.  M and N are derived
from the input tensor at forward time, and kernels are cached by
``(M, N)`` to avoid rebuilds.
"""

from math import prod
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up

from ..op import Op
from ._multidim import flatten_for_multidim, normalize_dim, restore_multidim_shape

__all__ = ["_SoftmaxBaseOp"]


class _SoftmaxBaseOp(Op):
    """Base class for softmax-family ops.

    Handles shared validation, reshape, pad/trim logic. Subclasses only
    need to set ``_op_kind``, ``_kernel_key``, ``_kernel_class`` and
    override output reshaping if needed.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (default -1).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind: str  # set by subclass
    _kernel_key: str  # set by subclass
    _kernel_class: type  # set by subclass
    _supports_multidim: bool = False  # override to True in reduced-dim ops (e.g. LogSumExpOp)

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: Union[int, List[int], None] = -1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if not self._supports_multidim and not isinstance(dim, int):
            raise TypeError(
                f"{self.__class__.__name__} expects 'dim' to be an int; "
                f"got {type(dim).__name__}."
            )
        self.dtype = dtype
        self.dim = dim
        self.keepdim = False
        self._tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, object] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_class}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, x: torch.Tensor) -> None:
        """Validate input tensor."""
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the softmax-family op.

        Accepts arbitrary-dim input along the configured dim.
        Supports ``dim=list[int]`` for multi-dim reduction (logsumexp).
        """
        self._validate(x)
        orig_shape = x.shape

        # --- multi-dim path (includes dim=None for full reduction) ---
        if isinstance(self.dim, (list, tuple)) or self.dim is None:
            if not self._supports_multidim:
                raise ValueError(
                    f"{type(self).__name__} does not support multi-dim reduction. "
                    "Use a scalar dim."
                )
            dims = normalize_dim(self.dim, x.ndim)
            x, orig_shape, _kept = flatten_for_multidim(x, dims)
            N = x.shape[-1]
            M = prod(x.shape[:-1])
            x = x.reshape(M, N)
            kernel = self._get_or_create_kernel(M, N)
            N_padded = align_up(N, DEFAULT_ALIGNMENT)
            if N_padded != N:
                x = F.pad(x, (0, N_padded - N), value=float("-inf"))
            y = kernel(x)
            if N_padded != N:
                y = y[:, :N] if y.ndim == 2 else y
            return restore_multidim_shape(y, orig_shape, dims, self.keepdim)

        # --- single-dim path ---
        # Validate and normalize dim (match PyTorch IndexError behavior).
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
        needs_transpose = dim != x.ndim - 1
        if needs_transpose:
            x = x.movedim(dim, -1)

        x = x.contiguous().reshape(M, N)

        # Get or create cached kernel for this (M, N).
        kernel = self._get_or_create_kernel(M, N)

        # Pad hidden dim to alignment.
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        if N_padded != N:
            x = F.pad(x, (0, N_padded - N), value=float("-inf"))

        y = kernel(x)

        # Trim padding.
        if N_padded != N:
            y = y[:, :N] if y.ndim == 2 else y

        return self._reshape_output(y, orig_shape, dim, needs_transpose)

    def _get_or_create_kernel(self, M: int, N: int) -> object:
        """Return a cached kernel for (M, N), creating one if needed."""
        key = (M, N)
        if key not in self._kernel_cache:
            kernel_cls = self.kernel_map[self._kernel_key]
            self._kernel_cache[key] = kernel_cls(
                M, N, self._op_kind, self.dtype, tune=self._tune
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    # Output reshaping
    # ------------------------------------------------------------------

    def _reshape_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
        dim: int,
        needs_transpose: bool,
    ) -> torch.Tensor:
        """Restore original shape.

        Default (softmax/log_softmax): output same shape as input.
        Reduced-dim ops (logsumexp): remove or keep dim based on keepdim.
        """
        # y is (M, N) or (M,) depending on op kind.
        if y.ndim == 2:
            # Same-shape ops: rebuild the transposed shape, then move dim back.
            if needs_transpose:
                transposed_shape = list(orig_shape)
                transposed_shape.append(transposed_shape.pop(dim))
                y = y.reshape(transposed_shape)
                y = y.movedim(-1, dim)
            else:
                y = y.reshape(orig_shape)
        else:
            # Reduced-dim ops (logsumexp): (M,) -> remove or keep dim.
            if self.keepdim:
                kept_shape = list(orig_shape)
                kept_shape[dim] = 1
                y = y.reshape(kept_shape)
            else:
                reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
                y = y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)

        return y
