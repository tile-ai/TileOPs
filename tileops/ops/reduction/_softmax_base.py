"""Base class for softmax-family operators (L2 Op layer).

Provides the shared validate -> reshape -> pad -> kernel -> trim -> reshape
pattern for softmax, log_softmax, and logsumexp ops.

Supports two construction paths:

- **Legacy path** (M, N provided): caller pre-computes M and N, kernel is
  built once at init time.  Used by LogSoftmaxOp / LogSumExpOp tests and
  benchmarks today.

- **Spec path** (dim provided, M/N omitted): matches the manifest signature
  ``op(dtype, dim=-1)``.  M and N are derived from the input tensor at
  forward time, and kernels are cached by ``(M, N)`` to avoid rebuilds.
"""

from math import prod
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up

from ..op import Op

__all__ = ["_SoftmaxBaseOp"]


class _SoftmaxBaseOp(Op):
    """Base class for softmax-family ops.

    Handles shared validation, reshape, pad/trim logic. Subclasses only
    need to set ``_op_kind``, ``_kernel_key``, ``_kernel_class`` and
    override output reshaping if needed.

    Args:
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction dimension (spec path).  Ignored when M/N are given.
        M: Number of rows (legacy path).
        N: Hidden dimension (legacy path).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind: str  # set by subclass
    _kernel_key: str  # set by subclass
    _kernel_class: type  # set by subclass

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        dim: int = -1,
        M: Optional[int] = None,
        N: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if M is not None and N is not None:
            # Legacy path: M, N precomputed by caller.
            self._legacy = True
            self.M = M
            self.N = N
            self.dim = -1  # legacy always reduces last dim
        else:
            # Spec path: dim provided, M/N computed at forward time.
            self._legacy = False
            self.dim = dim
            self.M: Optional[int] = None  # type: ignore[assignment]
            self.N: Optional[int] = None  # type: ignore[assignment]

        self.dtype = dtype

        # Kernel creation: immediate for legacy, deferred for spec.
        if self._legacy:
            self.N_padded = align_up(self.N, DEFAULT_ALIGNMENT)
            self.dispatch_kernel(kernel_map)
            self.kernel = self.kernel_map[self._kernel_key](
                M,
                N,
                self._op_kind,
                dtype,
                tune=tune,
            )
        else:
            self._kernel_map_override = kernel_map
            self._tune = tune
            self._kernel_cache: Dict[tuple, object] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_class}

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_legacy(self, x: torch.Tensor) -> None:
        """Validate input tensor for legacy path."""
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.shape[-1] != self.N:
            raise ValueError(f"Expected hidden dim {self.N}, got {x.shape[-1]}")

    def _validate_spec(self, x: torch.Tensor) -> None:
        """Validate input tensor for spec path."""
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the softmax-family op.

        Legacy path accepts 1D-4D input along dim=-1.
        Spec path accepts arbitrary-dim input along the configured dim.
        """
        if self._legacy:
            return self._forward_legacy(x)
        return self._forward_spec(x)

    def _forward_legacy(self, x: torch.Tensor) -> torch.Tensor:
        """Original forward — fixed M, N, dim=-1."""
        self._validate_legacy(x)
        orig_shape = x.shape

        # Flatten leading dims: (..., N) -> (M, N)
        x = x.contiguous().reshape(-1, self.N)
        M_actual = x.shape[0]
        if M_actual != self.M:
            raise ValueError(
                f"Expected M={self.M} (product of leading dims), got {M_actual}"
            )

        # Pad hidden dim to alignment
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N), value=float("-inf"))

        y = self.kernel(x)

        return self._reshape_output(y, orig_shape)

    def _forward_spec(self, x: torch.Tensor) -> torch.Tensor:
        """Spec forward — computes M, N from x.shape + self.dim."""
        self._validate_spec(x)
        orig_shape = x.shape

        # Normalize dim.
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

        return self._reshape_output_spec(y, orig_shape, dim, needs_transpose)

    def _get_or_create_kernel(self, M: int, N: int) -> object:
        """Return a cached kernel for (M, N), creating one if needed."""
        key = (M, N)
        if key not in self._kernel_cache:
            self.dispatch_kernel(self._kernel_map_override)
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
    ) -> torch.Tensor:
        """Trim padding and restore original leading dims (legacy path)."""
        if self.N_padded != self.N:
            y = y[:, : self.N]
        return y.reshape(orig_shape)

    def _reshape_output_spec(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
        dim: int,
        needs_transpose: bool,
    ) -> torch.Tensor:
        """Restore original shape for spec path.

        Default (softmax/log_softmax): output same shape as input.
        Subclasses (logsumexp) may override this.
        """
        # y is (M, N) or (M,) depending on op kind.
        if y.ndim == 2:
            # Same-shape ops: rebuild the transposed shape, then move dim back.
            if needs_transpose:
                # Transposed shape: move reduction dim from last to orig position.
                transposed_shape = list(orig_shape)
                transposed_shape.append(transposed_shape.pop(dim))
                y = y.reshape(transposed_shape)
                y = y.movedim(-1, dim)
            else:
                y = y.reshape(orig_shape)
        else:
            # Reduced-dim ops (logsumexp): (M,) -> remove dim from shape.
            reduced_shape = [s for i, s in enumerate(orig_shape) if i != dim]
            y = y.squeeze() if len(reduced_shape) == 0 else y.reshape(reduced_shape)

        return y
