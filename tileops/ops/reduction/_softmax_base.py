"""Base class for softmax-family operators (L2 Op layer).

Provides the shared validate -> reshape -> pad -> kernel -> trim -> reshape
pattern for softmax, log_softmax, and logsumexp ops.
"""

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

    Supports two calling conventions:

    **Spec interface** (preferred)::

        op = SoftmaxOp(dim=-1, dtype=torch.float16)
        y = op(x)  # M, N derived dynamically from x.shape and dim

    **Legacy interface** (backward-compatible)::

        op = SoftmaxOp(M=1024, N=4096, dtype=torch.float16)
        y = op(x)  # M, N fixed at construction time

    Args:
        dim: Reduction dimension (default -1). Ignored when M/N provided.
        dtype: Data type (float32, float16, or bfloat16).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
        M: (Legacy) Number of rows. When provided with N, uses eager mode.
        N: (Legacy) Hidden dimension. When provided with M, uses eager mode.
    """

    _op_kind: str  # set by subclass
    _kernel_key: str  # set by subclass
    _kernel_class: type  # set by subclass

    def __init__(
        self,
        dim: int = -1,
        dtype: torch.dtype = torch.float16,
        *,
        M: Optional[int] = None,
        N: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.dtype = dtype
        self.tune = tune
        self.dispatch_kernel(kernel_map)

        if M is not None and N is not None:
            # Legacy eager mode: pre-create kernel at init time
            self._legacy = True
            self.dim = -1
            self.M = M
            self.N = N
            self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
            self.kernel = self.kernel_map[self._kernel_key](
                M,
                N,
                self._op_kind,
                dtype,
                tune=tune,
            )
        else:
            # Spec mode: dim-based, kernel created per-forward
            self._legacy = False
            self.dim = dim

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_class}

    def _validate(self, x: torch.Tensor) -> None:
        """Validate input tensor."""
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the softmax-family op.

        Accepts 1D-4D input. Operates along the configured dim.

        Args:
            x: Input tensor.

        Returns:
            Output tensor. Same shape as input for softmax/log_softmax,
            or input shape without the reduce dim for logsumexp.
        """
        self._validate(x)
        orig_shape = x.shape

        if self._legacy:
            return self._forward_legacy(x, orig_shape)
        return self._forward_spec(x, orig_shape)

    def _forward_legacy(self, x: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        """Legacy path: M/N fixed at init, dim always -1."""
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
        return self._reshape_output(y, orig_shape, dim=len(orig_shape) - 1)

    def _forward_spec(self, x: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        """Spec path: dim-based, M/N computed dynamically."""
        # Normalize dim
        dim = self.dim % x.ndim
        N = x.shape[dim]
        N_padded = align_up(N, DEFAULT_ALIGNMENT)

        # Move reduce dim to last position if needed
        x = x.transpose(dim, -1).contiguous() if dim != x.ndim - 1 else x.contiguous()

        # Flatten to 2D: (M, N)
        M = x.numel() // N
        x = x.reshape(M, N)

        # Pad hidden dim to alignment
        if N_padded != N:
            x = F.pad(x, (0, N_padded - N), value=float("-inf"))

        # Create kernel (underlying JIT compilation is lru_cached)
        kernel = self.kernel_map[self._kernel_key](
            M, N, self._op_kind, self.dtype, tune=self.tune
        )
        y = kernel(x)

        return self._reshape_output(
            y, orig_shape, dim=dim, N=N, N_padded=N_padded
        )

    def _reshape_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
        dim: int,
        N: Optional[int] = None,
        N_padded: Optional[int] = None,
    ) -> torch.Tensor:
        """Trim padding and restore original shape.

        For softmax/log_softmax the output has the same shape as the input.
        Subclasses (e.g. logsumexp) override for reduction semantics.
        """
        # Resolve N/N_padded for legacy path
        if N is None:
            N = self.N
        if N_padded is None:
            N_padded = self.N_padded

        # Trim padding
        if N_padded != N:
            y = y[:, :N]

        if dim == len(orig_shape) - 1:
            # Reduce dim was already last -- just restore leading dims
            return y.reshape(orig_shape)

        # Reduce dim was transposed to last; reconstruct the transposed shape
        # then transpose back.
        transposed_shape = list(orig_shape)
        transposed_shape[dim], transposed_shape[-1] = (
            transposed_shape[-1],
            transposed_shape[dim],
        )
        y = y.reshape(transposed_shape)
        y = y.transpose(dim, -1).contiguous()
        return y
