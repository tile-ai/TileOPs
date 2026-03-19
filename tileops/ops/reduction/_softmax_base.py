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

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        dtype: Data type (float32, float16, or bfloat16).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind: str  # set by subclass
    _kernel_key: str  # set by subclass
    _kernel_class: type  # set by subclass

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.M = M
        self.N = N
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._kernel_key](
            M,
            N,
            self._op_kind,
            dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_class}

    def _validate(self, x: torch.Tensor) -> None:
        """Validate input tensor."""
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.shape[-1] != self.N:
            raise ValueError(f"Expected hidden dim {self.N}, got {x.shape[-1]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the softmax-family op.

        Accepts 1D-4D input. Operates along dim=-1.

        Args:
            x: Input tensor with last dim == N.

        Returns:
            Output tensor. Same shape as input for softmax/log_softmax,
            or input shape without last dim for logsumexp.
        """
        self._validate(x)
        orig_shape = x.shape

        # Flatten leading dims: (..., N) -> (M, N)
        x = x.contiguous().reshape(-1, self.N)
        M_actual = x.shape[0]
        if M_actual != self.M:
            raise ValueError(f"Expected M={self.M} (product of leading dims), got {M_actual}")

        # Pad hidden dim to alignment
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N), value=float("-inf"))

        y = self.kernel(x)

        return self._reshape_output(y, orig_shape)

    def _reshape_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
    ) -> torch.Tensor:
        """Trim padding and restore original leading dims."""
        if self.N_padded != self.N:
            y = y[:, : self.N]
        return y.reshape(orig_shape)
