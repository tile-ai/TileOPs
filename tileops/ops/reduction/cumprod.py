"""Cumulative product operator (L2 Op layer).

Provides:
  - CumprodFwdOp: y = cumprod(x, dim=-1)

Follows the validate -> reshape -> kernel -> trim -> reshape pattern
and supports 1D-4D input with dim=-1. Output has the same shape as input.
Alignment padding is handled inside the kernel via masked loads.
"""

from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.cumulative import CumulativeKernel

from ..op import Op

__all__ = ["CumprodFwdOp"]


class CumprodFwdOp(Op):
    """Cumulative product operator: y = cumprod(x, dim=-1).

    Output has the same shape and dtype as input.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        dtype: Data type (float32, float16, or bfloat16).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        >>> op = CumprodFwdOp(M=1024, N=4096, dtype=torch.float16)
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> y = op(x)  # shape: (1024, 4096)
    """

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
        self.kernel = self.kernel_map["cumulative_fwd"](
            M,
            N,
            "prod",
            dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"cumulative_fwd": CumulativeKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the cumulative product op.

        Accepts 1D-4D input. Operates along dim=-1.

        Args:
            x: Input tensor with last dim == N.

        Returns:
            Output tensor with the same shape as input.
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.shape[-1] != self.N:
            raise ValueError(f"Expected last dim {self.N}, got {x.shape[-1]}")

        orig_shape = x.shape

        # Flatten leading dims: (..., N) -> (M, N)
        x = x.contiguous().reshape(-1, self.N)
        M_actual = x.shape[0]
        if M_actual != self.M:
            raise ValueError(f"Expected M={self.M} (product of leading dims), got {M_actual}")

        # Alignment padding is handled inside the kernel via masked loads.
        y = self.kernel(x)

        # Trim padding (kernel output is N_padded-wide) and restore shape
        if self.N_padded != self.N:
            y = y[:, : self.N]
        return y.reshape(orig_shape)
