"""AnyOp: returns bool indicating if any element is non-zero along dim=-1.

The Op layer validates inputs, reshapes to 2D (M_flat, N), pads to alignment
(with 0, which is neutral for OR/any), calls the kernel, and reshapes the
output back. Output dtype is always bool.

Supports any numeric dtype as input including torch.bool (bool inputs are
cast to float32 internally before the TileLang kernel call).
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.logical_reduce import LogicalReduceKernel

from ..op import Op

__all__ = ["AnyOp"]


class AnyOp(Op):
    """Any reduction along dim=-1, returning bool.

    Follows the validate -> reshape -> pad -> kernel -> reshape pattern.
    Padded positions use 0 (False), which is neutral for OR/any.

    Supports any numeric dtype including torch.bool. Bool inputs are cast to
    float32 internally because TileLang does not support bool as a shared
    memory storage dtype.

    Args:
        M: Product of all leading dimensions.
        N: Last dimension size.
        dtype: Input data type (any dtype including torch.bool).
        kernel_map: Optional custom kernel map.
        tune: Whether to autotune the kernel.
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
        self.kernel = self.kernel_map["logical_reduce"](
            M,
            N,
            "any",
            dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"logical_reduce": LogicalReduceKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute any along dim=-1.

        Args:
            x: Input tensor with last dim == N.

        Returns:
            Bool tensor with shape == x.shape[:-1].
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if x.shape[-1] != self.N:
            raise ValueError(f"Expected last dim {self.N}, got {x.shape[-1]}")

        orig_shape = x.shape[:-1]  # output shape (leading dims)
        x = x.contiguous().reshape(-1, self.N)
        M_actual = x.shape[0]
        if M_actual != self.M:
            raise ValueError(f"Expected M={self.M} (product of leading dims), got {M_actual}")

        # Cast bool to float32: TileLang cannot handle bool as a shared-memory
        # storage dtype. The kernel is compiled for float32 in this case.
        if x.dtype == torch.bool:
            x = x.to(torch.float32)

        # Pad to alignment with 0 (False is neutral for OR/any)
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N), value=0.0)

        y = self.kernel(x)

        return y.reshape(orig_shape)
