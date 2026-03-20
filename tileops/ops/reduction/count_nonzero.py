"""CountNonzeroOp: counts non-zero elements along dim=-1, returning int64.

The Op layer validates inputs, reshapes to 2D (M_flat, N), pads to alignment
(with 0, which is neutral for sum/count), calls the kernel, and reshapes the
output back. Output dtype is always int64.

Supports any numeric dtype as input including torch.bool, int32, int64, and
complex types. Inputs with unsupported TileLang storage dtypes (bool, int32,
int64, complex64, complex128) are pre-converted to float32 before the kernel
call.
"""

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

__all__ = ["CountNonzeroOp"]


class CountNonzeroOp(Op):
    """Count nonzero reduction along dim=-1, returning int64.

    Follows the validate -> reshape -> pad -> kernel -> reshape pattern.
    Padded positions use 0, which is neutral for sum/count.

    Supports any numeric dtype including torch.bool, int32, int64, and complex
    types. Inputs with unsupported TileLang storage dtypes (bool, int32, int64,
    complex64, complex128) are pre-converted to float32 in forward().

    Args:
        M: Product of all leading dimensions.
        N: Last dimension size.
        dtype: Input data type (float16, bfloat16, float32, int32, int64,
               bool, complex64, complex128).
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
            "count_nonzero",
            dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"logical_reduce": LogicalReduceKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute count_nonzero along dim=-1.

        Args:
            x: Input tensor with last dim == N.

        Returns:
            Int64 tensor with shape == x.shape[:-1].
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

        # Pre-convert unsupported storage dtypes (bool, int32, int64, complex)
        # to float32. TileLang cannot handle these as shared-memory storage
        # dtypes; the kernel is compiled for float32 in those cases.
        if x.dtype in _UNSUPPORTED_STORAGE_DTYPES:
            x = to_logical_float32(x)

        # Pad to alignment with 0 (zero is neutral for sum/count)
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N), value=0.0)

        y = self.kernel(x)

        return y.reshape(orig_shape)
