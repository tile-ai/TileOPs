"""InfNormOp: computes infinity norm (max absolute value) along dim=-1.

The Op layer validates inputs, reshapes to 2D (M_flat, N), pads to alignment
(with 0.0, which is neutral for max of absolute values), calls the kernel,
and reshapes the output back. Output dtype matches input dtype; internal
computation in fp32.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.vector_norm import VectorNormKernel

from ..op import Op

__all__ = ["InfNormOp"]


class InfNormOp(Op):
    """Infinity norm reduction along dim=-1.

    Follows the validate -> reshape -> pad -> kernel -> reshape pattern.
    Padded positions use 0.0 (neutral for max of absolute values).

    Args:
        M: Product of all leading dimensions.
        N: Last dimension size.
        dtype: Input data type (float16, bfloat16, float32).
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
        self.kernel = self.kernel_map["vector_norm"](
            M,
            N,
            "inf",
            dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"vector_norm": VectorNormKernel}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute infinity norm along dim=-1.

        Args:
            x: Input tensor with last dim == N.

        Returns:
            Tensor with shape == x.shape[:-1], same dtype as input.
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

        # Pad to alignment with 0.0 (neutral for max of absolute values)
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))

        y = self.kernel(x)

        return y.reshape(orig_shape)
