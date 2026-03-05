from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.norm import RmsNormKernel

from .op import Op

__all__ = ["RmsNormOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class RmsNormOp(Op):
    """Standalone RMS Norm operator.

    y = x * rsqrt(mean(x^2, dim=-1) + eps) * weight

    Supports arbitrary leading dimensions (3D+) via flatten/unflatten.
    Handles non-contiguous inputs and non-power-of-two hidden dims.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        dtype: Data type (float16 or bfloat16).
        eps: Epsilon for numerical stability (default 1e-6).
    """

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        eps: float = 1e-6,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.M = M
        self.N = N
        self.dtype = dtype
        self.eps = eps
        self.N_padded = _align_up(N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["rms_norm"](
            M, N, eps, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"rms_norm": RmsNormKernel}

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.N:
            raise ValueError(
                f"Expected hidden dim {self.N}, got {x.shape[-1]}"
            )
        if weight.shape[-1] != self.N:
            raise ValueError(
                f"Expected weight dim {self.N}, got {weight.shape[-1]}"
            )

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, self.N)

        # Pad hidden dim to 256-element alignment if needed
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))
            weight = F.pad(weight, (0, self.N_padded - self.N))

        y = self.kernel(x, weight)

        # Trim padding
        if self.N_padded != self.N:
            y = y[:, :self.N]

        return y.reshape(orig_shape)
