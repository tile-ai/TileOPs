from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.norm import AdaLayerNormKernel

from ..op import Op

__all__ = ["AdaLayerNormOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class AdaLayerNormOp(Op):
    """Adaptive LayerNorm (AdaLN) operator.

    y = scale * LayerNorm(x) + shift

    scale and shift are per-token tensors of shape (M, N), pre-computed
    by the caller from a conditioning signal. Linear projection from
    conditioning input to scale/shift is the caller's responsibility.

    Supports arbitrary leading dimensions (3D+) via flatten/unflatten.
    Handles non-contiguous inputs and non-power-of-two hidden dims.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        dtype: Data type (float32, float16, or bfloat16).
        eps: Epsilon for numerical stability (default 1e-5).
    """

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        eps: float = 1e-5,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.M = M
        self.N = N
        self.dtype = dtype
        self.eps = eps
        self.N_padded = _align_up(N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["ada_layer_norm"](
            M, N, eps, dtype, has_gate=False, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ada_layer_norm": AdaLayerNormKernel}

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
    ) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if not scale.is_cuda:
            raise ValueError("scale must be a CUDA tensor")
        if not shift.is_cuda:
            raise ValueError("shift must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(
                f"Expected x.dtype {self.dtype}, got {x.dtype}"
            )
        if scale.dtype != self.dtype:
            raise ValueError(
                f"Expected scale.dtype {self.dtype}, got {scale.dtype}"
            )
        if shift.dtype != self.dtype:
            raise ValueError(
                f"Expected shift.dtype {self.dtype}, got {shift.dtype}"
            )
        if x.shape[-1] != self.N:
            raise ValueError(
                f"Expected hidden dim {self.N}, got {x.shape[-1]}"
            )

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, self.N)
        scale = scale.contiguous().reshape(-1, self.N)
        shift = shift.contiguous().reshape(-1, self.N)
        M_actual = x.shape[0]
        if M_actual != self.M:
            raise ValueError(
                f"Expected M={self.M} (product of leading dims), got {M_actual}"
            )

        # Pad hidden dim to 256-element alignment if needed
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))
            scale = F.pad(scale, (0, self.N_padded - self.N))
            shift = F.pad(shift, (0, self.N_padded - self.N))

        y = self.kernel(x, scale, shift)

        # Trim padding
        if self.N_padded != self.N:
            y = y[:, :self.N]

        return y.reshape(orig_shape)
