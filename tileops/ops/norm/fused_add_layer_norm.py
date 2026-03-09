from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.norm import FusedAddLayerNormKernel

from ..op import Op

__all__ = ["FusedAddLayerNormOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class FusedAddLayerNormOp(Op):
    """Fused Add + LayerNorm operator.

    y = LayerNorm(x + residual)

    Returns dual outputs (y, x + residual):
    - y: the normalized result
    - x + residual: the pre-norm sum, needed by downstream residual connections

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
        self.kernel = self.kernel_map["fused_add_layer_norm"](
            M, N, eps, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fused_add_layer_norm": FusedAddLayerNormKernel}

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if not residual.is_cuda:
            raise ValueError("residual must be a CUDA tensor")
        if not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if not bias.is_cuda:
            raise ValueError("bias must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(
                f"Expected x.dtype {self.dtype}, got {x.dtype}"
            )
        if residual.dtype != self.dtype:
            raise ValueError(
                f"Expected residual.dtype {self.dtype}, got {residual.dtype}"
            )
        if weight.dtype != self.dtype:
            raise ValueError(
                f"Expected weight.dtype {self.dtype}, got {weight.dtype}"
            )
        if bias.dtype != self.dtype:
            raise ValueError(
                f"Expected bias.dtype {self.dtype}, got {bias.dtype}"
            )
        if weight.ndim != 1:
            raise ValueError(
                f"Expected weight to be 1D, got {weight.ndim}D"
            )
        if bias.ndim != 1:
            raise ValueError(
                f"Expected bias to be 1D, got {bias.ndim}D"
            )
        if x.shape[-1] != self.N:
            raise ValueError(
                f"Expected hidden dim {self.N}, got {x.shape[-1]}"
            )
        if residual.shape != x.shape:
            raise ValueError(
                f"Expected residual shape {x.shape}, got {residual.shape}"
            )
        if weight.shape[0] != self.N:
            raise ValueError(
                f"Expected weight dim {self.N}, got {weight.shape[0]}"
            )
        if bias.shape[0] != self.N:
            raise ValueError(
                f"Expected bias dim {self.N}, got {bias.shape[0]}"
            )

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, self.N)
        residual = residual.contiguous().reshape(-1, self.N)
        M_actual = x.shape[0]
        if M_actual != self.M:
            raise ValueError(
                f"Expected M={self.M} (product of leading dims), got {M_actual}"
            )

        # Pad hidden dim to 256-element alignment if needed
        if self.N_padded != self.N:
            x = F.pad(x, (0, self.N_padded - self.N))
            residual = F.pad(residual, (0, self.N_padded - self.N))
            weight = F.pad(weight, (0, self.N_padded - self.N))
            bias = F.pad(bias, (0, self.N_padded - self.N))

        outputs = self.kernel(x, residual, weight, bias)
        y, pre_norm = outputs[0], outputs[1]

        # Trim padding
        if self.N_padded != self.N:
            y = y[:, :self.N]
            pre_norm = pre_norm[:, :self.N]

        return y.reshape(orig_shape), pre_norm.reshape(orig_shape)
