"""InstanceNorm forward operator.

InstanceNorm is a special case of GroupNorm where G = C (each channel
is its own group). This operator delegates to GroupNormKernel with G=C.

User-facing API mirrors torch.nn.functional.instance_norm:

    op = InstanceNormOp(N=batch, C=channels, spatial=(H, W), dtype=dtype)
    y = op(x, weight, bias)

Input tensors accept shape (N, C, *spatial).
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.norm import GroupNormKernel

from ..op import Op

__all__ = ["InstanceNormOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class InstanceNormOp(Op):
    """InstanceNorm forward operator.

    y = (x - mean) / sqrt(var + eps) * weight + bias

    where mean and var are computed over *spatial dimensions for each
    (batch, channel) independently. Equivalent to GroupNorm with G=C.

    Args:
        N: Batch size.
        C: Number of channels.
        spatial: Spatial dimensions tuple (H, W, ...).
        dtype: Data type (float32, float16, or bfloat16).
        eps: Epsilon for numerical stability (default 1e-5).
        kernel_map: Optional kernel override dict.
        tune: If True, autotune tile configs.
    """

    def __init__(
        self,
        N: int,
        C: int,
        spatial: tuple,
        dtype: torch.dtype,
        eps: float = 1e-5,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.G = C  # InstanceNorm: each channel is its own group
        self.dtype = dtype
        self.eps = eps
        self.spatial_size = math.prod(spatial)
        # For InstanceNorm (G=C): D = (C/C) * spatial_size = spatial_size
        self.D = self.spatial_size
        self.M = N * C  # number of rows = N * G = N * C
        self.D_padded = _align_up(self.D, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["group_norm"](
            self.M, self.D, eps, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"group_norm": GroupNormKernel}

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Run InstanceNorm forward.

        Args:
            x: Input tensor of shape (N, C, *spatial).
            weight: Affine scale of shape (C,).
            bias: Affine shift of shape (C,).

        Returns:
            Normalized output of the same shape as x.
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if not bias.is_cuda:
            raise ValueError("bias must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(
                f"Expected x.dtype {self.dtype}, got {x.dtype}"
            )
        if weight.dtype != self.dtype:
            raise ValueError(
                f"Expected weight.dtype {self.dtype}, got {weight.dtype}"
            )
        if bias.dtype != self.dtype:
            raise ValueError(
                f"Expected bias.dtype {self.dtype}, got {bias.dtype}"
            )
        if weight.ndim != 1 or weight.shape[0] != self.C:
            raise ValueError(
                f"Expected weight shape ({self.C},), got {weight.shape}"
            )
        if bias.ndim != 1 or bias.shape[0] != self.C:
            raise ValueError(
                f"Expected bias shape ({self.C},), got {bias.shape}"
            )

        orig_shape = x.shape
        x = x.contiguous()

        # Reshape: (N, C, *spatial) -> (N*C, spatial_size)
        x_2d = x.reshape(self.M, self.D)

        # Unit weight and zero bias for the kernel (affine applied after)
        unit_weight = torch.ones(self.D, dtype=self.dtype, device=x.device)
        zero_bias = torch.zeros(self.D, dtype=self.dtype, device=x.device)

        # Pad to alignment
        if self.D_padded != self.D:
            x_2d = F.pad(x_2d, (0, self.D_padded - self.D))
            unit_weight = F.pad(unit_weight, (0, self.D_padded - self.D))
            zero_bias = F.pad(zero_bias, (0, self.D_padded - self.D))

        # Run kernel: produces (x - mean) / sqrt(var + eps)
        y_2d = self.kernel(x_2d, unit_weight, zero_bias)

        # Trim padding
        if self.D_padded != self.D:
            y_2d = y_2d[:, :self.D]

        # Reshape back: (N*C, spatial_size) -> (N, C, *spatial)
        y = y_2d.reshape(orig_shape)

        # Apply per-channel affine: y = y * weight + bias
        affine_shape = [1, self.C] + [1] * len(self.spatial)
        y = y * weight.reshape(affine_shape) + bias.reshape(affine_shape)

        return y
