"""GroupNorm forward operator.

Wraps GroupNormKernel in the standard TileOPs Op interface.

User-facing API mirrors torch.nn.functional.group_norm:

    op = GroupNormOp(N=batch, C=channels, spatial=(H, W), G=groups, dtype=dtype)
    y = op(x, weight, bias)

Input tensors accept shape (N, C, *spatial); the op reshapes to
(N*G, D_padded) internally where D = (C/G) * spatial_size.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.norm import GroupNormKernel

from ..op import Op

__all__ = ["GroupNormOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class GroupNormOp(Op):
    """GroupNorm forward operator.

    y = (x - mean) / sqrt(var + eps) * weight + bias

    where mean and var are computed over (C/G, *spatial) for each group.

    Supports arbitrary spatial dimensions (1D, 2D, 3D+).
    Handles non-contiguous inputs via explicit contiguous() call.

    Args:
        N: Batch size.
        C: Number of channels.
        spatial: Spatial dimensions tuple (H, W, ...).
        G: Number of groups. Must divide C evenly.
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
        G: int,
        dtype: torch.dtype,
        eps: float = 1e-5,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if C % G != 0:
            raise ValueError(f"C={C} must be divisible by G={G}")
        self.N = N
        self.C = C
        self.spatial = spatial
        self.G = G
        self.dtype = dtype
        self.eps = eps
        self.spatial_size = math.prod(spatial)
        self.D = (C // G) * self.spatial_size  # row length before padding
        self.M = N * G  # number of rows
        self.D_padded = _align_up(self.D, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["group_norm"](
            self.M, self.D, eps, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"group_norm": GroupNormKernel}

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Run GroupNorm forward.

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
        # Ensure contiguous and reshape to (N, G, C/G, *spatial)
        x = x.contiguous()

        # Reshape: (N, C, *spatial) -> (N, G, C/G, *spatial) -> (N*G, (C/G)*spatial_size)
        cpg = self.C // self.G  # channels per group
        x_reshaped = x.reshape(self.N, self.G, cpg, *self.spatial)
        x_2d = x_reshaped.reshape(self.M, self.D)

        # The kernel broadcasts 1D weight/bias across all rows, but GroupNorm
        # needs per-group affine parameters. Run kernel with unit weight/zero bias
        # to normalize, then apply per-channel affine transform afterwards.
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

        # Reshape back: (N*G, D) -> (N, G, cpg, *spatial) -> (N, C, *spatial)
        y = y_2d.reshape(self.N, self.G, cpg, *self.spatial)
        y = y.reshape(orig_shape)

        # Apply per-channel affine: y = y * weight + bias
        # weight and bias are (C,), need to broadcast to (N, C, *spatial)
        affine_shape = [1, self.C] + [1] * len(self.spatial)
        y = y * weight.reshape(affine_shape) + bias.reshape(affine_shape)

        return y
