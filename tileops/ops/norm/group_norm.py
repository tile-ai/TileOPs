"""GroupNorm Op.

Wraps GroupNormKernel in a standard TileOPs Op interface.

User-facing API mirrors torch.nn.functional.group_norm:

    op = GroupNormOp(N=batch, C=channels, spatial=(H, W), G=groups, dtype=dtype)
    y = op(x, weight, bias)

Input tensors accept shape (N, C, *spatial). The op reshapes to
(N*G, C/G * spatial_size) for the kernel and restores the original shape
on output.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel import Kernel
from tileops.kernels.norm.group_norm import GroupNormKernel
from tileops.kernels.norm.group_norm.fwd import ALIGNMENT, SMEM_ROW_LIMIT, _align_up

from ..op import Op

__all__ = ["GroupNormOp"]


class GroupNormOp(Op):
    """GroupNorm forward operator.

    y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

    where normalization is computed per group over (C/G, *spatial).

    Args:
        N: Batch size.
        C: Number of channels.
        spatial: Spatial dimensions as a tuple (H, W, ...).
        G: Number of groups (must divide C evenly).
        dtype: Data type (float32, float16, or bfloat16).
        eps: Epsilon for numerical stability (default 1e-5).
        tune: If True, autotune tile config.
        kernel_map: Optional kernel override dict.
    """

    def __init__(
        self,
        N: int,
        C: int,
        spatial: Tuple[int, ...],
        G: int,
        dtype: torch.dtype,
        eps: float = 1e-5,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.G = G
        self.dtype = dtype
        self.eps = eps

        if C % G != 0:
            raise ValueError(f"C={C} must be divisible by G={G}")

        self.channels_per_group = C // G
        self.spatial_size = math.prod(spatial) if spatial else 1
        self.N_row = self.channels_per_group * self.spatial_size
        self.M = N * G

        self.N_padded = _align_up(self.N_row, ALIGNMENT)
        self.use_rowwise = self.N_padded <= SMEM_ROW_LIMIT

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["group_norm"](
            self.M, self.N_row, eps, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"group_norm": GroupNormKernel}

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (N, C, *spatial) -> (N*G, N_row)."""
        # (N, C, *spatial) -> (N, G, C/G, *spatial) -> (N*G, C/G * spatial_size)
        x = x.contiguous()
        new_shape = (self.N, self.G, self.channels_per_group, *self.spatial)
        x = x.reshape(new_shape)
        # Flatten last dims: (N, G, C/G * spatial_size)
        x = x.reshape(self.N, self.G, self.N_row)
        # -> (N*G, N_row)
        x = x.reshape(self.M, self.N_row)
        return x

    def _expand_param(self, param: torch.Tensor) -> torch.Tensor:
        """Expand a per-channel parameter (C,) to kernel row layout (M, N_row).

        Reshapes (C,) -> (G, C/G) -> repeat spatial -> (N*G, N_row).
        """
        t = param.reshape(self.G, self.channels_per_group)
        if self.spatial_size > 1:
            t = t.unsqueeze(-1).expand(self.G, self.channels_per_group, self.spatial_size)
            t = t.reshape(self.G, self.N_row)
        t = t.unsqueeze(0).expand(self.N, self.G, self.N_row).reshape(self.M, self.N_row)
        return t.contiguous()

    def _expand_weight_bias(
        self, weight: torch.Tensor, bias: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand per-channel weight/bias to match kernel row layout.

        weight, bias: (C,) -> (M, N_row)
        """
        return self._expand_param(weight), self._expand_param(bias)

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if not bias.is_cuda:
            raise ValueError("bias must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if weight.dtype != self.dtype:
            raise ValueError(f"Expected weight.dtype {self.dtype}, got {weight.dtype}")
        if bias.dtype != self.dtype:
            raise ValueError(f"Expected bias.dtype {self.dtype}, got {bias.dtype}")

        expected_shape = (self.N, self.C, *self.spatial)
        if x.shape != expected_shape:
            raise ValueError(f"Expected x.shape {expected_shape}, got {x.shape}")
        if weight.shape != (self.C,):
            raise ValueError(f"Expected weight.shape ({self.C},), got {weight.shape}")
        if bias.shape != (self.C,):
            raise ValueError(f"Expected bias.shape ({self.C},), got {bias.shape}")

        orig_shape = x.shape

        # Reshape input to (M, N_row)
        x_flat = self._reshape_input(x)

        # Expand weight/bias to (M, N_row)
        w_flat, b_flat = self._expand_weight_bias(weight, bias)

        if self.use_rowwise:
            # Pad to N_padded for the rowwise kernel
            if self.N_padded != self.N_row:
                x_flat = F.pad(x_flat, (0, self.N_padded - self.N_row))
                w_flat = F.pad(w_flat, (0, self.N_padded - self.N_row))
                b_flat = F.pad(b_flat, (0, self.N_padded - self.N_row))

            y = self.kernel(x_flat, w_flat, b_flat)

            # Trim padding
            if self.N_padded != self.N_row:
                y = y[:, :self.N_row]
        else:
            y = self.kernel(x_flat, w_flat, b_flat)

        # Reshape back to (N, C, *spatial)
        y = y.reshape(self.N, self.G, self.channels_per_group, *self.spatial)
        y = y.reshape(orig_shape)

        return y
