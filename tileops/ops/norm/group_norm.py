"""GroupNorm forward operator.

Wraps GroupNormKernel in the standard TileOPs Op interface.

User-facing API mirrors torch.nn.functional.group_norm:

    op = GroupNormFwdOp(
        N=batch, C=channels, spatial=(H, W), num_groups=groups, dtype=dtype,
    )
    y = op(x, weight, bias)

Input tensors accept shape (N, C, *spatial); the op reshapes to
(N*num_groups, D_padded) internally where
D = (C/num_groups) * spatial_size.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import GroupNormKernel

from ..op_base import Op
from .norm_base import ALIGNMENT, align_up

__all__ = ["GroupNormFwdOp"]


class GroupNormFwdOp(Op):
    """Group Normalization forward operator.

    Computes group normalization over ``(C/num_groups, *spatial)`` slices:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot w + b

    where the mean and variance are computed per group over
    ``(C/num_groups, *spatial)`` elements.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary spatial dimensions (1-D, 2-D, 3-D+).
        Handles non-contiguous inputs via explicit ``contiguous()`` call.
        Hidden dimension is padded to 256-element alignment internally.

    Args:
        N: Batch size.
        C: Number of channels.
        spatial: Spatial dimensions tuple ``(H, W, ...)``.
        num_groups: Number of groups (manifest ``params.num_groups``).
            Must divide *C* evenly.
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        N: int,
        C: int,
        spatial: tuple,
        num_groups: int,
        dtype: torch.dtype,
        eps: float = 1e-5,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if C % num_groups != 0:
            raise ValueError(
                f"C={C} must be divisible by num_groups={num_groups}"
            )
        self.N = N
        self.C = C
        self.spatial = spatial
        self.num_groups = num_groups
        self.dtype = dtype
        self.eps = eps
        self.spatial_size = math.prod(spatial)
        # row length before padding
        self.D = (C // num_groups) * self.spatial_size
        self.M = N * num_groups  # number of rows
        self.D_padded = align_up(self.D, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["group_norm"](
            self.M, self.D, eps, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"group_norm": GroupNormKernel}

    def eval_roofline(self) -> tuple[int, int]:
        elem_bytes = self.dtype.itemsize
        return (
            5 * self.N * self.C * self.spatial_size,
            (2 * self.N * self.C * self.spatial_size + 2 * self.C) * elem_bytes,
        )

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    ) -> torch.Tensor:
        """Apply group normalization.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.
            weight: Affine scale of shape ``(C,)`` on CUDA.
            bias: Affine shift of shape ``(C,)`` on CUDA.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch,
                or shapes are incompatible with the configured dimensions.
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
        # Ensure contiguous and reshape to (N, num_groups, C/num_groups, *spatial)
        x = x.contiguous()

        # Reshape: (N, C, *spatial)
        # -> (N, num_groups, C/num_groups, *spatial)
        # -> (N*num_groups, (C/num_groups)*spatial_size)
        cpg = self.C // self.num_groups  # channels per group
        x_reshaped = x.reshape(self.N, self.num_groups, cpg, *self.spatial)
        x_2d = x_reshaped.reshape(self.M, self.D)

        # The kernel broadcasts 1D weight/bias across all rows, but GroupNorm
        # needs per-group affine parameters. Run kernel with unit weight/zero
        # bias to normalize, then apply per-channel affine afterwards.
        unit_weight = torch.ones(
            self.D_padded, dtype=self.dtype, device=x.device,
        )
        zero_bias = torch.zeros(
            self.D_padded, dtype=self.dtype, device=x.device,
        )

        # Pad to alignment
        if self.D_padded != self.D:
            x_2d = F.pad(x_2d, (0, self.D_padded - self.D))

        # Run kernel: produces (x - mean) / sqrt(var + eps)
        y_2d = self.kernel(x_2d, unit_weight, zero_bias)

        # Trim padding
        if self.D_padded != self.D:
            y_2d = y_2d[:, :self.D]

        # Reshape back: (N*num_groups, D) -> (N, num_groups, cpg, *spatial)
        # -> (N, C, *spatial)
        y = y_2d.reshape(self.N, self.num_groups, cpg, *self.spatial)
        y = y.reshape(orig_shape)

        # Apply per-channel affine: y = y * weight + bias
        # weight and bias are (C,), need to broadcast to (N, C, *spatial)
        affine_shape = [1, self.C] + [1] * len(self.spatial)
        y = y * weight.reshape(affine_shape) + bias.reshape(affine_shape)

        return y
