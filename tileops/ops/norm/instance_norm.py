"""InstanceNorm Op.

InstanceNorm is mathematically a special case of GroupNorm where G=C
(each channel is its own group). This op reuses GroupNormKernel with G=C.

User-facing API mirrors torch.nn.functional.instance_norm:

    op = InstanceNormOp(N=batch, C=channels, spatial=(H, W), dtype=dtype)
    y = op(x, weight, bias)
"""

from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.norm.group_norm import GroupNormKernel

from ..op import Op
from .group_norm import GroupNormOp

__all__ = ["InstanceNormOp"]


class InstanceNormOp(Op):
    """InstanceNorm forward operator.

    Delegates to GroupNormOp with G=C (each channel is its own group).

    Args:
        N: Batch size.
        C: Number of channels.
        spatial: Spatial dimensions as a tuple (H, W, ...).
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
        dtype: torch.dtype,
        eps: float = 1e-5,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype
        self.eps = eps

        # InstanceNorm = GroupNorm with G=C
        self._group_norm_op = GroupNormOp(
            N=N, C=C, spatial=spatial, G=C, dtype=dtype, eps=eps,
            tune=tune, kernel_map=kernel_map,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # Required by Op ABC. Not used directly -- delegation to GroupNormOp
        # handles kernel dispatch internally.
        return {"group_norm": GroupNormKernel}

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return self._group_norm_op.forward(x, weight, bias)
