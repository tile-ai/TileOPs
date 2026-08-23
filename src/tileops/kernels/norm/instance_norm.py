"""InstanceNorm kernels.

InstanceNorm is the special case of GroupNorm with G = C (each channel is
its own group). The math reduces exactly to the GroupNorm row-wise kernel
on a reshape of ``(N, C, *spatial) -> (N*C, spatial_size)``.

The TileLang bodies are GroupNorm's; these classes exist so that
`tileops.ops.norm.instance_norm` and the manifest can name an
InstanceNorm-specific kernel, and so that both take the five inputs
``InstanceNormFwdOp``'s signature declares.
"""

from typing import Optional

import torch

from .group_norm import GroupNormKernel, GroupNormNoAffineKernel

__all__ = ["InstanceNormKernel", "InstanceNormNoAffineKernel"]


class InstanceNormKernel(GroupNormKernel):
    """InstanceNorm forward kernel with a per-channel affine.

    GroupNorm's kernel with ``num_groups=C`` and ``channels_per_group=1``. The running
    statistics are slots of the op's signature that this kernel does not read: it
    normalizes by the statistics of this call's input.
    """

    def forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(x, weight, bias)


class InstanceNormNoAffineKernel(GroupNormNoAffineKernel):
    """InstanceNorm forward kernel without affine scale/shift.

    GroupNorm's no-affine kernel with ``G = C``. The running statistics and the affine
    pair are slots of the op's signature that this kernel does not read.
    """

    def forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(x, weight, bias)
