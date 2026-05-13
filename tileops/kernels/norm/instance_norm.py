"""InstanceNorm kernels.

InstanceNorm is the special case of GroupNorm with G = C (each channel is
its own group). The math reduces exactly to the GroupNorm row-wise kernel
on a reshape of ``(N, C, *spatial) -> (N*C, spatial_size)``.

This module exposes thin aliases over the GroupNorm kernels so that
op modules and the manifest can name the InstanceNorm-specific kernel
without duplicating the TileLang body.
"""

from .group_norm import GroupNormKernel, GroupNormNoAffineKernel

__all__ = ["InstanceNormKernel", "InstanceNormNoAffineKernel"]


class InstanceNormKernel(GroupNormKernel):
    """InstanceNorm forward kernel with affine scale/shift.

    Alias of :class:`GroupNormKernel` invoked with ``G = C``; the underlying
    row-wise kernel is identical. Defined here so that
    :mod:`tileops.ops.norm.instance_norm` and the manifest can refer to an
    InstanceNorm-specific kernel symbol.
    """


class InstanceNormNoAffineKernel(GroupNormNoAffineKernel):
    """InstanceNorm forward kernel without affine scale/shift.

    Alias of :class:`GroupNormNoAffineKernel` invoked with ``G = C``; the
    underlying row-wise kernel is identical. Defined here so that
    :mod:`tileops.ops.norm.instance_norm` and the manifest can refer to an
    InstanceNorm-specific kernel symbol.
    """
