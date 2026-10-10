"""GroupNorm forward operator.

User-facing API follows torch.nn.functional.group_norm:

    op = GroupNormFwdOp(num_groups=groups)
    y = op(x, weight, bias)   # affine
    y = op(x)                 # torch.nn.GroupNorm(affine=False)

Input tensors accept shape (N, C, *spatial); the kernel reshapes to
(N*num_groups, D) internally where D = (C/num_groups) * spatial_size.
"""

import math
from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.norm import GroupNormKernel, GroupNormNoAffineKernel, GroupNormSplitKernel
from tileops.kernels.norm.call_spec import GroupNormCall, GroupNormFwdInterface
from tileops.ops.norm.affine import affine_or_constant
from tileops.ops.op_base import Op

__all__ = ["GroupNormFwdOp"]


class GroupNormFwdOp(Op):
    """Group Normalization forward operator.

    Computes group normalization over ``(C/num_groups, *spatial)`` slices:

    $$
    y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
    \\cdot w + b
    $$

    where the mean and variance are computed per group over
    ``(C/num_groups, *spatial)`` elements. ``weight`` and ``bias`` are independent, as in
    torch: an absent one scales by one or shifts by zero.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "group_norm": GroupNormKernel,
        "group_norm_no_affine": GroupNormNoAffineKernel,
        "group_norm_split": GroupNormSplitKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "group_norm": GroupNormFwdInterface
    }

    def __init__(
        self,
        num_groups: int,
        eps: float = 1e-5,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            num_groups: Number of groups; it divides the channel count.
            eps: Epsilon for numerical stability.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dictionary.
            tune: If ``True``, autotune tile configurations.
        """
        self.num_groups = num_groups
        self.eps = eps
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply group normalization.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)``.
            weight: Affine scale of shape $[C]$, or ``None``.
            bias: Affine shift of shape $[C]$, or ``None``.

        Returns:
            Normalized tensor of the same shape as *x*.
        """
        channels = x.shape[1]
        affine = weight is not None or bias is not None
        if affine:
            weight = affine_or_constant(weight, (channels,), 1.0, x.dtype, x.device)
            bias = affine_or_constant(bias, (channels,), 0.0, x.dtype, x.device)
        x = x.contiguous()
        call = GroupNormCall(
            device=x.device,
            c=channels,
            spatial=math.prod(x.shape[2:]),
            num_groups=self.num_groups,
            eps=self.eps,
            dtype=x.dtype,
            passes_affine=affine,
        )
        kernel = self.kernel_for("group_norm", call)
        # The affine kernel derives each element's channel from its position
        # in the row, so the per-channel affine is applied inside the kernel.
        return kernel(x, weight, bias)
