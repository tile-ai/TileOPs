"""InstanceNorm forward operator.

Instance Normalization (IN) is a special case of Group Normalization (GN)
where ``num_groups = C`` (each channel is its own group). The affine path
delegates to `GroupNormKernel` with that grouping.

User-facing API follows `torch.nn.functional.instance_norm`:

    op = InstanceNormFwdOp()
    y = op(x, running_mean, running_var, weight, bias)

Every tensor after ``x`` is optional. ``use_input_stats=False`` normalizes by the running
statistics; with ``use_input_stats=True`` passed running statistics are updated in place.

Input tensors accept shape ``(N, C, *spatial)``.
"""

import math
from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.norm import (
    BatchNormFwdInferKernel,
    InstanceNormFwdTrainKernel,
    InstanceNormFwdTrainSingleKernel,
    InstanceNormKernel,
    InstanceNormNoAffineKernel,
)
from tileops.kernels.norm.call_spec import (
    BatchNormCall,
    InstanceNormFwdInferInterface,
    InstanceNormFwdInterface,
    InstanceNormFwdTrainInterface,
)

from ..op_base import Op
from .norm_base import affine_or_constant

__all__ = ["InstanceNormFwdOp"]


class InstanceNormFwdOp(Op):
    """Instance Normalization forward operator.

    Computes instance normalization over spatial dimensions for each
    ``(batch, channel)`` independently:

    $$
    y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
    \\cdot w + b
    $$

    where the mean and variance are computed over ``*spatial`` for each sample-channel
    pair, or read from the running statistics when ``use_input_stats=False``. An absent
    ``weight`` scales by one and an absent ``bias`` shifts by zero. With
    ``use_input_stats=True`` passed running statistics take the batch mean of the instance
    means and unbiased variances with ``momentum``. As in torch, the running statistics are
    read and updated in the input's dtype.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "instance_norm": InstanceNormKernel,
        "instance_norm_no_affine": InstanceNormNoAffineKernel,
        "instance_norm_train_single": InstanceNormFwdTrainSingleKernel,
        "instance_norm_train": InstanceNormFwdTrainKernel,
        "instance_norm_running_stats": BatchNormFwdInferKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "instance_norm": InstanceNormFwdInterface,
        "instance_norm_train": InstanceNormFwdTrainInterface,
        "instance_norm_infer": InstanceNormFwdInferInterface,
    }

    def __init__(
        self,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            use_input_stats: Whether each instance is normalized by its own statistics
                (the default) or by the running statistics.
            momentum: Weight of this call's statistics in the running-statistics update.
            eps: Epsilon for numerical stability.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dictionary.
            tune: If ``True``, autotune tile configurations.
        """
        self.use_input_stats = use_input_stats
        self.momentum = momentum
        self.eps = eps
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    def forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply instance normalization.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)``.
            running_mean: Per-channel running mean of shape $[C]$, ``torch.float32``.
                Required when ``use_input_stats=False``; updated in place otherwise.
            running_var: Per-channel running variance, passed exactly when
                ``running_mean`` is.
            weight: Affine scale of shape $[C]$ in ``x``'s dtype, or ``None``.
            bias: Affine shift of shape $[C]$ in ``x``'s dtype, or ``None``.

        Returns:
            Normalized tensor of the same shape as *x*.
        """
        return self._call_boundary(x, running_mean, running_var, weight, bias)

    def _eager_forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernels and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        batch, channels = x.shape[0], x.shape[1]
        spatial = math.prod(x.shape[2:])
        tracks = running_mean is not None
        if x.numel() == 0:
            if self.use_input_stats and tracks and batch == 0:
                # The batch mean over no instances, as torch takes it.
                running_mean.fill_(math.nan)
                running_var.fill_(math.nan)
            return torch.empty_like(x)
        x = x.contiguous()
        call = BatchNormCall(
            device=x.device,
            n=batch,
            c=channels,
            spatial=spatial,
            dtype=x.dtype,
            eps=self.eps,
            momentum=self.momentum,
            input_dtype_params=True,
            has_weight=weight is not None,
            has_bias=bias is not None,
        )
        if not self.use_input_stats or tracks:
            role = "instance_norm_train" if self.use_input_stats else "instance_norm_infer"
            view = x.view(batch, channels, spatial)
            weight = None if weight is None else weight.contiguous()
            bias = None if bias is None else bias.contiguous()
            # The training kernel writes the running statistics in place, so a strided
            # buffer is served through a contiguous copy that is written back.
            stats = tuple(stat.contiguous() for stat in (running_mean, running_var))
            kernel = self.kernel_for(role, (view, *stats, weight, bias), call)
            self.kernel = kernel
            y = kernel(view, *stats, weight, bias)
            if self.use_input_stats:
                for caller, used in zip((running_mean, running_var), stats, strict=True):
                    if used is not caller:
                        caller.copy_(used)
            return y.view(x.shape)
        if weight is not None or bias is not None:
            weight = affine_or_constant(weight, (channels,), 1.0, x.dtype, x.device)
            bias = affine_or_constant(bias, (channels,), 0.0, x.dtype, x.device)
        # Row m of the (N*C, spatial_size) view is channel m % C throughout, so the affine
        # kernel applies the per-channel affine itself.
        rows = x.view(batch * channels, spatial)
        kernel = self.kernel_for("instance_norm", (rows, weight, bias), call)
        self.kernel = kernel
        return kernel(rows, running_mean, running_var, weight, bias).view(x.shape)
