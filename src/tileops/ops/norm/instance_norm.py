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
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm import InstanceNormKernel, InstanceNormNoAffineKernel
from tileops.kernels.norm.batch_norm import BatchNormFwdInferKernel
from tileops.kernels.reduction.reduce import ReduceKernel

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
        "instance_norm_running_stats": BatchNormFwdInferKernel,
        "instance_stats": ReduceKernel,
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
        if not self.use_input_stats:
            return self._normalize_by_running_stats(x, running_mean, running_var, weight, bias)
        if tracks:
            self._update_running_stats(x, running_mean, running_var)
        affine = weight is not None or bias is not None
        if affine:
            weight = affine_or_constant(weight, (channels,), 1.0, x.dtype, x.device)
            bias = affine_or_constant(bias, (channels,), 0.0, x.dtype, x.device)
        kernel = self.kernel_for(
            "instance_norm", (x, weight, bias), (spatial, x.dtype, affine, channels)
        )
        self.kernel = kernel
        # Row m of the (N*C, spatial_size) view is channel m % C throughout, so the affine
        # kernel applies the per-channel affine itself.
        return kernel(x, running_mean, running_var, weight, bias)

    def _normalize_by_running_stats(self, x, running_mean, running_var, weight, bias):
        """Each channel's affine map of the running statistics: batch norm's inference."""
        channels = x.shape[1]
        weight = affine_or_constant(weight, (channels,), 1.0, x.dtype, x.device).float()
        bias = affine_or_constant(bias, (channels,), 0.0, x.dtype, x.device).float()
        # torch reads the statistics in the input's dtype.
        stats = tuple(stat.to(x.dtype).float() for stat in (running_mean, running_var))
        call = (channels, x.numel() // channels, x.dtype, math.prod(x.shape[2:]))
        kernel = self.kernel_for("instance_norm_running_stats", (x, *stats, weight, bias), call)
        return kernel(x, *stats, weight, bias)

    def _update_running_stats(self, x, running_mean, running_var) -> None:
        """Move the running statistics toward the batch mean of the instance statistics.

        torch updates them in the input's dtype and writes that rounding back.
        """
        axes = tuple(range(2, x.ndim))
        rows, width = x.shape[0] * x.shape[1], math.prod(x.shape[2:])
        call = (tuple(x.shape), axes, x.dtype, x.device.index, rows, width)
        var, mean = self.kernel_for("instance_stats", (x,), call)(x)
        momentum = self.momentum
        for running, batch in ((running_mean, mean), (running_var, var)):
            moved = (1 - momentum) * running.to(x.dtype).float() + momentum * batch.float().mean(0)
            running.copy_(moved.to(x.dtype))

    def entry_for(self, role: str, call: tuple) -> Entry:
        """Each role has one implementation; the affine form picks the normalizing one."""
        if role == "instance_stats":
            shape, axes, dtype, device_index, rows, width = call
            cls = self.kernel_map["instance_stats"]
            return call, lambda: cls(
                rows,
                width,
                "var_mean",
                dtype,
                reduce_axes=axes,
                correction=1,
                tune=self.tune,
                device_index=device_index,
            )
        if role == "instance_norm_running_stats":
            channels, length, dtype, spatial = call
            cls = self.kernel_map["instance_norm_running_stats"]
            return call, lambda: cls(channels, length, dtype, self.eps, tune=self.tune, S=spatial)
        d, dtype, affine, channels = call
        if affine:
            cls = self.kernel_map["instance_norm"]
            return call, lambda: cls(d, self.eps, dtype, channels, 1, tune=self.tune)
        cls = self.kernel_map["instance_norm_no_affine"]
        return call, lambda: cls(d, self.eps, dtype, tune=self.tune)
