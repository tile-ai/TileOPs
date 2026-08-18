"""InstanceNorm forward operator.

Instance Normalization (IN) is a special case of Group Normalization (GN)
where ``num_groups = C`` (each channel is its own group). The affine path
delegates to :class:`GroupNormKernel` with that grouping.

User-facing API mirrors :func:`torch.nn.functional.instance_norm`:

    op = InstanceNormFwdOp()
    y = op(x, running_mean, running_var, weight, bias)

Every tensor after ``x`` is optional. Passing ``weight`` and ``bias`` applies
the per-channel affine; passing the running stats is what makes
``use_input_stats=False`` callable.

Input tensors accept shape ``(N, C, *spatial)``.
"""

import math
from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import (
    GroupNormKernel,
    InstanceNormNoAffineKernel,
)

from ..op_base import Op

__all__ = ["InstanceNormFwdOp"]


class InstanceNormFwdOp(Op):
    """Instance Normalization forward operator.

    Computes instance normalization over spatial dimensions for each
    ``(batch, channel)`` independently:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot w + b

    where the mean and variance are computed over ``*spatial`` for each
    sample-channel pair, and the trailing affine applies only when ``weight``
    and ``bias`` are passed. Equivalent to Group Normalization with
    ``num_groups = C``.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary spatial dimensions (1-D, 2-D, 3-D+). The affine
        call delegates to :class:`GroupNormKernel` with one group per channel,
        which applies the per-channel affine itself; without affine it
        delegates to :class:`InstanceNormNoAffineKernel`.

    Args:
        use_input_stats: Mirrors ``torch.nn.functional.instance_norm``. When
            ``True`` (the default), per-instance statistics are computed from
            the input. ``False`` normalizes by the passed running stats, and
            is implemented for the affine-free call only.
        momentum: Mirrors ``torch.nn.functional.instance_norm``. Stored on the
            op instance for API parity with PyTorch but unused: neither path
            updates the running stats.
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N: Optional[int] = None
        self.C: Optional[int] = None
        self.spatial: Optional[Tuple[int, ...]] = None
        self.dtype: Optional[torch.dtype] = None
        self.use_input_stats = use_input_stats
        self.momentum = momentum
        self.eps = eps
        self.tune = tune
        self.spatial_size: Optional[int] = None
        self.D: Optional[int] = None
        self.M: Optional[int] = None
        self._running_stats_broadcast_shape: Optional[list[int]] = None
        self.dispatch_kernel(kernel_map)
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[tuple] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "group_norm": GroupNormKernel,
            "instance_norm_no_affine": InstanceNormNoAffineKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "InstanceNormFwdOp.eval_roofline() requires a prior forward() call"
            )
        N, C, spatial_size, dtype, affine, tracks_stats = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        flops = (5 if affine else 3) * N * C * spatial_size
        nbytes = (
            2 * N * C * spatial_size + (2 * C if affine else 0)
        ) * elem_bytes + (4 * C * 4 if tracks_stats else 0)
        return flops, nbytes

    def _validate_dtypes(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Validate input dtypes against the manifest dtype union.

        Manifest declares ``x.dtype`` as ``float32 | float16 | bfloat16``, with
        ``weight`` / ``bias`` matching it and the running stats ``float32``.

        Args:
            x: Input tensor.
            running_mean: Per-channel running mean, or ``None``.
            running_var: Per-channel running variance, or ``None``.
            weight: Affine scale, or ``None``.
            bias: Affine shift, or ``None``.

        Raises:
            ValueError: If any dtype is outside its supported set, or a passed
                affine tensor does not match ``x.dtype``.
        """
        allowed = (torch.float32, torch.float16, torch.bfloat16)
        if x.dtype not in allowed:
            raise ValueError(f"x.dtype must be one of {allowed}, got {x.dtype}")
        for name, t in (("weight", weight), ("bias", bias)):
            if t is None:
                continue
            if t.dtype != x.dtype:
                raise ValueError(
                    f"Expected {name}.dtype == {x.dtype}, got {t.dtype}"
                )
        for name, t in (("running_mean", running_mean), ("running_var", running_var)):
            if t is None:
                continue
            if t.dtype != torch.float32:
                raise ValueError(
                    f"Expected {name}.dtype torch.float32, got {t.dtype}"
                )

    def _validate_affine(
        self, name: str, t: torch.Tensor, x_device: torch.device, C: int,
    ) -> None:
        """Validate device and shape of an affine tensor."""
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.device != x_device:
            raise ValueError(f"Expected {name} on {x_device}, got {t.device}")
        if t.ndim != 1 or t.shape[0] != C:
            raise ValueError(f"Expected {name} shape ({C},), got {tuple(t.shape)}")

    def _validate_running_stats(
        self, name: str, t: torch.Tensor, x_device: torch.device, C: int,
    ) -> None:
        """Validate device, dtype, and shape of a running-stats tensor."""
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.device != x_device:
            raise ValueError(f"Expected {name} on {x_device}, got {t.device}")
        if t.dtype != torch.float32:
            raise ValueError(
                f"Expected {name}.dtype torch.float32, got {t.dtype}"
            )
        if t.ndim != 1 or t.shape[0] != C:
            raise ValueError(f"Expected {name} shape ({C},), got {tuple(t.shape)}")

    def _resolve_spec(
        self, x: torch.Tensor
    ) -> Tuple[int, int, Tuple[int, ...], int, int, int, torch.dtype]:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.ndim < 2:
            raise ValueError("x must have shape (N, C, *spatial)")
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                "x.dtype must be float32, float16, or bfloat16, "
                f"got {x.dtype}"
            )
        N, C, *spatial_list = x.shape
        spatial = tuple(spatial_list)
        spatial_size = math.prod(spatial)
        D = spatial_size
        M = N * C
        return N, C, spatial, spatial_size, D, M, x.dtype

    def _bind_spec(
        self,
        N: int,
        C: int,
        spatial: Tuple[int, ...],
        spatial_size: int,
        D: int,
        M: int,
        dtype: torch.dtype,
        affine: bool,
        tracks_stats: bool,
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.spatial_size = spatial_size
        self.D = D
        self.M = M
        self._running_stats_broadcast_shape = [1, C] + [1] * len(spatial)
        self.dtype = dtype
        self._last_roofline_spec = (
            N, C, spatial_size, dtype, affine, tracks_stats,
        )

    def _get_kernel(
        self,
        M: int,
        D: int,
        C: int,
        dtype: torch.dtype,
        device_index: Optional[int],
        affine: bool,
    ) -> Kernel:
        if affine:
            # One group per channel, so a row's every element belongs to the
            # same channel: num_groups=C with channels_per_group=1.
            key = ("group_norm", M, D, C, dtype, device_index, self.eps, self.tune)
            kernel = self.get_or_build_kernel(
                "group_norm",
                key=key,
                build=lambda: self.kernel_map["group_norm"](
                    M, D, self.eps, dtype, C, 1, tune=self.tune,
                ),
            )
        else:
            key = ("no_affine", M, D, dtype, device_index, self.eps, self.tune)
            kernel = self.get_or_build_kernel(
                "instance_norm_no_affine",
                key=key,
                build=lambda: self.kernel_map["instance_norm_no_affine"](
                    M, D, self.eps, dtype, tune=self.tune,
                ),
            )
        self.kernel = kernel
        return kernel

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
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.
            running_mean: Per-channel running mean of shape ``(C,)``, dtype
                ``torch.float32``, on ``x``'s device. Required when
                ``use_input_stats=False``.
            running_var: Per-channel running variance, same constraints.
            weight: Affine scale of shape ``(C,)`` on CUDA, ``x``'s dtype.
                Must be passed together with ``bias``.
            bias: Affine shift, same constraints as ``weight``.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If a tensor is not on CUDA, a dtype mismatches, a shape
                is incompatible, one half of a pair is passed, or
                ``use_input_stats=False`` without running stats.
            NotImplementedError: If ``use_input_stats=False`` is combined with
                the affine tensors.
        """
        if (weight is None) != (bias is None):
            raise ValueError(
                "weight and bias are one switch: pass both for the affine "
                "path, or neither"
            )
        if (running_mean is None) != (running_var is None):
            raise ValueError(
                "running_mean and running_var are one switch: pass both or "
                "neither"
            )
        affine = weight is not None
        tracks_stats = running_mean is not None
        if not self.use_input_stats:
            if not tracks_stats:
                raise ValueError(
                    "use_input_stats=False normalizes by the running stats, so "
                    "running_mean and running_var must be passed"
                )
            if affine:
                raise NotImplementedError(
                    "use_input_stats=False is implemented for the affine-free "
                    "call only"
                )

        if self.use_input_stats and running_mean is not None:
            raise ValueError(
                "running_mean and running_var normalize the input only when "
                "use_input_stats=False. With use_input_stats=True torch "
                "updates them in place, which this inference op does not "
                "implement. If these were meant as the affine pair, pass them "
                "as weight= and bias=."
            )

        self._validate_dtypes(x, running_mean, running_var, weight, bias)
        N, C, spatial, spatial_size, D, M, dtype = self._resolve_spec(x)
        if affine:
            self._validate_affine("weight", weight, x.device, C)
            self._validate_affine("bias", bias, x.device, C)
        if tracks_stats:
            self._validate_running_stats("running_mean", running_mean, x.device, C)
            self._validate_running_stats("running_var", running_var, x.device, C)
        self._bind_spec(
            N, C, spatial, spatial_size, D, M, dtype, affine, tracks_stats,
        )

        if not self.use_input_stats:
            # Eval-mode path: y = (x - running_mean[c]) / sqrt(running_var[c] + eps).
            # Pure elementwise per-channel; matches torch.nn.functional.instance_norm
            # (use_input_stats=False) numerics bit-for-bit (verified) by computing in
            # fp32 then casting to x.dtype.
            mean_b = running_mean.reshape(self._running_stats_broadcast_shape)
            var_b = running_var.reshape(self._running_stats_broadcast_shape)
            y = (x.float() - mean_b) * torch.rsqrt(var_b + self.eps)
            return y.to(x.dtype)

        orig_shape = x.shape
        x_2d = x.contiguous().reshape(M, D)
        kernel = self._get_kernel(M, D, C, dtype, x.device.index, affine)

        # Row m of the (N*C, spatial_size) view is channel m % C throughout,
        # so the affine kernel applies the per-channel affine itself.
        y_2d = kernel(x_2d, weight, bias) if affine else kernel(x_2d)

        # Reshape back: (N*C, spatial_size) -> (N, C, *spatial)
        return y_2d.reshape(orig_shape)
