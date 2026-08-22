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
from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import InstanceNormKernel, InstanceNormNoAffineKernel

from ..compile_boundary import get_instance
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
        eps: Epsilon for numerical stability (manifest ``params.eps``).
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
            in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    #: The operator this op registers; a test asserts the graph holds nothing else.
    compile_op_names: ClassVar[Tuple[str, ...]] = ("tileops::norm_instance_norm_fwd",)

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
        self.dtype: Optional[torch.dtype] = None
        self.use_input_stats = use_input_stats
        self.momentum = momentum
        self.eps = eps
        self.target = target
        self.tune = tune
        self._running_stats_broadcast_shape: Optional[list[int]] = None
        self.dispatch_kernel(kernel_map)
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[tuple] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "instance_norm": InstanceNormKernel,
            "instance_norm_no_affine": InstanceNormNoAffineKernel,
        }

    def _infer_output_shapes(
        self,
        x_shape: Tuple[int, ...],
        running_mean_shape: Optional[Tuple[int, ...]],
        running_var_shape: Optional[Tuple[int, ...]],
        weight_shape: Optional[Tuple[int, ...]],
        bias_shape: Optional[Tuple[int, ...]],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: ``output.shape == x.shape``."""
        return {"output": tuple(x_shape)}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError("InstanceNormFwdOp.eval_roofline() requires a prior forward() call")
        N, C, spatial_size, dtype, affine, tracks_stats = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        flops = (5 if affine else 3) * N * C * spatial_size
        nbytes = (2 * N * C * spatial_size + (2 * C if affine else 0)) * elem_bytes + (
            4 * C * 4 if tracks_stats else 0
        )
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
                raise ValueError(f"Expected {name}.dtype == {x.dtype}, got {t.dtype}")
        for name, t in (("running_mean", running_mean), ("running_var", running_var)):
            if t is None:
                continue
            if t.dtype != torch.float32:
                raise ValueError(f"Expected {name}.dtype torch.float32, got {t.dtype}")

    def _validate_affine(
        self,
        name: str,
        t: torch.Tensor,
        x_device: torch.device,
        C: int,
    ) -> None:
        """Validate device and shape of an affine tensor."""
        if t.device != x_device:
            raise ValueError(f"Expected {name} on {x_device}, got {t.device}")
        if t.ndim != 1 or t.shape[0] != C:
            raise ValueError(f"Expected {name} shape ({C},), got {tuple(t.shape)}")

    def _validate_running_stats(
        self,
        name: str,
        t: torch.Tensor,
        x_device: torch.device,
        C: int,
    ) -> None:
        """Validate device, dtype, and shape of a running-stats tensor."""
        if t.device != x_device:
            raise ValueError(f"Expected {name} on {x_device}, got {t.device}")
        if t.dtype != torch.float32:
            raise ValueError(f"Expected {name}.dtype torch.float32, got {t.dtype}")
        if t.ndim != 1 or t.shape[0] != C:
            raise ValueError(f"Expected {name} shape ({C},), got {tuple(t.shape)}")

    def _resolve_spec(
        self, x: torch.Tensor
    ) -> Tuple[int, int, Tuple[int, ...], int, int, torch.dtype]:
        if x.ndim < 2:
            raise ValueError("x must have shape (N, C, *spatial)")
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"x.dtype must be float32, float16, or bfloat16, got {x.dtype}")
        N, C, *spatial_list = x.shape
        spatial = tuple(spatial_list)
        spatial_size = math.prod(spatial)
        return N, C, spatial, spatial_size, spatial_size, x.dtype

    def _bind_spec(
        self,
        N: int,
        C: int,
        spatial: Tuple[int, ...],
        spatial_size: int,
        dtype: torch.dtype,
        affine: bool,
        tracks_stats: bool,
    ) -> None:
        """Bind what ``eval_roofline`` and the running-stats path read off this call."""
        self._running_stats_broadcast_shape = [1, C] + [1] * len(spatial)
        self.dtype = dtype
        self._last_roofline_spec = (
            N,
            C,
            spatial_size,
            dtype,
            affine,
            tracks_stats,
        )

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
            running_mean: Per-channel running mean of shape ``(C,)``, dtype
                ``torch.float32``, on ``x``'s device. Required when
                ``use_input_stats=False``.
            running_var: Per-channel running variance, same constraints.
            weight: Affine scale of shape ``(C,)``, ``x``'s dtype. Must be passed
                together with ``bias``.
            bias: Affine shift, same constraints as ``weight``.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: A dtype mismatches, a shape is incompatible, one half of a pair
                is passed, or ``use_input_stats=False`` without running stats.
            NotImplementedError: ``use_input_stats=False`` combined with the affine
                tensors. Both raised from inside the operator, by :meth:`_eager_forward`.
        """
        return _norm_instance_norm_fwd(
            x, running_mean, running_var, weight, bias, self._instance_key
        )

    def _eager_forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        if (weight is None) != (bias is None):
            raise ValueError(
                "weight and bias are one switch: pass both for the affine path, or neither"
            )
        if (running_mean is None) != (running_var is None):
            raise ValueError("running_mean and running_var are one switch: pass both or neither")
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
                    "use_input_stats=False is implemented for the affine-free call only"
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
        N, C, spatial, spatial_size, D, dtype = self._resolve_spec(x)
        if affine:
            self._validate_affine("weight", weight, x.device, C)
            self._validate_affine("bias", bias, x.device, C)
        if tracks_stats:
            self._validate_running_stats("running_mean", running_mean, x.device, C)
            self._validate_running_stats("running_var", running_var, x.device, C)
        self._bind_spec(N, C, spatial, spatial_size, dtype, affine, tracks_stats)

        if not self.use_input_stats:
            # Eval-mode path: y = (x - running_mean[c]) / sqrt(running_var[c] + eps).
            # Pure elementwise per-channel; matches torch.nn.functional.instance_norm
            # (use_input_stats=False) numerics bit-for-bit (verified) by computing in
            # fp32 then casting to x.dtype.
            mean_b = running_mean.reshape(self._running_stats_broadcast_shape)
            var_b = running_var.reshape(self._running_stats_broadcast_shape)
            y = (x.float() - mean_b) * torch.rsqrt(var_b + self.eps)
            return y.to(x.dtype)

        # Handed over as the manifest declares it; the layout a kernel wants is its own business.
        x = x.contiguous()
        if affine:
            weight = weight.contiguous()
            bias = bias.contiguous()
        if tracks_stats:
            running_mean = running_mean.contiguous()
            running_var = running_var.contiguous()
        # The affine pair picks the implementation, so it belongs in the key; both are
        # fetched under one name, which is what a target is asked to serve. One group per
        # channel, so a row's every element belongs to the same channel.
        slot = "instance_norm" if affine else "instance_norm_no_affine"
        kernel = self.get_or_build_kernel(
            "instance_norm",
            (x, running_mean, running_var, weight, bias),
            key=(D, dtype, affine),  # this instance's in-tree cache key
            build=lambda: (
                self.kernel_map[slot](D, self.eps, dtype, C, 1, tune=self.tune)
                if affine
                else self.kernel_map[slot](D, self.eps, dtype, tune=self.tune)
            ),
        )
        self.kernel = kernel

        # Row m of the (N*C, spatial_size) view is channel m % C throughout, so the affine
        # kernel applies the per-channel affine itself. Every declared input keeps its slot:
        # this kernel reads no running statistics, and an absent optional input is ``None``.
        return kernel(x, running_mean, running_var, weight, bias)


@torch.library.custom_op("tileops::norm_instance_norm_fwd", mutates_args=())
def _norm_instance_norm_fwd(
    x: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(x, running_mean, running_var, weight, bias)


@_norm_instance_norm_fwd.register_fake
def _norm_instance_norm_fwd_fake(
    x: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(
        tuple(x.shape),
        None if running_mean is None else tuple(running_mean.shape),
        None if running_var is None else tuple(running_var.shape),
        None if weight is None else tuple(weight.shape),
        None if bias is None else tuple(bias.shape),
    )
    # ``new_empty``, not ``empty_like``: a non-contiguous input's strides must not reach the fake.
    return x.new_empty(shapes["output"])
