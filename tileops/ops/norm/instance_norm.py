"""InstanceNorm forward operator.

Instance Normalization (IN) is a special case of Group Normalization (GN)
where ``num_groups = C`` (each channel is its own group). This operator
delegates to :class:`GroupNormKernel` with that grouping.

User-facing API mirrors :func:`torch.nn.functional.instance_norm`:

    op = InstanceNormFwdOp(N=batch, C=channels, spatial=(H, W), dtype=dtype)
    y = op(x, weight, bias)

Input tensors accept shape ``(N, C, *spatial)``.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import (
    GroupNormKernel,
    InstanceNormNoAffineKernel,
)

from ..op_base import Op
from .norm_base import ALIGNMENT, align_up

__all__ = ["InstanceNormFwdOp", "InstanceNormFwdOpNoAffine"]

# Largest candidate block_m in InstanceNormNoAffineKernel.autotune_configs.
# The op pads M to a multiple of this value so the kernel's full-tile
# T.copy never crosses the M boundary regardless of the selected block_m.
_M_BLOCK_ALIGN = 16


class InstanceNormFwdOp(Op):
    """Instance Normalization forward operator.

    Computes instance normalization over spatial dimensions for each
    ``(batch, channel)`` independently:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot w + b

    where the mean and variance are computed over ``*spatial`` for each
    sample-channel pair. Equivalent to Group Normalization with
    ``num_groups = C``.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary spatial dimensions (1-D, 2-D, 3-D+). Delegates
        to :class:`GroupNormKernel` with one group per channel. Hidden
        dimension is padded to 256-element alignment internally. The
        running-stats variant (``use_input_stats=False``) is out of scope
        for this op.

    Args:
        N: Batch size.
        C: Number of channels.
        spatial: Spatial dimensions tuple ``(H, W, ...)``.
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        use_input_stats: Mirrors ``torch.nn.functional.instance_norm``. When
            ``True`` (the default and only supported value), per-batch
            statistics are computed from the input. ``False`` (the
            running-stats / eval-mode path) is deferred and raises
            ``NotImplementedError``.
        momentum: Mirrors ``torch.nn.functional.instance_norm``. Stored on
            the op instance for API parity with PyTorch but unused on
            the per-batch (``use_input_stats=True``) path.
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.

    Raises:
        NotImplementedError: If ``use_input_stats=False`` is requested
            (the deferred running-stats path).
    """

    def __init__(
        self,
        N: int,
        C: int,
        spatial: tuple,
        dtype: torch.dtype,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if not use_input_stats:
            raise NotImplementedError(
                "use_input_stats=False (the running-stats / eval-mode path) "
                "is not supported by InstanceNormFwdOp; only "
                "use_input_stats=True (per-batch statistics) is implemented."
            )
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype
        self.use_input_stats = use_input_stats
        self.momentum = momentum
        self.eps = eps
        self.spatial_size = math.prod(spatial)
        # InstanceNorm: each channel is its own group (num_groups = C)
        # so D = (C/C) * spatial_size = spatial_size and M = N * C.
        self.D = self.spatial_size
        self.M = N * C
        self.D_padded = align_up(self.D, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["group_norm"](
            self.M, self.D, eps, dtype, tune=tune,
        )
        # The compiled kernel binds to the active CUDA device at construction
        # time, so subsequent forwards must receive inputs on that same
        # device. Capture it here so forward() can raise a clean error
        # rather than letting the kernel layer surface an opaque
        # device-mismatch failure.
        self._kernel_device = torch.device("cuda", torch.cuda.current_device())
        # Pre-allocate forward-time constants. The kernel binds 1D weight/bias
        # row-broadcast inputs; per-channel affine doesn't fit that layout, so
        # the kernel call uses identity (unit/zero) buffers and the affine is
        # applied after. These tensors are immutable for the op's lifetime.
        self._unit_weight = torch.ones(
            self.D_padded, dtype=dtype, device=self._kernel_device,
        )
        self._zero_bias = torch.zeros(
            self.D_padded, dtype=dtype, device=self._kernel_device,
        )
        self._expected_shape = (self.N, self.C, *self.spatial)
        self._affine_shape = (1, self.C) + (1,) * len(self.spatial)

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
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Apply instance normalization.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.
            weight: Affine scale of shape ``(C,)`` on CUDA. Required; the
                affine-free path is :class:`InstanceNormFwdOpNoAffine`.
            bias: Affine shift of shape ``(C,)`` on CUDA. Required; the
                affine-free path is :class:`InstanceNormFwdOpNoAffine`.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If any tensor is not on CUDA, dtypes mismatch, or
                shapes are incompatible with the configured dimensions.
        """
        if not isinstance(weight, torch.Tensor):
            raise ValueError(
                "weight is required; use InstanceNormFwdOpNoAffine for the "
                "affine-free path"
            )
        if not isinstance(bias, torch.Tensor):
            raise ValueError(
                "bias is required; use InstanceNormFwdOpNoAffine for the "
                "affine-free path"
            )
        self._validate_dtypes(x, weight, bias)
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.device != self._kernel_device:
            raise ValueError(
                f"Device mismatch: op was constructed for {self._kernel_device} "
                f"but x is on {x.device}. Construct a separate op instance per "
                f"CUDA device."
            )
        if tuple(x.shape) != self._expected_shape:
            raise ValueError(
                f"Expected x shape {self._expected_shape}, got {tuple(x.shape)}"
            )
        if not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if weight.device != x.device:
            raise ValueError(
                f"Expected weight on {x.device}, got {weight.device}"
            )
        if weight.ndim != 1 or weight.shape[0] != self.C:
            raise ValueError(
                f"Expected weight shape ({self.C},), got {weight.shape}"
            )
        if not bias.is_cuda:
            raise ValueError("bias must be a CUDA tensor")
        if bias.device != x.device:
            raise ValueError(
                f"Expected bias on {x.device}, got {bias.device}"
            )
        if bias.ndim != 1 or bias.shape[0] != self.C:
            raise ValueError(
                f"Expected bias shape ({self.C},), got {bias.shape}"
            )

        orig_shape = x.shape
        x = x.contiguous()
        x_2d = x.reshape(self.M, self.D)

        if self.D_padded != self.D:
            x_2d = F.pad(x_2d, (0, self.D_padded - self.D))

        # Kernel broadcasts 1D weight/bias row-wise; per-channel affine is
        # applied after, so run the kernel with identity (unit/zero) buffers.
        y_2d = self.kernel(x_2d, self._unit_weight, self._zero_bias)

        # Trim padding
        if self.D_padded != self.D:
            y_2d = y_2d[:, :self.D]

        # Reshape back: (N*C, spatial_size) -> (N, C, *spatial)
        y = y_2d.reshape(orig_shape)

        y = y * weight.reshape(self._affine_shape) + bias.reshape(self._affine_shape)
        return y


class InstanceNormFwdOpNoAffine(Op):
    """Instance Normalization forward without affine scale/shift.

    Computes instance normalization without the trailing weight/bias affine:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}

    where the mean and variance are computed over ``*spatial`` for each
    sample-channel pair. Mirrors
    ``torch.nn.functional.instance_norm(x, weight=None, bias=None, ...)``
    and ``torch.nn.InstanceNorm*(affine=False)`` (the InstanceNorm default).

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Args:
        N: Batch size.
        C: Number of channels.
        spatial: Spatial dimensions tuple ``(H, W, ...)``.
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        use_input_stats: Mirrors ``torch.nn.functional.instance_norm``. When
            ``True`` (the default), per-instance statistics are computed from
            the input. When ``False``, the supplied ``running_mean`` and
            ``running_var`` are used to normalize (eval-mode / inference path).
        momentum: Mirrors ``torch.nn.functional.instance_norm``. Stored on
            the op instance for API parity with PyTorch but unused on the
            forward path (no running-stat update on ``use_input_stats=True``;
            no running-stat update on ``use_input_stats=False`` either).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        N: int,
        C: int,
        spatial: tuple,
        dtype: torch.dtype,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype
        self.use_input_stats = use_input_stats
        self.momentum = momentum
        self.eps = eps
        self.spatial_size = math.prod(spatial)
        self.D = self.spatial_size
        self.M = N * C
        # Eval-mode broadcast layout for running stats: [1, C, 1, ...] (one 1 per spatial dim).
        self._running_stats_broadcast_shape = [1, C] + [1] * len(spatial)
        self.D_padded = align_up(self.D, ALIGNMENT)
        # Kernel launches T.ceildiv(M, block_m) programs, each copying a full
        # block_m-row tile. Pad M to a multiple of the largest candidate
        # block_m so the tail program never reads/writes past the input.
        self.M_padded = align_up(self.M, _M_BLOCK_ALIGN)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["instance_norm_no_affine"](
            self.M_padded, self.D, eps, dtype, tune=tune,
        )
        self._kernel_device = torch.device("cuda", torch.cuda.current_device())

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"instance_norm_no_affine": InstanceNormNoAffineKernel}

    def eval_roofline(self) -> tuple[int, int]:
        elem_bytes = self.dtype.itemsize
        return (
            3 * self.N * self.C * self.spatial_size,
            2 * self.N * self.C * self.spatial_size * elem_bytes,
        )

    def _validate_dtypes(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> None:
        """Validate input dtypes against the manifest dtype union.

        Manifest declares ``x.dtype`` as ``float32 | float16 | bfloat16``;
        ``running_mean`` and ``running_var`` are ``float32`` only. The
        configured op dtype must be drawn from the same union as ``x`` and
        match the input.

        Args:
            x: Input tensor.
            running_mean: Per-channel running mean tensor.
            running_var: Per-channel running variance tensor.

        Raises:
            ValueError: If any dtype is outside its supported set, or
                ``x.dtype`` does not match ``self.dtype``.
        """
        allowed = (torch.float32, torch.float16, torch.bfloat16)
        if self.dtype not in allowed:
            raise ValueError(
                f"self.dtype must be one of {allowed}, got {self.dtype}"
            )
        if x.dtype not in allowed:
            raise ValueError(
                f"x.dtype must be one of {allowed}, got {x.dtype}"
            )
        if x.dtype != self.dtype:
            raise ValueError(
                f"Expected x.dtype {self.dtype}, got {x.dtype}"
            )
        if running_mean.dtype != torch.float32:
            raise ValueError(
                f"Expected running_mean.dtype torch.float32, got {running_mean.dtype}"
            )
        if running_var.dtype != torch.float32:
            raise ValueError(
                f"Expected running_var.dtype torch.float32, got {running_var.dtype}"
            )

    def _validate_running_stats(
        self, name: str, t: torch.Tensor, x_device: torch.device,
    ) -> None:
        """Validate device, dtype, and shape of a running-stats tensor."""
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.device != x_device:
            raise ValueError(
                f"Expected {name} on {x_device}, got {t.device}"
            )
        if t.dtype != torch.float32:
            raise ValueError(
                f"Expected {name}.dtype torch.float32, got {t.dtype}"
            )
        if t.ndim != 1 or t.shape[0] != self.C:
            raise ValueError(
                f"Expected {name} shape ({self.C},), got {tuple(t.shape)}"
            )

    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> torch.Tensor:
        """Apply instance normalization without affine.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.
            running_mean: Per-channel running mean of shape ``(C,)``, dtype
                ``torch.float32``, on the same CUDA device as ``x``. Used
                only when ``use_input_stats=False``; ignored otherwise but
                must still be supplied (R16: no ``Optional[Tensor]``).
            running_var: Per-channel running variance of shape ``(C,)``,
                dtype ``torch.float32``, on the same CUDA device as ``x``.
                Used only when ``use_input_stats=False``; ignored otherwise
                but must still be supplied.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch, or
                shapes are incompatible with the configured dimensions.
        """
        self._validate_dtypes(x, running_mean, running_var)
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.device != self._kernel_device:
            raise ValueError(
                f"Device mismatch: op was constructed for {self._kernel_device} "
                f"but x is on {x.device}. Construct a separate op instance per "
                f"CUDA device."
            )
        expected_shape = (self.N, self.C, *self.spatial)
        if tuple(x.shape) != expected_shape:
            raise ValueError(
                f"Expected x shape {expected_shape}, got {tuple(x.shape)}"
            )
        self._validate_running_stats("running_mean", running_mean, x.device)
        self._validate_running_stats("running_var", running_var, x.device)

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
        x = x.contiguous()
        x_2d = x.reshape(self.M, self.D)

        if self.D_padded != self.D:
            x_2d = F.pad(x_2d, (0, self.D_padded - self.D))
        if self.M_padded != self.M:
            # Pad along dim 0 (M); padded rows are dropped after the kernel.
            x_2d = F.pad(x_2d, (0, 0, 0, self.M_padded - self.M))

        y_2d = self.kernel(x_2d)

        if self.M_padded != self.M:
            y_2d = y_2d[:self.M, :]
        if self.D_padded != self.D:
            y_2d = y_2d[:, :self.D]

        return y_2d.reshape(orig_shape)
