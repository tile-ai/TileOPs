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
        eps: float = 1e-5,
        use_input_stats: bool = True,
        momentum: float = 0.1,
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
        # Affine-identity tensors are reused across forward calls when the
        # caller passes weight=None / bias=None. Cached on the op instance
        # and invalidated on (dtype, device) change of the input.
        self._cached_unit_weight: Optional[torch.Tensor] = None
        self._cached_zero_bias: Optional[torch.Tensor] = None
        self._cached_affine_key: Optional[tuple] = None

    def _get_affine_identity(
        self, dtype: torch.dtype, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached unit_weight / zero_bias for the given (dtype, device)."""
        key = (dtype, device)
        if self._cached_affine_key != key:
            self._cached_unit_weight = torch.ones(
                self.D_padded, dtype=dtype, device=device,
            )
            self._cached_zero_bias = torch.zeros(
                self.D_padded, dtype=dtype, device=device,
            )
            self._cached_affine_key = key
        return self._cached_unit_weight, self._cached_zero_bias

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
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply instance normalization.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.
            weight: Optional affine scale of shape ``(C,)`` on CUDA. When
                ``None``, the affine scale defaults to all-ones (no scaling).
            bias: Optional affine shift of shape ``(C,)`` on CUDA. When
                ``None``, the affine shift defaults to all-zeros (no shift).

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch,
                or shapes are incompatible with the configured dimensions.
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.device != self._kernel_device:
            raise ValueError(
                f"Device mismatch: op was constructed for {self._kernel_device} "
                f"but x is on {x.device}. Construct a separate op instance per "
                f"CUDA device."
            )
        if x.dtype != self.dtype:
            raise ValueError(
                f"Expected x.dtype {self.dtype}, got {x.dtype}"
            )
        if weight is not None:
            if not weight.is_cuda:
                raise ValueError("weight must be a CUDA tensor")
            if weight.device != x.device:
                raise ValueError(
                    f"Expected weight on {x.device}, got {weight.device}"
                )
            if weight.dtype != self.dtype:
                raise ValueError(
                    f"Expected weight.dtype {self.dtype}, got {weight.dtype}"
                )
            if weight.ndim != 1 or weight.shape[0] != self.C:
                raise ValueError(
                    f"Expected weight shape ({self.C},), got {weight.shape}"
                )
        if bias is not None:
            if not bias.is_cuda:
                raise ValueError("bias must be a CUDA tensor")
            if bias.device != x.device:
                raise ValueError(
                    f"Expected bias on {x.device}, got {bias.device}"
                )
            if bias.dtype != self.dtype:
                raise ValueError(
                    f"Expected bias.dtype {self.dtype}, got {bias.dtype}"
                )
            if bias.ndim != 1 or bias.shape[0] != self.C:
                raise ValueError(
                    f"Expected bias shape ({self.C},), got {bias.shape}"
                )

        orig_shape = x.shape
        x = x.contiguous()

        # Reshape: (N, C, *spatial) -> (N*C, spatial_size)
        x_2d = x.reshape(self.M, self.D)

        # Unit weight and zero bias for the kernel (affine applied after).
        # Reuse cached identity tensors only on the affine-free path
        # (weight is None or bias is None); the user-supplied path allocates
        # fresh tensors to preserve byte-identical behavior.
        if weight is None or bias is None:
            unit_weight, zero_bias = self._get_affine_identity(
                x.dtype, x.device,
            )
        else:
            unit_weight = torch.ones(
                self.D_padded, dtype=x.dtype, device=x.device,
            )
            zero_bias = torch.zeros(
                self.D_padded, dtype=x.dtype, device=x.device,
            )

        # Pad to alignment
        if self.D_padded != self.D:
            x_2d = F.pad(x_2d, (0, self.D_padded - self.D))

        # Run kernel: produces (x - mean) / sqrt(var + eps)
        y_2d = self.kernel(x_2d, unit_weight, zero_bias)

        # Trim padding
        if self.D_padded != self.D:
            y_2d = y_2d[:, :self.D]

        # Reshape back: (N*C, spatial_size) -> (N, C, *spatial)
        y = y_2d.reshape(orig_shape)

        # Apply per-channel affine: y = y * weight + bias when supplied.
        # Both args default to identity (no-op) when None. The combined
        # expression matches the user-supplied path bit-for-bit.
        affine_shape = [1, self.C] + [1] * len(self.spatial)
        if weight is not None and bias is not None:
            y = y * weight.reshape(affine_shape) + bias.reshape(affine_shape)
        elif weight is not None:
            y = y * weight.reshape(affine_shape)
        elif bias is not None:
            y = y + bias.reshape(affine_shape)

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
            ``True`` (the default), per-instance statistics are computed
            from the input. When ``False`` (the running-stats / eval-mode
            path), the supplied ``running_mean`` and ``running_var``
            tensors are used as the normalization statistics, broadcast
            across the batch and spatial dimensions.
        momentum: Mirrors ``torch.nn.functional.instance_norm``. Stored on
            the op instance for API parity with PyTorch but unused at the
            op layer (running-stat updates are the caller's responsibility).
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
        eps: float = 1e-5,
        use_input_stats: bool = True,
        momentum: float = 0.1,
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
        # Running statistics for the use_input_stats=False path. Bound via
        # set_running_stats(); left None on the per-instance-stats path.
        self._running_mean: Optional[torch.Tensor] = None
        self._running_var: Optional[torch.Tensor] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"instance_norm_no_affine": InstanceNormNoAffineKernel}

    def eval_roofline(self) -> tuple[int, int]:
        elem_bytes = self.dtype.itemsize
        return (
            3 * self.N * self.C * self.spatial_size,
            2 * self.N * self.C * self.spatial_size * elem_bytes,
        )

    def _validate_dtypes(self, x: torch.Tensor) -> None:
        """Validate ``x.dtype`` against the manifest dtype union.

        Manifest declares ``x.dtype`` as ``float32 | float16 | bfloat16``
        and the configured op dtype must match the input.

        Args:
            x: Input tensor.

        Raises:
            ValueError: If ``x.dtype`` is outside the supported union or
                does not match ``self.dtype``.
        """
        allowed = (torch.float32, torch.float16, torch.bfloat16)
        if x.dtype not in allowed:
            raise ValueError(
                f"x.dtype must be one of {allowed}, got {x.dtype}"
            )
        if x.dtype != self.dtype:
            raise ValueError(
                f"Expected x.dtype {self.dtype}, got {x.dtype}"
            )

    def set_running_stats(
        self,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> None:
        """Bind the running statistics consumed on the ``use_input_stats=False`` path.

        Mirrors how ``torch.nn.InstanceNorm*`` carries ``running_mean`` /
        ``running_var`` as module buffers: the op stores references to the
        caller's tensors and reads them on each forward. The op does not
        update the buffers; the caller is responsible for the momentum-
        based update before calling forward.

        Args:
            running_mean: Channel-wise running mean of shape ``(C,)`` on
                the kernel device, with dtype matching the op.
            running_var: Channel-wise running variance of shape ``(C,)`` on
                the kernel device, with dtype matching the op.

        Raises:
            ValueError: On dtype, shape, or device mismatch.
        """
        for name, t in (("running_mean", running_mean),
                        ("running_var", running_var)):
            if not t.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            if t.device != self._kernel_device:
                raise ValueError(
                    f"Expected {name} on {self._kernel_device}, got {t.device}"
                )
            if t.dtype != self.dtype:
                raise ValueError(
                    f"Expected {name}.dtype {self.dtype}, got {t.dtype}"
                )
            if t.ndim != 1 or t.shape[0] != self.C:
                raise ValueError(
                    f"Expected {name} shape ({self.C},), got {tuple(t.shape)}"
                )
        self._running_mean = running_mean
        self._running_var = running_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply instance normalization without affine.

        On the ``use_input_stats=True`` path, per-instance statistics are
        computed from the input. On the ``use_input_stats=False`` path,
        the running statistics previously bound via
        :meth:`set_running_stats` are broadcast across the batch and
        spatial dimensions and used as the normalization statistics,
        matching ``torch.nn.functional.instance_norm`` with
        ``use_input_stats=False``.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch, or
                shapes are incompatible with the configured dimensions, or
                if ``use_input_stats=False`` and running stats have not
                been bound.
        """
        self._validate_dtypes(x)
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

        if not self.use_input_stats:
            if self._running_mean is None or self._running_var is None:
                raise ValueError(
                    "use_input_stats=False requires running_mean and running_var "
                    "to be bound via set_running_stats() before forward()."
                )
            return self._forward_running_stats(
                x, self._running_mean, self._running_var,
            )

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

    def _forward_running_stats(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize using supplied per-channel running statistics.

        Pure elementwise: ``y = (x - rm[None, :, None...]) / sqrt(rv + eps)``.
        Math is promoted to fp32 to keep fp16 / bf16 numerics aligned with
        PyTorch's reference path; the result is cast back to ``x.dtype``.
        """
        broadcast_shape = [1, self.C] + [1] * len(self.spatial)
        rm = running_mean.float().reshape(broadcast_shape)
        rv = running_var.float().reshape(broadcast_shape)
        rstd = torch.rsqrt(rv + self.eps)
        y = (x.float() - rm) * rstd
        return y.to(x.dtype)
