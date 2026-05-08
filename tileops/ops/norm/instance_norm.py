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
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype
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
        if not use_input_stats:
            raise NotImplementedError(
                "use_input_stats=False (running-stats path) is out of scope"
            )
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

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"instance_norm_no_affine": InstanceNormNoAffineKernel}

    def eval_roofline(self) -> tuple[int, int]:
        elem_bytes = self.dtype.itemsize
        return (
            3 * self.N * self.C * self.spatial_size,
            2 * self.N * self.C * self.spatial_size * elem_bytes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply instance normalization without affine.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If *x* is not a CUDA tensor, lives on a different
                device than the op was constructed for, or its dtype does
                not match the configured dtype.
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
