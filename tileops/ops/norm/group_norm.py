"""GroupNorm forward operator.

Wraps GroupNormKernel in the standard TileOPs Op interface.

User-facing API mirrors torch.nn.functional.group_norm:

    op = GroupNormFwdOp(num_groups=groups)
    y = op(x, weight, bias)

Input tensors accept shape (N, C, *spatial); the op reshapes to
(N*num_groups, D_padded) internally where
D = (C/num_groups) * spatial_size.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import GroupNormKernel, GroupNormNoAffineKernel

from ..op_base import Op
from .norm_base import ALIGNMENT, align_up

__all__ = ["GroupNormFwdOp", "GroupNormFwdOpNoAffine"]

# Largest candidate block_m in GroupNormNoAffineKernel.autotune_configs.
# The op pads M to a multiple of this value so the kernel's full-tile
# T.copy never crosses the M boundary regardless of the selected block_m.
_M_BLOCK_ALIGN = 16


class GroupNormFwdOp(Op):
    """Group Normalization forward operator.

    Computes group normalization over ``(C/num_groups, *spatial)`` slices:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot w + b

    where the mean and variance are computed per group over
    ``(C/num_groups, *spatial)`` elements.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary spatial dimensions (1-D, 2-D, 3-D+).
        Handles non-contiguous inputs via explicit ``contiguous()`` call.
        Hidden dimension is padded to 256-element alignment internally.

    Args:
        num_groups: Number of groups (manifest ``params.num_groups``).
            Must divide *C* evenly.
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        num_groups: int,
        eps: float = 1e-5,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N: Optional[int] = None
        self.C: Optional[int] = None
        self.spatial: Optional[Tuple[int, ...]] = None
        self.num_groups = num_groups
        self.dtype: Optional[torch.dtype] = None
        self.eps = eps
        self.tune = tune
        self.spatial_size: Optional[int] = None
        self.D: Optional[int] = None
        self.M: Optional[int] = None
        self.D_padded: Optional[int] = None
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self._constant_cache: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[tuple[int, int, int, torch.dtype]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"group_norm": GroupNormKernel}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "GroupNormFwdOp.eval_roofline() requires a prior forward() call"
            )
        N, C, spatial_size, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            5 * N * C * spatial_size,
            (2 * N * C * spatial_size + 2 * C) * elem_bytes,
        )

    def _resolve_spec(
        self, x: torch.Tensor
    ) -> Tuple[int, int, Tuple[int, ...], int, int, int, int, int, torch.dtype]:
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
        if C % self.num_groups != 0:
            raise ValueError(
                f"C={C} must be divisible by num_groups={self.num_groups}"
            )
        spatial_size = math.prod(spatial)
        cpg = C // self.num_groups
        D = cpg * spatial_size
        M = N * self.num_groups
        D_padded = align_up(D, ALIGNMENT)
        return N, C, spatial, spatial_size, cpg, D, M, D_padded, x.dtype

    def _bind_spec(
        self,
        N: int,
        C: int,
        spatial: Tuple[int, ...],
        spatial_size: int,
        D: int,
        M: int,
        D_padded: int,
        dtype: torch.dtype,
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.spatial_size = spatial_size
        self.D = D
        self.M = M
        self.D_padded = D_padded
        self.dtype = dtype
        self._last_roofline_spec = (N, C, spatial_size, dtype)

    def _get_kernel_and_constants(
        self,
        M: int,
        D: int,
        D_padded: int,
        dtype: torch.dtype,
        device: torch.device,
        device_index: Optional[int],
    ) -> Tuple[Kernel, torch.Tensor, torch.Tensor]:
        key = (M, D, D_padded, dtype, device_index, self.eps, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["group_norm"](
                M, D, self.eps, dtype, tune=self.tune,
            )
        if key not in self._constant_cache:
            self._constant_cache[key] = (
                torch.ones(D_padded, dtype=dtype, device=device),
                torch.zeros(D_padded, dtype=dtype, device=device),
            )
        kernel = self._kernel_cache[key]
        self.kernel = kernel
        unit_weight, zero_bias = self._constant_cache[key]
        return kernel, unit_weight, zero_bias

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Apply group normalization.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.
            weight: Affine scale of shape ``(C,)`` on CUDA. Required; the
                affine-free path is :class:`GroupNormFwdOpNoAffine`.
            bias: Affine shift of shape ``(C,)`` on CUDA. Required; the
                affine-free path is :class:`GroupNormFwdOpNoAffine`.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If any tensor is not on CUDA, dtypes mismatch, or
                shapes are incompatible with the configured dimensions.
        """
        (
            N,
            C,
            spatial,
            spatial_size,
            cpg,
            D,
            M,
            D_padded,
            dtype,
        ) = self._resolve_spec(x)
        if not isinstance(weight, torch.Tensor):
            raise ValueError(
                "weight is required; use GroupNormFwdOpNoAffine for the "
                "affine-free path"
            )
        if not isinstance(bias, torch.Tensor):
            raise ValueError(
                "bias is required; use GroupNormFwdOpNoAffine for the "
                "affine-free path"
            )
        if not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if weight.device != x.device:
            raise ValueError(
                f"Expected weight on {x.device}, got {weight.device}"
            )
        if weight.dtype != dtype:
            raise ValueError(
                f"Expected weight.dtype {dtype}, got {weight.dtype}"
            )
        if weight.ndim != 1 or weight.shape[0] != C:
            raise ValueError(
                f"Expected weight shape ({C},), got {weight.shape}"
            )
        if not bias.is_cuda:
            raise ValueError("bias must be a CUDA tensor")
        if bias.device != x.device:
            raise ValueError(
                f"Expected bias on {x.device}, got {bias.device}"
            )
        if bias.dtype != dtype:
            raise ValueError(
                f"Expected bias.dtype {dtype}, got {bias.dtype}"
            )
        if bias.ndim != 1 or bias.shape[0] != C:
            raise ValueError(
                f"Expected bias shape ({C},), got {bias.shape}"
            )

        self._bind_spec(N, C, spatial, spatial_size, D, M, D_padded, dtype)
        kernel, unit_weight, zero_bias = self._get_kernel_and_constants(
            M, D, D_padded, dtype, x.device, x.device.index,
        )
        orig_shape = x.shape
        x = x.contiguous()
        x_reshaped = x.reshape(
            N, self.num_groups, cpg, *spatial,
        )
        x_2d = x_reshaped.reshape(M, D)

        if D_padded != D:
            x_2d = F.pad(x_2d, (0, D_padded - D))

        # Kernel broadcasts 1D weight/bias row-wise; GroupNorm's per-group
        # affine doesn't fit that layout, so run the kernel with identity
        # (unit/zero) and apply per-channel affine after.
        y_2d = kernel(x_2d, unit_weight, zero_bias)

        if D_padded != D:
            y_2d = y_2d[:, :D]

        y = y_2d.reshape(N, self.num_groups, cpg, *spatial)
        y = y.reshape(orig_shape)

        affine_shape = (1, C) + (1,) * len(spatial)
        y = y * weight.reshape(affine_shape) + bias.reshape(affine_shape)
        return y


class GroupNormFwdOpNoAffine(Op):
    """Group Normalization forward without affine scale/shift.

    Computes group normalization without the trailing weight/bias affine:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}

    where the mean and variance are computed per group over
    ``(C/num_groups, *spatial)`` elements. Mirrors
    ``torch.nn.functional.group_norm(x, num_groups, weight=None, bias=None)``
    and ``torch.nn.GroupNorm(affine=False)``.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Args:
        num_groups: Number of groups (manifest ``params.num_groups``).
            Must divide *C* evenly.
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        num_groups: int,
        eps: float = 1e-5,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.N: Optional[int] = None
        self.C: Optional[int] = None
        self.spatial: Optional[Tuple[int, ...]] = None
        self.num_groups = num_groups
        self.dtype: Optional[torch.dtype] = None
        self.eps = eps
        self.tune = tune
        self.spatial_size: Optional[int] = None
        self.D: Optional[int] = None
        self.M: Optional[int] = None
        self.D_padded: Optional[int] = None
        self.M_padded: Optional[int] = None
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[tuple[int, int, int, torch.dtype]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"group_norm_no_affine": GroupNormNoAffineKernel}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "GroupNormFwdOpNoAffine.eval_roofline() requires a prior forward() call"
            )
        N, C, spatial_size, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            3 * N * C * spatial_size,
            2 * N * C * spatial_size * elem_bytes,
        )

    def _resolve_spec(
        self, x: torch.Tensor
    ) -> Tuple[int, int, Tuple[int, ...], int, int, int, int, int, int, torch.dtype]:
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
        if C % self.num_groups != 0:
            raise ValueError(
                f"C={C} must be divisible by num_groups={self.num_groups}"
            )
        spatial_size = math.prod(spatial)
        cpg = C // self.num_groups
        D = cpg * spatial_size
        M = N * self.num_groups
        D_padded = align_up(D, ALIGNMENT)
        M_padded = align_up(M, _M_BLOCK_ALIGN)
        return N, C, spatial, spatial_size, cpg, D, M, D_padded, M_padded, x.dtype

    def _bind_spec(
        self,
        N: int,
        C: int,
        spatial: Tuple[int, ...],
        spatial_size: int,
        D: int,
        M: int,
        D_padded: int,
        M_padded: int,
        dtype: torch.dtype,
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.spatial_size = spatial_size
        self.D = D
        self.M = M
        self.D_padded = D_padded
        self.M_padded = M_padded
        self.dtype = dtype
        self._last_roofline_spec = (N, C, spatial_size, dtype)

    def _get_kernel(
        self,
        M_padded: int,
        D: int,
        dtype: torch.dtype,
        device_index: Optional[int],
    ) -> Kernel:
        key = (M_padded, D, dtype, device_index, self.eps, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["group_norm_no_affine"](
                M_padded, D, self.eps, dtype, tune=self.tune,
            )
        kernel = self._kernel_cache[key]
        self.kernel = kernel
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply group normalization without affine.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If *x* is not a CUDA tensor, lives on a different
                device than the op was constructed for, or its dtype does
                not match the configured dtype.
        """
        (
            N,
            C,
            spatial,
            spatial_size,
            cpg,
            D,
            M,
            D_padded,
            M_padded,
            dtype,
        ) = self._resolve_spec(x)
        self._bind_spec(N, C, spatial, spatial_size, D, M, D_padded, M_padded, dtype)
        kernel = self._get_kernel(M_padded, D, dtype, x.device.index)

        orig_shape = x.shape
        x = x.contiguous()
        x_reshaped = x.reshape(N, self.num_groups, cpg, *spatial)
        x_2d = x_reshaped.reshape(M, D)

        if D_padded != D:
            x_2d = F.pad(x_2d, (0, D_padded - D))
        if M_padded != M:
            # Pad along dim 0 (M); padded rows are dropped after the kernel.
            x_2d = F.pad(x_2d, (0, 0, 0, M_padded - M))

        y_2d = kernel(x_2d)

        if M_padded != M:
            y_2d = y_2d[:M, :]
        if D_padded != D:
            y_2d = y_2d[:, :D]

        y = y_2d.reshape(N, self.num_groups, cpg, *spatial)
        return y.reshape(orig_shape)
