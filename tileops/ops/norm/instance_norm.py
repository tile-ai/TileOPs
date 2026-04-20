"""InstanceNorm forward operator.

InstanceNorm is a special case of GroupNorm where G = C (each channel
is its own group). This operator delegates to GroupNormKernel with G=C.

User-facing API mirrors torch.nn.functional.instance_norm:

    op = InstanceNormFwdOp(C=channels, dtype=dtype)
    y = op(x, weight, bias)

Input tensors accept shape ``(N, C, *spatial)``.  ``N`` and ``*spatial``
are derived at forward time; kernels are cached by
``(M=N*C, D=prod(spatial))``.
"""

from math import prod
from typing import Dict, Hashable, Optional, Tuple

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import GroupNormKernel

from ..op_base import Op

__all__ = ["InstanceNormFwdOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class InstanceNormFwdOp(Op):
    """Instance Normalization forward operator.

    Computes instance normalization over spatial dimensions for each
    ``(batch, channel)`` independently:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot w + b

    where the mean and variance are computed over ``*spatial`` for each
    sample-channel pair. Equivalent to Group Normalization with ``G = C``.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary spatial dimensions (1-D, 2-D, 3-D+).
        Delegates to :class:`GroupNormKernel` with ``G = C``.
        Hidden dimension is padded to 256-element alignment internally.

    Args:
        C: Number of channels (committed at construction per manifest
            ``static_dims``; forward validates ``x.shape[1] == C``).
        dtype: Data type (``torch.float32``, ``torch.float16``, or
            ``torch.bfloat16``).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    _static_axes = frozenset({(0, 1)})

    def __init__(
        self,
        *,
        C: int,
        dtype: torch.dtype,
        eps: float = 1e-5,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.C = C
        self.dtype = dtype
        self.eps = eps
        self._tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[Hashable, Kernel] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"group_norm": GroupNormKernel}

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
        """Kernel cache key: (M, D) with M = N * C and D = prod(spatial).

        Matches the kernel construction projection in ``forward``: kernels are
        keyed by the same ``(M, D)`` tuple used at ``_get_or_create_kernel``,
        so input shapes that differ only in how spatial dims split (e.g.
        ``(N, C, H, W)`` vs ``(N, C, H*W)``) share one cached kernel.
        """
        x_shape = input_shapes[0]
        N = x_shape[0]
        spatial = x_shape[2:]
        D = prod(spatial) if spatial else 1
        M = N * self.C
        return (M, D)

    def _get_or_create_kernel(self, M: int, D: int) -> Kernel:
        key = (M, D)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self.kernel_map["group_norm"](
                M, D, self.eps, self.dtype, tune=self._tune,
            )
            self._kernel_cache[key] = kernel
        return kernel

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Apply instance normalization.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.
            weight: Affine scale of shape ``(C,)`` on CUDA.
            bias: Affine shift of shape ``(C,)`` on CUDA.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch,
                or shapes are incompatible with the configured dimensions.
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if not bias.is_cuda:
            raise ValueError("bias must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(
                f"Expected x.dtype {self.dtype}, got {x.dtype}"
            )
        if weight.dtype != self.dtype:
            raise ValueError(
                f"Expected weight.dtype {self.dtype}, got {weight.dtype}"
            )
        if bias.dtype != self.dtype:
            raise ValueError(
                f"Expected bias.dtype {self.dtype}, got {bias.dtype}"
            )
        if x.ndim < 3:
            raise ValueError(
                f"Expected x.ndim >= 3 (N, C, *spatial), got {x.ndim}"
            )
        # static_dims validation: x.shape[1] == C (committed at ctor).
        if x.shape[1] != self.C:
            raise ValueError(
                f"Expected channel dim {self.C}, got {x.shape[1]}"
            )
        if weight.ndim != 1 or weight.shape[0] != self.C:
            raise ValueError(
                f"Expected weight shape ({self.C},), got {weight.shape}"
            )
        if bias.ndim != 1 or bias.shape[0] != self.C:
            raise ValueError(
                f"Expected bias shape ({self.C},), got {bias.shape}"
            )

        orig_shape = x.shape
        N = orig_shape[0]
        spatial = orig_shape[2:]
        spatial_size = prod(spatial) if spatial else 1
        # For InstanceNorm (G=C): D = spatial_size, M = N * C
        D = spatial_size
        M = N * self.C
        D_padded = _align_up(D, ALIGNMENT)
        kernel = self._get_or_create_kernel(M, D)

        x = x.contiguous()

        # Reshape: (N, C, *spatial) -> (N*C, spatial_size)
        x_2d = x.reshape(M, D)

        # Unit weight and zero bias for the kernel (affine applied after)
        unit_weight = torch.ones(D_padded, dtype=self.dtype, device=x.device)
        zero_bias = torch.zeros(D_padded, dtype=self.dtype, device=x.device)

        # Pad to alignment
        if D_padded != D:
            x_2d = F.pad(x_2d, (0, D_padded - D))

        # Run kernel: produces (x - mean) / sqrt(var + eps)
        y_2d = kernel(x_2d, unit_weight, zero_bias)

        # Trim padding
        if D_padded != D:
            y_2d = y_2d[:, :D]

        # Reshape back: (N*C, spatial_size) -> (N, C, *spatial)
        y = y_2d.reshape(orig_shape)

        # Apply per-channel affine: y = y * weight + bias
        affine_shape = [1, self.C] + [1] * len(spatial)
        y = y * weight.reshape(affine_shape) + bias.reshape(affine_shape)

        return y
