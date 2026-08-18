"""GroupNorm forward operator.

Wraps GroupNormKernel in the standard TileOPs Op interface.

User-facing API mirrors torch.nn.functional.group_norm:

    op = GroupNormFwdOp(num_groups=groups)
    y = op(x, weight, bias)   # affine
    y = op(x)                 # torch.nn.GroupNorm(affine=False)

Input tensors accept shape (N, C, *spatial); the op reshapes to
(N*num_groups, D) internally where D = (C/num_groups) * spatial_size.
"""

import math
from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import GroupNormKernel, GroupNormNoAffineKernel

from ..op_base import Op

__all__ = ["GroupNormFwdOp"]


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
        The per-channel affine is applied inside the kernel, so the op does
        no post-kernel arithmetic.

    ``weight`` and ``bias`` are one switch: pass both for the affine form,
    pass neither for ``torch.nn.GroupNorm(affine=False)``. Passing one alone
    is an error — the manifest states the same in ``shape_rules``.

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
        self.dispatch_kernel(kernel_map)
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[
            tuple[int, int, int, torch.dtype, bool]
        ] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "group_norm": GroupNormKernel,
            "group_norm_no_affine": GroupNormNoAffineKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "GroupNormFwdOp.eval_roofline() requires a prior forward() call"
            )
        N, C, spatial_size, dtype, affine = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            (5 if affine else 3) * N * C * spatial_size,
            (2 * N * C * spatial_size + (2 * C if affine else 0)) * elem_bytes,
        )

    def _resolve_spec(
        self, x: torch.Tensor
    ) -> Tuple[int, int, Tuple[int, ...], int, int, int, int, torch.dtype]:
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
        return N, C, spatial, spatial_size, cpg, D, M, x.dtype

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
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.spatial_size = spatial_size
        self.D = D
        self.M = M
        self.dtype = dtype
        self._last_roofline_spec = (N, C, spatial_size, dtype, affine)

    def _get_kernel(
        self,
        M: int,
        D: int,
        cpg: int,
        dtype: torch.dtype,
        device_index: Optional[int],
        affine: bool,
    ) -> Kernel:
        # Presence of weight/bias picks the implementation; the two kernels
        # take different constructor arguments, so each keys its own cache.
        if affine:
            key = (M, D, cpg, dtype, device_index, self.eps, self.tune)
            kernel = self.get_or_build_kernel(
                "group_norm",
                key=key,
                build=lambda: self.kernel_map["group_norm"](
                    M, D, self.eps, dtype, self.num_groups, cpg,
                    tune=self.tune,
                ),
            )
        else:
            key = (M, D, dtype, device_index, self.eps, self.tune)
            kernel = self.get_or_build_kernel(
                "group_norm_no_affine",
                key=key,
                build=lambda: self.kernel_map["group_norm_no_affine"](
                    M, D, self.eps, dtype, tune=self.tune,
                ),
            )
        self.kernel = kernel
        return kernel

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply group normalization.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.
            weight: Affine scale of shape ``(C,)`` on CUDA, or ``None``.
            bias: Affine shift of shape ``(C,)`` on CUDA, or ``None``.
                ``weight`` and ``bias`` are one switch: give both or neither.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If only one of ``weight`` / ``bias`` is given, if a
                tensor is not on CUDA, if dtypes mismatch, or if shapes are
                incompatible with the configured dimensions.
        """
        (
            N,
            C,
            spatial,
            spatial_size,
            cpg,
            D,
            M,
            dtype,
        ) = self._resolve_spec(x)
        # weight and bias are a single affine switch.
        if (weight is None) != (bias is None):
            given, missing = (
                ("weight", "bias") if bias is None else ("bias", "weight")
            )
            raise ValueError(
                f"weight and bias are one switch: got {given} without "
                f"{missing}. Pass both for the affine form, or neither for "
                f"torch.nn.GroupNorm(affine=False)"
            )
        affine = weight is not None
        if affine:
            for name, t in (("weight", weight), ("bias", bias)):
                if not t.is_cuda:
                    raise ValueError(f"{name} must be a CUDA tensor")
                if t.device != x.device:
                    raise ValueError(
                        f"Expected {name} on {x.device}, got {t.device}"
                    )
                if t.dtype != dtype:
                    raise ValueError(
                        f"Expected {name}.dtype {dtype}, got {t.dtype}"
                    )
                if t.ndim != 1 or t.shape[0] != C:
                    raise ValueError(
                        f"Expected {name} shape ({C},), got {tuple(t.shape)}"
                    )

        self._bind_spec(N, C, spatial, spatial_size, D, M, dtype, affine)
        kernel = self._get_kernel(M, D, cpg, dtype, x.device.index, affine)
        orig_shape = x.shape
        x_2d = x.contiguous().reshape(M, D)

        # The affine kernel derives each element's channel from its position
        # in the row, so the per-channel affine is applied inside the kernel.
        y_2d = kernel(x_2d, weight, bias) if affine else kernel(x_2d)

        return y_2d.reshape(orig_shape)
