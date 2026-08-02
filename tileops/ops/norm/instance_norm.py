"""InstanceNorm forward operator.

Instance Normalization (IN) is a special case of Group Normalization (GN)
where ``num_groups = C`` (each channel is its own group). This operator
delegates to :class:`GroupNormKernel` with that grouping.

User-facing API mirrors :func:`torch.nn.functional.instance_norm`:

    op = InstanceNormFwdOp()
    y = op(x, weight, bias)

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

__all__ = ["InstanceNormFwdOp", "InstanceNormNoAffineFwdOp"]


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
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[tuple[int, int, int, torch.dtype]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"group_norm": GroupNormKernel}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "InstanceNormFwdOp.eval_roofline() requires a prior forward() call"
            )
        N, C, spatial_size, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            5 * N * C * spatial_size,
            (2 * N * C * spatial_size + 2 * C) * elem_bytes,
        )

    def _validate_dtypes(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        allowed = (torch.float32, torch.float16, torch.bfloat16)
        expected_dtype = x.dtype
        for name, t in (("x", x), ("weight", weight), ("bias", bias)):
            if t.dtype not in allowed:
                raise ValueError(f"{name}.dtype must be one of {allowed}, got {t.dtype}")
            if t.dtype != expected_dtype:
                raise ValueError(
                    f"Expected {name}.dtype == {expected_dtype}, "
                    f"got {t.dtype}"
                )

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
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.spatial_size = spatial_size
        self.D = D
        self.M = M
        self.dtype = dtype
        self._last_roofline_spec = (N, C, spatial_size, dtype)

    def _get_kernel(
        self,
        M: int,
        D: int,
        dtype: torch.dtype,
        device_index: Optional[int],
    ) -> Kernel:
        key = (M, D, dtype, device_index, self.eps, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["group_norm"](
                M, D, self.eps, dtype, tune=self.tune,
            )
        kernel = self._kernel_cache[key]
        self.kernel = kernel
        return kernel

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
                affine-free path is :class:`InstanceNormNoAffineFwdOp`.
            bias: Affine shift of shape ``(C,)`` on CUDA. Required; the
                affine-free path is :class:`InstanceNormNoAffineFwdOp`.

        Returns:
            Normalized tensor of the same shape as *x*.

        Raises:
            ValueError: If any tensor is not on CUDA, dtypes mismatch, or
                shapes are incompatible with the configured dimensions.
        """
        if not isinstance(weight, torch.Tensor):
            raise ValueError(
                "weight is required; use InstanceNormNoAffineFwdOp for the "
                "affine-free path"
            )
        if not isinstance(bias, torch.Tensor):
            raise ValueError(
                "bias is required; use InstanceNormNoAffineFwdOp for the "
                "affine-free path"
            )
        self._validate_dtypes(x, weight, bias)
        N, C, spatial, spatial_size, D, M, dtype = self._resolve_spec(x)
        if not weight.is_cuda:
            raise ValueError("weight must be a CUDA tensor")
        if weight.device != x.device:
            raise ValueError(
                f"Expected weight on {x.device}, got {weight.device}"
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
        if bias.ndim != 1 or bias.shape[0] != C:
            raise ValueError(
                f"Expected bias shape ({C},), got {bias.shape}"
            )

        self._bind_spec(N, C, spatial, spatial_size, D, M, dtype)
        kernel = self._get_kernel(M, D, dtype, x.device.index)
        orig_shape = x.shape
        x = x.contiguous()
        x_2d = x.reshape(M, D)

        # The kernel broadcasts its affine row-wise; the per-channel affine
        # is applied after, so normalize without an affine here.
        y_2d = kernel(x_2d)

        # Reshape back: (N*C, spatial_size) -> (N, C, *spatial)
        y = y_2d.reshape(orig_shape)

        affine_shape = (1, C) + (1,) * len(spatial)
        y = y * weight.reshape(affine_shape) + bias.reshape(affine_shape)
        return y


class InstanceNormNoAffineFwdOp(Op):
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
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[tuple[int, int, int, torch.dtype]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"instance_norm_no_affine": InstanceNormNoAffineKernel}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "InstanceNormNoAffineFwdOp.eval_roofline() requires a prior forward() call"
            )
        N, C, spatial_size, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            3 * N * C * spatial_size,
            2 * N * C * spatial_size * elem_bytes,
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
        if x.dtype not in allowed:
            raise ValueError(
                f"x.dtype must be one of {allowed}, got {x.dtype}"
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
        self, name: str, t: torch.Tensor, x_device: torch.device, C: int,
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
        if t.ndim != 1 or t.shape[0] != C:
            raise ValueError(
                f"Expected {name} shape ({C},), got {tuple(t.shape)}"
            )

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
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.spatial_size = spatial_size
        self.D = D
        self.M = M
        self._running_stats_broadcast_shape = [1, C] + [1] * len(spatial)
        self.dtype = dtype
        self._last_roofline_spec = (N, C, spatial_size, dtype)

    def _get_kernel(
        self,
        M: int,
        D: int,
        dtype: torch.dtype,
        device_index: Optional[int],
    ) -> Kernel:
        key = (M, D, dtype, device_index, self.eps, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["instance_norm_no_affine"](
                M, D, self.eps, dtype, tune=self.tune,
            )
        kernel = self._kernel_cache[key]
        self.kernel = kernel
        return kernel

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
        (
            N,
            C,
            spatial,
            spatial_size,
            D,
            M,
            dtype,
        ) = self._resolve_spec(x)
        self._validate_running_stats("running_mean", running_mean, x.device, C)
        self._validate_running_stats("running_var", running_var, x.device, C)
        self._bind_spec(N, C, spatial, spatial_size, D, M, dtype)

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
        x_2d = x.reshape(M, D)

        kernel = self._get_kernel(M, D, dtype, x.device.index)
        y_2d = kernel(x_2d)

        return y_2d.reshape(orig_shape)
