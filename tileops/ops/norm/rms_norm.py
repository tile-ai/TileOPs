from typing import Dict, Optional, Sequence

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm import RMSNormKernel

from .norm_base import RowNormOp, normalized_shape_to_n

__all__ = ["RMSNormFwdOp"]

_DEFAULT_EPS = 1e-6


class RMSNormFwdOp(RowNormOp):
    """Standalone Root Mean Square (RMS) Norm operator.

    Computes ``y = x * rsqrt(mean(x ** 2, dim) + eps) * weight``.

    Mirrors :func:`torch.nn.functional.rms_norm`: ``normalized_shape`` is the
    trailing-axis shape tuple over which the reduction runs; ``eps=None``
    selects the dtype default.

    Args:
        normalized_shape: Trailing axes the reduction runs over (manifest
            ``params.normalized_shape``). Either this or legacy ``N`` must be
            set.
        eps: Epsilon for numerical stability (manifest ``params.eps``).
            ``None`` uses the implementation default (``1e-6``).
        dtype: Data type (float16 or bfloat16).
        N: Legacy single-axis reduction size; supplied when callers cannot
            yet pass a tuple. Mutually exclusive with ``normalized_shape``.
        dim: Reduction axis (default -1) when using legacy ``N`` form.
            When ``normalized_shape`` is set, the reduction always runs over
            the trailing axes.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        >>> op = RMSNormFwdOp(normalized_shape=(4096,), dtype=torch.float16)
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> w = torch.randn(4096, dtype=torch.float16, device="cuda")
        >>> y = op(x, w)  # shape: (1024, 4096)
    """

    _kernel_key = "rms_norm"
    _kernel_cls = RMSNormKernel

    def __init__(
        self,
        normalized_shape: Optional[Sequence[int]] = None,
        eps: Optional[float] = None,
        *,
        dtype: torch.dtype,
        N: Optional[int] = None,
        dim: int = -1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        n_resolved = normalized_shape_to_n(normalized_shape, n_fallback=N)
        self.normalized_shape = (
            tuple(int(d) for d in normalized_shape)
            if normalized_shape is not None else None
        )
        eps_resolved = _DEFAULT_EPS if eps is None else float(eps)
        super().__init__(
            N=n_resolved,
            dtype=dtype,
            dim=dim,
            eps=eps_resolved,
            kernel_map=kernel_map,
            tune=tune,
        )

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        dim_norm = self._validate_and_normalize_dim(x, weight)
        x, post_move_shape = self._flatten_to_2d(x, dim_norm)
        M = x.shape[0]
        if self._needs_pad:
            x = self._pad_row(x)
            weight = self._pad_vec(weight)
        y = self._get_kernel(M)(x, weight)
        self._last_roofline_mn = (M, self.N)
        return self._trim_and_unflatten(y, post_move_shape, dim_norm, ndim)
