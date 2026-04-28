import torch

from tileops.kernels.norm import RMSNormKernel

from .norm_base import RowNormOp

__all__ = ["RMSNormFwdOp"]


class RMSNormFwdOp(RowNormOp):
    """Standalone Root Mean Square (RMS) Norm operator.

    Computes ``y = x * rsqrt(mean(x ** 2, dim) + eps) * weight``.

    Args:
        N: Reduction dimension size (statically committed at ctor;
            corresponds to manifest ``static_dims.N = "x.shape[dim]"``).
        dtype: Data type (float16 or bfloat16).
        dim: Reduction axis (default -1). Negative values are normalized
            at forward time.
        eps: Epsilon for numerical stability (default ``1e-6``).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        >>> op = RMSNormFwdOp(N=4096, dtype=torch.float16)
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> w = torch.randn(4096, dtype=torch.float16, device="cuda")
        >>> y = op(x, w)  # shape: (1024, 4096)
    """

    _kernel_key = "rms_norm"
    _kernel_cls = RMSNormKernel

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
