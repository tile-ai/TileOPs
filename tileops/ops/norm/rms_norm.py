import torch

from tileops.kernels.norm import RMSNormKernel

from .norm_base import RowNormOp

__all__ = ["RMSNormFwdOp"]


class RMSNormFwdOp(RowNormOp):
    """Standalone RMS Norm operator.

    y = x * rsqrt(mean(x^2, dim) + eps) * weight
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
