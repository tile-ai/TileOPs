import torch

from tileops.kernels.norm import RmsNormKernel

from .base import RowNormOp

__all__ = ["RMSNormFwdOp"]


class RMSNormFwdOp(RowNormOp):
    """Standalone RMS Norm operator.

    y = x * rsqrt(mean(x^2, dim=-1) + eps) * weight
    """

    _kernel_key = "rms_norm"
    _kernel_cls = RmsNormKernel

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        self._validate_cuda_dtype("x", x)
        self._validate_hidden_dim(x)
        self._validate_cuda_dtype("weight", weight)
        self._validate_1d("weight", weight)
        orig_shape = x.shape
        x = self._flatten_and_check_M(x)
        if self._needs_pad:
            x = self._pad_row(x)
            weight = self._pad_vec(weight)
        return self._trim_and_reshape(self.kernel(x, weight), orig_shape)
