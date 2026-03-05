from typing import Dict, Optional

import torch

from tileops.kernels.norm import InstanceNormKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["InstanceNormOp"]


class InstanceNormOp(Op):

    def __init__(self,
                 num_channels: int,
                 eps: float = 1e-5,
                 affine: bool = True,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.dtype = dtype
        self._affine_ones: Optional[torch.Tensor] = None
        self._affine_zeros: Optional[torch.Tensor] = None

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["instance_norm_kernel"](
            num_channels=self.num_channels,
            spatial_size=1,  # actual spatial size is rebound in forward
            eps=self.eps,
            dtype=self.dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"instance_norm_kernel": InstanceNormKernel}

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError(f"instance_norm expects input ndim >= 3, got {x.ndim}.")
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"input channel mismatch: expected {self.num_channels}, got {x.shape[1]}.")
        if x.dtype != self.dtype:
            raise ValueError(f"input dtype mismatch: expected {self.dtype}, got {x.dtype}.")

        x_contig = x.contiguous()
        n, c = x_contig.shape[:2]
        spatial_size = 1
        for dim in x_contig.shape[2:]:
            spatial_size *= dim
        x_2d = x_contig.reshape(n * c, spatial_size)

        # Recreate kernel when spatial size changes.
        if spatial_size != self.kernel.spatial_size:
            self.kernel = self.kernel_map["instance_norm_kernel"](
                num_channels=self.num_channels,
                spatial_size=spatial_size,
                eps=self.eps,
                dtype=self.dtype,
                tune=False,
            )

        if self.affine:
            if weight is None or bias is None:
                raise ValueError("weight and bias are required when affine=True.")
            if weight.shape != (self.num_channels,) or bias.shape != (self.num_channels,):
                raise ValueError(
                    f"weight/bias shape must be ({self.num_channels},), got {weight.shape}/{bias.shape}.")
            gamma = weight
            beta = bias
        else:
            if self._affine_ones is None or self._affine_ones.device != x.device or self._affine_ones.dtype != self.dtype:
                self._affine_ones = torch.ones(
                    self.num_channels, device=x.device, dtype=self.dtype)
                self._affine_zeros = torch.zeros(
                    self.num_channels, device=x.device, dtype=self.dtype)
            gamma = self._affine_ones
            beta = self._affine_zeros

        y_2d = self.kernel(x_2d, gamma, beta)
        return y_2d.reshape_as(x_contig)
