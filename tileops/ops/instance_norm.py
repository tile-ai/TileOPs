from typing import Dict, Optional

import torch

from tileops.kernels.instance_norm import InstanceNormKernel
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

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["instance_norm_kernel"](
            num_channels=self.num_channels,
            eps=self.eps,
            affine=self.affine,
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
        return self.kernel(x, weight, bias)
