from typing import Dict, Optional

import torch

from top.kernels.gemv import GemvKernel
from top.kernels.kernel import Kernel

from .op import Op

__all__ = ['GemvOp']


class GemvOp(Op):

    def __init__(self,
                 n: int,
                 k: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.N = n
        self.K = k

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gemv_kernel"](n, k, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gemv_kernel": GemvKernel}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b)
