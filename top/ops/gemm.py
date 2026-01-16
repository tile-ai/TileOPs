from typing import Dict, Optional

import torch

from top.kernels.gemm import GemmKernel
from top.kernels.kernel import Kernel

from .op import Op

__all__ = ['GemmOp']


class GemmOp(Op):

    def __init__(self,
                 m: int,
                 n: int,
                 k: int,
                 trans_a: bool = False,
                 trans_b: bool = False,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.M = m
        self.N = n
        self.K = k

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gemm_kernel"](
            m, n, k, self.dtype, tune=tune, trans_a=trans_a, trans_b=trans_b)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gemm_kernel": GemmKernel}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b)
