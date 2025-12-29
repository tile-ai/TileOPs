import torch
from .op import Op
from top.kernels.kernel import Kernel
from top.kernels.gemm import gemm_kernel
from typing import Optional, Dict

__all__ = ['GemmOp']


class GemmOp(Op):

    def __init__(self,
                 m: int,
                 n: int,
                 k: int,
                 trans_a=False,
                 trans_b=False,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.M = m
        self.N = n
        self.K = k

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gemm_kernel"](
            m, n, k, self.dtype, tune=tune, trans_a=trans_a, trans_b=trans_b)

    @property
    def default_kernel_map(self):
        return {"gemm_kernel": gemm_kernel}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.kernel(a, b)