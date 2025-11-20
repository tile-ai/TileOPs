import torch
from .op import Op
from top.kernels import gemm_kernel, Kernel
from typing import Optional, Dict

__all__ = ['Gemm']


class Gemm(Op):

    def __init__(self,
                 M: int,
                 N: int,
                 K: int,
                 trans_A=False,
                 trans_B=False,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.M = M
        self.N = N
        self.K = K

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gemm_kernel"](M, N, K, self.dtype, tune=tune, trans_A=trans_A, trans_B=trans_B)

    @property
    def default_kernel_map(self):
        return {"gemm_kernel": gemm_kernel}

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.kernel(A, B)
