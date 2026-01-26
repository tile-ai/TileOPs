from typing import Dict, Optional

import torch

from top.kernels.deepseek_mla import Fp8QuantKernel
from top.kernels.kernel import Kernel

from .op import Op

__all__ = ["Fp8QuantOp"]


class Fp8QuantOp(Op):

    def __init__(self,
                 seq_len_kv,
                 index_dim,
                 in_dtype,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.seq_len_kv = seq_len_kv
        self.index_dim = index_dim
        self.in_dtype = in_dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["Fp8QuantKernel"](self.seq_len_kv, self.index_dim,
                                                        self.in_dtype, self.tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"Fp8QuantKernel": Fp8QuantKernel}

    def forward(self, input_tensor) -> tuple(torch.Tensor, torch.Tensor):
        return self.kernel(input_tensor)
