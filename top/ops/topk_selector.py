from typing import Dict, Optional

import torch

from top.kernels.deepseek_mla import TopkSelectorKernel
from top.kernels.kernel import Kernel

from .op import Op

__all__ = ["TopkSelectorOp"]

class TopkSelectorOp(Op):
    def __init__(self,
                 batch: int,
                seq_len: int,
                topk: int,
                in_dtype: str,
                out_dtype: str,
                kernel_map: Optional[Dict[str, Kernel]] = None,
                tune: bool = False) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype


        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["TopkSelectorKernel"](
            self.batch, self.seq_len, self.topk, self.in_dtype, self.out_dtype
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"TopkSelectorKernel": TopkSelectorKernel}

    def forward(self, index_score, starts, ends) -> torch.Tensor:

        return self.kernel(index_score, starts, ends)