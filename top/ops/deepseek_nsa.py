from typing import Dict, Optional

import torch

from top.kernels.deepseek_nsa.mean_pooling_fwd import MeanPoolingFwdKernel
from top.kernels.kernel import Kernel
from top.ops.op import Op

__all__ = ["MeanPoolingForwardOp"]


class MeanPoolingForwardOp(Op):

    def __init__(self,
                 batch_size: int,
                 total_seqlen: int,
                 total_chunks: int,
                 heads: int,
                 dim: int,
                 chunk_size: int,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch_size = batch_size
        self.total_seqlen = total_seqlen
        self.total_chunks = total_chunks
        self.heads = heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.tune = tune

        self.dispatch_kernel(kernel_map)

        kernel_args = {
            "batch_size": self.batch_size,
            "total_seqlen": self.total_seqlen,
            "total_chunks": self.total_chunks,
            "heads": self.heads,
            "dim": self.dim,
            "chunk_size": self.chunk_size,
            "tune": self.tune,
        }
        self.kernel = self.kernel_map["mean_pooling_fwd_kernel"](**kernel_args)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mean_pooling_fwd_kernel": MeanPoolingFwdKernel}

    def forward(
        self,
        x_unpad: torch.Tensor,
        cu_seqlens: torch.Tensor,
        chunk_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.kernel(x_unpad, cu_seqlens, chunk_indices)
