from typing import Dict, Optional

import torch

from top.kernels.deepseek_nsa.mean_pooling_fwd import MeanPoolingFwdKernel
from top.kernels.kernel import Kernel
from top.ops.op import Op

__all__ = ["MeanPoolingForwardOp"]


class MeanPoolingForwardOp(Op):

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        heads: int,
        dim: int,
        chunk_size: int,
        chunks_per_bacth: int,
        seq_num: int,
        use_offsets: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.heads = heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunks_per_bacth = chunks_per_bacth
        self.seq_num = seq_num
        self.use_offsets = use_offsets
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.tune = tune

        self.dispatch_kernel(kernel_map)

        kernel_args = {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "heads": self.heads,
            "dim": self.dim,
            "chunk_size": self.chunk_size,
            "chunks_per_bacth": self.chunks_per_bacth,
            "seq_num": self.seq_num,
            "use_offsets": self.use_offsets,
            "dtype": self.dtype,
            "accum_dtype": self.accum_dtype,
            "tune": self.tune,
        }
        self.kernel = self.kernel_map["mean_pooling_fwd_kernel"](**kernel_args)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mean_pooling_fwd_kernel": MeanPoolingFwdKernel}

    def forward(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.kernel(x, offsets, indices=indices)
