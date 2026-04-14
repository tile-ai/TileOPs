from typing import Optional, Tuple

import torch

from workloads.workload_base import WorkloadBase


class MeanPoolingTest(WorkloadBase):

    def __init__(self, batch_size: int, seq_len: int, heads: int, dim: int, chunk_size: int,
                 chunks_per_bacth: int, seq_num: int, use_offsets: int,
                 dtype: torch.dtype, accum_dtype: torch.dtype,
                 offsets: Optional[torch.Tensor] = None,
                 indices: Optional[torch.Tensor] = None):
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
        self.offsets = offsets
        self.indices = indices

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(
            self.batch_size, self.seq_len, self.heads, self.dim,
            device='cuda', dtype=self.dtype)
        return x, self.offsets, self.indices
