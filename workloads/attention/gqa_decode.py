from typing import Tuple

import torch

from workloads.workload_base import WorkloadBase


class GroupedQueryAttentionDecodeTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                 dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(self.batch, self.heads, self.dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len_kv, self.heads_kv, self.dim, device='cuda', dtype=self.dtype)
        V = torch.randn(
            self.batch, self.seq_len_kv, self.heads_kv, self.dim, device='cuda', dtype=self.dtype)
        return Q, K, V
