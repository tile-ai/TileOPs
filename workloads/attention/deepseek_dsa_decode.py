from typing import Tuple

import torch

from workloads.workload_base import WorkloadBase


class DsaDecodeTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, seq_len: int, seq_len_kv: int, dim: int,
                 dim_tail: int, topk: int, stride_kv: int, heads_kv: int, q_start_index_s: int,
                 sm_scale: float = None, is_causal: bool = True,
                 dtype: torch.dtype = torch.float16) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_tail = dim_tail
        self.topk = topk
        self.stride_kv = stride_kv
        self.heads_kv = heads_kv
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.dtype = dtype
        self.q_start_index_s = q_start_index_s

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim + self.dim_tail,
            device='cuda',
            dtype=self.dtype)
        kv = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim + self.dim_tail,
            device='cuda',
            dtype=self.dtype)
        indices = torch.full((self.batch, self.seq_len, self.heads_kv, self.topk),
                             self.seq_len_kv,
                             dtype=torch.int32,
                             device='cuda')
        for b in range(self.batch):
            for t in range(self.seq_len):
                for h in range(self.heads_kv):
                    i_i = torch.randperm(
                        min(
                            max(1, ((t + int(self.q_start_index_s)) // self.stride_kv)),
                            self.seq_len_kv))[:self.topk]
                    indices[b, t, h, :len(i_i)] = i_i
        return q, kv, indices
