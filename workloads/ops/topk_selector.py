from typing import Tuple

import torch

from workloads.base import WorkloadBase


class TopkSelectorTest(WorkloadBase):

    def __init__(self, batch: int, seq_len: int, seq_len_kv: int, kv_group: int, topk: int,
                 in_dtype: torch.dtype, out_dtype: torch.dtype):
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index_score = torch.randn(
            self.batch,
            self.seq_len,
            self.seq_len_kv,
            self.kv_group,
            dtype=self.in_dtype,
            device="cuda")
        starts = torch.zeros(self.batch, self.seq_len, dtype=self.out_dtype, device="cuda")
        ends = torch.ones(self.batch, self.seq_len, dtype=self.out_dtype,
                          device="cuda") * self.seq_len_kv
        return index_score, starts, ends

    def ref_program(self, index_score: torch.Tensor, starts: torch.Tensor,
                    ends: torch.Tensor) -> torch.Tensor:
        # index_score: (batch, seq_len, seq_len_kv, kv_group); topk over seq_len_kv (dim=2)
        indexes_ref = torch.topk(index_score, self.topk, dim=2)[1]
        # Match kernel/output layout: (batch, seq_len, kv_group, topk)
        return indexes_ref.permute(0, 1, 3, 2)
