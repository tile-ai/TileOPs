
from typing import Tuple

import torch

from workloads.workload_base import WorkloadBase


class MhaDecodePagedTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int,
                 page_size: int, is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
            self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = self.seqlen_kv // self.page_size
        real_seqlen_kv = torch.ones(
            (self.batch,), dtype=torch.int32, device="cuda") * self.seqlen_kv
        q = torch.randn(
            self.batch, self.seqlen_q, self.heads, self.dim, device="cuda", dtype=self.dtype)
        k = torch.randn(self.seqlen_kv, self.heads, self.dim, device="cuda", dtype=self.dtype)
        v = torch.randn(self.seqlen_kv, self.heads, self.dim, device="cuda", dtype=self.dtype)
        # Identity block_table: logical page i -> physical page i (contiguous layout)
        block_table = torch.arange(
            num_pages, dtype=torch.int32, device="cuda").unsqueeze(0).expand(self.batch, -1)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        block_table = block_table.contiguous()
        real_seqlen_kv = real_seqlen_kv.contiguous()

        return q, k, v, real_seqlen_kv, block_table
