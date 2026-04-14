
from typing import Tuple

import torch

from workloads.workload_base import WorkloadBase


class GroupedQueryAttentionDecodePagedTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, seqlen_kv: int, dim: int,
                 page_size: int, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.dtype = dtype

    def gen_inputs(
            self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = self.seqlen_kv // self.page_size
        real_seqlen_kv = torch.randint(
            self.page_size, self.seqlen_kv + 1, (self.batch,), dtype=torch.int32, device="cuda")
        real_seqlen_kv = (real_seqlen_kv // self.page_size) * self.page_size
        real_seqlen_kv[0] = min(real_seqlen_kv[0].item(), self.seqlen_kv)

        q = torch.randn(self.batch, self.heads, self.dim, dtype=self.dtype, device="cuda")
        k = torch.randn(self.seqlen_kv, self.heads_kv, self.dim, dtype=self.dtype, device="cuda")
        v = torch.randn(self.seqlen_kv, self.heads_kv, self.dim, dtype=self.dtype, device="cuda")
        block_table = torch.arange(
            num_pages, dtype=torch.int32, device="cuda").unsqueeze(0).expand(self.batch, -1)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        block_table = block_table.contiguous()
        real_seqlen_kv = real_seqlen_kv.contiguous()

        return q, k, v, real_seqlen_kv, block_table
