from typing import List, Tuple

import torch

from workloads.base import WorkloadBase


class GqaSlidingWindowVarlenFwdTest(WorkloadBase):

    def __init__(
        self,
        batch: int,
        seqlens_q: List[int],
        seqlens_k: List[int],
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        wl: int,
        wr: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.seqlens_q = seqlens_q
        self.seqlens_k = seqlens_k
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.wl = wl
        self.wr = wr
        self.dtype = dtype

    def gen_inputs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, int]:
        total_q = sum(self.seqlens_q)
        total_k = sum(self.seqlens_k)
        q = torch.randn(total_q, self.heads, self.dim,
                        dtype=self.dtype, device="cuda") * 0.1
        k = torch.randn(total_k, self.heads_kv, self.dim,
                        dtype=self.dtype, device="cuda") * 0.1
        v = torch.randn(total_k, self.heads_kv, self.dim,
                        dtype=self.dtype, device="cuda") * 0.1

        cu_seqlens_q = torch.tensor(
            [0] + list(torch.cumsum(
                torch.tensor(self.seqlens_q), 0).tolist()),
            dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.tensor(
            [0] + list(torch.cumsum(
                torch.tensor(self.seqlens_k), 0).tolist()),
            dtype=torch.int32, device="cuda")
        max_seqlen_q = max(self.seqlens_q)

        return q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q
