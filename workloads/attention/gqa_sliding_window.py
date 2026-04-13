from typing import Tuple

import torch

from workloads.workload_base import WorkloadBase


class GqaSlidingWindowFwdTest(WorkloadBase):

    def __init__(
        self,
        batch: int,
        seq: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        wl: int,
        wr: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.seq = seq
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.wl = wl
        self.wr = wr
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(self.batch, self.seq, self.heads,    self.dim,
                        dtype=self.dtype, device="cuda") * 0.1
        k = torch.randn(self.batch, self.seq, self.heads_kv, self.dim,
                        dtype=self.dtype, device="cuda") * 0.1
        v = torch.randn(self.batch, self.seq, self.heads_kv, self.dim,
                        dtype=self.dtype, device="cuda") * 0.1
        return q, k, v
