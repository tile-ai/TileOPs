import torch

from workloads.workload_base import WorkloadBase


class MhaDecodeTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                 dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(
            self.batch, self.seq_len_q, self.heads, self.dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len_kv, self.heads, self.dim, device='cuda', dtype=self.dtype)
        V = torch.randn(
            self.batch, self.seq_len_kv, self.heads, self.dim, device='cuda', dtype=self.dtype)
        return Q, K, V
