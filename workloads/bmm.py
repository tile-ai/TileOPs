import torch

from workloads.workload_base import WorkloadBase


class BmmTest(WorkloadBase):
    """Workload for batched matmul: a=[B,M,K], b=[B,K,N] -> d=[B,M,N]."""

    def __init__(self, batch: int, m: int, n: int, k: int, dtype: torch.dtype):
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.batch, self.m, self.k, device="cuda", dtype=self.dtype)
        b = torch.randn(self.batch, self.k, self.n, device="cuda", dtype=self.dtype)
        return a, b
