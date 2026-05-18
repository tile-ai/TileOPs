import torch

from workloads.workload_base import WorkloadBase


class GemmTest(WorkloadBase):

    def __init__(self, m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool = False,
                 trans_b: bool = False):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        shape_a = (self.k, self.m) if self.trans_a else (self.m, self.k)
        a = torch.randn(*shape_a, device='cuda', dtype=self.dtype)
        shape_b = (self.n, self.k) if self.trans_b else (self.k, self.n)
        b = torch.randn(*shape_b, device='cuda', dtype=self.dtype)
        return a, b
