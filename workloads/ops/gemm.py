from typing import Tuple

import torch

from workloads.base import WorkloadBase


class GemmTest(WorkloadBase):

    def __init__(self, m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool = False,
                 trans_b: bool = False):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        shape_a = (self.k, self.m) if self.trans_a else (self.m, self.k)
        a = torch.randn(*shape_a, device='cuda', dtype=self.dtype)
        shape_b = (self.n, self.k) if self.trans_b else (self.k, self.n)
        b = torch.randn(*shape_b, device='cuda', dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)
