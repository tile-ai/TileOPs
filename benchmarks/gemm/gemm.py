from typing import Any, Tuple

import torch

from benchmarks.benchmark import Benchmark
from top.ops import GemmOp


class GemmBenchmark(Benchmark):

    op_type = GemmOp

    def __init__(self,
                 m: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype,
                 trans_a: bool = False,
                 trans_b: bool = False):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

    @property
    def total_flops(self) -> float:
        return 2.0 * self.m * self.n * self.k

    @property
    def total_memory(self) -> int:
        return (self.m * self.k + self.k * self.n + self.m * self.n) * self.dtype.itemsize

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        A = torch.randn(self.m, self.k, device='cuda', dtype=self.dtype)
        B = torch.randn(self.k, self.n, device='cuda', dtype=self.dtype)
        return A, B

    def ref_program(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            A = A.T
        if self.trans_b:
            B = B.T
        return torch.matmul(A, B)


class MatMulBenchmark(Benchmark):

    def __init__(self, m: int, n: int, k: int, dtype: torch.dtype, grad: bool = True):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.grad = grad

    @property
    def total_flops(self) -> float:
        return 6.0 * self.m * self.n * self.k

    @property
    def total_memory(self) -> int:
        return 3 * (self.m * self.k + self.k * self.n + self.m * self.n) * self.dtype.itemsize

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        A = torch.randn(self.m, self.k, device='cuda', dtype=self.dtype, requires_grad=self.grad)
        B = torch.randn(self.k, self.n, device='cuda', dtype=self.dtype, requires_grad=self.grad)
        return A, B

    def ref_program(self, A: torch.Tensor, B: torch.Tensor) -> Any:
        output = torch.matmul(A, B)
        if not self.grad:
            return output
        else:
            loss = output.sum()
            loss.backward()
        return output, A.grad, B.grad
