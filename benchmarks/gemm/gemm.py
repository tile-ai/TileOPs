from typing import Tuple, Union

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
        if self.trans_a:
            a = torch.randn(self.k, self.m, device='cuda', dtype=self.dtype)
        else:
            a = torch.randn(self.m, self.k, device='cuda', dtype=self.dtype)

        if self.trans_b:
            b = torch.randn(self.n, self.k, device='cuda', dtype=self.dtype)
        else:
            b = torch.randn(self.k, self.n, device='cuda', dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)

    def baseline_profile(self, *inputs, warmup=100, rep=10, device="cuda:0") -> None:
        return super().baseline_profile(
            self.ref_program, *inputs, backend="torch", warmup=warmup, rep=rep, device=device)


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
        a = torch.randn(self.m, self.k, device='cuda', dtype=self.dtype, requires_grad=self.grad)
        b = torch.randn(self.k, self.n, device='cuda', dtype=self.dtype, requires_grad=self.grad)
        return a, b

    def ref_program(
            self, a: torch.Tensor, b: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        output = torch.matmul(a, b)
        if not self.grad:
            return output
        loss = output.sum()
        loss.backward()
        return output, a.grad, b.grad
