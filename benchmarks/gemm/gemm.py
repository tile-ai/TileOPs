from benchmarks.benchmark import Benchmark
from top.ops import GemmOp
import torch


class GemmBenchmark(Benchmark):

    op_type = GemmOp

    def __init__(self, m, n, k, dtype, trans_a=False, trans_b=False):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

    @property
    def total_flops(self):
        return 2.0 * self.m * self.n * self.k

    @property
    def total_memory(self):
        return (self.m * self.k + self.k * self.n + self.m * self.n) * self.dtype.itemsize

    def gen_inputs(self):
        A = torch.randn(self.m, self.k, device='cuda', dtype=self.dtype)
        B = torch.randn(self.k, self.n, device='cuda', dtype=self.dtype)
        return A, B

    def ref_program(self, A: torch.Tensor, B: torch.Tensor):
        if self.trans_a:
            A = A.T
        if self.trans_b:
            B = B.T
        return torch.matmul(A, B)


class MatMulBenchmark(Benchmark):

    def __init__(self, m, n, k, dtype, grad=True):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.grad = grad

    @property
    def total_flops(self):
        return 6.0 * self.m * self.n * self.k

    @property
    def total_memory(self):
        return 3 * (self.m * self.k + self.k * self.n + self.m * self.n) * self.dtype.itemsize

    def gen_inputs(self):
        A = torch.randn(self.m, self.k, device='cuda', dtype=self.dtype, requires_grad=self.grad)
        B = torch.randn(self.k, self.n, device='cuda', dtype=self.dtype, requires_grad=self.grad)
        return A, B

    def ref_program(self, A: torch.Tensor, B: torch.Tensor):
        output = torch.matmul(A, B)
        if not self.grad:
            return output
        else:
            loss = output.sum()
            loss.backward()
        return output, A.grad, B.grad
