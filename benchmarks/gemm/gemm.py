from benchmarks.benchmark import Benchmark
from top.ops import Gemm
import torch


class gemm_benchmark(Benchmark):

    op_type = Gemm

    def __init__(self, M, N, K, dtype, trans_A=False, trans_B=False):
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.trans_A = trans_A
        self.trans_B = trans_B

    @property
    def total_flops(self):
        return 2.0 * self.M * self.N * self.K

    @property
    def total_memory(self):
        return (self.M * self.K + self.K * self.N + self.M * self.N) * self.dtype.itemsize

    def gen_inputs(self):
        A = torch.randn(self.M, self.K, device='cuda', dtype=self.dtype)
        B = torch.randn(self.N, self.K, device='cuda', dtype=self.dtype)
        return A, B

    def ref_program(self, A: torch.Tensor, B: torch.Tensor):
        if self.trans_A:
            A = A.transpose(-2, -1)
        if self.trans_B:
            print(2)
            B = B.T
        return torch.matmul(A, B)
    

class matmul_benchmark(Benchmark): 

    def __init__(self, M, N, K, dtype, grad=True):
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype
        self.grad = grad
    @property
    def total_flops(self):
        return 6.0 * self.M * self.N * self.K

    @property
    def total_memory(self):
        return 3 * (self.M * self.K + self.K * self.N + self.M * self.N) * self.dtype.itemsize

    def gen_inputs(self):
        A = torch.randn(self.M, self.K, device='cuda', dtype=self.dtype, requires_grad=self.grad)
        B = torch.randn(self.K, self.N, device='cuda', dtype=self.dtype, requires_grad=self.grad)
        return A, B

    def ref_program(self, A: torch.Tensor, B: torch.Tensor):
        output = torch.matmul(A, B)
        if not self.grad:
            return output
        else:
            loss = output.sum()
            loss.backward()
        return output, A.grad, B.grad
