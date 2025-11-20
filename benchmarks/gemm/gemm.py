from benchmarks.benchmark import Benchmark
from top.ops import Gemm
from top.functions import matmul
from top.layers import Linear
import torch


class BaseGemmBenchmark(Benchmark):
    """Base class for GEMM benchmarks with common functionality"""

    def __init__(self, M, N, K, dtype):
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype

    def gen_inputs(self):
        """Generic method to generate input tensors"""
        A = torch.randn(self.M, self.K, device='cuda', dtype=self.dtype)
        B = torch.randn(self.K, self.N, device='cuda', dtype=self.dtype)
        return A, B


class gemm_benchmark(BaseGemmBenchmark):
    """GEMM operation benchmark"""

    op_type = Gemm

    def __init__(self, M, N, K, dtype, trans_A=False, trans_B=False):
        super().__init__(M, N, K, dtype)
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
        B = torch.randn(self.K, self.N, device='cuda', dtype=self.dtype)
        return A, B

    def ref_program(self, A: torch.Tensor, B: torch.Tensor):
        if self.trans_A:
            A = A.T
        if self.trans_B:
            B = B.T
        return torch.matmul(A, B)


class matmul_benchmark(BaseGemmBenchmark):
    """MatMul operation benchmark (with gradient computation support)"""

    function_type = matmul

    def __init__(self, M, N, K, dtype, grad=True):
        super().__init__(M, N, K, dtype)
        self.grad = grad

    @property
    def total_flops(self):
        return 6.0 * self.M * self.N * self.K if self.grad else 2.0 * self.M * self.N * self.K

    @property
    def total_memory(self):
        return 3 * (self.M * self.K + self.K * self.N +
                    self.M * self.N) * self.dtype.itemsize if self.grad else (
                        self.M * self.K + self.K * self.N + self.M * self.N) * self.dtype.itemsize

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


class linear_benchmark(BaseGemmBenchmark):
    """Linear layer benchmark"""

    function_type = Linear

    def __init__(self, batch_size, out_features, in_features, dtype, grad=True):
        super().__init__(batch_size, out_features, in_features, dtype)
        self.batch_size = batch_size
        self.out_features = out_features
        self.in_features = in_features
        self.grad = grad

    @property
    def total_flops(self):
        # Forward pass: 2 * batch_size * out_features * in_features
        # Backward pass: 4 * batch_size * out_features * in_features
        return 6.0 * self.batch_size * self.out_features * self.in_features if self.grad else 2.0 * self.batch_size * self.out_features * self.in_features

    @property
    def total_memory(self):
        # Memory usage for input, weight and output
        input_memory = self.batch_size * self.in_features * self.dtype.itemsize
        weight_memory = self.in_features * self.out_features * self.dtype.itemsize
        output_memory = self.batch_size * self.out_features * self.dtype.itemsize
        base_memory = input_memory + weight_memory + output_memory
        return 3 * base_memory if self.grad else base_memory

    def gen_inputs(self):
        # Generate input data
        x = torch.randn(
            self.batch_size,
            self.in_features,
            device='cuda',
            dtype=self.dtype,
            requires_grad=self.grad)
        return (x,)

    def ref_program(self, x: torch.Tensor):
        # Use torch.nn.Linear as reference implementation
        ref_linear = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=False,
            dtype=self.dtype,
            device='cuda')

        # Copy weights to ensure consistency in comparison
        with torch.no_grad():
            ref_linear.weight.copy_(self.test_linear.weight.T)

        output = ref_linear(x)

        if not self.grad:
            return output
        else:
            loss = output.sum()
            loss.backward()
            return output, x.grad
