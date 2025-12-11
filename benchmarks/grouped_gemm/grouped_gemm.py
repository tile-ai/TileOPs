from benchmarks.benchmark import Benchmark
from top.ops.grouped_gemm import grouped_gemm_nt, grouped_gemm_nn, grouped_gemm_tn, grouped_gemm_tt
import torch
import math


class grouped_gemm_nt_benchmark(Benchmark):
    """Benchmark for forward grouped GEMM: A @ B^T -> C"""
    op_type = grouped_gemm_nt

    def __init__(self, batch_sum, batch_count, N, K, dtype):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.batch_sizes_list = self._generate_batch_sizes()
        self.padding_M = 128

    def _generate_batch_sizes(self):
        base_size = self.batch_sum // self.batch_count
        remainder = self.batch_sum % self.batch_count
        batch_sizes = [base_size] * self.batch_count
        for i in range(remainder):
            batch_sizes[i] += 1
        return batch_sizes

    @property
    def total_flops(self):
        total_flops = 2.0 * self.batch_sum * self.K * self.N
        return total_flops

    @property
    def total_memory(self):
        memory_A = self.batch_sum * self.K * self.dtype.itemsize
        memory_B = self.batch_count * self.K * self.N * self.dtype.itemsize
        memory_Y = self.batch_sum * self.N * self.dtype.itemsize
        return memory_A + memory_B + memory_Y

    def gen_inputs(self):
        batch_sizes_list = self.batch_sizes_list
        N = self.N
        K = self.K
        padding_M = self.padding_M
        device = 'cuda'
        dtype = self.dtype
        batch_sum = sum(batch_sizes_list)
        batch_count = len(batch_sizes_list)
        batch_offsets_list = [0]
        batch_padded_offsets_list = [0]
        for i in range(batch_count - 1):
            batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
        for i in range(batch_count - 1):
            batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                             math.ceil((batch_sizes_list[i] + 1) / padding_M) *
                                             padding_M)

        A = torch.randn(batch_sum, K, device=device, dtype=dtype)
        B = torch.randn(batch_count, N, K, device=device, dtype=dtype)
        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(
            batch_padded_offsets_list, device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets, batch_padded_offsets

    def ref_program(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor):
        assert A.shape[0] == sum(batch_sizes)
        assert B.shape[0] == len(batch_sizes)
        output = torch.empty((sum(batch_sizes), B.shape[1]), device=A.device, dtype=A.dtype)
        start = 0
        for i, size in enumerate(batch_sizes):
            end = start + size
            part_a = A[start:end]
            part_b = B[i].transpose(0, 1).contiguous()
            output[start:end] = torch.mm(part_a, part_b)
            start = end
        return output


class grouped_gemm_nn_benchmark(Benchmark):
    """Benchmark for backward dA grouped GEMM: A @ B -> C"""
    op_type = grouped_gemm_nn

    def __init__(self, batch_sum, batch_count, N, K, dtype):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.batch_sizes_list = self._generate_batch_sizes()
        self.padding_M = 128

    def _generate_batch_sizes(self):
        base_size = self.batch_sum // self.batch_count
        remainder = self.batch_sum % self.batch_count
        batch_sizes = [base_size] * self.batch_count
        for i in range(remainder):
            batch_sizes[i] += 1
        return batch_sizes

    @property
    def total_flops(self):
        total_flops = 2.0 * self.batch_sum * self.K * self.N
        return total_flops

    @property
    def total_memory(self):
        memory_A = self.batch_sum * self.N * self.dtype.itemsize
        memory_B = self.batch_count * self.N * self.K * self.dtype.itemsize
        memory_Y = self.batch_sum * self.K * self.dtype.itemsize
        return memory_A + memory_B + memory_Y

    def gen_inputs(self):
        batch_sizes_list = self.batch_sizes_list
        N = self.N
        K = self.K
        padding_M = self.padding_M
        device = 'cuda'
        dtype = self.dtype
        batch_sum = sum(batch_sizes_list)
        batch_count = len(batch_sizes_list)

        batch_offsets_list = [0]
        batch_padded_offsets_list = [0]
        for i in range(batch_count - 1):
            batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
        for i in range(batch_count - 1):
            batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                             math.ceil((batch_sizes_list[i] + 1) / padding_M) *
                                             padding_M)

        A = torch.randn(batch_sum, K, device=device, dtype=dtype)
        B = torch.randn(batch_count, K, N, device=device, dtype=dtype)
        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(
            batch_padded_offsets_list, device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets, batch_padded_offsets

    def ref_program(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor):
        assert A.shape[0] == sum(batch_sizes)
        assert B.shape[0] == len(batch_sizes)
        output = torch.empty((sum(batch_sizes), B.shape[2]), device=A.device, dtype=A.dtype)
        start = 0
        for i, size in enumerate(batch_sizes):
            end = start + size
            part_a = A[start:end]
            part_b = B[i]
            output[start:end] = torch.mm(part_a, part_b)
            start = end
        return output


class grouped_gemm_tn_benchmark(Benchmark):
    """Benchmark for backward dB grouped GEMM: A^T @ B -> C"""
    op_type = grouped_gemm_tn

    def __init__(self, batch_sum, batch_count, N, K, dtype):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.batch_sizes_list = self._generate_batch_sizes()
        self.padding_M = 128

    def _generate_batch_sizes(self):
        base_size = self.batch_sum // self.batch_count
        remainder = self.batch_sum % self.batch_count
        batch_sizes = [base_size] * self.batch_count
        for i in range(remainder):
            batch_sizes[i] += 1
        return batch_sizes

    @property
    def total_flops(self):
        total_flops = 2.0 * self.batch_sum * self.K * self.N
        return total_flops

    @property
    def total_memory(self):
        memory_A = self.K * self.batch_sum * self.dtype.itemsize
        memory_B = self.batch_sum * self.N * self.dtype.itemsize
        memory_Y = self.batch_count * self.K * self.N * self.dtype.itemsize
        return memory_A + memory_B + memory_Y

    def gen_inputs(self):
        batch_sizes_list = self.batch_sizes_list
        N = self.N
        K = self.K
        padding_M = self.padding_M
        device = 'cuda'
        dtype = self.dtype
        batch_sum = sum(batch_sizes_list)
        batch_count = len(batch_sizes_list)

        batch_offsets_list = [0]
        batch_padded_offsets_list = [0]
        for i in range(batch_count - 1):
            batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
        for i in range(batch_count - 1):
            batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                             math.ceil((batch_sizes_list[i] + 1) / padding_M) *
                                             padding_M)

        A = torch.randn(batch_sum, N, device=device, dtype=dtype)
        B = torch.randn(batch_sum, K, device=device, dtype=dtype)
        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(
            batch_padded_offsets_list, device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets, batch_padded_offsets

    def ref_program(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor):
        batch_sum_A = A.shape[0]
        batch_sum_B = B.shape[0]
        total_batch = int(batch_sizes.sum().item())
        assert batch_sum_A == total_batch, f"A.shape[0]={batch_sum_A} != sum(batch_sizes)={total_batch}"
        assert batch_sum_B == total_batch, f"B.shape[0]={batch_sum_B} != sum(batch_sizes)={total_batch}"

        N, K = A.shape[1], B.shape[1]
        batch_count = len(batch_sizes)
        output = torch.zeros((batch_count, N, K), device=A.device, dtype=A.dtype)

        start = 0
        for i, size in enumerate(batch_sizes):
            end = start + size
            part_a = A[start:end, :]
            part_b = B[start:end, :]
            output[i] = torch.mm(part_a.transpose(0, 1), part_b)
            start = end
        return output


class grouped_gemm_tt_benchmark(Benchmark):
    op_type = grouped_gemm_tt

    def __init__(self, batch_sum, batch_count, N, K, dtype):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.batch_sizes_list = self._generate_batch_sizes()
        self.padding_M = 128

    def _generate_batch_sizes(self):
        base_size = self.batch_sum // self.batch_count
        remainder = self.batch_sum % self.batch_count
        batch_sizes = [base_size] * self.batch_count
        for i in range(remainder):
            batch_sizes[i] += 1
        return batch_sizes

    @property
    def total_flops(self):
        return 2.0 * self.batch_sum * self.N * self.K

    @property
    def total_memory(self):
        memory_A = self.batch_sum * self.N * self.dtype.itemsize
        memory_B = self.K * self.batch_sum * self.dtype.itemsize
        memory_Y = self.batch_count * self.N * self.K * self.dtype.itemsize
        return memory_A + memory_B + memory_Y

    def gen_inputs(self):
        batch_sizes_list = self.batch_sizes_list
        N = self.N
        K = self.K
        padding_M = self.padding_M
        device = 'cuda'
        dtype = self.dtype
        batch_sum = sum(batch_sizes_list)
        batch_count = len(batch_sizes_list)

        batch_offsets_list = [0]
        batch_padded_offsets_list = [0]
        for i in range(batch_count - 1):
            batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
        for i in range(batch_count - 1):
            batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                             math.ceil((batch_sizes_list[i] + 1) / padding_M) *
                                             padding_M)

        A = torch.randn(batch_sum, N, device=device, dtype=dtype)
        B = torch.randn(K, batch_sum, device=device, dtype=dtype)
        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(
            batch_padded_offsets_list, device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets, batch_padded_offsets

    def ref_program(self, A, B, batch_sizes, batch_offsets, batch_padded_offsets):
        batch_sum_A = A.shape[0]
        batch_sum_B = B.shape[1]
        total_batch = int(batch_sizes.sum().item())
        assert batch_sum_A == total_batch
        assert batch_sum_B == total_batch
        N = A.shape[1]
        K = B.shape[0]
        batch_count = len(batch_sizes)
        output = torch.zeros((batch_count, N, K), device=A.device, dtype=A.dtype)

        start = 0
        for i, size in enumerate(batch_sizes):
            size = int(size.item())
            end = start + size
            dO_slice = A[start:end, :]
            A_T_slice = B[:, start:end]
            output[i] = torch.mm(dO_slice.transpose(0, 1), A_T_slice.transpose(0, 1))
            start = end

        return output


class grouped_gemm_benchmark(Benchmark):

    def __init__(self, batch_sum, batch_count, N, K, dtype, grad=True):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.grad = grad
        self.batch_sizes_list = self._generate_batch_sizes()
        self.padding_M = 128
        self.nt_benchmark = grouped_gemm_nt_benchmark(batch_sum, batch_count, N, K, dtype)
        self.nn_benchmark = grouped_gemm_nn_benchmark(batch_sum, batch_count, N, K, dtype)
        self.tn_benchmark = grouped_gemm_tn_benchmark(batch_sum, batch_count, N, K, dtype)

    @property
    def total_flops(self):
        return (self.nt_benchmark.total_flops + self.nn_benchmark.total_flops +
                self.tn_benchmark.total_flops)

    @property
    def total_memory(self):
        return (self.nt_benchmark.total_memory + self.nn_benchmark.total_memory +
                self.tn_benchmark.total_memory)

    def _generate_batch_sizes(self):
        base_size = self.batch_sum // self.batch_count
        remainder = self.batch_sum % self.batch_count
        batch_sizes = [base_size] * self.batch_count
        for i in range(remainder):
            batch_sizes[i] += 1
        return batch_sizes

    def gen_inputs(self):
        return self.nt_benchmark.gen_inputs()

    def ref_program(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor):
        forward_output = self.nt_benchmark.ref_program(A, B, batch_sizes, batch_offsets,
                                                       batch_padded_offsets)
        grad_output = torch.ones_like(forward_output)
        dA = self.nn_benchmark.ref_program(grad_output, B, batch_sizes, batch_offsets,
                                           batch_padded_offsets)
        # no need to transpose grad_output here, it is transposed in tn layout ref_program
        dB = self.tn_benchmark.ref_program(grad_output, A, batch_sizes, batch_offsets,
                                           batch_padded_offsets)

        return forward_output, dA, dB, None, None, None
