import math

import torch

from workloads.workload_base import WorkloadBase


def _generate_batch_sizes(batch_sum: int, batch_count: int):
    base_size = batch_sum // batch_count
    remainder = batch_sum % batch_count
    batch_sizes = [base_size] * batch_count
    for i in range(remainder):
        batch_sizes[i] += 1
    return batch_sizes

def _generate_offsets(batch_sizes_list, padding_M):
    batch_count = len(batch_sizes_list)
    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count - 1):
        batch_padded_offsets_list.append(
            batch_padded_offsets_list[-1]
            + math.ceil((batch_sizes_list[i] + 1) / padding_M) * padding_M)
    return batch_offsets_list, batch_padded_offsets_list

class GroupedGemmTest(WorkloadBase):

    def __init__(self, batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                 transpose_a: bool, transpose_b: bool):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.batch_sizes_list = _generate_batch_sizes(batch_sum, batch_count)
        self.padding_M = 128

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        batch_sizes_list = self.batch_sizes_list
        N, K = self.N, self.K
        device = 'cuda'
        dtype = self.dtype
        batch_sum = sum(batch_sizes_list)
        batch_count = len(batch_sizes_list)
        batch_offsets_list, batch_padded_offsets_list = _generate_offsets(
            batch_sizes_list, self.padding_M)

        if not self.transpose_a:
            # NT / NN: A is (batch_sum, K)
            A = torch.randn(batch_sum, K, device=device, dtype=dtype)
            if self.transpose_b:
                # NT: B is (batch_count, N, K)
                B = torch.randn(batch_count, N, K, device=device, dtype=dtype)
            else:
                # NN: B is (batch_count, K, N)
                B = torch.randn(batch_count, K, N, device=device, dtype=dtype)
        else:
            # TN / TT: A is (batch_sum, N)
            A = torch.randn(batch_sum, N, device=device, dtype=dtype)
            if self.transpose_b:
                # TT: B is (K, batch_sum)
                B = torch.randn(K, batch_sum, device=device, dtype=dtype)
            else:
                # TN: B is (batch_sum, K)
                B = torch.randn(batch_sum, K, device=device, dtype=dtype)

        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(
            batch_padded_offsets_list, device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets, batch_padded_offsets


class GroupedGemmUniformWorkload(WorkloadBase):
    """Uniform routing: every expert gets ``numel // num_experts`` rows.

    Produces the NT-layout tensors (``C = A @ B[e]^T``) plus the per-expert
    size and offset tables. Derived index tables that only one baseline needs
    are built by the caller.
    """

    def __init__(self, numel: int, num_experts: int, n: int, k: int, dtype: torch.dtype):
        if numel % num_experts:
            raise ValueError(f"numel={numel} not divisible by num_experts={num_experts}")
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.dtype = dtype
        self.rows_per_expert = numel // num_experts

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(42)
        dev = "cuda"
        a = torch.randn(self.numel, self.k, dtype=self.dtype, device=dev) * 0.02
        b = torch.randn(self.num_experts, self.n, self.k, dtype=self.dtype, device=dev) * 0.02
        sizes = torch.full(
            (self.num_experts,), self.rows_per_expert, dtype=torch.int32, device=dev
        )
        offsets = torch.zeros(self.num_experts, dtype=torch.int32, device=dev)
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
        return a, b, sizes, offsets
