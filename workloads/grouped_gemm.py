import math
from typing import Tuple

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

class GroupedGemmCompleteTest:
    """Parameter holder for GroupedGemmCompleteBenchmark (forward NT + backward NN + backward TN).

    The benchmark test function profiles each variant (NT/NN/TN) individually
    using GroupedGemmTest; this class exists so the benchmark can access
    batch_sum, batch_count, N, K, and dtype for FLOPS/memory calculations.
    """

    def __init__(self, batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype

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

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
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
