import math
from typing import Tuple

import torch
import pytest

from tests.test_base import TestBase, FixtureBase
from tileops.ops.grouped_gemm import GroupedGemmNNOp, GroupedGemmNTOp, GroupedGemmTNOp, GroupedGemmTTOp


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NT variant: A @ B^T -> C
# ---------------------------------------------------------------------------

class GroupedGemmNTFixture(FixtureBase):
    PARAMS = [
        ("batch_sum, batch_count, N, K, dtype, tune", [
            (16384, 4, 4864, 4096, torch.float16, False),
        ]),
    ]


class GroupedGemmNTTest(TestBase):

    def __init__(self, batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
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

        A = torch.randn(batch_sum, K, device=device, dtype=dtype)
        B = torch.randn(batch_count, N, K, device=device, dtype=dtype)
        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(
            batch_padded_offsets_list, device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets, batch_padded_offsets

    def ref_program(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor,
                    batch_padded_offsets: torch.Tensor) -> torch.Tensor:
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


@GroupedGemmNTFixture
def test_grouped_gemm_nt(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                         tune: bool) -> None:
    test = GroupedGemmNTTest(batch_sum, batch_count, N, K, dtype)
    op = GroupedGemmNTOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    test.check(op, *test.gen_inputs())


# ---------------------------------------------------------------------------
# NN variant: A @ B -> C
# ---------------------------------------------------------------------------

class GroupedGemmNNFixture(FixtureBase):
    PARAMS = [
        ("batch_sum, batch_count, N, K, dtype, tune", [
            (16384, 4, 4864, 4096, torch.float16, False),
        ]),
    ]


class GroupedGemmNNTest(TestBase):

    def __init__(self, batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
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

        A = torch.randn(batch_sum, K, device=device, dtype=dtype)
        B = torch.randn(batch_count, K, N, device=device, dtype=dtype)
        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(
            batch_padded_offsets_list, device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets, batch_padded_offsets

    def ref_program(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor,
                    batch_padded_offsets: torch.Tensor) -> torch.Tensor:
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


@GroupedGemmNNFixture
def test_grouped_gemm_nn(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                         tune: bool) -> None:
    test = GroupedGemmNNTest(batch_sum, batch_count, N, K, dtype)
    op = GroupedGemmNNOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    test.check(op, *test.gen_inputs())


# ---------------------------------------------------------------------------
# TN variant: A^T @ B -> C
# ---------------------------------------------------------------------------

class GroupedGemmTNFixture(FixtureBase):
    PARAMS = [
        ("batch_sum, batch_count, N, K, dtype, tune", [
            (16384, 4, 4864, 4096, torch.float16, False),
        ]),
    ]


class GroupedGemmTNTest(TestBase):

    def __init__(self, batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.batch_sizes_list = _generate_batch_sizes(batch_sum, batch_count)
        self.padding_M = 128

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
        batch_sizes_list = self.batch_sizes_list
        N, K = self.N, self.K
        device = 'cuda'
        dtype = self.dtype
        batch_sum = sum(batch_sizes_list)
        batch_offsets_list, batch_padded_offsets_list = _generate_offsets(
            batch_sizes_list, self.padding_M)

        A = torch.randn(batch_sum, N, device=device, dtype=dtype)
        B = torch.randn(batch_sum, K, device=device, dtype=dtype)
        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(
            batch_padded_offsets_list, device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets, batch_padded_offsets

    def ref_program(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor,
                    batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        batch_sum_A = A.shape[0]
        batch_sum_B = B.shape[0]
        total_batch = int(batch_sizes.sum().item())
        assert batch_sum_A == total_batch, \
            f"A.shape[0]={batch_sum_A} != sum(batch_sizes)={total_batch}"
        assert batch_sum_B == total_batch, \
            f"B.shape[0]={batch_sum_B} != sum(batch_sizes)={total_batch}"

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


@GroupedGemmTNFixture
def test_grouped_gemm_tn(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                         tune: bool) -> None:
    test = GroupedGemmTNTest(batch_sum, batch_count, N, K, dtype)
    op = GroupedGemmTNOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    test.check(op, *test.gen_inputs())


# ---------------------------------------------------------------------------
# TT variant: A^T @ B^T -> C
# ---------------------------------------------------------------------------

class GroupedGemmTTFixture(FixtureBase):
    PARAMS = [
        ("batch_sum, batch_count, N, K, dtype, tune", [
            (16384, 4, 4864, 4096, torch.float16, False),
        ]),
    ]


class GroupedGemmTTTest(TestBase):

    def __init__(self, batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.batch_sizes_list = _generate_batch_sizes(batch_sum, batch_count)
        self.padding_M = 128

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
        batch_sizes_list = self.batch_sizes_list
        N, K = self.N, self.K
        device = 'cuda'
        dtype = self.dtype
        batch_sum = sum(batch_sizes_list)
        batch_offsets_list, batch_padded_offsets_list = _generate_offsets(
            batch_sizes_list, self.padding_M)

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


@GroupedGemmTTFixture
def test_grouped_gemm_tt(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                         tune: bool) -> None:
    test = GroupedGemmTTTest(batch_sum, batch_count, N, K, dtype)
    op = GroupedGemmTTOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    test.check(op, *test.gen_inputs())


# ---------------------------------------------------------------------------
# Complete (GroupedGemmFunc — forward + backward)
# ---------------------------------------------------------------------------

class GroupedGemmCompleteFixture(FixtureBase):
    PARAMS = [
        ("batch_sum, batch_count, N, K, dtype, tune", [
            (16384, 4, 4864, 4096, torch.float16, False),
        ]),
    ]


class GroupedGemmCompleteTest(TestBase):
    """Tests the complete GroupedGemmFunc (forward + backward)."""

    def __init__(self, batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.batch_sizes_list = _generate_batch_sizes(batch_sum, batch_count)
        self.padding_M = 128
        self._nt_test = GroupedGemmNTTest(batch_sum, batch_count, N, K, dtype)
        self._nn_test = GroupedGemmNNTest(batch_sum, batch_count, N, K, dtype)
        self._tn_test = GroupedGemmTNTest(batch_sum, batch_count, N, K, dtype)

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
        return self._nt_test.gen_inputs()

    def ref_program(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor):
        forward_output = self._nt_test.ref_program(A, B, batch_sizes, batch_offsets,
                                                   batch_padded_offsets)
        grad_output = torch.ones_like(forward_output)
        dA = self._nn_test.ref_program(grad_output, B, batch_sizes, batch_offsets,
                                       batch_padded_offsets)
        dB = self._tn_test.ref_program(grad_output, A, batch_sizes, batch_offsets,
                                       batch_padded_offsets)
        return forward_output, dA, dB, None, None, None


@GroupedGemmCompleteFixture
def test_grouped_gemm_complete(batch_sum: int, batch_count: int, N: int, K: int,
                               dtype: torch.dtype, tune: bool) -> None:
    from tileops.functions.grouped_gemm import GroupedGemmFunc

    fn = GroupedGemmFunc(batch_sum, batch_count, N, K, dtype, tune=tune)
    test = GroupedGemmCompleteTest(batch_sum, batch_count, N, K, dtype)
    inputs = test.gen_inputs()
    inputs[0].requires_grad_(True)  # A
    inputs[1].requires_grad_(True)  # B
    test.check_fn(fn, *inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
