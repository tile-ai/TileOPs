import math
from typing import Optional

import torch

from tileops.ops.gemm.grouped_gemm import GroupedGemmFwdOp
from workloads.workload_base import WorkloadBase


def _generate_batch_sizes(batch_sum: int, batch_count: int):
    base_size = batch_sum // batch_count
    remainder = batch_sum % batch_count
    batch_sizes = [base_size] * batch_count
    for i in range(remainder):
        batch_sizes[i] += 1
    return batch_sizes


class GroupedGemmWorkload(WorkloadBase):
    """Grouped GEMM operands under one row layout.

    ``padded`` starts every group on a row block, so ``a`` carries the padding rows
    between groups; the reference defines them, because the kernel reads whole
    tiles and multiplies those rows by the group they trail. Tight rows carry no
    padding and ``gen_inputs`` returns no padded table.
    """

    def __init__(
        self,
        batch_sum: int,
        batch_count: int,
        N: int,
        K: int,
        dtype: torch.dtype,
        transpose_a: bool,
        transpose_b: bool,
        padded: bool = False,
    ):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.padded = padded
        self.batch_sizes_list = _generate_batch_sizes(batch_sum, batch_count)
        self.padding_M = GroupedGemmFwdOp.row_block

        def span(size: int) -> int:
            return math.ceil(size / self.padding_M) * self.padding_M if padded else size

        # Where each group starts in a, and where the last one ends.
        self.group_starts_list = [0]
        for size in self.batch_sizes_list[:-1]:
            self.group_starts_list.append(self.group_starts_list[-1] + span(size))
        self.total_rows = self.group_starts_list[-1] + span(self.batch_sizes_list[-1])

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        batch_sizes_list = self.batch_sizes_list
        N, K = self.N, self.K
        device = "cuda"
        dtype = self.dtype
        batch_sum = sum(batch_sizes_list)
        batch_count = len(batch_sizes_list)
        rows = self.total_rows

        if not self.transpose_a:
            # NT / NN: A is (rows, K), rows == batch_sum unless the layout pads
            A = torch.randn(rows, K, device=device, dtype=dtype)
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
        # Under either layout these are where the groups start; a padded call
        # names them again to state that they sit on a row block.
        batch_offsets = torch.tensor(self.group_starts_list, device=device, dtype=torch.int32)
        if not self.padded:
            return A, B, batch_sizes, batch_offsets
        return A, B, batch_sizes, batch_offsets, batch_offsets

    def ref_program(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.transpose_a:
            # NT / NN: output is (rows, N), one product per group over the rows the
            # layout gives it -- a padded group owns the padding that trails it.
            ends = self.group_starts_list[1:] + [A.shape[0]]
            spans = list(zip(self.group_starts_list, ends, strict=True))
            assert A.shape[0] == self.total_rows
            assert B.shape[0] == len(batch_sizes)
            cols = B.shape[1] if self.transpose_b else B.shape[2]
            output = torch.empty((A.shape[0], cols), device=A.device, dtype=A.dtype)
            for i, (start, end) in enumerate(spans):
                b_i = B[i].transpose(0, 1).contiguous() if self.transpose_b else B[i]
                output[start:end] = torch.mm(A[start:end], b_i)
        else:
            # TN / TT: output is (batch_count, N, K)
            total_batch = int(batch_sizes.sum().item())
            assert A.shape[0] == total_batch
            N = A.shape[1]
            batch_count = len(batch_sizes)

            if self.transpose_b:
                # TT: A^T @ B^T
                K = B.shape[0]
                assert B.shape[1] == total_batch
                output = torch.zeros((batch_count, N, K), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[i] = torch.mm(
                        A[start:end].transpose(0, 1), B[:, start:end].transpose(0, 1)
                    )
                    start = end
            else:
                # TN: A^T @ B
                K = B.shape[1]
                assert B.shape[0] == total_batch
                output = torch.zeros((batch_count, N, K), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[i] = torch.mm(A[start:end].transpose(0, 1), B[start:end])
                    start = end
        return output
