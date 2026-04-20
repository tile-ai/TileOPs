
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.grouped_gemm import GroupedGemmOp
from workloads.grouped_gemm import (
    GroupedGemmTest as _GroupedGemmTestWorkload,
)


class GroupedGemmTest(_GroupedGemmTestWorkload, TestBase):
    def ref_program(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor,
                    batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        if not self.transpose_a:
            # NT / NN: output is (batch_sum, N)
            if self.transpose_b:
                # NT: A @ B^T
                assert A.shape[0] == sum(batch_sizes)
                assert B.shape[0] == len(batch_sizes)
                output = torch.empty((sum(batch_sizes), B.shape[1]), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[start:end] = torch.mm(A[start:end], B[i].transpose(0, 1).contiguous())
                    start = end
            else:
                # NN: A @ B
                assert A.shape[0] == sum(batch_sizes)
                assert B.shape[0] == len(batch_sizes)
                output = torch.empty((sum(batch_sizes), B.shape[2]), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[start:end] = torch.mm(A[start:end], B[i])
                    start = end
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
                    output[i] = torch.mm(A[start:end].transpose(0, 1),
                                         B[:, start:end].transpose(0, 1))
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


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Parametrized grouped GEMM test
# ---------------------------------------------------------------------------

class GroupedGemmFixture(FixtureBase):
    PARAMS = [
        ("batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b, tune", [
            pytest.param(
                16384, 4, 4864, 4096, torch.float16, False, True, False,
                marks=pytest.mark.smoke,
            ),
        ]),
    ]


@GroupedGemmFixture
def test_grouped_gemm(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                      transpose_a: bool, transpose_b: bool, tune: bool) -> None:
    test = GroupedGemmTest(batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b)
    op = GroupedGemmOp(
        batch_sum, batch_count, N, K, dtype, transpose_a=transpose_a, transpose_b=transpose_b,
        tune=tune)
    test.check(op, *test.gen_inputs())


# ---------------------------------------------------------------------------
# Complete variant: forward (NT) + backward dA (NN) + backward dB (TN)
# ---------------------------------------------------------------------------

class GroupedGemmCompleteFixture(FixtureBase):
    PARAMS = [
        ("batch_sum, batch_count, N, K, dtype, tune", [
            pytest.param(16384, 4, 4864, 4096, torch.float16, False, marks=pytest.mark.smoke),
        ]),
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
