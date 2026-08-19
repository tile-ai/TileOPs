import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.grouped_gemm import GroupedGemmFwdOp
from workloads.grouped_gemm import (
    GroupedGemmWorkload,
)


class GroupedGemmTest(GroupedGemmWorkload, TestBase):
    pass


# Shared helper


# Parametrized grouped GEMM test


class GroupedGemmFixture(FixtureBase):
    PARAMS = [
        (
            "batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b, tune",
            [
                pytest.param(
                    16384,
                    4,
                    4864,
                    4096,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    16384,
                    4,
                    4864,
                    4096,
                    torch.float16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    16384,
                    4,
                    4864,
                    4096,
                    torch.float16,
                    True,
                    False,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    16384,
                    4,
                    4864,
                    4096,
                    torch.float16,
                    True,
                    True,
                    False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@GroupedGemmFixture
def test_grouped_gemm(
    batch_sum: int,
    batch_count: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    transpose_a: bool,
    transpose_b: bool,
    tune: bool,
) -> None:
    test = GroupedGemmTest(batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b)
    op = GroupedGemmFwdOp(transpose_a=transpose_a, transpose_b=transpose_b, tune=tune)
    test.check(op, *test.gen_inputs())


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
