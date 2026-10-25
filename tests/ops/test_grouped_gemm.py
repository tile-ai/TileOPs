import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.gemm.grouped_gemm import GroupedGEMMFwdOp
from workloads.gemm import (
    GroupedGEMMWorkload,
)


class GroupedGEMMTest(GroupedGEMMWorkload, TestBase):
    pass


# Shared helper


# Parametrized grouped GEMM test


class GroupedGEMMFixture(FixtureBase):
    PARAMS = [
        (
            "batch_sum, batch_count, N, K, dtype, trans_a, trans_b, tune",
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
                    4099,
                    6,
                    4000,
                    4096,
                    torch.float16,
                    False,
                    True,
                    False,
                    marks=pytest.mark.smoke,
                    id="uneven-unaligned",
                ),
                pytest.param(
                    4099,
                    6,
                    4000,
                    4096,
                    torch.float16,
                    False,
                    False,
                    False,
                    marks=pytest.mark.smoke,
                    id="nn-uneven-unaligned",
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


@GroupedGEMMFixture
def test_grouped_gemm(
    batch_sum: int,
    batch_count: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    trans_a: bool,
    trans_b: bool,
    tune: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The kernel accumulates in fp32; by default PyTorch lets cuBLAS reduce fp16 in fp16.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False)
    test = GroupedGEMMTest(batch_sum, batch_count, N, K, dtype, trans_a, trans_b)
    op = GroupedGEMMFwdOp(trans_a=trans_a, trans_b=trans_b)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("trans_b", [True, False], ids=["tn", "tt"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_k_grouped_gemm_with_ragged_groups(
    dtype: torch.dtype, trans_b: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Groups split K: ragged group lengths, an empty group and output edges off the tile grid.

    A group whose length is not a multiple of 8 starts mid-vector, and the empty group's
    output block is zero.
    """
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False)
    sizes = [100, 0, 300, 64, 7, 1]
    test = GroupedGEMMTest(sum(sizes), len(sizes), 200, 136, dtype, True, trans_b)
    test.batch_sizes_list = sizes
    op = GroupedGEMMFwdOp(trans_a=True, trans_b=trans_b)
    test.check(op, *test.gen_inputs())


# Which kernel serves which call
