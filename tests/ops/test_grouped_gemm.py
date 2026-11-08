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


@pytest.mark.parametrize("trans_b", [True, False], ids=["nt", "nn"])
@pytest.mark.parametrize(
    "sizes, n, k",
    [
        # Empty groups first and last end where the next group starts and at the last row.
        pytest.param(
            [0, 100, 0, 300, 128, 7, 64, 0], 256, 512, marks=pytest.mark.smoke, id="empty-ends"
        ),
        # Mean 428 rows a group at N 2048: the ping-pong tile past its band.
        pytest.param(
            [360, 34, 103, 889, 352, 155, 1993, 31, 363, 749, 160, 899, 142, 558, 0, 55],
            2048,
            768,
            marks=pytest.mark.full,
            id="skewed-pingpong",
        ),
        # A mean of 38 rows a group: the short-group tile over a K tail of 64 and an output
        # tail of 40 columns.
        pytest.param(
            [40, 0, 64, 17, 60, 64, 3, 50],
            1000,
            1088,
            marks=pytest.mark.full,
            id="one-row-tile-tails",
        ),
    ],
)
def test_m_grouped_gemm_group_starts(sizes: list, n: int, k: int, trans_b: bool) -> None:
    """Groups split the rows of ``a``: the kernel reads ``batch_offsets`` as group starts."""
    test = GroupedGEMMTest(sum(sizes), len(sizes), n, k, torch.bfloat16, False, trans_b)
    test.batch_sizes_list = sizes
    op = GroupedGEMMFwdOp(trans_a=False, trans_b=trans_b)
    test.check(op, *test.gen_inputs())


# Which kernel serves which call
