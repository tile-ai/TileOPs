import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.gemm.grouped_gemm import GroupedGEMMFwdOp
from workloads.gemm import (
    GroupedGEMMWorkload,
)


class GroupedGEMMTest(GroupedGEMMWorkload, TestBase):
    pass


class GroupedGEMMFixture(FixtureBase):
    PARAMS = [
        (
            "batch_sum, batch_count, N, K, dtype, trans_a, trans_b",
            [
                pytest.param(
                    16384,
                    4,
                    4864,
                    4096,
                    torch.float16,
                    False,
                    True,
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
                    marks=pytest.mark.full,
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
                    marks=pytest.mark.full,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The kernel accumulates in fp32; by default PyTorch lets cuBLAS reduce fp16 in fp16.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False)
    test = GroupedGEMMTest(batch_sum, batch_count, N, K, dtype, trans_a, trans_b)
    test.check(GroupedGEMMFwdOp(trans_a=trans_a, trans_b=trans_b), *test.gen_inputs())


@pytest.mark.parametrize(
    "dtype, trans_b",
    [
        pytest.param(torch.float16, True, marks=pytest.mark.smoke, id="fp16-tn"),
        pytest.param(torch.bfloat16, True, marks=pytest.mark.smoke, id="bf16-tn"),
        pytest.param(torch.float16, False, marks=pytest.mark.full, id="fp16-tt"),
        pytest.param(torch.bfloat16, False, marks=pytest.mark.full, id="bf16-tt"),
    ],
)
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


_EMPTY_ENDS = [0, 100, 0, 300, 128, 7, 64, 0]
_SKEWED = [360, 34, 103, 889, 352, 155, 1993, 31, 363, 749, 160, 899, 142, 558, 0, 55]
_ONE_ROW_TILE = [40, 0, 64, 17, 60, 64, 3, 50]


@pytest.mark.parametrize(
    "sizes, n, k, trans_b",
    [
        # Empty first and last groups end where the next group starts and at the last row.
        pytest.param(_EMPTY_ENDS, 256, 512, True, marks=pytest.mark.smoke, id="empty-ends-nt"),
        pytest.param(_EMPTY_ENDS, 256, 512, False, marks=pytest.mark.full, id="empty-ends-nn"),
        # Enough groups that their row ends leave the 192-wide ping-pong tile a deeper ring
        # only when it stages its output by 64-column box; it needs a K-major B, so NT only.
        pytest.param(
            [193] * 1280, 192, 64, True, marks=pytest.mark.full, id="many-groups-box-staged-nt"
        ),
        # The short-group tile over a K tail of 64 and an output tail of 40 columns.
        pytest.param(
            _ONE_ROW_TILE, 1000, 1088, True, marks=pytest.mark.full, id="one-row-tile-tails-nt"
        ),
        pytest.param(
            _ONE_ROW_TILE, 1000, 1088, False, marks=pytest.mark.full, id="one-row-tile-tails-nn"
        ),
    ],
)
def test_m_grouped_gemm_group_starts(sizes: list, n: int, k: int, trans_b: bool) -> None:
    """Groups split the rows of ``a``: the kernel reads ``batch_offsets`` as group starts."""
    test = GroupedGEMMTest(sum(sizes), len(sizes), n, k, torch.bfloat16, False, trans_b)
    test.batch_sizes_list = sizes
    op = GroupedGEMMFwdOp(trans_a=False, trans_b=trans_b)
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize(
    "n, dtype",
    [
        pytest.param(1040, torch.float16, marks=pytest.mark.smoke, id="narrow-fp16"),
        pytest.param(1040, torch.bfloat16, marks=pytest.mark.smoke, id="narrow-bf16"),
        pytest.param(2024, torch.float16, marks=pytest.mark.full, id="box-staged-fp16"),
        pytest.param(2024, torch.bfloat16, marks=pytest.mark.full, id="box-staged-bf16"),
    ],
)
def test_m_grouped_gemm_skewed_pingpong_tails(n: int, dtype: torch.dtype) -> None:
    """Skewed groups past the ping-pong band, which takes a K-major B, with an output tail.

    N = 1040 runs the 192-wide tile, its last column tile 80 columns. N = 2024 runs the
    256-wide tile staged by 64-column box; its last box holds 40 columns, which each group's
    ragged last tile stores in 16-byte vectors.
    """
    test = GroupedGEMMTest(sum(_SKEWED), len(_SKEWED), n, 768, dtype, False, True)
    test.batch_sizes_list = _SKEWED
    op = GroupedGEMMFwdOp(trans_a=False, trans_b=True)
    test.check(op, *test.gen_inputs())
