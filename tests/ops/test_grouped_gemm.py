import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.kernels.gemm.grouped import GroupedGemmCall
from tileops.ops.gemm.grouped_gemm import GroupedGemmFwdOp
from workloads.gemm import (
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The kernel accumulates in fp32; by default PyTorch lets cuBLAS reduce fp16 in fp16.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False)
    test = GroupedGemmTest(batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b)
    op = GroupedGemmFwdOp(transpose_a=transpose_a, transpose_b=transpose_b, tune=tune)
    test.check(op, *test.gen_inputs())


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("transpose_b", [True, False], ids=["tn", "tt"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_k_grouped_gemm_with_ragged_groups(
    dtype: torch.dtype, transpose_b: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Groups split K: ragged group lengths, an empty group and output edges off the tile grid.

    A group whose length is not a multiple of 8 starts mid-vector, and the empty group's
    output block is zero.
    """
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False)
    sizes = [100, 0, 300, 64, 7, 1]
    test = GroupedGemmTest(sum(sizes), len(sizes), 200, 136, dtype, True, transpose_b)
    test.batch_sizes_list = sizes
    op = GroupedGemmFwdOp(transpose_a=True, transpose_b=transpose_b)
    test.check(op, *test.gen_inputs())


# Which kernel serves which call


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "numel, n, k, transpose_a, transpose_b, expected",
    [
        (4096, 4096, 4096, False, True, "GroupedGemmPersistentKernel"),
        (
            4096,
            4000,
            4096,
            False,
            True,
            "GroupedGemmPersistentKernel",
        ),  # N off the tile grid still runs
        (4096, 4096, 4096, False, False, "GroupedGemmPersistentKernel"),  # NN
        (4096, 4096, 4096, True, False, "GroupedGemmPersistentKernel"),  # TN
        (4099, 4096, 4096, True, True, "GroupedGemmKernel"),  # TT: b's row pitch is the K sum
        (4096, 4096, 4100, False, True, "GroupedGemmKernel"),  # K TMA cannot address
    ],
)
@pytest.mark.smoke
def test_selection_prefers_the_template_where_tma_can_address_the_operands(
    numel: int, n: int, k: int, transpose_a: bool, transpose_b: bool, expected: str
):
    """The SM90 template serves every layout with 8-aligned extents; the general kernel the rest."""
    op = GroupedGemmFwdOp(transpose_a=transpose_a, transpose_b=transpose_b)
    call = GroupedGemmCall(
        arch=90,
        numel=numel,
        num_experts=16,
        n=n,
        k=k,
        dtype=torch.float16,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )
    key = op.select_implementation("grouped_gemm", call)
    assert op.kernel_map[key].__name__ == expected
