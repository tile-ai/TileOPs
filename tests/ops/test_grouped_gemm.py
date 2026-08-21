import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.grouped_gemm import GroupedGemmKernel
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


# What `tune=True` measures


class _FakeKernelParam:
    """The part of TileLang's ``KernelParam`` its tensor supplier reads."""

    def __init__(self, dtype: str, shape: list[int]) -> None:
        self.dtype = dtype
        self.shape = shape

    def torch_dtype(self):
        return getattr(torch, self.dtype)

    def __getattr__(self, name):  # is_unsigned / is_float8 / is_float4 / is_boolean
        return lambda: False


@pytest.mark.smoke
def test_supply_prog_keeps_every_row_in_the_k_loop():
    """Random int32 metadata drops the NT/NN guard sum to ~0 and every tile skips the K-loop."""
    batch_sum, batch_count, n, k = 64, 8, 32, 32
    kernel = GroupedGemmKernel(batch_sum, batch_count, n, k, torch.float16)
    # TileLang supplies inputs only, so the ``out_idx=[2]`` output is absent.
    params = [
        _FakeKernelParam("float16", [batch_sum, k]),
        _FakeKernelParam("float16", [batch_count, n, k]),
        *(_FakeKernelParam("int32", [batch_count]) for _ in range(3)),
    ]
    supplied = kernel.autotune_supply_prog(params)

    assert [list(t.shape) for t in supplied] == [p.shape for p in params]
    sizes, offsets, padded_offsets = supplied[2:]
    assert int(sizes.sum()) == batch_sum
    assert int(offsets[0]) == 0 and int(offsets[-1]) == batch_sum - int(sizes[-1])
    assert int(padded_offsets[-1]) + int(sizes[-1]) == batch_sum

    # A fourth such parameter must fail rather than silently receive the offsets.
    with pytest.raises(RuntimeError, match="expects 3 int32"):
        kernel.autotune_supply_prog(params + [_FakeKernelParam("int32", [batch_count])])
