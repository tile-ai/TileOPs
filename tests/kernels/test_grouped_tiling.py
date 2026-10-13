"""``GroupTiling``'s tile-count prefix sums match a host prefix sum.

Each case builds a one-CTA program that writes the prefix sum to global memory, from
group lengths that include empty groups.
"""

import pytest
import tilelang
import tilelang.language as T
import torch

from tileops.kernels.grouped_tiling import GroupTiling

pytestmark = [pytest.mark.cuda_only]

_BLOCK_M = 64


def _program(tiling: GroupTiling, from_offsets: bool, threads: int, index_dtype: str):
    groups = tiling.num_groups
    extent = groups + 1 if from_offsets else groups
    build = tiling.cumsum_offsets if from_offsets else tiling.cumsum

    @tilelang.jit(out_idx=[1])
    def program():
        @T.prim_func
        def main(src: T.Tensor((extent,), index_dtype), out: T.Tensor((groups + 1,), "int32")):
            with T.Kernel(1, threads=threads):
                s_cum = T.alloc_shared([groups + 1], "int32")
                build(src, s_cum)
                for g in T.Parallel(groups + 1):
                    out[g] = s_cum[g]

        return main

    return program()


@pytest.mark.parametrize(
    "groups, threads, rows_per_offset, index_dtype",
    [
        # The loop every thread runs for the few groups below the scan.
        pytest.param(8, 128, 1, "int32", marks=pytest.mark.smoke),
        # The fewest groups the scan takes: one warp, its run ending short.
        pytest.param(9, 128, 1, "int32", marks=pytest.mark.full),
        # Two warps carry into each other, and a row unit of two rows.
        pytest.param(33, 384, 2, "int32", marks=pytest.mark.full),
        # Every warp takes eight runs and the last warp's runs end short.
        pytest.param(1000, 128, 1, "int32", marks=pytest.mark.full),
        # Offsets read at 64 bits.
        pytest.param(1000, 256, 1, "int64", marks=pytest.mark.full),
    ],
)
def test_cumsum_offsets(groups, threads, rows_per_offset, index_dtype):
    torch.manual_seed(0)
    lengths = torch.randint(0, 3 * _BLOCK_M, (groups,))
    lengths[3::7] = 0
    offsets = torch.zeros(groups + 1, dtype=getattr(torch, index_dtype))
    offsets[1:] = lengths.cumsum(0)
    tiling = GroupTiling(groups, _BLOCK_M, rows_per_offset=rows_per_offset)
    got = _program(tiling, True, threads, index_dtype)(offsets.cuda())
    tiles = (lengths * rows_per_offset + _BLOCK_M - 1) // _BLOCK_M
    expected = torch.cat([torch.zeros(1, dtype=torch.int64), tiles.cumsum(0)])
    assert torch.equal(got.cpu().long(), expected)


@pytest.mark.full
def test_cumsum_sizes():
    groups = 1000
    torch.manual_seed(0)
    sizes = torch.randint(0, 3 * _BLOCK_M, (groups,), dtype=torch.int32)
    sizes[3::7] = 0
    got = _program(GroupTiling(groups, _BLOCK_M), False, 256, "int32")(sizes.cuda())
    tiles = (sizes.long() + _BLOCK_M - 1) // _BLOCK_M
    expected = torch.cat([torch.zeros(1, dtype=torch.int64), tiles.cumsum(0)])
    assert torch.equal(got.cpu().long(), expected)
