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

# Tile height both tests build the tiling with and compute the host tile counts from.
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
        pytest.param(8, 128, 1, "int32", marks=pytest.mark.full),
        # The fewest groups the scan takes: one warp, its run ending short.
        pytest.param(9, 128, 1, "int32", marks=pytest.mark.full),
        # Two warps carry into each other, and a row unit of two rows.
        pytest.param(33, 384, 2, "int32", marks=pytest.mark.full),
        # Four warps of two runs: the third warp's second run is empty, the fourth has none.
        pytest.param(129, 128, 1, "int32", marks=pytest.mark.smoke),
        # Three passes of two warps carry into each other, offsets read at 64 bits.
        pytest.param(5000, 64, 1, "int64", marks=pytest.mark.full),
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


@pytest.mark.parametrize(
    "groups",
    [
        # The loop every thread runs.
        pytest.param(8, marks=pytest.mark.full),
        # The scan, two warps carrying into each other.
        pytest.param(33, marks=pytest.mark.smoke),
    ],
)
def test_cumsum_by_caller_tile_counts(groups):
    """A caller's own tile count, here read from two buffers, sums like the built-in ones."""
    torch.manual_seed(0)
    lengths = torch.randint(0, 3 * _BLOCK_M, (groups,), dtype=torch.int32)
    extra = torch.randint(0, 4, (groups,), dtype=torch.int32)
    tiling = GroupTiling(groups, _BLOCK_M)

    def tiles(src, g):
        sizes, adds = src
        return (sizes[g] + T.int32(_BLOCK_M - 1)) // T.int32(_BLOCK_M) + adds[g]

    build = tiling.cumsum_by(tiles)

    @tilelang.jit(out_idx=[2])
    def program():
        @T.prim_func
        def main(
            sizes: T.Tensor((groups,), "int32"),
            adds: T.Tensor((groups,), "int32"),
            out: T.Tensor((groups + 1,), "int32"),
        ):
            with T.Kernel(1, threads=384):
                s_cum = T.alloc_shared([groups + 1], "int32")
                build((sizes, adds), s_cum)
                for g in T.Parallel(groups + 1):
                    out[g] = s_cum[g]

        return main

    got = program()(lengths.cuda(), extra.cuda())
    tiles_host = (lengths.long() + _BLOCK_M - 1) // _BLOCK_M + extra.long()
    expected = torch.cat([torch.zeros(1, dtype=torch.int64), tiles_host.cumsum(0)])
    assert torch.equal(got.cpu().long(), expected)
