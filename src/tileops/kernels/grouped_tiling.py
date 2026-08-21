"""Mapping a flat tile id onto the group that owns it.

A grouped GEMM lays its rows out per group, so an M tile belongs to one group
only if the tiles are enumerated per group rather than over the whole row range.
Enumerating them globally lets a tile straddle a group boundary, and a tile that
resolves a single group computes only the first of the ones it covers.

Both primitives here run inside a kernel and take the caller's buffers, so the
caller keeps its own scheduler: a static wave, a plain grid, or a separate
scheduling launch all decode a tile id the same way.
"""

import tilelang.language as T

__all__ = ["make_group_tile_cumsum", "make_group_tile_decode", "tile_upper_bound"]


def tile_upper_bound(numel: int, num_groups: int, block_m: int) -> int:
    """Tiles ``numel`` rows can need once every group starts a fresh tile.

    Each group rounds up to a whole tile, so the worst case adds one partial tile
    per group. Sizes arrive on the device, which is why this is a bound and the
    exact count is read from the prefix sum at run time.
    """
    return numel // block_m + num_groups


def make_group_tile_cumsum(num_groups: int, block_m: int):
    """Build the macro that writes the per-group tile-count prefix sum.

    ``s_cum[g + 1] - s_cum[g]`` is the tiles group *g* needs, so ``s_cum`` maps a
    tile id to its group by search, and ``s_cum[num_groups]`` is how many tiles
    the call actually has.

    Args:
        num_groups: Groups the rows are spread over.
        block_m: Rows one tile covers.
    """

    @T.macro
    def group_tile_cumsum(sizes, s_cum):
        s_cum[0] = T.int32(0)
        for g in T.serial(num_groups):
            s_cum[g + 1] = s_cum[g] + (sizes[g] + T.int32(block_m - 1)) // T.int32(block_m)

    return group_tile_cumsum


def make_group_tile_decode(num_groups: int, block_m: int):
    """Build the macro that resolves one M tile id into its group and first row.

    ``row`` is the tile's first row within its group, so the caller reads its
    start in the packed layout as ``offsets[group] + row`` and clamps the tile
    with ``sizes[group] - row``.

    Args:
        num_groups: Groups the rows are spread over.
        block_m: Rows one tile covers.
    """
    log2_up = max(1, (num_groups - 1).bit_length())

    @T.macro
    def group_tile_decode(m_tile, s_cum, lo, hi, group, row):
        lo[0] = T.int32(0)
        hi[0] = T.int32(num_groups - 1)
        for _ in T.serial(log2_up):
            mid = (lo[0] + hi[0]) >> T.int32(1)
            if s_cum[mid + 1] <= m_tile:
                lo[0] = mid + T.int32(1)
            else:
                hi[0] = mid
        group[0] = lo[0]
        row[0] = (m_tile - s_cum[lo[0]]) * T.int32(block_m)

    return group_tile_decode
