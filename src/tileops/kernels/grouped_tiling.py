"""Mapping a flat tile id onto the group that owns it.

Rows are laid out per group, so an M tile belongs to one group only if the tiles
are enumerated per group. Enumerated over the whole row range instead, a tile can
straddle a boundary, and one that resolves a single group computes only the first
of the ones it covers.

The primitives run inside a kernel and take the caller's buffers, so each caller
keeps its own scheduler.
"""

import tilelang.language as T

__all__ = ["make_group_tile_cumsum", "make_group_tile_decode", "tile_upper_bound"]


def tile_upper_bound(numel: int, num_groups: int, block_m: int) -> int:
    """Tiles ``numel`` rows can need, at most one partial tile per group.

    Sizes arrive on the device, so a grid takes this bound and reads the exact
    count off ``s_cum[num_groups]``.
    """
    return numel // block_m + num_groups


def make_group_tile_cumsum(num_groups: int, block_m: int):
    """Build the macro writing the per-group tile-count prefix sum into ``s_cum``.

    ``s_cum[g + 1] - s_cum[g]`` is the tiles group *g* needs, and
    ``s_cum[num_groups]`` is how many the call has.
    """

    @T.macro
    def group_tile_cumsum(sizes, s_cum):
        s_cum[0] = T.int32(0)
        for g in T.serial(num_groups):
            s_cum[g + 1] = s_cum[g] + (sizes[g] + T.int32(block_m - 1)) // T.int32(block_m)

    return group_tile_cumsum


def make_group_tile_decode(num_groups: int, block_m: int):
    """Build the macro resolving one M tile id into its group and first row.

    ``row`` is the tile's first row within its group: the caller reads the packed
    start as ``offsets[group] + row`` and clamps the tile with
    ``sizes[group] - row``.
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
