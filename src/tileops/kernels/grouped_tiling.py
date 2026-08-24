"""Mapping a flat tile id onto the group that owns it.

Rows are laid out per group, so an M tile belongs to one group only if the tiles
are enumerated per group. Enumerated over the whole row range instead, a tile can
straddle a boundary, and one that resolves a single group computes only the first
of the ones it covers.

The prefix sum and the search read the same tile counts, so one object holds the
group count and tile height both of them are built from, and the macros take the
caller's buffers so each kernel keeps its own scheduler.
"""

import tilelang.language as T

__all__ = ["GroupTiling"]


class GroupTiling:
    """Tiles enumerated per group, for ``num_groups`` groups of ``block_m`` rows.

    Example:
        ```python linenums="1"
        tiling = GroupTiling(num_groups=16, block_m=64)
        tiling.tile_upper_bound(4096) # 80
        ```
    """

    def __init__(self, num_groups: int, block_m: int) -> None:
        self.num_groups = num_groups
        self.block_m = block_m
        self._search_steps = max(1, (num_groups - 1).bit_length())

    def tile_upper_bound(self, numel: int) -> int:
        """Tiles ``numel`` rows can need, at most one partial tile per group.

        Sizes arrive on the device, so a grid takes this bound and drops the
        tiles past the count the kernel finds at run time.
        """
        return numel // self.block_m + self.num_groups

    @property
    def cumsum(self):
        """Macro writing the per-group tile-count prefix sum into ``s_cum``.

        ``s_cum[g + 1] - s_cum[g]`` is the tiles group *g* needs, and
        ``s_cum[num_groups]`` is how many the call has.
        """
        num_groups, block_m = self.num_groups, self.block_m

        @T.macro
        def group_tile_cumsum(sizes, s_cum):
            s_cum[0] = T.int32(0)
            for g in T.serial(num_groups):
                s_cum[g + 1] = s_cum[g] + (sizes[g] + T.int32(block_m - 1)) // T.int32(block_m)

        return group_tile_cumsum

    @property
    def decode(self):
        """Macro searching ``s_cum`` for the group owning one M tile id.

        ``row`` is the tile's first row within its group: the caller reads the
        packed start as ``offsets[group] + row``.
        """
        num_groups, block_m, steps = self.num_groups, self.block_m, self._search_steps

        @T.macro
        def group_tile_decode(m_tile, s_cum, lo, hi, group, row):
            lo[0] = T.int32(0)
            hi[0] = T.int32(num_groups - 1)
            for _ in T.serial(steps):
                mid = (lo[0] + hi[0]) >> T.int32(1)
                if s_cum[mid + 1] <= m_tile:
                    lo[0] = mid + T.int32(1)
                else:
                    hi[0] = mid
            group[0] = lo[0]
            row[0] = (m_tile - s_cum[lo[0]]) * T.int32(block_m)

        return group_tile_decode
