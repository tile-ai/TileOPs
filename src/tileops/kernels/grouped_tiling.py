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
from tilelang.language.kernel import KernelLaunchFrame

from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["GroupTiling"]


class GroupTiling:
    """Tiles enumerated per group, for ``num_groups`` groups of ``block_m`` rows.

    ``rows_per_offset`` is how many rows one unit of a packed offset carries, for a caller
    whose row axis holds more than one row per packed token: a grouped-query attention tile
    that folds a KV head's query heads into its rows carries ``heads // heads_kv`` of them.

    Example:
        ```python linenums="1"
        tiling = GroupTiling(num_groups=16, block_m=64)
        tiling.tile_upper_bound(4096) # 80
        ```
    """

    def __init__(self, num_groups: int, block_m: int, rows_per_offset: int = 1) -> None:
        self.num_groups = num_groups
        self.block_m = block_m
        self.rows_per_offset = rows_per_offset
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
        block_m = self.block_m

        def tiles(sizes, g):
            return (sizes[g] + T.int32(block_m - 1)) // T.int32(block_m)

        return _tile_prefix_sum(self.num_groups, tiles)

    @property
    def cumsum_offsets(self):
        """Build the tile-count prefix sum from packed row offsets.

        ``(offsets[g + 1] - offsets[g]) * rows_per_offset`` is the row count of group ``g``.
        Varlen operators already receive that representation, so they should
        not materialize a second sizes tensor merely to schedule row tiles.
        The offsets are read at whatever integer width they arrive in.
        """
        block_m, rows = self.block_m, self.rows_per_offset

        def tiles(offsets, g):
            size = T.cast(offsets[g + 1] - offsets[g], "int32") * T.int32(rows)
            return (size + T.int32(block_m - 1)) // T.int32(block_m)

        return _tile_prefix_sum(self.num_groups, tiles)

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


def _tile_prefix_sum(num_groups, tiles):
    """Macro writing ``s_cum[g + 1] = tiles(src, 0) + ... + tiles(src, g)`` and ``s_cum[0] = 0``.

    Every CTA of the caller's grid computes the whole sum, so the CTA's threads share
    it: each warp scans a contiguous run of groups, one group a lane per round, with
    shuffles, and adds the totals of the warps before it. The macro ends with a block
    barrier, and every thread of the block must reach it.
    """
    # Up to this many groups the scan's shuffles and barriers cost paged attention up to
    # 5% over a loop on every thread (H200); from 16 groups on the scan is no slower.
    serial_groups = 8
    if num_groups <= serial_groups:

        @T.macro
        def group_tile_prefix_sum(src, s_cum):
            s_cum[0] = T.int32(0)
            for g in T.serial(num_groups):
                s_cum[g + 1] = s_cum[g] + tiles(src, g)

        return group_tile_prefix_sum

    @T.macro
    def group_tile_prefix_scan(src, s_cum):
        threads, rest_y, rest_z = KernelLaunchFrame.Current().get_thread_extents()
        if threads % WARP_LANES != 0 or rest_y * rest_z != 1:
            raise ValueError(f"the scan needs whole warps along threadIdx.x, got {threads}")
        # Only as many warps as there are runs of WARP_LANES groups take part. A thread
        # holds one tile count a round in registers, and the rounds unroll: past
        # `max_rounds` rounds the CTA scans in passes, each carrying the passes before it.
        # 32 rounds is 4096 groups on 128 threads, and holds 56 registers (H200).
        max_rounds = 32
        runs = -(-num_groups // WARP_LANES)
        warps = min(threads // WARP_LANES, runs)
        passes = -(-runs // (warps * max_rounds))
        rounds = -(-runs // (warps * passes))
        span = warps * rounds * WARP_LANES
        tx = T.get_thread_binding()
        lane = tx % WARP_LANES
        warp = tx // WARP_LANES
        vals = T.alloc_local([rounds], "int32")
        running = T.alloc_local([1], "int32")
        carry = T.alloc_local([1], "int32")
        for p in T.serial(passes):
            first = p * span + warp * WARP_LANES * rounds + lane
            if tx < warps * WARP_LANES:
                for r in T.unroll(rounds):
                    vals[r] = T.int32(0)
                    if first + r * WARP_LANES < num_groups:
                        vals[r] = tiles(src, first + r * WARP_LANES)
                for r in T.unroll(rounds):
                    for step in T.unroll(WARP_SHUFFLE_STAGES):
                        up = T.shfl_up(vals[r], T.int32(1) << step)
                        if lane >= (T.int32(1) << step):
                            vals[r] = vals[r] + up
                running[0] = T.int32(0)
                for r in T.unroll(rounds):
                    vals[r] = vals[r] + running[0]
                    running[0] = T.shfl_sync(vals[r], WARP_LANES - 1)
                # s_cum[p * span + 1 + warp] holds each warp's total until every warp
                # has read it.
                if warps > 1 and lane == WARP_LANES - 1:
                    s_cum[p * span + warp + 1] = running[0]
            carry[0] = T.int32(0)
            # The previous pass ended with a barrier, so its last sum is final.
            if passes > 1 and p > 0:
                carry[0] = s_cum[p * span]
            if warps > 1:
                T.sync_threads()
                for w in T.serial(warps - 1):
                    if w < warp:
                        carry[0] = carry[0] + s_cum[p * span + w + 1]
                T.sync_threads()
            if tx < warps * WARP_LANES:
                for r in T.unroll(rounds):
                    if first + r * WARP_LANES < num_groups:
                        s_cum[first + r * WARP_LANES + 1] = vals[r] + carry[0]
            if passes > 1:
                T.sync_threads()
        if tx == 0:
            s_cum[0] = T.int32(0)
        T.sync_threads()

    return group_tile_prefix_scan
