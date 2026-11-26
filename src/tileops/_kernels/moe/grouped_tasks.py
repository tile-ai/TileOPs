"""Task walk and launch-config plumbing shared by the persistent M-grouped GEMM kernels.

A persistent CTA walks tasks a grid apart, ``num_sms * wave + pid``, or ``grid * wave + worker``
over the working CTAs of a grid that balances its tasks at run time; each task is one
``block_m``-row tile of one expert's rows by one ``block_n``-column tile. ``task_walk`` builds
the TileLang pieces that resolve a task id to its expert and tile for masked slabs and for tight
or aligned contiguous runs, store a task's output tile, and describe row and expert-weight tiles
to TMA; ``table_task_walk`` finds the same tasks' experts from a per-CTA prefix of the experts'
row tiles instead; ``GroupedGEMMConfigMixin`` holds the config checks the kernels share.
"""

from collections.abc import Mapping
from types import SimpleNamespace

import tilelang
import tilelang.language as T

from tileops._kernels.constants import TMA_INTERLEAVE_NONE, TMA_OOB_FILL_NONE

__all__ = [
    "CURSOR",
    "GroupedGEMMConfigMixin",
    "table_task_walk",
    "task_walk",
    "walk_refusal",
    "walk_smem_bytes",
]

# Cursor fields: expert, preceding M tiles, expert tiles, first row, row count.
CURSOR = SimpleNamespace(g=0, m=1, tiles=2, row0=3, rows=4)


def task_walk(
    *,
    num_groups: int,
    block_m: int,
    block_n: int,
    n: int,
    num_sms: int,
    max_m: int | None,
    packing: str | None,
    metadata_kind: str | None,
    alignment: int,
):
    """The task walk of one kernel build.

    Args:
        num_groups: Experts.
        block_m: Rows of a row tile; under aligned packing it divides ``alignment``.
        block_n: Columns of a column tile, ``n_tiles`` of which make the fastest task index.
        n: Columns of the output.
        num_sms: Persistent CTAs.
        max_m: A masked slab's rows; ``None`` for contiguous rows.
        packing: ``"tight"``, ``"aligned"``, or ``None`` for masked slabs.
        metadata_kind: ``"per_row"`` or ``"physical_psum"`` for contiguous rows.
        alignment: An aligned layout's run alignment.

    Returns:
        A namespace of the launch extents (``m``, the dynamic row count of contiguous rows;
        ``rows_shape``, ``ragged_slab``, ``max_waves``, ``cursor_len``),
        ``walk(layout_or_ends, task_id, cursor)`` (resolves a task to an expert, ``num_groups``
        marking exhaustion), ``tile_of(task_id, cursor, tile)``, ``slab_tiles(count)`` (a
        masked slab's row tiles, its count clamped to ``[0, max_m]``), the task's clamped
        ``expert(cursor)``, ``first_row(cursor, tile)`` and ``first_col(tile)``,
        ``store_tile(...)``, ``row_tiles_descriptor(...)``, ``expert_rows_descriptor(...)``,
        ``recover_ends(layout, ends, tx, threads)`` and the flag ``recover`` (tight per-row
        ids need each expert's end recovered first).
    """
    contiguous = packing is not None
    tight = packing == "tight"
    # Aligned per-row ids name each row tile's expert; tight ones need each expert's end.
    rows_walk = metadata_kind == "per_row" and not tight
    recover = metadata_kind == "per_row" and tight
    # M tiles grouped to share weight tiles within a wave.
    swizzle_m_tiles = 16
    n_tiles = tilelang.cdiv(n, block_n)
    # The last N tile runs past N: its loads read zeros and its stores are clipped.
    ragged_n = n % block_n != 0
    swizzle_span = swizzle_m_tiles * n_tiles
    if contiguous:
        m = T.dynamic("m")
        m_tiles = T.ceildiv(m, block_m)
        # A tight expert's last tile may hold fewer than block_m rows; aligned runs tile their
        # padded extent exactly.
        max_waves = T.ceildiv((m_tiles + (num_groups if tight else 0)) * n_tiles, num_sms)
        rows_shape = (m,)
        tiles_per_slab = ragged_slab = None
    else:
        m = None
        tiles_per_slab = tilelang.cdiv(max_m, block_m)
        # The sequence is never longer than every tile of every slab.
        max_waves = tilelang.cdiv(num_groups * tiles_per_slab * n_tiles, num_sms)
        # The last tile of a slab whose rows end inside it reads no scale past max_m.
        ragged_slab = max_m % block_m != 0
        rows_shape = (num_groups, max_m)
    cur = CURSOR

    def slab_tiles(count):
        """The row tiles of a masked slab of ``count`` rows, ``count`` clamped to ``[0, max_m]``."""
        return (T.max(T.min(count, max_m), 0) + block_m - 1) // block_m

    @T.macro
    def walk_to(layout, task_id, cursor):
        """Resolve a masked task to an expert; ``num_groups`` marks exhaustion."""
        while cursor[cur.g] < num_groups:
            cursor[cur.tiles] = slab_tiles(layout[cursor[cur.g]])
            if task_id < (cursor[cur.m] + cursor[cur.tiles]) * n_tiles:
                break
            cursor[cur.m] = cursor[cur.m] + cursor[cur.tiles]
            cursor[cur.g] = cursor[cur.g] + 1

    @T.macro
    def walk_ends(ends, task_id, cursor):
        """Resolve a task to an expert run, retaining its row bounds and tile count."""
        while cursor[cur.g] < num_groups:
            prev_end = T.if_then_else(cursor[cur.g] > 0, ends[T.max(cursor[cur.g] - 1, 0)], 0)
            if tight:
                cursor[cur.row0] = prev_end
            else:
                cursor[cur.row0] = T.ceildiv(prev_end, alignment) * alignment
            cursor[cur.rows] = T.max(ends[cursor[cur.g]] - cursor[cur.row0], 0)
            cursor[cur.tiles] = (cursor[cur.rows] + block_m - 1) // block_m
            if task_id < (cursor[cur.m] + cursor[cur.tiles]) * n_tiles:
                break
            cursor[cur.m] = cursor[cur.m] + cursor[cur.tiles]
            cursor[cur.g] = cursor[cur.g] + 1

    @T.macro
    def walk_rows(layout, task_id, cursor):
        """Resolve an aligned row tile from its first expert id, with N tiles fastest.

        Id ``num_groups`` denotes padding; each row tile occupies one cursor run.
        """
        cursor[cur.g] = num_groups
        if task_id < m_tiles * n_tiles:
            cursor[cur.m] = task_id // n_tiles
            cursor[cur.tiles] = 1
            cursor[cur.row0] = cursor[cur.m] * block_m
            cursor[cur.rows] = block_m
            cursor[cur.g] = T.min(layout[cursor[cur.row0]], num_groups)

    @T.macro
    def tile_of(task_id, cursor, tile):
        """Map a valid expert-local task to its M/N tile.

        Requires ``0 <= local < m_tiles * n_tiles`` and ``m_tiles >= 1``.
        """
        local = task_id - cursor[cur.m] * n_tiles
        expert_tiles = cursor[cur.tiles]
        if not contiguous and tiles_per_slab <= swizzle_m_tiles:
            # One group spans every M tile of the expert. A runtime divisor costs a long
            # division, so each of the tiles_per_slab values the count can take gets its own.
            for i in T.unroll(tiles_per_slab):
                if expert_tiles == i + 1:
                    tile[0] = T.truncmod(local, i + 1)
                    tile[1] = T.truncdiv(local, i + 1)
        else:
            first = T.truncdiv(local, swizzle_span) * swizzle_m_tiles
            in_group = T.truncmod(local, swizzle_span)
            group_m_tiles = T.min(swizzle_m_tiles, expert_tiles - first)
            tile[0] = first + T.truncmod(in_group, group_m_tiles)
            tile[1] = T.truncdiv(in_group, group_m_tiles)

    walk = walk_rows if rows_walk else walk_ends if contiguous else walk_to

    # Bounds inline in each use, which spares TileLang's guards on the loads they index.
    def expert(cursor):
        return T.max(T.min(cursor[cur.g], num_groups - 1), 0)

    def first_row(cursor, tile):
        if contiguous:
            return cursor[cur.row0] + tile[0] * block_m
        return T.max(T.min(tile[0], tiles_per_slab - 1), 0) * block_m

    def first_col(tile):
        return T.max(T.min(tile[1], n_tiles - 1), 0) * block_n

    @T.macro
    def store_tile(output_desc, shared_c, C, cursor, tile, box_cols, box_count, tx, threads):
        """Store the task's staged ``block_m x block_n`` bfloat16 tile from ``shared_c``, boxes of
        ``box_cols`` columns: one TMA store a box by threads ``tx < box_count``, boxes past N
        skipped; a tight tail tile, by rows, so the next expert's rows stay as they are."""
        g = T.meta_var(expert(cursor))
        m_start = T.meta_var(first_row(cursor, tile))
        n_start = T.meta_var(first_col(tile))
        if ragged_n:
            store_box = T.And(tx < box_count, n_start + tx * box_cols < n)
        else:
            store_box = tx < box_count
        short_tile = T.meta_var(cursor[cur.rows] - tile[0] * block_m < block_m) if tight else False
        if not contiguous:
            if store_box:
                T.call_extern(
                    "handle",
                    "tileops::moe_fp8_tma_store_3d_issue",
                    output_desc,
                    T.address_of(shared_c[tx * block_m * box_cols]),
                    n_start + tx * box_cols,
                    m_start,
                    g,
                )
                T.tma_store_arrive()
        elif short_tile:
            T.call_extern(
                "handle",
                f"tileops::moe_fp8_store_rows_bf16_swizzled<{block_m}, {block_n}>",
                T.address_of(shared_c[0]),
                T.address_of(C[m_start, n_start]),
                n,
                T.min(cursor[cur.rows] - tile[0] * block_m, m - m_start),
                n - n_start,
                tx,
                threads,
            )
        else:
            if store_box:
                T.call_extern(
                    "handle",
                    "tileops::moe_fp8_tma_store_2d_issue",
                    output_desc,
                    T.address_of(shared_c[tx * block_m * box_cols]),
                    n_start + tx * box_cols,
                    m_start,
                )
                T.tma_store_arrive()

    @T.macro
    def recover_ends(layout, ends, tx, threads):
        """Each expert's end: the first row whose id is past it, ids not decreasing."""
        for group_step in T.serial(T.ceildiv(num_groups, threads)):
            group = group_step * threads + tx
            if group < num_groups:
                lo = T.alloc_var("int32", init=0)
                hi = T.alloc_var("int32", init=m)
                for _ in T.serial(32):
                    if lo < hi:
                        mid = (lo + hi) >> 1
                        if layout[mid] <= group:
                            lo = mid + 1
                        else:
                            hi = mid
                ends[group] = lo

    def row_tiles_descriptor(dtype, tensor, row, row_bytes, box_row, swizzle, promotion):
        """A TMA descriptor of ``block_m``-row tiles of a tensor laid out like ``a``.

        Its rows are ``row`` elements over ``row_bytes`` bytes: ``[m, row]`` for contiguous
        rows, ``[E, max_m, row]`` for masked slabs; a box is ``box_row`` elements of each of
        ``block_m`` rows.
        """
        if contiguous:
            return T.create_tma_descriptor(
                dtype,
                2,
                tensor.data,
                row,
                m,
                1,
                row_bytes,
                box_row,
                block_m,
                1,
                1,
                TMA_INTERLEAVE_NONE,
                swizzle,
                promotion,
                TMA_OOB_FILL_NONE,
            )
        return T.create_tma_descriptor(
            dtype,
            3,
            tensor.data,
            row,
            max_m,
            num_groups,
            1,
            row_bytes,
            max_m * row_bytes,
            box_row,
            block_m,
            1,
            1,
            1,
            1,
            TMA_INTERLEAVE_NONE,
            swizzle,
            promotion,
            TMA_OOB_FILL_NONE,
        )

    def expert_rows_descriptor(dtype, tensor, row, row_bytes, box_row, swizzle, promotion):
        """A TMA descriptor of ``block_n``-row tiles of an ``[E, n, row]`` tensor of rows of
        ``row`` elements over ``row_bytes`` bytes; a box is ``box_row`` elements of each of
        ``block_n`` rows."""
        return T.create_tma_descriptor(
            dtype,
            3,
            tensor.data,
            row,
            n,
            num_groups,
            1,
            row_bytes,
            n * row_bytes,
            box_row,
            block_n,
            1,
            1,
            1,
            1,
            TMA_INTERLEAVE_NONE,
            swizzle,
            promotion,
            TMA_OOB_FILL_NONE,
        )

    return SimpleNamespace(
        m=m,
        rows_shape=rows_shape,
        ragged_slab=ragged_slab,
        max_waves=max_waves,
        cursor_len=5 if contiguous else 3,
        walk=walk,
        slab_tiles=slab_tiles,
        tile_of=tile_of,
        expert=expert,
        first_row=first_row,
        first_col=first_col,
        store_tile=store_tile,
        recover_ends=recover_ends,
        row_tiles_descriptor=row_tiles_descriptor,
        expert_rows_descriptor=expert_rows_descriptor,
        recover=recover,
    )


def walk_smem_bytes(num_groups: int, packing: str | None, metadata_kind: str | None) -> int:
    """Shared memory ``table_task_walk`` keeps beside a kernel's tiles: each expert's tile end,
    and under tight per-row ids each expert's row end; aligned per-row ids name each row tile's
    expert and keep none."""
    if packing == "aligned" and metadata_kind == "per_row":
        return 0
    return 4 * num_groups * (2 if packing == "tight" and metadata_kind == "per_row" else 1)


def walk_refusal(
    num_groups: int, packing: str | None, metadata_kind: str | None, room: int
) -> str | None:
    """Why ``num_groups`` experts' table-walk state does not fit in the ``room`` bytes of shared
    memory a kernel leaves beside its default tile on the shallowest ring, or ``None``."""
    per_expert = walk_smem_bytes(1, packing, metadata_kind)
    if not per_expert or per_expert * num_groups <= room:
        return None
    return (
        f"keeps task-walk state in shared memory, {per_expert} bytes an expert: at most "
        f"{room // per_expert} experts fit beside the default tile, got {num_groups}"
    )


def table_task_walk(
    *,
    num_groups: int,
    block_m: int,
    block_n: int,
    n: int,
    num_sms: int,
    max_m: int | None,
    packing: str | None,
    metadata_kind: str | None,
    alignment: int,
):
    """``task_walk``'s tasks and tiles, each task's expert found from shared memory.

    A CTA first sums the experts' row tiles into shared memory, unless aligned per-row ids name
    each row tile's expert; a walk then passes up to a warp's width of experts a step, and over
    contiguous rows the CTA runs as many waves as its tasks, so neither grows with experts that
    hold no rows. The FP8 kernel keeps ``task_walk``: the table's prologue runs between the
    layout load and its first TMA load, where ``task_walk`` compares one expert's tiles, and
    costs its decode rows more than the walk saves.

    Takes ``task_walk``'s arguments.

    Returns:
        ``task_walk``'s launch extents (``m``, ``rows_shape``, ``ragged_slab``,
        ``cursor_len``), ``tile_of``, ``expert``, ``first_row``, ``first_col``, ``store_tile``
        and descriptors, with ``prepare(layout, tx, threads)`` (the walk's ``source``, filled
        by all of the block's ``threads``), ``waves(source, worker, grid)``,
        ``walk(source, task_id, cursor)`` (walks a zeroed ``cursor`` on to a task's expert,
        ``num_groups`` marking exhaustion; a cursor's tasks come in increasing order) and
        ``walk_to_running(source, task_id, cursor, tile)`` (the first running task a wave apart
        from ``task_id`` on, resolved and tiled).
    """
    tasks = task_walk(
        num_groups=num_groups,
        block_m=block_m,
        block_n=block_n,
        n=n,
        num_sms=num_sms,
        max_m=max_m,
        packing=packing,
        metadata_kind=metadata_kind,
        alignment=alignment,
    )
    contiguous = packing is not None
    tight = packing == "tight"
    rows_walk = metadata_kind == "per_row" and not tight
    recover = metadata_kind == "per_row" and tight
    n_tiles = tilelang.cdiv(n, block_n)
    m_tiles = T.ceildiv(tasks.m, block_m) if contiguous else None
    runs = -(-num_groups // 32)
    cur = CURSOR

    def run_of(ends, group):
        """The first row and the row count of contiguous expert ``group``'s run."""
        prev_end = T.if_then_else(group > 0, ends[T.max(group - 1, 0)], 0)
        row0 = prev_end if tight else T.ceildiv(prev_end, alignment) * alignment
        return row0, T.max(ends[group] - row0, 0)

    def expert_tiles(rows, group):
        """The row tiles of expert ``group``."""
        if contiguous:
            return (run_of(rows, group)[1] + block_m - 1) // block_m
        return tasks.slab_tiles(rows[group])

    @T.macro
    def tile_prefix(rows, tile_ends, tx, threads):
        """``tile_ends[g]``, the row tiles of experts ``0 .. g``: each warp scans runs of a warp's
        width of experts by shuffles, then each expert adds the totals of the runs before its
        own. Past a warp's width of runs, one warp first scans the runs' totals the same way, a
        warp's width of runs a step, so each run's last expert holds its sum, and every other
        expert adds the sum of the last expert of the run before its own."""
        warps = threads // 32
        lane = tx % 32
        tiles = T.alloc_var("int32")
        for step in T.unroll(-(-runs // warps)):
            group = (step * warps + tx // 32) * 32 + lane
            tiles = 0
            if group < num_groups:
                tiles = expert_tiles(rows, group)
            for offset in (1, 2, 4, 8, 16):
                shifted = T.shfl_up(tiles, offset)
                if lane >= offset:
                    tiles = tiles + shifted
            if group < num_groups:
                tile_ends[group] = tiles
        if runs > 32:
            carry = T.alloc_var("int32", init=0)
            T.sync_threads()
            if tx < 32:
                for step in T.unroll(-(-runs // 32)):
                    run = step * 32 + lane
                    last = T.min(run * 32 + 31, num_groups - 1)
                    tiles = tile_ends[last]
                    for offset in (1, 2, 4, 8, 16):
                        shifted = T.shfl_up(tiles, offset)
                        if lane >= offset:
                            tiles = tiles + shifted
                    if run < runs:
                        tile_ends[last] = carry + tiles
                    carry = carry + T.shfl_sync(tiles, 31)
            T.sync_threads()
            for step in T.unroll(-(-num_groups // threads)):
                group = step * threads + tx
                if T.And(T.And(group >= 32, lane != 31), group < num_groups - 1):
                    tile_ends[group] = tile_ends[group] + tile_ends[group // 32 * 32 - 1]
        elif runs > 1:
            per_thread = -(-num_groups // threads)
            held = T.alloc_local((per_thread,), "int32")
            T.sync_threads()
            for step in T.unroll(per_thread):
                group = step * threads + tx
                held[step] = tile_ends[T.min(group, num_groups - 1)]
                for run in T.unroll(runs - 1):
                    if run < group // 32:
                        held[step] = held[step] + tile_ends[run * 32 + 31]
            T.sync_threads()
            for step in T.unroll(per_thread):
                group = step * threads + tx
                if group < num_groups:
                    tile_ends[group] = held[step]

    def prepare(layout, tx, threads):
        """Fill the shared memory the walk reads and sync the block's ``threads``, every one of
        which runs this; return the walk's source: the rows' metadata, each expert's end
        recovered under tight per-row ids, and each expert's tile end unless aligned per-row ids
        name each row tile's expert."""
        rows = layout
        if recover:
            rows = T.alloc_shared((num_groups,), "int32", scope="shared")
            tasks.recover_ends(layout, rows, tx, threads)
            T.sync_threads()
        tile_ends = None
        if not rows_walk:
            tile_ends = T.alloc_shared((num_groups,), "int32", scope="shared")
            tile_prefix(rows, tile_ends, tx, threads)
        T.sync_threads()
        return SimpleNamespace(rows=rows, tile_ends=tile_ends)

    def task_count(source):
        """The tasks of the call: every row tile of every expert, or of the rows under per-row
        ids, by every column tile."""
        if rows_walk:
            return m_tiles * n_tiles
        return source.tile_ends[num_groups - 1] * n_tiles

    def waves(source, worker, grid):
        """The waves CTA ``worker`` of ``grid`` runs, ``worker < grid``: over contiguous rows as
        many as its tasks; over masked slabs ``max_waves``, the walk marking each task past the
        last as not running."""
        if not contiguous:
            return tasks.max_waves
        count = T.ceildiv(task_count(source) - worker, grid)
        return count if rows_walk else T.alloc_var("int32", init=count)

    @T.macro
    def walk_tile_ends(source, task_id, cursor):
        """Walk ``cursor`` on to a task's expert, retaining its tile count and, for contiguous
        rows, its row bounds; ``num_groups`` marks a task past every expert's tiles. Every lane
        of the warp runs this: each step, lane ``i`` compares the ``i``-th expert's tile end
        from the cursor on, and every expert ending at or before the task's row tile is passed,
        up to a warp's width a step."""
        row_tile = task_id // n_tiles
        lane = T.get_lane_idx()
        passed = T.alloc_var("int32", init=32)
        while T.And(passed == 32, cursor[cur.g] < num_groups):
            member = cursor[cur.g] + lane
            end = source.tile_ends[T.min(member, num_groups - 1)]
            passed = T.Cast(
                "int32", T.popcount(T.ballot(T.And(member < num_groups, end <= row_tile)))
            )
            if passed > 0:
                cursor[cur.m] = T.shfl_sync(end, passed - 1)
            cursor[cur.tiles] = T.shfl_sync(end, T.min(passed, 31)) - cursor[cur.m]
            cursor[cur.g] = cursor[cur.g] + passed
        if contiguous:
            running = T.bind(cursor[cur.g] < num_groups)
            if running:
                row0, rows = run_of(source.rows, cursor[cur.g])
                cursor[cur.row0] = row0
                cursor[cur.rows] = rows

    @T.macro
    def walk_rows(source, task_id, cursor):
        """Resolve an aligned row tile from its first expert id, as ``task_walk`` does."""
        tasks.walk(source.rows, task_id, cursor)

    walk = walk_rows if rows_walk else walk_tile_ends

    @T.macro
    def walk_to_running(source, task_id, cursor, tile):
        """Walk ``cursor`` on to the first task from ``task_id`` on, a wave apart, that runs, and
        map it to ``tile``; ``cursor`` marks exhaustion when none of the CTA's tasks does. Only
        an aligned row tile of padding is skipped: every other task before the last runs."""
        next_task = T.alloc_var("int32", init=task_id)
        walk(source, next_task, cursor)
        if rows_walk:
            while T.And(cursor[cur.g] >= num_groups, next_task + num_sms < task_count(source)):
                next_task = next_task + num_sms
                walk(source, next_task, cursor)
        if cursor[cur.g] < num_groups:
            tasks.tile_of(next_task, cursor, tile)

    return SimpleNamespace(
        m=tasks.m,
        rows_shape=tasks.rows_shape,
        ragged_slab=tasks.ragged_slab,
        cursor_len=tasks.cursor_len,
        prepare=prepare,
        waves=waves,
        walk=walk,
        walk_to_running=walk_to_running,
        tile_of=tasks.tile_of,
        expert=tasks.expert,
        first_row=tasks.first_row,
        first_col=tasks.first_col,
        store_tile=tasks.store_tile,
        row_tiles_descriptor=tasks.row_tiles_descriptor,
        expert_rows_descriptor=tasks.expert_rows_descriptor,
    )


class GroupedGEMMConfigMixin:
    """Launch-config checks of a persistent M-grouped GEMM kernel.

    The kernel class defines ``_CONFIG_KEYS`` (``block_m``, ``block_n``, ``num_stages`` and
    ``num_sms`` first), their domains ``_BLOCK_M``, ``_BLOCK_N`` and ``_NUM_STAGES``,
    ``config_refusal(config, **inputs)`` and ``_refusal_inputs()`` (the device and layout facts
    ``config_refusal`` takes beside the config), and sets ``call``, ``sm_count`` before
    configuring.
    """

    _CONFIG_KEYS: tuple[str, ...]
    _BLOCK_M: tuple[int, ...]
    _BLOCK_N: tuple[int, ...]
    _NUM_STAGES: range

    @classmethod
    def _common_refusal(
        cls, config: Mapping[str, object], *, sm_count: int, alignment: int
    ) -> str | None:
        """The first of the key, integer, tile, ring, grid and alignment checks *config* fails,
        or ``None``: each kernel's ``config_refusal`` runs these before its own."""
        if set(config) != set(cls._CONFIG_KEYS):
            return f"needs exactly the keys {list(cls._CONFIG_KEYS)}, got {sorted(config)}"
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in config.values()):
            return "takes integer values"
        block_m, block_n, stages, num_sms = (config[key] for key in cls._CONFIG_KEYS[:4])
        if block_m not in cls._BLOCK_M:
            return f"block_m must be one of {cls._BLOCK_M}, got {block_m}"
        if block_n not in cls._BLOCK_N:
            return f"block_n must be one of {cls._BLOCK_N}, got {block_n}"
        if stages not in cls._NUM_STAGES:
            return (
                f"num_stages must lie in [{cls._NUM_STAGES.start}, {cls._NUM_STAGES.stop - 1}], "
                f"got {stages}"
            )
        if not 1 <= num_sms <= sm_count:
            return f"num_sms must lie in [1, {sm_count}], the device's SMs, got {num_sms}"
        if alignment % block_m:
            return f"block_m must divide the layout's alignment {alignment}, got {block_m}"
        return None

    def _refusal_inputs(self) -> dict:
        raise NotImplementedError

    def _check_config(self, config: Mapping[str, object]) -> None:
        """Validate a launch configuration against device and layout limits."""
        reason = self.config_refusal(config, **self._refusal_inputs())
        if reason is not None:
            raise ValueError(f"{type(self).__name__} cannot run config {dict(config)}: {reason}")

    def init_config(self, config: dict | None = None) -> None:
        """Set the config as the base class does, then check it.

        Raises:
            ValueError: *config* names a key the kernel does not take, or the resulting
                config is one ``config_refusal`` refuses.
        """
        if config is not None and not set(config) <= set(self._CONFIG_KEYS):
            raise ValueError(
                f"{type(self).__name__} takes the config keys {list(self._CONFIG_KEYS)}, "
                f"got {sorted(config)}"
            )
        super().init_config(config)
        self._check_config(self.config)

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        super().autotune(warmup=warmup, rep=rep)
        self._check_config(self.config)

    def _slab_tiles(self, block_m: int, block_n: int) -> int:
        """The tasks of every tile of every masked slab, each slab full."""
        call = self.call
        return call.num_groups * -(-call.max_m // block_m) * -(-call.n // block_n)

    def _balanced_sms(self, block_m: int, block_n: int) -> int:
        """Minimize idle blocks for masked slabs; use all SMs for contiguous layouts."""
        if self.call.kind == "contiguous":
            return self.sm_count
        tiles = self._slab_tiles(block_m, block_n)
        if tiles == 0:
            return 1
        return -(-tiles // -(-tiles // self.sm_count))
