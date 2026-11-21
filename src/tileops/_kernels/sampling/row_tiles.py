"""What the sampling kernels that walk a row a tile at a time share.

The two masks that settle a row's maximum before masking it share their launch plan, their
read of a row and their masked write; the two draws share the barrier that meets a split
row's totals. ``vector_width`` sits here beside the macros that read it rather than on the
call record, which holds no TileLang.
"""

from types import SimpleNamespace

import tilelang.language as T

from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = [
    "INF_BITS",
    "QUIET_NAN",
    "gather_part_totals",
    "load_vector",
    "mask_plan",
    "mask_tiles",
    "offset_refusal",
    "row_split",
    "vector_width",
]

# A float32's magnitude bits, the largest of them that is not a NaN, and a NaN.
MAGNITUDE_BITS: int = 0x7FFFFFFF
INF_BITS: int = 0x7F800000
QUIET_NAN: int = 0x7FC00000


def vector_width(vocab: int, itemsize: int) -> int:
    """Elements of a 16-byte vector, or 1 where a row's bytes are not a whole number of them.

    Every row starts on a vector only when the row's bytes are; a row that does not is read
    and written element by element.
    """
    return VECTOR_ACCESS_BYTES // itemsize if vocab * itemsize % VECTOR_ACCESS_BYTES == 0 else 1


def row_split(row_tiles: int, batch: int, sm_count: int) -> int:
    """Blocks to put on one row of ``row_tiles`` tiles, for a batch of ``batch`` rows.

    A row is split only while the batch leaves blocks idle, and never past its own tiles:
    the grid barrier a split takes needs the whole grid resident, which this keeps it.
    """
    return max(1, min(row_tiles, sm_count // max(batch, 1)))


@T.macro
def load_vector(dst, slot, src, at, vec: int, evict_first: bool):
    """Read one ``vec``-element vector of *src* at *at* into slot *slot* of *dst*."""
    if vec == 1:
        dst[slot, 0] = src[at]
    elif evict_first:
        T.call_extern(
            "handle",
            "tileops::load16_evict_first",
            T.address_of(dst[slot, 0]),
            T.address_of(src[at]),
        )
    else:
        T.call_extern(
            "handle",
            "tileops::load16",
            T.address_of(dst[slot, 0]),
            T.address_of(src[at]),
        )


@T.macro
def fold_extremes(top, seen, src, slot, vec: int):
    """Fold one slot into the running value maximum and magnitude maximum.

    The magnitude maximum rides alongside because ``max`` drops a NaN where ``torch.amax``
    propagates it, and a magnitude above an infinity's is one.
    """
    for c in T.unroll(vec):
        value = T.cast(src[slot, c], "float32")
        top[0] = T.max(top[0], value)
        seen[0] = T.max(seen[0], T.reinterpret(value, "uint32") & T.uint32(MAGNITUDE_BITS))


@T.macro
def store_masked(dst, src, slot, cut, at, vec: int, dtype: str):
    """Write one slot to *dst* at *at*, every value below *cut* replaced by ``-inf``."""
    for c in T.unroll(vec):
        value = T.cast(src[slot, c], "float32")
        src[slot, c] = T.if_then_else(
            value < cut, T.cast(-T.infinity("float32"), dtype), src[slot, c]
        )
    if vec == 1:
        dst[at] = src[slot, 0]
    else:
        for c in T.vectorized(vec):
            dst[at + c] = src[slot, c]


@T.macro
def block_extremes(top, seen, warp_top, warp_seen, warps: int):
    """Leave the block's maxima in ``top[0]`` and ``seen[0]`` of every thread."""
    tx = T.get_thread_binding()
    for stage in T.serial(WARP_SHUFFLE_STAGES):
        reach = T.int32(WARP_LANES // 2) >> stage
        top[0] = T.max(top[0], T.shfl_xor(top[0], reach, width=WARP_LANES))
        seen[0] = T.max(seen[0], T.shfl_xor(seen[0], reach, width=WARP_LANES))
    if tx % WARP_LANES == 0:
        warp_top[tx // WARP_LANES] = top[0]
        warp_seen[tx // WARP_LANES] = seen[0]
    T.sync_threads()
    top[0] = warp_top[0]
    seen[0] = warp_seen[0]
    for w in T.serial(1, warps):
        top[0] = T.max(top[0], warp_top[w])
        seen[0] = T.max(seen[0], warp_seen[w])


@T.macro
def gather_part_totals(
    seg, lane, partial, chunks, acc, bx, line, leaves: int, leaf_span: int, parts: int, threads: int
):
    """Publish the block's ``seg`` total to ``partial`` and read its row's ``parts`` totals
    into ``chunks``, across one grid barrier.

    Warp 0 sums the total exactly as a draw's descent sums the same entries, a lane's
    ``leaf_span`` of them each, so the prefix a block takes from the barrier and the prefix
    it then descends agree entry for entry.
    """
    tx = T.get_thread_binding()
    if tx < WARP_LANES:
        acc[0] = T.cast(0, "float64")
        for i in T.serial(leaf_span):
            if tx * leaf_span + i < leaves:
                acc[0] = acc[0] + seg[tx * leaf_span + i]
        lane[tx] = acc[0]
    T.sync_threads()
    if tx == 0:
        acc[0] = T.cast(0, "float64")
        for l in T.serial(WARP_LANES):
            acc[0] = acc[0] + lane[l]
        partial[bx] = acc[0]
    T.sync_grid()
    for i in T.serial(-(-parts // threads)):
        if i * threads + tx < parts:
            chunks[i * threads + tx] = partial[line * parts + i * threads + tx]
    T.sync_threads()


def mask_plan(vectors: int, batch: int, sm_count: int, threads: int, min_threads: int):
    """A logit mask's threads a block, blocks a row, and tiles a block's chunk.

    A row of ``vectors`` vectors takes ``threads`` a block, halved down to ``min_threads``
    while the batch's tiles leave SMs idle, and then splits across blocks by ``row_split``.
    """
    while threads > min_threads and batch * -(-vectors // threads) < sm_count:
        threads //= 2
    row_tiles = max(1, -(-vectors // threads))
    parts = row_split(row_tiles, batch, sm_count)
    return threads, parts, -(-row_tiles // parts)


def offset_refusal(call, threads: int) -> str | None:
    """Why int32 element offsets cannot address a mask's ``threads``-thread tiles, or ``None``.

    The offsets run up to one tile past ``B * V``, and a vector is at most
    ``VECTOR_ACCESS_BYTES`` elements.
    """
    if call.batch * call.vocab > 2**31 - 1 - threads * VECTOR_ACCESS_BYTES:
        return f"indexes elements with int32, and B * V = {call.batch * call.vocab}"
    return None


def mask_tiles(
    vocab: int,
    dtype: str,
    vec: int,
    threads: int,
    parts: int,
    reg_tiles: int,
    smem_tiles: int,
    pace: int,
) -> SimpleNamespace:
    """A logit mask's chunk of a row and the macros that read it and write it masked.

    One block owns one contiguous chunk of one row, ``parts`` blocks to a row. A tile of a
    chunk is one ``vec``-element vector per thread; of them the first ``reg_tiles`` stay in
    registers (``kept``) and the next ``smem_tiles`` in shared memory (``parked``), and the
    rest (``streamed``) are read again on each later walk.
    """
    # Vectors of a row; a row divides into whole ones, since ``vec`` falls back to 1.
    full = vocab // vec
    row_tiles = -(-full // threads)
    chunk = -(-row_tiles // parts)
    warps = threads // WARP_LANES
    kept = min(chunk, reg_tiles)
    parked = min(chunk - kept, smem_tiles)
    streamed = chunk - kept - parked
    parked_rounds = -(-parked // pace)
    streamed_rounds = -(-streamed // pace)

    @T.macro
    def row_extremes(
        logits,
        start,
        head,
        line,
        bx,
        in_regs,
        in_smem,
        cur,
        top,
        seen,
        warp_top,
        warp_seen,
        partial,
    ):
        """Read the chunk, holding what it keeps on chip, and leave the row's maxima in
        ``top[0]`` and ``seen[0]``. ``parts > 1`` meets a row's blocks in ``partial`` across
        one grid barrier."""
        tx = T.get_thread_binding()
        top[0] = -T.infinity("float32")
        seen[0] = T.uint32(0)
        # A tile the block holds is read evict-first, since nothing reads it from memory
        # again; a tile it will read again stays cacheable for that read.
        for j in T.unroll(kept):
            if head + j * threads < full:
                load_vector(in_regs, j, logits, start + (head + j * threads) * vec, vec, True)
        for j in T.unroll(kept):
            if head + j * threads < full:
                fold_extremes(top, seen, in_regs, j, vec)
        for r in T.serial(parked_rounds):
            for j in T.unroll(pace):
                t = kept + r * pace + j
                if t < kept + parked and head + t * threads < full:
                    load_vector(cur, j, logits, start + (head + t * threads) * vec, vec, True)
            for j in T.unroll(pace):
                t = kept + r * pace + j
                if t < kept + parked and head + t * threads < full:
                    fold_extremes(top, seen, cur, j, vec)
                    for c in T.vectorized(vec):
                        in_smem[t - kept, tx * vec + c] = cur[j, c]
        for r in T.serial(streamed_rounds):
            for j in T.unroll(pace):
                t = kept + parked + r * pace + j
                if t < chunk and head + t * threads < full:
                    load_vector(cur, j, logits, start + (head + t * threads) * vec, vec, False)
            for j in T.unroll(pace):
                t = kept + parked + r * pace + j
                if t < chunk and head + t * threads < full:
                    fold_extremes(top, seen, cur, j, vec)
        block_extremes(top, seen, warp_top, warp_seen, warps)

        if parts > 1:
            if tx == 0:
                partial[bx] = T.if_then_else(
                    seen[0] > T.uint32(INF_BITS),
                    T.reinterpret(T.uint32(QUIET_NAN), "float32"),
                    top[0],
                )
            T.sync_grid()
            # One chunk maximum per thread, then one block reduction: every thread reading
            # all of them instead serializes that many loads.
            top[0] = -T.infinity("float32")
            seen[0] = T.uint32(0)
            for c in T.serial(-(-parts // threads)):
                if c * threads + tx < parts:
                    value = partial[line * parts + c * threads + tx]
                    top[0] = T.max(top[0], value)
                    seen[0] = T.max(
                        seen[0], T.reinterpret(value, "uint32") & T.uint32(MAGNITUDE_BITS)
                    )
            block_extremes(top, seen, warp_top, warp_seen, warps)

    @T.macro
    def write_masked(masked, logits, start, head, in_regs, in_smem, cur, cut):
        """Write the chunk to ``masked``, every value below ``cut[0]`` replaced by ``-inf``."""
        tx = T.get_thread_binding()
        # The tiles read a second time go first, while L2 still holds them.
        for r in T.serial(streamed_rounds):
            for j in T.unroll(pace):
                t = kept + parked + r * pace + j
                if t < chunk and head + t * threads < full:
                    load_vector(cur, j, logits, start + (head + t * threads) * vec, vec, True)
            for j in T.unroll(pace):
                t = kept + parked + r * pace + j
                if t < chunk and head + t * threads < full:
                    store_masked(
                        masked, cur, j, cut[0], start + (head + t * threads) * vec, vec, dtype
                    )
        for t in T.serial(kept, kept + parked):
            if head + t * threads < full:
                for c in T.vectorized(vec):
                    cur[0, c] = in_smem[t - kept, tx * vec + c]
                store_masked(masked, cur, 0, cut[0], start + (head + t * threads) * vec, vec, dtype)
        for j in T.unroll(kept):
            if head + j * threads < full:
                store_masked(
                    masked, in_regs, j, cut[0], start + (head + j * threads) * vec, vec, dtype
                )

    return SimpleNamespace(
        full=full,
        chunk=chunk,
        warps=warps,
        kept=kept,
        parked=parked,
        streamed_rounds=streamed_rounds,
        row_extremes=row_extremes,
        write_masked=write_masked,
    )
