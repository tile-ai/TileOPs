"""What every sampling kernel that walks a row of logits a tile at a time shares.

``vector_width`` sits here beside the macros that read it rather than on the call record,
which holds no TileLang.
"""

import tilelang.language as T

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = [
    "INF_BITS",
    "MAGNITUDE_BITS",
    "QUIET_NAN",
    "block_extremes",
    "fold_extremes",
    "load_vector",
    "row_split",
    "store_masked",
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
            "tl::tileops_load16_evict_first",
            T.address_of(dst[slot, 0]),
            T.address_of(src[at]),
        )
    else:
        T.call_extern(
            "handle",
            "tl::tileops_load16",
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
