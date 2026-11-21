"""Row helpers shared by the MoE programs: tile ownership and row-copy widths."""

import tilelang.language as T

from tileops._kernels.constants import MAX_BLOCK_THREADS

__all__ = ["aligned_group", "row_threads"]


@T.macro
def aligned_group(layout, row, groups):
    """The expert at an in-bounds tile start, or -1 for padding.

    The caller uses a tile height that divides the segment alignment, so one
    tile cannot cross from one expert to another.
    """
    group = layout[row]
    return T.if_then_else((group >= 0) & (group < groups), group, -1)


def row_threads(hidden_size: int, vector_elems: int, cap: int = MAX_BLOCK_THREADS) -> int:
    """Threads that copy one row ``vector_elems`` elements each: the most, up to *cap*,
    dividing ``hidden_size``."""
    threads = min(cap, hidden_size // vector_elems)
    while threads > 0 and hidden_size % threads != 0:
        threads -= 1
    return max(threads, 1)
