"""Register arrangements the FA3 FP8 attention helpers of ``fp8_gqa_helper.h`` assume.

The helpers read and write the WGMMA accumulators as raw register arrays, so a kernel
that calls them must annotate those fragments with the arrangement the PTX produced.
Each function states one of them for the warpgroup whose first thread is
``thread_offset``.
"""

import tilelang

__all__ = [
    "fa3_acc_fragment",
    "fa3_qk_acc_column",
    "fa3_qk_row_fragment",
]


def fa3_acc_fragment(cols: int, thread_offset: int) -> tilelang.layout.Fragment:
    """The WGMMA accumulator's arrangement for a 64 x *cols* f32 tile.

    The score accumulator (``cols == block_n``) and the value accumulator
    (``cols == dim``) come out of the same PTX shape, so one arrangement states both.
    """
    if cols % 8:
        raise ValueError("an FA3 accumulator fragment annotation requires cols % 8 == 0.")
    col_phase = cols // 8

    def forward_fn(i, j):
        rv = j // 4
        thread = thread_offset + (i // 16) * 32 + (i % 8) * 4 + (j % 4)
        index = (rv % col_phase) * 4 + ((i % 16) // 8) * 2 + rv // col_phase
        return thread, index

    return tilelang.layout.Fragment([64, cols], forward_fn=forward_fn)


def fa3_qk_acc_column(j: int) -> int:
    """The key column fragment index *j* of :func:`fa3_acc_fragment` holds.

    The fragment orders a row's registers lane first, then 8-column group, then pair,
    which a row reduction does not see; a column-dependent mask does. The WGMMA
    accumulator puts lane ``j % 4`` of group ``(j // 4) % 28`` on columns
    ``8 * group + 2 * lane``, and the pair ``j // 112`` on the next one.
    """
    return 8 * ((j // 4) % 28) + 2 * (j % 4) + j // 112


def fa3_qk_row_fragment(thread_offset: int) -> tilelang.layout.Fragment:
    """The arrangement of one value per row of the score tile, replicated across the quad."""

    def forward_fn(i, rep):
        thread = thread_offset + (i // 16) * 32 + (i % 8) * 4 + rep
        index = (i % 16) // 8
        return thread, index

    return tilelang.layout.Fragment([64], forward_fn=forward_fn, replicate=4)
