"""Expert ownership of aligned row tiles shared by the MMA programs."""

import tilelang.language as T

__all__ = ["aligned_group"]


@T.macro
def aligned_group(layout, row, groups):
    """The expert at an in-bounds tile start, or -1 for padding.

    The caller uses a tile height that divides the segment alignment, so one
    tile cannot cross from one expert to another.
    """
    group = layout[row]
    return T.if_then_else((group >= 0) & (group < groups), group, -1)
