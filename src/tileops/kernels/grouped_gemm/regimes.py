"""The rows-per-group regimes the persistent 3WG grouped-GEMM kernels select on.

A grouped GEMM whose groups are shorter than a tile leaves most of that tile idle,
and a schedule with a shorter block wins. Kernels of this family ask which regime
their shape is in and pick a schedule from the answer.

The boundaries are measured, not derived: on H200 with bf16, a short-group schedule
gains 4-7% inside them and costs 34-39% above them.
"""

__all__ = ["rows_per_group_regime"]

_SHORT_MAX_ROWS_PER_GROUP = 32  # above this, groups fill tiles and the default wins
_THIN_MAX_ROWS_PER_GROUP = 16  # at or below, a group cannot fill a cooperative split


def rows_per_group_regime(numel: int, num_groups: int) -> str | None:
    """How this shape's groups sit against a tile.

    Args:
        numel: Rows in total, across all groups.
        num_groups: Groups the rows are spread over.

    Returns:
        ``"thin"`` when a group cannot fill a cooperative split, ``"short"`` when it
        is shorter than a default tile, or ``None`` when groups fill tiles.
    """
    rows_per_group = numel / max(1, num_groups)
    if rows_per_group <= _THIN_MAX_ROWS_PER_GROUP:
        return "thin"
    if rows_per_group <= _SHORT_MAX_ROWS_PER_GROUP:
        return "short"
    return None
