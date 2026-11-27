"""Row helpers shared by the MoE programs: tile ownership, row-copy widths and the layouts
the permute kernels serve, and the expert counts they share."""

import tilelang.language as T
import torch

from tileops._kernels.constants import MAX_BLOCK_THREADS

__all__ = ["aligned_group", "count_routes", "permute_refusal", "row_threads", "zero_counts"]


def permute_refusal(layout_key: str, dtype: torch.dtype) -> str | None:
    """Why a permute kernel cannot serve rows of ``dtype`` in layout ``layout_key``, or
    ``None``: it serves float16 or bfloat16 rows, tight physical-psum or aligned per-row."""
    layouts = ("aligned_per_row", "tight_physical_psum")
    if layout_key not in layouts:
        return f"serves layouts {list(layouts)}, not {layout_key}"
    if dtype not in (torch.bfloat16, torch.float16):
        return f"requires float16 or bfloat16, got {dtype}"
    return None


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


@T.macro
def zero_counts(counts, num_experts, threads, tx):
    """Zero ``counts[0:num_experts]`` over a block of ``threads``."""
    for i in T.serial(T.ceildiv(num_experts, threads)):
        expert = i * threads + tx
        if expert < num_experts:
            counts[expert] = T.int32(0)


@T.macro
def count_routes(counts, flat_ids, numel, threads, tx):
    """Add each of the ``numel`` routes to its expert's count, over a block of ``threads``."""
    for i in T.serial(T.ceildiv(numel, threads)):
        idx = i * threads + tx
        if idx < numel:
            T.atomic_add(counts[flat_ids[idx]], 1)
