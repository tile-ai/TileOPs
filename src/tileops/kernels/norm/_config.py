"""Tile-config selection shared by the row-wise norm kernels.

These kernels hold a ``(block_m, N_padded)`` row block in register fragments and
reduce along the row. When ``N_padded`` is not a power of two, TileLang's layout
inference places a partitioned layout for some ``block_m`` but falls back to a
replicated one for others -- each thread owns a whole row, spills to local
memory, and the cross-thread ``AllReduce`` degenerates into a serial loop. That
is 5-16x slower yet numerically correct, so no correctness test catches it.

``select_row_config`` pins ``block_m=1``: with one row per CTA the per-row
uniformity the reduction needs always holds, so the collapse is impossible for
any N/threads/dtype -- structural, not a tuned value. ``threads=128`` is the
measured optimum. ``select_row_configs`` keeps ``block_m`` as a swept autotune
knob so the kernel interface is not narrowed; configs that collapse lose on their
measured runtime.
"""

import torch

__all__ = ["select_row_config", "select_row_configs"]

# Powers of two only (tl::AllReduce is an XOR butterfly) that also divide
# N_padded, or layout inference reports "no available layout". CUDA caps at 1024.
_CANDIDATE_THREADS = (128, 256, 512, 1024)

# block_m values offered to autotune; block_m=1 is always the safe default.
_CANDIDATE_BLOCK_M = (1, 2, 4, 8)

_DEFAULT_THREADS = 128  # measured optimum; see the module docstring


def _feasible_threads(n_padded: int, dtype: torch.dtype = torch.float16) -> list[int]:
    """Thread counts that divide the row and keep loads 128-bit vectorizable.

    128-bit needs ``16 // element_size`` elements per thread (8 for fp16/bf16,
    4 for fp32). If no candidate meets that floor (small rows), fall back to any
    thread count that divides the row so the autotune space is never empty.
    """
    min_elements = 16 // torch.tensor([], dtype=dtype).element_size()
    candidates = [t for t in _CANDIDATE_THREADS if n_padded % t == 0]
    vectorizable = [t for t in candidates if n_padded // t >= min_elements]
    return vectorizable or candidates


def select_row_config(n_padded: int) -> dict:
    """Structurally collapse-free default ``{block_m, threads}`` for a row reduction."""
    # N_padded is a multiple of the 256-element alignment, so 128 always divides it.
    return {"block_m": 1, "threads": _DEFAULT_THREADS}


def select_row_configs(
    n_padded: int, dtype: torch.dtype = torch.float16, num_buffers: int = 1
) -> list[dict]:
    """Autotune space: block_m x usable thread counts, default always included.

    block_m is swept (not pinned) so the interface is preserved; configs that
    collapse are rejected by the autotuner's own measured runtime. block_m is
    capped by the 48 KB shared-memory budget for ``num_buffers`` row-sized
    buffers, and the default is always a member so the list is never empty.
    """
    threads = _feasible_threads(n_padded, dtype)
    smem_per_row = n_padded * torch.tensor([], dtype=dtype).element_size()
    max_block_m = (48 * 1024) // (num_buffers * smem_per_row)
    configs = [
        {"block_m": block_m, "threads": t}
        for block_m in _CANDIDATE_BLOCK_M
        if block_m <= max_block_m
        for t in threads
    ]
    default = select_row_config(n_padded)
    if default not in configs:
        configs.insert(0, default)
    return configs
