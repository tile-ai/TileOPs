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

__all__ = ["select_row_config", "select_row_configs"]

# Powers of two only (tl::AllReduce is an XOR butterfly) that also divide
# N_padded, or layout inference reports "no available layout". CUDA caps at 1024.
_CANDIDATE_THREADS = (128, 256, 512, 1024)

# block_m values offered to autotune; block_m=1 is always the safe default.
_CANDIDATE_BLOCK_M = (1, 2, 4, 8)

_DEFAULT_THREADS = 128  # measured optimum; see the module docstring


def _feasible_threads(n_padded: int) -> list[int]:
    """Thread counts that divide the row and keep loads vectorizable (>=8/thread)."""
    return [
        threads
        for threads in _CANDIDATE_THREADS
        if n_padded % threads == 0 and n_padded // threads >= 8
    ]


def select_row_config(n_padded: int) -> dict:
    """Structurally collapse-free default ``{block_m, threads}`` for a row reduction."""
    threads = _DEFAULT_THREADS
    if n_padded % threads:
        # N_padded is a multiple of the 256-element alignment, so 128 always
        # divides it; this is a guard, not a live path.
        threads = next(
            (t for t in _CANDIDATE_THREADS if n_padded % t == 0), _DEFAULT_THREADS
        )
    return {"block_m": 1, "threads": threads}


def select_row_configs(n_padded: int) -> list[dict]:
    """Autotune space: block_m x usable thread counts, default always included.

    block_m is swept (not pinned) so the interface is preserved; configs that
    collapse are rejected by the autotuner's own measured runtime.
    """
    threads = _feasible_threads(n_padded)
    configs = [
        {"block_m": block_m, "threads": t}
        for block_m in _CANDIDATE_BLOCK_M
        for t in threads
    ]
    default = select_row_config(n_padded)
    if default not in configs:
        configs.insert(0, default)
    return configs
