"""What the row-wise norm kernels share: tile-config selection and the row reduction.

These kernels hold a ``(block_m, N_padded)`` row block in register fragments and
reduce along the row. When ``N_padded`` is not a power of two, TileLang's layout
inference places a partitioned layout for some ``block_m`` but falls back to a
replicated one for others -- each thread owns a whole row, spills to local
memory, and the cross-thread ``AllReduce`` degenerates into a serial loop. That
is 5-16x slower yet numerically correct, so no correctness test catches it.

``select_row_config`` pins ``block_m=1``: with one row per CTA the per-row
uniformity the reduction needs always holds, so the collapse is impossible for
any N/threads/dtype -- structural, not a tuned value. ``threads=128`` is what it
pins, because a row padded to the 256-element alignment often divides by nothing
wider. ``select_row_configs`` keeps ``block_m`` as a swept autotune knob so the
kernel interface is not narrowed; configs that collapse lose on their measured
runtime.

A row of a few thousand elements is latency-bound, not instruction-bound: what
decides it is how many loads one thread has in flight. ``select_row_config_by_width``
sizes the block from :data:`_TARGET_ELEMENTS_PER_THREAD` for that reason, over widths
that reach down to one warp for a row short enough to keep a thread's share small.
"""

import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.tiling import ALIGNMENT

__all__ = [
    "CANDIDATE_THREADS_BY_WIDTH",
    "make_row_reduce",
    "row_padding",
    "select_row_config",
    "select_row_config_by_width",
    "select_row_configs",
]

# Powers of two only (tl::AllReduce is an XOR butterfly) that also divide
# N_padded, or layout inference reports "no available layout". CUDA caps at 1024.
_CANDIDATE_THREADS = (128, 256, 512, 1024)

# Row width at or below which a block narrower than 128 is offered. Above it a
# narrow block hands one thread hundreds of elements, and layout inference
# answers that with the replicated layout this module's docstring describes.
NARROW_ROW = 2048

# The widths a row sized by :data:`_TARGET_ELEMENTS_PER_THREAD` can land on.
# Separate from :data:`_CANDIDATE_THREADS` so a block narrower than one warp
# reaches only the kernels that ask for this tuple, and only for a row short
# enough to keep a thread's share of it small.
CANDIDATE_THREADS_BY_WIDTH = (32, 64) + _CANDIDATE_THREADS


def widths_for_row(n_padded: int) -> tuple:
    """Block widths a row of *n_padded* elements may be split across."""
    return CANDIDATE_THREADS_BY_WIDTH if n_padded <= NARROW_ROW else _CANDIDATE_THREADS


# block_m values offered to autotune; block_m=1 is always the safe default.
_CANDIDATE_BLOCK_M = (1, 2, 4, 8)

_DEFAULT_THREADS = 128  # divides every aligned row; see the module docstring

_ROW_SMEM_BUDGET_BYTES = 48 * 1024


def row_padding(n: int, elem_bytes: int) -> int:
    """Row width to pad *n* to: the next power of two while it fits in shared.

    Only a power of two is divided by every entry of :data:`_CANDIDATE_THREADS`;
    the 256-element alignment leaves an odd factor that caps a block at eight
    warps, which costs 3% on a row of 7168. Never below :data:`ALIGNMENT`, which
    the 128-thread default needs.
    """
    pow2 = 1 << (n - 1).bit_length()
    if pow2 * elem_bytes <= _ROW_SMEM_BUDGET_BYTES:
        return max(pow2, ALIGNMENT)
    return -(-n // ALIGNMENT) * ALIGNMENT


# Row elements a thread carries. Enough of them keep loads in flight to cover
# the latency the row is bound by.
_TARGET_ELEMENTS_PER_THREAD = 32


def select_row_config_by_width(n_padded: int) -> dict:
    """``{block_m, threads}`` for a row reduction, sized by the row itself.

    For a row padded to a power of two, which admits every candidate width.
    """
    threads = n_padded // _TARGET_ELEMENTS_PER_THREAD
    for candidate in sorted(widths_for_row(n_padded), reverse=True):
        if candidate <= threads and n_padded % candidate == 0:
            return {"block_m": 1, "threads": candidate}
    return {"block_m": 1, "threads": _DEFAULT_THREADS}


def _feasible_threads(
    n_padded: int, dtype: torch.dtype = torch.float16, widths: tuple = _CANDIDATE_THREADS
) -> list[int]:
    """Thread counts that divide the row and keep loads 128-bit vectorizable.

    128-bit needs ``16 // element_size`` elements per thread (8 for fp16/bf16,
    4 for fp32). If no candidate meets that floor (small rows), fall back to any
    thread count that divides the row so the autotune space is never empty.

    Args:
        n_padded: Padded row width.
        dtype: Element type the row is stored in.
        widths: Block widths to draw from.
    """
    min_elements = VECTOR_ACCESS_BYTES // torch.tensor([], dtype=dtype).element_size()
    candidates = [t for t in widths if n_padded % t == 0]
    vectorizable = [t for t in candidates if n_padded // t >= min_elements]
    return vectorizable or candidates


def select_row_config() -> dict:
    """Structurally collapse-free default ``{block_m, threads}`` for a row reduction.

    Takes no width: a row is padded to a multiple of the 256-element alignment,
    which 128 threads always divide.
    """
    return {"block_m": 1, "threads": _DEFAULT_THREADS}


def select_row_configs(
    n_padded: int,
    dtype: torch.dtype = torch.float16,
    num_buffers: int = 1,
    widths: tuple = _CANDIDATE_THREADS,
) -> list[dict]:
    """Autotune space: block_m x usable thread counts, default always included.

    block_m is swept (not pinned) so the interface is preserved; configs that
    collapse are rejected by the autotuner's own measured runtime. block_m is
    capped by the 48 KB shared-memory budget for ``num_buffers`` row-sized
    buffers, which bounds the kernels that stage a row block in shared memory;
    for one that holds the block in registers the cap is simply not the binding
    limit. The default is always a member so the list is never empty.

    Args:
        n_padded: Padded row width.
        dtype: Element type the row is stored in.
        num_buffers: Row-sized shared buffers the kernel holds live at once.
        widths: Block widths to draw from.
    """
    threads = _feasible_threads(n_padded, dtype, widths)
    smem_per_row = n_padded * torch.tensor([], dtype=dtype).element_size()
    max_block_m = (48 * 1024) // (num_buffers * smem_per_row)
    configs = [
        {"block_m": block_m, "threads": t}
        for block_m in _CANDIDATE_BLOCK_M
        if block_m <= max_block_m
        for t in threads
    ]
    default = select_row_config()
    if default not in configs:
        configs.insert(0, default)
    return configs


def make_row_reduce(block_m, n, n_padded, eps):
    """Create the macro reducing a loaded fp32 row block to mean and rstd.

    Consumes ``x_f32`` and overwrites it with the centered squares. The load
    stays at the call sites, which read the row block in the dtype the tensor
    holds and keep it for the output pass.

    Args:
        block_m: Rows per block.
        n: Row length.
        n_padded: *n* rounded up to :data:`ALIGNMENT`.
        eps: Epsilon for numerical stability.

    Returns:
        A ``@T.macro`` taking ``(x_f32, acc, mean_val, rstd)``.
    """
    pad_count = n_padded - n

    @T.macro
    def row_reduce(x_f32, acc, mean_val, rstd):
        T.reduce_sum(x_f32, acc, dim=1)
        for i in T.Parallel(block_m):
            mean_val[i] = acc[i] / float(n)

        # Rewrite x_f32 in-place with (x - mean)^2. Padded positions (x=0)
        # contribute mean^2, subtracted back out below.
        for i, j in T.Parallel(block_m, n_padded):
            x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

        T.reduce_sum(x_f32, acc, dim=1)
        for i in T.Parallel(block_m):
            rstd[i] = T.rsqrt(
                (acc[i] - float(pad_count) * mean_val[i] * mean_val[i]) / float(n) + eps
            )

    return row_reduce
