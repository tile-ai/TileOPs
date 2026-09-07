"""What the row-wise norm kernels share: tile-config selection and the row reduction.

These kernels hold a ``(block_m, n_padded)`` row block in register fragments and
reduce along the row. When ``n_padded`` is not a power of two, TileLang's layout
inference places a partitioned layout for some ``block_m`` and a replicated one
for others -- each thread owns a whole row, spills to local memory, and the
cross-thread ``AllReduce`` degenerates into a serial loop. That is 5-16x slower
yet numerically correct, so no correctness test catches it. Every config chosen
here pins ``block_m=1``, where the per-row uniformity the reduction needs holds
for any width, thread count and dtype.

Of the two row reductions here, :func:`make_row_reduce` walks the row twice and
holds one fp32 copy of it, and :func:`make_shifted_row_reduce` walks it once and
holds two: a kernel reading the row straight into fragments can afford the
second, one staging it through shared memory cannot.
"""

from typing import Optional

import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.tiling import ALIGNMENT

__all__ = [
    "CANDIDATE_THREADS_BY_WIDTH",
    "NARROW_ROW",
    "make_row_reduce",
    "make_shifted_row_reduce",
    "row_padding",
    "select_row_config",
    "select_row_config_by_width",
    "select_row_configs",
    "widths_for_row",
]

# Powers of two only (tl::AllReduce is an XOR butterfly) that also divide
# n_padded, or layout inference reports "no available layout". CUDA caps a block
# at 1024 threads.
_CANDIDATE_THREADS = (128, 256, 512, 1024)

# The widths a short row may also use. Separate from :data:`_CANDIDATE_THREADS`
# so a block narrower than one warp reaches only the kernels that ask for it.
CANDIDATE_THREADS_BY_WIDTH = (32, 64) + _CANDIDATE_THREADS

# Row width at or below which a block narrower than 128 is offered. Above it a
# narrow block hands one thread hundreds of columns, which layout inference
# answers with the replicated layout this module's docstring describes.
NARROW_ROW = 2048

_CANDIDATE_BLOCK_M = (1, 2, 4, 8)  # rows per block offered to autotune
_DEFAULT_THREADS = 128  # divides every row padded to a multiple of ALIGNMENT
_ROW_SMEM_BUDGET_BYTES = 48 * 1024

_TARGET_ELEMENTS_PER_THREAD = 32  # what select_row_config_by_width aims for


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


def widths_for_row(n_padded: int) -> tuple:
    """Block widths a row of *n_padded* columns may be split across."""
    return CANDIDATE_THREADS_BY_WIDTH if n_padded <= NARROW_ROW else _CANDIDATE_THREADS


def _feasible_threads(
    n_padded: int, dtype: torch.dtype = torch.float16, widths: tuple = _CANDIDATE_THREADS
) -> list[int]:
    """Thread counts that divide the row and keep loads 128-bit vectorizable.

    128-bit needs ``16 // element_size`` columns per thread (8 for fp16/bf16, 4
    for fp32). If no candidate meets that floor (small rows), fall back to any
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


def select_row_config_by_width(n_padded: int, widths: Optional[tuple] = None) -> dict:
    """``{block_m, threads}`` for a row reduction, sized by the row itself.

    Args:
        n_padded: Padded row width.
        widths: Block widths to choose from. A caller that narrows its autotune
            space passes the same tuple here, so the untuned default stays a
            member of it. When every width in it overshoots the target, the
            narrowest is the closest the tuple gets.
    """
    narrowed = widths is not None
    if widths is None:
        widths = widths_for_row(n_padded)
    usable = sorted(t for t in widths if n_padded % t == 0)
    target = n_padded // _TARGET_ELEMENTS_PER_THREAD
    for candidate in reversed(usable):
        if candidate <= target:
            return {"block_m": 1, "threads": candidate}
    if narrowed and usable:
        return {"block_m": 1, "threads": usable[0]}
    return select_row_config()


def select_row_configs(
    n_padded: int,
    dtype: torch.dtype = torch.float16,
    num_buffers: int = 1,
    widths: tuple = _CANDIDATE_THREADS,
    block_ms: tuple = _CANDIDATE_BLOCK_M,
) -> list[dict]:
    """Autotune space: *block_ms* x the thread counts *widths* admits.

    block_m defaults to the full sweep so the kernel interface is not narrowed;
    a caller whose kernel is too short for the autotuner to rank pins it
    instead. block_m is capped by the shared-memory budget for ``num_buffers``
    row-sized buffers, which binds only the kernels that stage a row block in
    shared memory.

    Args:
        n_padded: Padded row width.
        dtype: Element type the row is stored in.
        num_buffers: Row-sized shared buffers the kernel holds live at once.
        widths: Block widths to draw from.
        block_ms: Rows-per-block values to offer.
    """
    threads = _feasible_threads(n_padded, dtype, widths)
    smem_per_row = n_padded * torch.tensor([], dtype=dtype).element_size()
    max_block_m = _ROW_SMEM_BUDGET_BYTES // (num_buffers * smem_per_row)
    configs = [
        {"block_m": block_m, "threads": t}
        for block_m in block_ms
        if block_m <= max_block_m
        for t in threads
    ]
    # A row so wide that no offered block_m fits the budget still needs one
    # config to tune over.
    return configs or [select_row_config()]


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


def make_shifted_row_reduce(block_m, n, eps):
    """Create the macro turning an already-shifted row block into mean and rstd.

    Reads the two fragments the load fills and rewrites neither: ``d`` holds
    ``x - shift`` and ``sq`` holds its square. ``mean_val`` comes back shifted by
    the same amount, so the output pass reads ``d[i, j] - mean_val[i]`` and gets
    ``x - E[x]``.

    Two things the caller supplies:

    - ``shift`` is a value from the row itself. Without it the difference of the
      two sums loses the variance to cancellation as soon as a row's mean
      outgrows its spread.
    - A pad column holds zero in both fragments, which is why no correction term
      for the padding appears here.

    Args:
        block_m: Rows per block.
        n: Row length.
        eps: Epsilon for numerical stability.

    Returns:
        A ``@T.macro`` taking ``(d, sq, acc_shifted, acc_squares, mean_val, rstd)``.
    """

    @T.macro
    def shifted_row_reduce(d, sq, acc_shifted, acc_squares, mean_val, rstd):
        # One accumulator each: sharing one puts a block-wide barrier between
        # the two sums and makes the second wait on the first.
        T.reduce_sum(d, acc_shifted, dim=1)
        T.reduce_sum(sq, acc_squares, dim=1)
        for i in T.Parallel(block_m):
            mean_val[i] = acc_shifted[i] / float(n)
            rstd[i] = T.rsqrt(acc_squares[i] / float(n) - mean_val[i] * mean_val[i] + eps)

    return shifted_row_reduce
