"""Shared row-normalization configuration and reduction macros.

Non-power-of-two fragment rows use ``block_m=1``: larger row blocks can produce
replicated layouts, register spills, and serial reductions. Whole 16-byte-vector
rows up to 1024 elements can instead use an unpadded warp reduction."""

from typing import Optional

import tilelang.language as T
import torch

from tileops._kernels.constants import STATIC_SHARED_BYTES, VECTOR_ACCESS_BYTES
from tileops._kernels.tiling import ALIGNMENT

__all__ = [
    "NARROW_ROW",
    "make_centred_row_reduce",
    "make_read_leads",
    "pairwise_sum",
    "raw_sums_ratio",
    "row_padding",
    "select_row_config",
    "select_row_config_by_width",
    "select_row_configs",
    "warp_row_vector",
    "widths_for_row",
]

# Powers of two only (tl::AllReduce is an XOR butterfly) that also divide
# n_padded, or layout inference reports "no available layout". CUDA caps a block
# at 1024 threads.
_CANDIDATE_THREADS = (128, 256, 512, 1024)

# Row width at or below which a block narrower than 128 is offered. Above it a
# narrow block hands one thread hundreds of columns, which layout inference
# answers with the replicated layout this module's docstring describes.
NARROW_ROW = 2048


def row_padding(n: int, elem_bytes: int) -> int:
    """Row width to pad *n* to: the next power of two while it fits in shared.

    Only a power of two is divided by every entry of :data:`_CANDIDATE_THREADS`;
    the 256-element alignment leaves an odd factor that caps a block at eight
    warps, which costs 3% on a row of 7168. Never below :data:`ALIGNMENT`, which
    the 128-thread default needs.
    """
    pow2 = 1 << (n - 1).bit_length()
    if pow2 * elem_bytes <= STATIC_SHARED_BYTES:
        return max(pow2, ALIGNMENT)
    return -(-n // ALIGNMENT) * ALIGNMENT


def widths_for_row(n_padded: int) -> tuple:
    """Block widths a row of *n_padded* columns may be split across."""
    # The widths a short row may also use. Separate from :data:`_CANDIDATE_THREADS`
    # so a block narrower than one warp reaches only the kernels that ask for it.
    candidate_threads_by_width = (32, 64) + _CANDIDATE_THREADS
    return candidate_threads_by_width if n_padded <= NARROW_ROW else _CANDIDATE_THREADS


def select_row_config() -> dict:
    """Structurally collapse-free default ``{block_m, threads}`` for a row reduction.

    Takes no width: a row is padded to a multiple of the 256-element alignment,
    which 128 threads always divide.
    """
    default_threads = 128
    return {"block_m": 1, "threads": default_threads}


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
    target_elements_per_thread = 32
    target = n_padded // target_elements_per_thread
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
    block_ms: tuple = (1, 2, 4, 8),
    every_width_past_budget: bool = False,
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
        every_width_past_budget: Offer every admitted width, one row a block, for a row
            no offered block_m fits; otherwise such a row gets the single default.
    """

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
        min_elements = VECTOR_ACCESS_BYTES // dtype.itemsize
        candidates = [t for t in widths if n_padded % t == 0]
        vectorizable = [t for t in candidates if n_padded // t >= min_elements]
        return vectorizable or candidates

    threads = _feasible_threads(n_padded, dtype, widths)
    smem_per_row = n_padded * dtype.itemsize
    max_block_m = STATIC_SHARED_BYTES // (num_buffers * smem_per_row)
    configs = [
        {"block_m": block_m, "threads": t}
        for block_m in block_ms
        if block_m <= max_block_m
        for t in threads
    ]
    if not configs and every_width_past_budget:
        configs = [{"block_m": 1, "threads": t} for t in threads]
    return configs or [select_row_config()]


def pairwise_sum(terms):
    """The sum of the expressions *terms*, added pairwise: log2 dependent adds where a serial
    fold takes ``len(terms) - 1``."""
    terms = list(terms)
    while len(terms) > 1:
        pairs = [a + b for a, b in zip(terms[::2], terms[1::2], strict=False)]
        terms = pairs + terms[len(terms) // 2 * 2 :]
    return terms[0]


def make_read_leads(block_m, rows, n):
    """Create the macro reading each row's first element, the ``K`` its statistics take.

    Every sum a kernel forms is taken less an element of its row, so it scales with the
    row's spread rather than its offset: the mean never rounds at the offset, and no sum
    overflows unless the variance times the row length does. A row past the last takes the
    last row's.

    The read sits inside a two-column loop: TileLang fails to compile a lone read at row
    strides from 32768 to 65535 (tile-ai/tilelang#3453).

    Args:
        block_m: Rows per block.
        rows: Rows of the ``(rows, n)`` tensor read.
        n: Row length.

    Returns:
        A ``@T.macro`` taking ``(x, pid_m, shift)``, filling the fp32 fragment ``shift``
        with the first elements of rows ``pid_m * block_m`` onwards.
    """

    @T.macro
    def read_leads(x, pid_m, shift):
        for i in T.Parallel(block_m):
            for j in T.unroll(2):
                if j == 0:
                    shift[i] = T.cast(
                        x[T.min(pid_m * block_m + i, rows - 1), T.min(j, n - 1)], "float32"
                    )

    return read_leads


def make_centred_row_reduce(block_m, n, n_padded, eps):
    """Create the macro reducing a held fp32 row block to its centre, offset and rstd.

    Corrected two-pass statistics. The first sum, of ``x - shift``, gives an approximate
    mean ``centre = shift + sum / n``, one fp32 value. The centred pass takes
    ``d = x - centre`` from the row again and sums both ``d`` and ``d^2``: ``c = sum(d) / n``
    is what the centre's rounding missed, and the variance is ``sum(d^2) / n - c^2``. The
    output is ``(x - centre) * rstd + offset`` with ``offset = -c * rstd``: no value is formed
    at the row's offset, and no rounding of the centre reaches the output.

    Consumes ``x_f32``, which the caller fills with ``x - shift``, zero past the row; it
    is left holding the squares of the centred row.

    Args:
        block_m: Rows per block.
        n: Row length.
        n_padded: *n* rounded up to :data:`ALIGNMENT`.
        eps: Epsilon for numerical stability.

    Returns:
        A ``@T.macro`` taking ``(x_row, x_f32, shift, acc, centre, offset, rstd)``:
        ``x_row`` the row block in its storage dtype (fragment or shared), ``shift`` the
        element each row's first sum is taken less, the others fp32 fragments of
        ``block_m`` the macro fills.
    """

    @T.macro
    def row_reduce(x_row, x_f32, shift, acc, centre, offset, rstd):
        T.reduce_sum(x_f32, acc, dim=1)
        # By the reciprocal: a divide by a width that is no power of two sits on the path
        # every block ends on.
        for i in T.Parallel(block_m):
            centre[i] = shift[i] + acc[i] * (1.0 / n)

        for i, j in T.Parallel(block_m, n_padded):
            x_f32[i, j] = T.if_then_else(j < n, T.cast(x_row[i, j], "float32") - centre[i], 0.0)
        T.reduce_sum(x_f32, acc, dim=1)
        for i in T.Parallel(block_m):
            offset[i] = acc[i] * (1.0 / n)

        for i, j in T.Parallel(block_m, n_padded):
            x_f32[i, j] = x_f32[i, j] * x_f32[i, j]
        T.reduce_sum(x_f32, acc, dim=1)
        for i in T.Parallel(block_m):
            rstd[i] = T.rsqrt(acc[i] * (1.0 / n) - offset[i] * offset[i] + eps)
            offset[i] = -offset[i] * rstd[i]

    return row_reduce


def warp_row_vector(n: int, elem_bytes: int) -> Optional[int]:
    """Elements of the 16-byte vector a lane reads, for a row one warp holds; else ``None``.

    The fragment body pads such a row to a power of two -- 784 columns to 1024 -- and guards
    every element past it; a warp guards only the vectors past the row. Measured on H200 at
    64 to 1024 columns, where it led or tied the fragment body at every dtype.
    """
    vec = VECTOR_ACCESS_BYTES // elem_bytes
    # At 2048 16-bit columns the fragment body led it on H200: by 3% at 128 rows with one
    # row a block, and at parity at 512 rows.
    widest = 1024
    if n % vec or n > widest:
        return None
    return vec


def raw_sums_ratio(dtype: str, adds: int) -> float:
    """The largest ``E[x^2] / var`` at which fp32 sums of ``x`` and ``x^2`` give a row's
    statistics within its output's accuracy.

    Each sum is ``adds`` dependent adds deep, so it rounds by at most ``adds`` units of fp32
    rounding ``u``: ``E[x^2] - mean^2`` is off by at most ``3 * adds * u * E[x^2]``, and the
    mean by ``adds * u * E|x| <= adds * u * sqrt(E[x^2])``. With ``R = E[x^2] / var`` held to
    ``tau / (6 * adds * u)``, the variance stays within half the accuracy ``tau`` and so do
    the normalized mean and the output's ``u * (|x| + |mean|) * rstd``. ``tau`` is a 16-bit
    output's unit roundoff, and for fp32 one past the tolerance its statistics are checked to.

    Args:
        dtype: TileLang dtype string of the row.
        adds: Dependent adds behind each sum.
    """
    tau = {"float32": 2.0**-17, "float16": 2.0**-11, "bfloat16": 2.0**-8}[dtype]
    return tau / (6.0 * adds * 2.0**-24)


def make_raw_sums_hold(dtype: str, adds: int):
    """Create the predicate under which fp32 sums of ``x`` and ``x^2`` give a row's statistics
    within its output's accuracy (`raw_sums_ratio`), so a kernel may keep them rather than
    take the shifted rule.

    Also required: ``E[x^2]`` finite (a square past fp32's range, or a NaN, fails every
    comparison) and the variance at least ``2^-100`` (no subnormal products).

    Args:
        dtype: TileLang dtype string of the row.
        adds: Dependent adds behind each sum.

    Returns:
        A function of fp32 expressions ``(square_mean, var)`` returning the predicate.
    """
    ratio = raw_sums_ratio(dtype, adds)

    def raw_sums_hold(square_mean, var):
        return T.And(T.And(square_mean <= 3.0e38, var >= 2.0**-100), square_mean <= ratio * var)

    return raw_sums_hold
