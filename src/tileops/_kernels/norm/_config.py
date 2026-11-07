"""Shared row-normalization configuration and reduction macros.

Non-power-of-two fragment rows use ``block_m=1``: larger row blocks can produce
replicated layouts, register spills, and serial reductions. Whole 16-byte-vector
rows up to 1024 elements can instead use an unpadded warp reduction."""

from typing import Optional

import tilelang.language as T
import torch

from tileops._kernels.constants import STATIC_SHARED_BYTES, VECTOR_ACCESS_BYTES
from tileops._kernels.tiling import ALIGNMENT
from tileops._utils import WARP_SHUFFLE_STAGES

__all__ = [
    "FP32_RANGE_BOUND",
    "FP32_RANGE_BOUND_BITS",
    "NARROW_ROW",
    "VARIANCE_FLOOR",
    "fold_chains",
    "make_centred_row_reduce",
    "make_corrected_row_reduce",
    "make_raw_sums_hold",
    "make_read_leads",
    "pairwise_sum",
    "raw_sums_ratio",
    "read_first",
    "row_padding",
    "select_row_config",
    "select_row_config_by_width",
    "select_row_configs",
    "warp_row_vector",
    "warp_sum",
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

# The magnitude past which a statistic counts as past fp32's range (its largest is 3.4e38),
# and its bit pattern, which a kernel compares where a NaN must also count as past it.
FP32_RANGE_BOUND = 3.0e38
FP32_RANGE_BOUND_BITS = 0x7F61B1E6
# The smallest variance the raw sums serve; below it a kernel takes the shifted rule.
VARIANCE_FLOOR = 2.0**-100


def row_padding(n: int, elem_bytes: int) -> int:
    """Pad to a power of two within shared-memory capacity, otherwise to ALIGNMENT."""
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
    """The sum of the expressions *terms*, added pairwise."""
    terms = list(terms)
    while len(terms) > 1:
        pairs = [a + b for a, b in zip(terms[::2], terms[1::2], strict=False)]
        terms = pairs + terms[len(terms) // 2 * 2 :]
    return terms[0]


def fold_chains(partial, first: int = 0, chains: int = 4):
    """The pairwise sum of a thread's add chains, ``partial[first:first + chains]``."""
    return pairwise_sum([partial[first + j] for j in range(chains)])


@T.macro
def warp_sum(acc, count: int, first: int = 0):
    """Add each of ``acc[first:first + count]`` across the warp's lanes, by XOR shuffles."""
    for st in T.unroll(WARP_SHUFFLE_STAGES):
        for j in T.unroll(first, first + count):
            acc[j] += T.shfl_xor(acc[j], T.shift_left(1, st))


@T.macro
def read_first(dst, index, element):
    """``dst[index] = element(0)``: a row's, channel's or piece's first element, where
    ``element(j)`` is the expression reading column *j*."""
    # FIXME(staged-rollout): a lone first-element read sits in a two-column loop
    #
    # Broken invariant: a scalar read compiles at any row stride.
    # Why: TileLang fails to compile a lone read at row strides from 32768 to 65535.
    # Cleanup: assign element(0) directly once tile-ai/tilelang#3453 is fixed.
    for j in T.unroll(2):
        if j == 0:
            dst[index] = element(j)


def make_read_leads(block_m, rows, n):
    """Read each row's first element, through `read_first`."""

    @T.macro
    def read_leads(x, pid_m, shift):
        for i in T.Parallel(block_m):
            read_first(
                shift,
                i,
                lambda j, i=i: T.cast(
                    x[T.min(pid_m * block_m + i, rows - 1), T.min(j, n - 1)], "float32"
                ),
            )

    return read_leads


def make_centred_row_reduce(block_m, n, n_padded, eps):
    """A macro ``(x_row, x_f32, shift, acc, centre, offset, rstd)`` taking corrected two-pass
    statistics of a row block: ``x_f32`` holds ``x - shift``; the output is
    ``(x - centre) * rstd + offset``."""

    @T.macro
    def row_reduce(x_row, x_f32, shift, acc, centre, offset, rstd):
        T.reduce_sum(x_f32, acc, dim=1)
        # By the reciprocal: a divide by a width that is no power of two is slow.
        for i in T.Parallel(block_m):
            centre[i] = shift[i] + acc[i] * (1.0 / n)

        for i, j in T.Parallel(block_m, n_padded):
            x_f32[i, j] = T.if_then_else(j < n, T.cast(x_row[i, j], "float32") - centre[i], 0.0)
        T.reduce_sum(x_f32, acc, dim=1)
        for i in T.Parallel(block_m):
            offset[i] = acc[i] * (1.0 / n)

        # Weight before summing: n * variance can overflow even when variance is finite.
        for i, j in T.Parallel(block_m, n_padded):
            x_f32[i, j] = x_f32[i, j] * (x_f32[i, j] * (1.0 / n))
        T.reduce_sum(x_f32, acc, dim=1)
        for i in T.Parallel(block_m):
            rstd[i] = T.rsqrt(acc[i] - offset[i] * offset[i] + eps)
            offset[i] = -offset[i] * rstd[i]

    return row_reduce


def make_corrected_row_reduce(block_m, n, n_padded, eps):
    """A macro ``(row, sq, shift, acc, acc_sq, ref, centre, rstd)`` taking corrected two-pass
    statistics of an fp32 row block less its first elements *shift*: it leaves ``row - ref`` in
    *row*, and the normalized row is ``(row - centre) * rstd``. ``acc_sq`` holds
    the mean square about ``ref``, shared with InstanceNorm's running-stat update."""
    padded = n != n_padded

    def _in_row(j, value):
        return T.if_then_else(j < n, value, T.cast(0.0, "float32")) if padded else value

    @T.macro
    def corrected_row_reduce(row, sq, shift, acc, acc_sq, ref, centre, rstd):
        for i, j in T.Parallel(block_m, n_padded):
            sq[i, j] = _in_row(j, row[i, j] - shift[i])
        T.reduce_sum(sq, acc, dim=1)
        for i in T.Parallel(block_m):
            ref[i] = shift[i] + acc[i] * (1.0 / n)
        for i, j in T.Parallel(block_m, n_padded):
            row[i, j] = _in_row(j, row[i, j] - ref[i])
            sq[i, j] = row[i, j] * (row[i, j] * (1.0 / n))
        # One accumulator each: sharing one puts a barrier between the two sums.
        T.reduce_sum(row, acc, dim=1)
        T.reduce_sum(sq, acc_sq, dim=1)
        for i in T.Parallel(block_m):
            centre[i] = acc[i] * (1.0 / n)
            rstd[i] = T.rsqrt(acc_sq[i] - centre[i] * centre[i] + eps)

    return corrected_row_reduce


def warp_row_vector(n: int, elem_bytes: int) -> Optional[int]:
    """Return a 16-byte vector width for aligned rows of at most 1024 elements."""
    vec = VECTOR_ACCESS_BYTES // elem_bytes
    # At 2048 16-bit columns the fragment body led it on H200: by 3% at 128 rows with one
    # row a block, and at parity at 512 rows.
    widest = 1024
    if n % vec or n > widest:
        return None
    return vec


def raw_sums_ratio(dtype: str, adds: int) -> float:
    """The largest ``E[x^2] / var`` at which fp32 sums of ``x`` and ``x^2``, each ``adds``
    dependent adds deep, keep a row's variance, mean and output within ``tau``:
    ``tau / (6 * adds * 2^-24)``. ``tau`` is half the dtype's accuracy for fp16 (``2^-11``)
    and bf16 (``2^-8``); for fp32 it is an error budget of ``2^-17``, not half its accuracy."""
    tau = {"float32": 2.0**-17, "float16": 2.0**-11, "bfloat16": 2.0**-8}[dtype]
    return tau / (6.0 * adds * 2.0**-24)


def make_raw_sums_hold(dtype: str, adds: int):
    """The predicate on ``(square_mean, var)`` under which a kernel keeps raw sums: within
    `raw_sums_ratio`, ``square_mean`` at most `FP32_RANGE_BOUND` and ``var`` at least
    `VARIANCE_FLOOR`."""
    ratio = raw_sums_ratio(dtype, adds)

    def raw_sums_hold(square_mean, var):
        return T.And(
            T.And(square_mean <= FP32_RANGE_BOUND, var >= VARIANCE_FLOOR),
            square_mean <= ratio * var,
        )

    return raw_sums_hold
