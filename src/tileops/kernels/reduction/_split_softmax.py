"""Split-row softmax statistics, shared by softmax, log_softmax, and logsumexp.

A handful of long rows cannot fill the device one block per row; the split
gives every row one block per segment. This module holds the gate, the
fp32 ``(max, sum)`` statistics pass, and the fold; the pass that writes each
op's output stays in that op's module.
"""

import functools
from math import prod

import tilelang
import tilelang.language as T

from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    FRAGMENT_ELEMS_PER_THREAD,
    align_up,
    ceildiv_int,
)

__all__ = [
    "SPLIT_BLOCKS_PER_SM",
    "edge_split_partials_kernel",
    "edge_split_view",
    "make_block_split_fold",
    "softmax_split_partials_kernel",
    "split_seg_n",
]

# Blocks per SM a split aims for; under this the grid runs the device empty.
SPLIT_BLOCKS_PER_SM = 2

# Rows shorter than this cannot amortize the fold pass. Measured threshold,
# not derived: below it the second launch outweighs the extra blocks.
_SPLIT_MIN_AMORTIZED_COLS = 16384

# Widest segment a partials block holds in its fragment.
_SPLIT_MAX_SEG_COLS = FRAGMENT_ELEMS_PER_THREAD * DEFAULT_THREADS


def split_seg_n(M: int, N: int, block_m: int, sm_count: int) -> int:
    """The split-row segment width, or 0 when one block per row is enough.

    Applies when the row grid of *block_m* rows a block leaves a device of *sm_count*
    SMs under-filled and a row is long enough to amortize the fold pass; the segment
    count targets ``SPLIT_BLOCKS_PER_SM`` blocks per SM and the width stays aligned and
    within the fragment cap.
    """
    target_blocks = SPLIT_BLOCKS_PER_SM * sm_count
    if align_up(N, DEFAULT_ALIGNMENT) < _SPLIT_MIN_AMORTIZED_COLS:
        return 0
    if ceildiv_int(M, block_m) >= target_blocks:
        return 0
    num_segs = max(1, ceildiv_int(target_blocks, M))
    seg_n = min(align_up(ceildiv_int(N, num_segs), DEFAULT_ALIGNMENT), _SPLIT_MAX_SEG_COLS)
    if ceildiv_int(N, seg_n) < 2:
        return 0
    return seg_n


@functools.lru_cache(maxsize=64)
def softmax_split_partials_kernel(M: int, N: int, seg_n: int, dtype: str, threads: int):
    """Per-segment softmax statistics: fp32 ``(max, sum)`` for a later fold.

    One block owns one ``seg_n``-column segment of one row and writes the
    segment's max and its sum of ``exp(x - max)``; a masked lane contributes
    ``exp(-inf) = 0``. A segment whose max is infinite sums ``exp(x)``
    instead, torch.logsumexp's shift: an all--inf segment then sums to zero
    rather than the NaN of ``exp(-inf - -inf)``, and a segment holding +inf
    sums to +inf, or to NaN when it also holds one.
    """
    num_segs = ceildiv_int(N, seg_n)

    @tilelang.jit(out_idx=[1, 2])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            seg_max: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            seg_sum: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
        ):
            with T.Kernel(num_segs, M, threads=threads) as (pid_s, pid_m):
                x_f32 = T.alloc_fragment((1, seg_n), "float32")
                m_s = T.alloc_fragment((1,), "float32")
                s_s = T.alloc_fragment((1,), "float32")
                shift = T.alloc_local((1,), "float32")

                for _, j in T.Parallel(1, seg_n):
                    x_f32[0, j] = T.if_then_else(
                        pid_s * seg_n + j < N,
                        T.cast(x[pid_m, pid_s * seg_n + j], "float32"),
                        -T.infinity("float32"),
                    )
                T.fill(m_s, -T.infinity("float32"))
                T.reduce_max(x_f32, m_s, dim=1, clear=False)
                shift[0] = T.if_then_else(
                    T.abs(m_s[0]) == T.infinity("float32"), T.cast(0.0, "float32"), m_s[0]
                )
                for _, j in T.Parallel(1, seg_n):
                    x_f32[0, j] = T.exp(x_f32[0, j] - shift[0])
                T.reduce_sum(x_f32, s_s, dim=1)

                seg_max[pid_m * num_segs + pid_s] = m_s[0]
                seg_sum[pid_m * num_segs + pid_s] = s_s[0]

        return main

    return _func


def make_block_split_fold(num_segs: int, threads: int, keep_inf: bool = False):
    """Create the macro folding one row's segment statistics across a whole block.

    Every lane folds a strided share of the ``num_segs`` pairs into a
    ``(1, threads)`` fragment, and two block reductions close it. A serial
    fold instead costs ``num_segs`` dependent loads and exponentials in one
    lane, which is the whole fold's latency.

    The statistics are read from global at *base*. A few hundred fp32 pairs
    that every block of one row reads sit in L2, and staging them through
    shared memory pays a barrier to restate that. The caller allocates
    *part_max* and *part_sum* as ``(1, threads)`` fp32 fragments and *row_max*
    and *row_sum* as ``(1,)`` fp32 fragments.

    Each segment's sum is rescaled by ``exp(seg_max - row_max)``, so a NaN
    anywhere in the row reaches the row sum, and a -inf segment below a
    finite row max contributes ``0 * exp(-inf) = 0``. Where both maxima are
    infinite, ``exp(inf - inf)`` leaves the row sum NaN: torch's softmax of
    an all--inf row or of a row holding +inf. *keep_inf* scales the segment
    holding the row max by one instead, which is torch's logsumexp: an
    all--inf row sums to 0 and a row holding +inf to the +inf its segment
    sums to, each still NaN when a NaN is present.
    """
    rounds = ceildiv_int(num_segs, threads)
    last = num_segs - 1

    @T.macro
    def fold(seg_max, seg_sum, base, part_max, part_sum, row_max, row_sum):
        for _, t in T.Parallel(1, threads):
            part_max[0, t] = -T.infinity("float32")
            for r in T.serial(rounds):
                # A lane past the last segment reads the last one; ``mine`` keeps it out.
                s = T.min(r * threads + t, last)
                mine = r * threads + t <= last
                part_max[0, t] = T.max(
                    part_max[0, t],
                    T.if_then_else(mine, seg_max[base + s], -T.infinity("float32")),
                )
        T.fill(row_max, -T.infinity("float32"))
        T.reduce_max(part_max, row_max, dim=1, clear=False)

        for _, t in T.Parallel(1, threads):
            part_sum[0, t] = 0.0
            for r in T.serial(rounds):
                s = T.min(r * threads + t, last)
                mine = r * threads + t <= last
                raw_gap = seg_max[base + s] - row_max[0]
                gap = (
                    T.if_then_else(seg_max[base + s] == row_max[0], T.cast(0.0, "float32"), raw_gap)
                    if keep_inf
                    else raw_gap
                )
                part_sum[0, t] = part_sum[0, t] + T.if_then_else(
                    mine, seg_sum[base + s] * T.exp(gap), T.cast(0.0, "float32")
                )
        T.reduce_sum(part_sum, row_sum, dim=1)

    return fold


def edge_split_view(
    shape: "tuple[int, ...]", k: int, j: int, threads: int
) -> "tuple[int, int, int] | None":
    """The ``(outer, kept, inner)`` view an edge-axis split reads, or None.

    An edge-axis reduction (a leading prefix of *k* axes plus a trailing
    suffix of *j* axes) leaves each kept row as ``outer`` contiguous runs of
    ``inner`` elements in the tensor's own layout, so the partials pass can
    read it without the permute ``rows_for_axes`` would pay. Eligible when a
    ``(1, inner)`` fragment builds (``threads`` divides ``inner``) and stays
    in registers.
    """
    outer = prod(shape[:k])
    inner = prod(shape[len(shape) - j :])
    kept = prod(shape[k : len(shape) - j])
    if inner % threads or inner > _SPLIT_MAX_SEG_COLS:
        return None
    return (outer, kept, inner)


@functools.lru_cache(maxsize=32)
def edge_split_partials_kernel(outer: int, kept: int, inner: int, dtype: str, threads: int):
    """Per-run softmax statistics over an ``(outer, kept, inner)`` view.

    One block owns one row's run ``x[s, m, :]`` and writes its fp32
    ``(max, sum)`` pair at ``m * outer + s`` -- row-major by kept row, the
    order ``make_block_split_fold`` reads. Semantics match
    ``softmax_split_partials_kernel``, including its shift of an infinite max.
    """

    @tilelang.jit(out_idx=[1, 2])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(outer, kept, inner), dtype],
            seg_max: T.Tensor[(kept * outer,), "float32"],  # noqa: F821
            seg_sum: T.Tensor[(kept * outer,), "float32"],  # noqa: F821
        ):
            with T.Kernel(outer, kept, threads=threads) as (pid_s, pid_m):
                x_f32 = T.alloc_fragment((1, inner), "float32")
                m_s = T.alloc_fragment((1,), "float32")
                s_s = T.alloc_fragment((1,), "float32")
                shift = T.alloc_local((1,), "float32")

                for _, i in T.Parallel(1, inner):
                    x_f32[0, i] = T.cast(x[pid_s, pid_m, i], "float32")
                T.fill(m_s, -T.infinity("float32"))
                T.reduce_max(x_f32, m_s, dim=1, clear=False)
                shift[0] = T.if_then_else(
                    T.abs(m_s[0]) == T.infinity("float32"), T.cast(0.0, "float32"), m_s[0]
                )
                for _, i in T.Parallel(1, inner):
                    x_f32[0, i] = T.exp(x_f32[0, i] - shift[0])
                T.reduce_sum(x_f32, s_s, dim=1)

                seg_max[pid_m * outer + pid_s] = m_s[0]
                seg_sum[pid_m * outer + pid_s] = s_s[0]

        return main

    return _func
