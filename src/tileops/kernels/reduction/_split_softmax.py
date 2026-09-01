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
import torch

from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    FRAGMENT_ELEMS_PER_THREAD,
    align_up,
    ceildiv_int,
)
from tileops.utils import WARP_LANES

__all__ = [
    "edge_split_partials_kernel",
    "edge_split_view",
    "fused_split_plan",
    "make_block_split_fold",
    "make_split_fold",
    "softmax_split_partials_kernel",
    "split_seg_n",
    "split_target_blocks",
]

# Blocks per SM a split aims for; under this the grid runs the device empty.
_OCCUPANCY_FACTOR = 2

# Rows shorter than this cannot amortize the fold pass. Measured threshold,
# not derived: below it the second launch outweighs the extra blocks.
_SPLIT_MIN_AMORTIZED_COLS = 16384

# Widest segment a partials block holds in its fragment.
_SPLIT_MAX_SEG_COLS = FRAGMENT_ELEMS_PER_THREAD * DEFAULT_THREADS


def split_target_blocks(device_index: "int | None" = None) -> int:
    """The block count a split aims for: ``_OCCUPANCY_FACTOR`` per SM.

    ``None`` reads the current device; without CUDA it falls back to an H200's
    SM count, mirroring ``device_smem_budget``'s auto-detect fallback.
    """
    try:
        if not torch.cuda.is_available():
            return _OCCUPANCY_FACTOR * 132
        if device_index is None:
            device_index = torch.cuda.current_device()
        return (
            _OCCUPANCY_FACTOR * torch.cuda.get_device_properties(device_index).multi_processor_count
        )
    except (RuntimeError, AssertionError):
        return _OCCUPANCY_FACTOR * 132


def split_seg_n(M: int, N: int, block_m: int, target_blocks: int) -> int:
    """The split-row segment width, or 0 when one block per row is enough.

    Applies when the row grid leaves the device under-filled and a row is
    long enough to amortize the fold pass; the segment count targets
    *target_blocks* and the width stays aligned and within the fragment cap.
    """
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
    segment's max and its sum of ``exp(x - max)``. With a finite segment max,
    a masked lane contributes ``exp(-inf) = 0``; a segment whose max is
    ``-inf`` (all lanes masked ``-inf``) writes an explicit zero sum, since
    ``exp(-inf - -inf)`` is NaN.
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

                for _, j in T.Parallel(1, seg_n):
                    x_f32[0, j] = T.if_then_else(
                        pid_s * seg_n + j < N,
                        T.cast(x[pid_m, pid_s * seg_n + j], "float32"),
                        -T.infinity("float32"),
                    )
                T.fill(m_s, -T.infinity("float32"))
                T.reduce_max(x_f32, m_s, dim=1, clear=False)
                for _, j in T.Parallel(1, seg_n):
                    x_f32[0, j] = T.exp(x_f32[0, j] - m_s[0])
                T.reduce_sum(x_f32, s_s, dim=1)

                seg_max[pid_m * num_segs + pid_s] = m_s[0]
                seg_sum[pid_m * num_segs + pid_s] = T.if_then_else(
                    m_s[0] == -T.infinity("float32"), T.cast(0.0, "float32"), s_s[0]
                )

        return main

    return _func


def make_split_fold(num_segs: int):
    """Create the macro folding one row's segment statistics.

    Reads ``num_segs`` fp32 pairs starting at *base* and leaves the row's max
    and rescaled sum in two scalar locals; *held* is a caller-allocated fp32
    scalar, since a macro argument is substituted per mention and an indexed
    read spelled twice would load twice. An all--inf segment contributes
    nothing; folding it through ``exp(-inf - row_max)`` would turn NaN when
    ``row_max`` is -inf too. An all--inf row leaves ``(-inf, 0)``, which reads
    as torch's NaN softmax and -inf logsumexp downstream.
    """

    @T.macro
    def fold(seg_max, seg_sum, base, row_max, row_sum, held):
        row_max[0] = -T.infinity("float32")
        for s in T.serial(num_segs):
            row_max[0] = T.max(row_max[0], seg_max[base + s])
        row_sum[0] = 0.0
        for s in T.serial(num_segs):
            held[0] = seg_max[base + s]
            row_sum[0] = row_sum[0] + T.if_then_else(
                held[0] == -T.infinity("float32"),
                T.cast(0.0, "float32"),
                seg_sum[base + s] * T.exp(held[0] - row_max[0]),
            )

    return fold


def make_block_split_fold(num_segs: int, threads: int):
    """Create the macro folding one row's segment statistics across a whole block.

    Every lane folds a strided share of the ``num_segs`` pairs into a
    ``(1, threads)`` fragment, and two block reductions close it. Use it where
    the block goes on to walk the row afterwards: a per-thread serial fold
    there costs ``num_segs`` dependent exponentials in every lane, and that
    cost is the block's rather than the data's. :func:`make_split_fold` stays
    the form for a fold that is a kernel's only work.

    The statistics are read from global at *base*. A few hundred fp32 pairs
    that every block of one row reads sit in L2, and staging them through
    shared memory pays a barrier to restate that. The caller allocates
    *part_max* and *part_sum* as ``(1, threads)`` fp32 fragments and *row_max*
    and *row_sum* as ``(1,)`` fp32 fragments. Segment semantics match
    :func:`make_split_fold`: an all--inf segment contributes nothing, and an
    all--inf row leaves ``(-inf, 0)``.
    """
    rounds = ceildiv_int(num_segs, threads)
    last = num_segs - 1

    def owned(r, t):
        """The segment lane *t* folds in round *r*, and whether it has one.

        A lane past the last segment reads the last one instead of running off
        the array; the flag is what keeps its value out of both reductions.
        """
        return T.min(r * threads + t, last), r * threads + t <= last

    @T.macro
    def fold(seg_max, seg_sum, base, part_max, part_sum, row_max, row_sum):
        for _, t in T.Parallel(1, threads):
            part_max[0, t] = -T.infinity("float32")
            for r in T.serial(rounds):
                s, mine = owned(r, t)
                part_max[0, t] = T.max(
                    part_max[0, t],
                    T.if_then_else(mine, seg_max[base + s], -T.infinity("float32")),
                )
        T.fill(row_max, -T.infinity("float32"))
        T.reduce_max(part_max, row_max, dim=1, clear=False)

        for _, t in T.Parallel(1, threads):
            part_sum[0, t] = 0.0
            for r in T.serial(rounds):
                s, mine = owned(r, t)
                part_sum[0, t] = part_sum[0, t] + T.if_then_else(
                    T.And(mine, seg_max[base + s] != -T.infinity("float32")),
                    seg_sum[base + s] * T.exp(seg_max[base + s] - row_max[0]),
                    T.cast(0.0, "float32"),
                )
        T.reduce_sum(part_sum, row_sum, dim=1)

    return fold


def fused_split_plan(M: int, N: int, seg_n: int) -> "int | None":
    """The thread width a one-kernel split runs at, or None when it cannot.

    A fused split keeps its segment in registers across a grid barrier, so it
    reads the row once where the two-kernel pair reads it twice. Two conditions
    bound it. The grid must be co-resident, since a cooperative launch wider
    than the device holds is refused outright; ``split_seg_n`` already aims at
    ``split_target_blocks``, and this rejects the shapes where the segment cap
    pushed it past that. The two fp32 fragments must also fit the same
    per-thread budget one fragment gets elsewhere, which is what picks the
    width: the narrowest power of two from ``WARP_LANES`` up that holds them.
    """
    num_segs = ceildiv_int(N, seg_n)
    if num_segs * M > split_target_blocks():
        return None
    threads = WARP_LANES
    while threads <= DEFAULT_THREADS:
        if 2 * seg_n <= FRAGMENT_ELEMS_PER_THREAD * threads:
            return threads
        threads *= 2
    return None


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
    order ``make_split_fold`` reads. Semantics match
    ``softmax_split_partials_kernel``: an all--inf run writes a zero sum.
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

                for _, i in T.Parallel(1, inner):
                    x_f32[0, i] = T.cast(x[pid_s, pid_m, i], "float32")
                T.fill(m_s, -T.infinity("float32"))
                T.reduce_max(x_f32, m_s, dim=1, clear=False)
                for _, i in T.Parallel(1, inner):
                    x_f32[0, i] = T.exp(x_f32[0, i] - m_s[0])
                T.reduce_sum(x_f32, s_s, dim=1)

                seg_max[pid_m * outer + pid_s] = m_s[0]
                seg_sum[pid_m * outer + pid_s] = T.if_then_else(
                    m_s[0] == -T.infinity("float32"), T.cast(0.0, "float32"), s_s[0]
                )

        return main

    return _func
