"""Split-row softmax statistics, shared by softmax, log_softmax, and logsumexp.

A handful of long rows cannot fill the device one block per row; the split
gives every row one block per segment. This module holds the gate, the
fp32 ``(max, sum)`` statistics pass, and the fold; the pass that writes each
op's output stays in that op's module.
"""

import functools

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

__all__ = [
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
