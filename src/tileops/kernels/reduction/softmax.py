"""Softmax / log-softmax forward kernels using TileLang.

Implements a 2-pass online softmax algorithm for two operations:
  - softmax:     y[i,j] = exp(x[i,j] - max_i) / sum_i(exp(x[i,j] - max_i))
  - log_softmax: y[i,j] = x[i,j] - max_i - log(sum_i(exp(x[i,j] - max_i)))

Two implementations, each stating the calls it serves over a :class:`SoftmaxCall`: a
split across blocks for a handful of long rows, and the general row kernel.

Supports arbitrarily large N dimensions by tiling over N when the full
N_padded does not fit in shared memory.  Uses the online softmax recurrence
(track running max and rescaled running sum) across N-tiles.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.  Boundary handling for non-aligned N is performed
inside the kernel via masked loads and -inf fills, eliminating host-side
``F.pad`` from the forward path.  In the multi-tile path, only the last
tile uses element-wise masked loads; all preceding tiles use the fast
vectorized T.copy path since their columns are fully in-bounds.
"""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    AUTOTUNE_THREADS,
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    FRAGMENT_ELEMS_PER_THREAD,
    BlockConfigPlanner,
    RowTiledAutotuneMixin,
    align_up,
    ceildiv_int,
    restore_same_shape,
    rows_for_axes,
)
from tileops.kernels.reduction._split_softmax import (
    SPLIT_BLOCKS_PER_SM,
    make_block_split_fold,
    softmax_split_partials_kernel,
    split_seg_n,
)
from tileops.kernels.reduction.call_spec import SoftmaxCall, SoftmaxFwdInterface
from tileops.utils import WARP_LANES

__all__ = [
    "SoftmaxKernel",
    "SoftmaxSplitKernel",
]


# Single-tile kernel (N fits in shared memory) -- original fast path


@functools.lru_cache(maxsize=64)
def _softmax_kernel_single(M: int, N: int, op_kind: str, dtype: str, out_dtype: str):
    """Build a single-tile softmax/log_softmax kernel (N fits in smem).

    Accepts an ``(M, N)`` input tensor.  When ``N`` is not a multiple of
    ``DEFAULT_ALIGNMENT``, the kernel uses element-wise ``T.if_then_else``
    loads that substitute ``-inf`` for out-of-bounds columns (kernel-side
    boundary handling).  When ``N`` is already aligned, the fast ``T.copy``
    path is used.

    softmax never reads the row again once it has exponentiated, so it goes global to
    fragment and back with nothing staged in between, dropping the shared round trip
    it would otherwise pay. log_softmax needs the row a second time, for
    ``(x - max) - log(sum)``,
    and shared memory is the cheaper of the two places to keep it -- the alternative is
    a second fragment of row width, which is what pushes a thread's slice into local
    memory.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    # Compile-time Python constant used for padding; it is still cast to
    # the kernel dtype where needed inside the generated kernel.
    _neg_inf = float("-inf")
    _stages_row = op_kind == "log_softmax"

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M, N_padded), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                staged = T.alloc_shared((block_m, N_padded if _stages_row else 1), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                row_max = T.alloc_fragment((block_m,), "float32")
                row_sum = T.alloc_fragment((block_m,), "float32")
                row_scale = T.alloc_fragment((block_m,), "float32")

                if _stages_row:
                    if _needs_pad:
                        # Element-wise load, masked for padding columns and the row tail.
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                staged[i, j] = T.if_then_else(
                                    T.And(pid_m * block_m + i < M, j < N),
                                    x[pid_m * block_m + i, j],
                                    T.cast(_neg_inf, dtype),
                                )
                    else:
                        T.copy(x[pid_m * block_m, 0], staged)
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(staged[i, j], "float32")
                elif _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_neg_inf, "float32"),
                            )
                else:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(x[pid_m * block_m + i, j], "float32")

                T.fill(row_max, -T.infinity("float32"))
                T.reduce_max(x_f32, row_max, dim=1, clear=False)

                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        x_f32[i, j] = T.exp(x_f32[i, j] - row_max[i])
                T.reduce_sum(x_f32, row_sum, dim=1)

                if op_kind == "softmax":
                    # One reciprocal per row, then a multiply per element.
                    for i in T.Parallel(block_m):
                        row_scale[i] = 1.0 / row_sum[i]
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = x_f32[i, j] * row_scale[i]
                else:
                    for i in T.Parallel(block_m):
                        row_scale[i] = T.log(row_sum[i])
                    # (x - max) - log(sum) avoids log(0) on padding; x comes back from shared.
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = (
                                T.cast(staged[i, j], "float32") - row_max[i] - row_scale[i]
                            )

                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        y[pid_m * block_m + i, j] = T.cast(x_f32[i, j], out_dtype)

        return main

    return _func


# Multi-tile kernel (N tiled over shared memory)


@functools.lru_cache(maxsize=64)
def _softmax_kernel_tiled(M: int, N: int, op_kind: str, dtype: str, out_dtype: str, tile_n: int):
    """Build a multi-tile softmax/log_softmax kernel.

    Uses online softmax recurrence across N-tiles:
      Pass 1 (all tiles): compute running max and rescaled running sum.
      Pass 2 (all tiles): normalize using global max and sum.

    The input tensor has the raw shape $[M \\times N]$ (no host-side padding).
    Boundary handling for the last tile (where ``t * tile_n + j`` may
    exceed ``N``) is performed inside the kernel via ``T.if_then_else``
    masked loads.  Output columns are ``total_cols = num_tiles * tile_n``.

    NOTE: Pass 2 uses a dedicated shared memory buffer AND dedicated register
    fragments. TileLang's allocator may alias both shared buffers and register
    fragments across T.Serial loop boundaries, corrupting pass-1 accumulators
    (row_max, row_sum) if the same names are reused.  The dual-buffer shared
    memory cost is accounted for by passing ``num_buffers=2`` to
    ``compute_tile_n``. An *out_dtype* other than *dtype* stages the output tile in
    a third buffer of its own.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    total_cols = num_tiles * tile_n
    # The last tile may extend beyond N; boundary masking is needed when
    # total_cols > N (which is always true when N is not aligned, and also
    # when tile_n does not evenly divide N_padded).
    _needs_mask = total_cols > N
    _neg_inf = float("-inf")

    if op_kind == "softmax":

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                y: T.Tensor[(M, total_cols), out_dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    # --- Pass 1 fragments ---
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")

                    row_max = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")
                    prev_max = T.alloc_fragment((block_m,), "float32")
                    tile_max = T.alloc_fragment((block_m,), "float32")
                    tile_sum = T.alloc_fragment((block_m,), "float32")

                    T.fill(row_max, -T.infinity("float32"))
                    T.fill(row_sum, 0.0)

                    # Pass 1: compute global max and sum using online recurrence
                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            # Only the last tile may have out-of-bounds columns.
                            # Use fast vectorized T.copy for all earlier tiles,
                            # and element-wise T.if_then_else only for the last.
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                with T.Else():
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            tile_f32[i, j] = T.if_then_else(
                                                T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                ),
                                                T.cast(_neg_inf, "float32"),
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.fill(tile_max, -T.infinity("float32"))
                        T.reduce_max(tile_f32, tile_max, dim=1, clear=False)

                        for i in T.Parallel(block_m):
                            prev_max[i] = row_max[i]
                            row_max[i] = T.max(row_max[i], tile_max[i])

                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                tile_f32[i, j] = T.exp(tile_f32[i, j] - row_max[i])
                        T.reduce_sum(tile_f32, tile_sum, dim=1)

                        for i in T.Parallel(block_m):
                            row_sum[i] = row_sum[i] * T.exp(prev_max[i] - row_max[i]) + tile_sum[i]

                    # Precompute reciprocal to replace division with
                    # multiplication in the per-element normalisation.
                    inv_sum = T.alloc_fragment((block_m,), "float32")
                    for i in T.Parallel(block_m):
                        inv_sum[i] = 1.0 / row_sum[i]

                    # --- Pass 2: dedicated shared + register fragments ---
                    # TileLang's allocator aliases both shared buffers and
                    # register fragments across T.Serial loop boundaries.
                    # Using separate allocations for pass 2 prevents
                    # corruption of pass-1 accumulators (row_max, row_sum).
                    # compute_tile_n accounts for 2x shared memory via
                    # num_buffers=2.
                    p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                    p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                    if out_dtype == dtype:
                        p2_out = p2_shared
                    else:
                        p2_out = T.alloc_shared((block_m, tile_n), out_dtype)

                    # Pass 2: normalize, then cast the tile back into the shared
                    # buffer it was read from
                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            p2_f32[i, j] = (
                                                T.exp(
                                                    T.cast(p2_shared[i, j], "float32") - row_max[i]
                                                )
                                                * inv_sum[i]
                                            )
                                with T.Else():
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            p2_f32[i, j] = T.if_then_else(
                                                T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                                                T.exp(
                                                    T.cast(
                                                        x[pid_m * block_m + i, t * tile_n + j],
                                                        "float32",
                                                    )
                                                    - row_max[i]
                                                )
                                                * inv_sum[i],
                                                0.0,
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    p2_f32[i, j] = (
                                        T.exp(T.cast(p2_shared[i, j], "float32") - row_max[i])
                                        * inv_sum[i]
                                    )

                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                p2_out[i, j] = T.cast(p2_f32[i, j], out_dtype)
                        T.copy(p2_out, y[pid_m * block_m, t * tile_n])

            return main

    else:  # log_softmax

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                y: T.Tensor[(M, total_cols), out_dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    # --- Pass 1 fragments ---
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")

                    row_max = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")
                    prev_max = T.alloc_fragment((block_m,), "float32")
                    tile_max = T.alloc_fragment((block_m,), "float32")
                    tile_sum = T.alloc_fragment((block_m,), "float32")

                    T.fill(row_max, -T.infinity("float32"))
                    T.fill(row_sum, 0.0)

                    # Pass 1: compute global max and sum
                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                with T.Else():
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            tile_f32[i, j] = T.if_then_else(
                                                T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                ),
                                                T.cast(_neg_inf, "float32"),
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.fill(tile_max, -T.infinity("float32"))
                        T.reduce_max(tile_f32, tile_max, dim=1, clear=False)

                        for i in T.Parallel(block_m):
                            prev_max[i] = row_max[i]
                            row_max[i] = T.max(row_max[i], tile_max[i])

                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                tile_f32[i, j] = T.exp(tile_f32[i, j] - row_max[i])
                        T.reduce_sum(tile_f32, tile_sum, dim=1)

                        for i in T.Parallel(block_m):
                            row_sum[i] = row_sum[i] * T.exp(prev_max[i] - row_max[i]) + tile_sum[i]

                    # Precompute log(sum) to avoid recomputing per-element
                    log_sum = T.alloc_fragment((block_m,), "float32")
                    for i in T.Parallel(block_m):
                        log_sum[i] = T.log(row_sum[i])

                    # --- Pass 2: dedicated shared + register fragments ---
                    # (Same aliasing workaround as softmax -- see note above.)
                    p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                    p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                    if out_dtype == dtype:
                        p2_out = p2_shared
                    else:
                        p2_out = T.alloc_shared((block_m, tile_n), out_dtype)

                    # Pass 2: log-normalize (cast + compute fused)
                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            p2_f32[i, j] = (
                                                T.cast(p2_shared[i, j], "float32")
                                                - row_max[i]
                                                - log_sum[i]
                                            )
                                with T.Else():
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            p2_f32[i, j] = T.if_then_else(
                                                T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                )
                                                - row_max[i]
                                                - log_sum[i],
                                                T.cast(_neg_inf, "float32"),
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    p2_f32[i, j] = (
                                        T.cast(p2_shared[i, j], "float32") - row_max[i] - log_sum[i]
                                    )

                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                p2_out[i, j] = T.cast(p2_f32[i, j], out_dtype)
                        T.copy(p2_out, y[pid_m * block_m, t * tile_n])

            return main

    return _func


@functools.lru_cache(maxsize=64)
def _softmax_split_finalize_kernel(
    M: int, N: int, op_kind: str, dtype: str, out_dtype: str, seg_n: int, threads: int
):
    """Fold per-segment ``(max, sum)`` and write one normalized segment per block.

    The statistics are folded across the block, and the row's scale -- a
    reciprocal for softmax, a log for log_softmax -- is taken once before the
    block walks its segment. Both are what ``_softmax_kernel_single`` already
    does: a per-thread serial fold costs ``num_segs`` dependent exponentials
    in every lane, and a divide or a log left inside the element loop is paid
    once per element.
    """
    num_segs = ceildiv_int(N, seg_n)
    fold = make_block_split_fold(num_segs, threads)

    @tilelang.jit(out_idx=[3])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            seg_max: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            seg_sum: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            y: T.Tensor[(M, N), out_dtype],
        ):
            with T.Kernel(num_segs, M, threads=threads) as (pid_s, pid_m):
                part_max = T.alloc_fragment((1, threads), "float32")
                part_sum = T.alloc_fragment((1, threads), "float32")
                row_max = T.alloc_fragment((1,), "float32")
                row_sum = T.alloc_fragment((1,), "float32")
                row_scale = T.alloc_local((1,), "float32")

                fold(seg_max, seg_sum, pid_m * num_segs, part_max, part_sum, row_max, row_sum)

                if op_kind == "softmax":
                    row_scale[0] = 1.0 / row_sum[0]
                else:
                    row_scale[0] = T.log(row_sum[0])

                for _, j in T.Parallel(1, seg_n):
                    col = pid_s * seg_n + j
                    with T.If(col < N):  # noqa: SIM117
                        with T.Then():
                            if op_kind == "softmax":
                                y[pid_m, col] = T.cast(
                                    T.exp(T.cast(x[pid_m, col], "float32") - row_max[0])
                                    * row_scale[0],
                                    out_dtype,
                                )
                            else:
                                y[pid_m, col] = T.cast(
                                    T.cast(x[pid_m, col], "float32") - row_max[0] - row_scale[0],
                                    out_dtype,
                                )

        return main

    return _func


@functools.lru_cache(maxsize=64)
def _softmax_fused_split_kernel(
    M: int, N: int, op_kind: str, dtype: str, out_dtype: str, seg_n: int, threads: int
):
    """Normalize a few long rows in one kernel, reading each row once.

    The split pair reads the row twice: once for the segment statistics, once
    to normalize. Here a block keeps its segment in registers across a grid
    barrier and rescales it in place, so the row moves one read and one write
    and the launch that separated the two passes goes away. A grid barrier is
    what makes that legal, and :func:`fused_split_plan` is what says the grid
    can take one.

    The rescale is the flash-attention identity, ``exp(x - row_max) =
    exp(x - seg_max) * exp(seg_max - row_max)``, which is why the segment's
    exponentials survive the fold. A segment of only ``-inf`` holds zeros
    rather than the NaN ``exp(-inf - -inf)`` would leave, so it contributes
    nothing; an all--inf row still reads NaN for softmax and log_softmax, as
    torch does.
    """
    num_segs = ceildiv_int(N, seg_n)
    fold = make_block_split_fold(num_segs, threads)

    @tilelang.jit(out_idx=[3])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            seg_max: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            seg_sum: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            y: T.Tensor[(M, N), out_dtype],
        ):
            with T.Kernel(num_segs, M, threads=threads) as (pid_s, pid_m):
                held = T.alloc_fragment((1, seg_n), "float32")
                shifted = T.alloc_fragment((1, seg_n), "float32")
                seg_m = T.alloc_fragment((1,), "float32")
                seg_s = T.alloc_fragment((1,), "float32")
                part_max = T.alloc_fragment((1, threads), "float32")
                part_sum = T.alloc_fragment((1, threads), "float32")
                row_max = T.alloc_fragment((1,), "float32")
                row_sum = T.alloc_fragment((1,), "float32")
                row_scale = T.alloc_local((1,), "float32")
                shift = T.alloc_local((1,), "float32")

                for _, j in T.Parallel(1, seg_n):
                    held[0, j] = T.if_then_else(
                        pid_s * seg_n + j < N,
                        T.cast(x[pid_m, pid_s * seg_n + j], "float32"),
                        -T.infinity("float32"),
                    )
                T.fill(seg_m, -T.infinity("float32"))
                T.reduce_max(held, seg_m, dim=1, clear=False)
                # A segment of only -inf shifts by zero instead of by its own
                # -inf, which would leave every lane the NaN of exp(-inf - -inf).
                # Shifting by zero leaves exp(-inf) = 0, so the segment sums to
                # zero and contributes nothing; the test is the row's, so it is
                # taken once here rather than at every element.
                shift[0] = T.if_then_else(
                    seg_m[0] == -T.infinity("float32"), T.cast(0.0, "float32"), seg_m[0]
                )
                for _, j in T.Parallel(1, seg_n):
                    shifted[0, j] = T.exp(held[0, j] - shift[0])
                T.reduce_sum(shifted, seg_s, dim=1)
                seg_max[pid_m * num_segs + pid_s] = seg_m[0]
                seg_sum[pid_m * num_segs + pid_s] = seg_s[0]

                T.sync_grid()

                fold(seg_max, seg_sum, pid_m * num_segs, part_max, part_sum, row_max, row_sum)

                if op_kind == "softmax":
                    row_scale[0] = T.exp(seg_m[0] - row_max[0]) / row_sum[0]
                else:
                    row_scale[0] = row_max[0] + T.log(row_sum[0])

                # Indexed with the expression rather than a name bound to it:
                # TileLang's parallel-loop verifier does not substitute the
                # binding, and reads the store as j-independent.
                for _, j in T.Parallel(1, seg_n):
                    with T.If(pid_s * seg_n + j < N):  # noqa: SIM117
                        with T.Then():
                            if op_kind == "softmax":
                                y[pid_m, pid_s * seg_n + j] = T.cast(
                                    shifted[0, j] * row_scale[0], out_dtype
                                )
                            else:
                                y[pid_m, pid_s * seg_n + j] = T.cast(
                                    held[0, j] - row_scale[0], out_dtype
                                )

        return main

    return _func


class _SoftmaxKernelBase(Kernel, SoftmaxFwdInterface):
    """The softmax family: the policy every candidate's region and plan reads."""

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def num_buffers(cls, call: SoftmaxCall) -> int:
        """Row-sized shared buffers the tiled kernel holds.

        Two, one per pass, since TileLang's allocator aliases one buffer across the
        passes; an output of another dtype adds its stage.
        """
        in_bytes = call.dtype.itemsize
        out_stage = 0 if call.out_dtype == call.dtype else -(-call.out_dtype.itemsize // in_bytes)
        return 2 + out_stage

    @classmethod
    def row_plan(cls, call: SoftmaxCall) -> "tuple[int, int]":
        """The row kernel's untuned ``(block_m, tile_n)``; ``tile_n == 0`` is one tile."""
        return cls._plan_rows(
            align_up(call.n, DEFAULT_ALIGNMENT),
            call.dtype.itemsize,
            call.smem_budget,
            cls.num_buffers(call),
        )

    @classmethod
    def split_seg_n(cls, call: SoftmaxCall) -> int:
        """The segment width a split of the rows takes, or 0 when the untuned row grid fills."""
        return split_seg_n(call.m, call.n, cls.row_plan(call)[0], call.sm_count)

    @classmethod
    def fused_split_threads(cls, call: SoftmaxCall) -> "int | None":
        """The thread width a one-kernel split of *call* runs at, or None when it cannot.

        A fused split keeps its segment in registers across a grid barrier, so it
        reads the row once where the two-kernel pair reads it twice. Two conditions
        bound it. The grid must be co-resident, since a cooperative launch wider
        than the device holds is refused outright; ``split_seg_n`` already aims at
        ``SPLIT_BLOCKS_PER_SM`` blocks per SM, and this rejects the shapes where the
        segment cap pushed it past that. The two fp32 fragments must also fit the same
        per-thread budget one fragment gets elsewhere, which is what picks the width:
        the narrowest power of two from ``WARP_LANES`` up that holds them.
        """
        seg_n = cls.split_seg_n(call)
        if ceildiv_int(call.n, seg_n) * call.m > SPLIT_BLOCKS_PER_SM * call.sm_count:
            return None
        threads = WARP_LANES
        while threads <= DEFAULT_THREADS:
            if 2 * seg_n <= FRAGMENT_ELEMS_PER_THREAD * threads:
                return threads
            threads *= 2
        return None

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _plan_rows(
        n_padded: int, elem_bytes: int, smem_budget: int, num_buffers: int
    ) -> "tuple[int, int]":
        """One tile keeps the *smallest* block_m: each extra row hands every thread
        another ``N_padded / threads`` registers until the fragment spills. Tiled rows
        take the block_m with strictly the fewest tiles, since fewer tiles means fewer
        global passes, and the smallest one on a tie, for occupancy."""
        planner = BlockConfigPlanner(n_padded, elem_bytes, smem_budget, num_buffers=num_buffers)
        threads = max(AUTOTUNE_THREADS)
        best_bm = 1
        best_tile_n = planner.tile_n_for(1, threads)
        for bm in [2, 4, 8, 16]:
            if not planner.layout_ok(bm, n_padded, DEFAULT_THREADS):
                continue
            try:
                tn = planner.tile_n_for(bm, threads)
            except ValueError:
                continue
            if tn == 0 and not planner.frag_fits(bm, n_padded, DEFAULT_THREADS):
                continue
            if best_tile_n == 0:
                continue
            if tn == 0 or ceildiv_int(n_padded, tn) < ceildiv_int(n_padded, best_tile_n):
                best_bm = bm
                best_tile_n = tn
        return best_bm, best_tile_n


class SoftmaxSplitKernel(_SoftmaxKernelBase):
    """Softmax / log-softmax of a handful of long rows, split into segments across blocks.

    Where the grid is co-resident and a segment fits twice in registers, one kernel
    keeps each segment in registers across a grid barrier and reads the row once;
    otherwise one launch writes each segment's fp32 ``(max, sum)``, shared with
    logsumexp, and a second folds them and normalizes one segment per block. Serves the
    calls with a :meth:`split_seg_n`.
    """

    @classmethod
    def applies(cls, call: SoftmaxCall) -> bool:
        return cls.split_seg_n(call) > 0

    def __init__(self, call: SoftmaxCall):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype
        seg_n = self.split_seg_n(call)
        self.num_segs = ceildiv_int(call.n, seg_n)
        out_dtype = self.dtype_to_str(call.out_dtype)
        fused_threads = self.fused_split_threads(call)
        if fused_threads is not None:
            self.fused = _softmax_fused_split_kernel(
                call.m, call.n, call.op_kind, self.dtype_str, out_dtype, seg_n, fused_threads
            )()
            return
        self.fused = None
        # split_seg_n's fragment cap assumes the default width.
        self.partials = softmax_split_partials_kernel(
            call.m, call.n, seg_n, self.dtype_str, DEFAULT_THREADS
        )()
        self.finalize = _softmax_split_finalize_kernel(
            call.m, call.n, call.op_kind, self.dtype_str, out_dtype, seg_n, DEFAULT_THREADS
        )()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``call.axis`` of the contiguous input *x*."""
        rows = rows_for_axes(x, (self.call.axis,))
        if self.fused is not None:
            stats = torch.empty(
                2, self.call.m * self.num_segs, dtype=torch.float32, device=x.device
            )
            y = self.fused(rows, stats[0], stats[1])
        else:
            seg_max, seg_sum = self.partials(rows)
            y = self.finalize(rows, seg_max, seg_sum)
        return restore_same_shape(y, self.call.shape, (self.call.axis,))


class SoftmaxKernel(RowTiledAutotuneMixin, _SoftmaxKernelBase):
    """Softmax / log-softmax of rows, ``block_m`` rows a block.

    The general implementation: it serves any call, and runs where no specialised one
    applies. A row one shared-memory tile holds is normalized from that tile; a longer
    row takes two passes over N-tiles with the online softmax recurrence (running max
    and rescaled sum). Non-aligned N is masked inside the kernel. Tunes ``tile_n``,
    ``block_m`` and ``threads``.

    ``forward`` takes the tensor the op declares and normalizes over ``call.axis``;
    moving that axis to the end, flattening to rows and putting the result back are
    this kernel's business.
    """

    general: bool = True
    _MAX_TILE_N_CANDIDATES = 3

    @classmethod
    def applies(cls, call: SoftmaxCall) -> bool:
        return True

    def __init__(self, call: SoftmaxCall):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.M = call.m
        self.N = call.n
        self.dtype = call.dtype
        self.out_dtype_str = self.dtype_to_str(call.out_dtype)
        self.N_padded = align_up(self.N, DEFAULT_ALIGNMENT)
        self._elem_bytes = call.dtype.itemsize
        self._smem_budget = call.smem_budget
        self._planner = BlockConfigPlanner(
            self.N_padded, self._elem_bytes, self._smem_budget, num_buffers=self.num_buffers(call)
        )
        self._block_m, self._tile_n = self.row_plan(call)
        self.kernel = self._build_row_kernel(self._tile_n)
        self.init_config(None)

    @property
    def default_config(self) -> dict:
        return {"block_m": self._block_m, "threads": DEFAULT_THREADS, "tile_n": self._tile_n}

    def _build_row_kernel(self, tile_n: int):
        if tile_n == 0:
            return _softmax_kernel_single(
                self.M, self.N, self.call.op_kind, self.dtype_str, self.out_dtype_str
            )
        return _softmax_kernel_tiled(
            self.M, self.N, self.call.op_kind, self.dtype_str, self.out_dtype_str, tile_n
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``call.axis`` of the contiguous input *x*.

        The prim_func writes an alignment-padded row; the surplus columns are trimmed.
        """
        program = self.kernel(self.config["block_m"], self.config["threads"])
        y = program(rows_for_axes(x, (self.call.axis,)))
        y = y[:, : self.N] if y.shape[1] > self.N else y
        return restore_same_shape(y, self.call.shape, (self.call.axis,))
