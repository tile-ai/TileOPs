"""Softmax / log-softmax forward kernels using TileLang.

Kernels for two operations:
  - softmax:     y[i,j] = exp(x[i,j] - max_i) / sum_i(exp(x[i,j] - max_i))
  - log_softmax: y[i,j] = x[i,j] - max_i - log(sum_i(exp(x[i,j] - max_i)))

Three implementations, each stating the calls it serves over a :class:`SoftmaxCall`: a
split across blocks for a handful of long rows, resident CTAs streaming the other rows
no shared-memory tile holds, and the general row kernel for rows one tile holds.
``softmax_on_chip`` adds a cluster kernel for rows of 32768 elements or more on SM90.

Boundary handling for non-aligned N is performed inside the kernel via masked loads
and -inf fills, eliminating host-side ``F.pad`` from the forward path.
"""

import functools

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel, vector_aligned
from tileops.kernels.reduction._primitives import (
    AUTOTUNE_THREADS,
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    FRAGMENT_ELEMS_PER_THREAD,
    BlockConfigPlanner,
    RowTiledAutotuneMixin,
    align_up,
    ceildiv_int,
    exp_shifted,
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
    "SoftmaxStreamingKernel",
]


@functools.lru_cache(maxsize=32)
def _softmax_streaming_kernel(M, N, op_kind, dtype, out_dtype, ctas, ctas_per_sm, held_bytes):
    """Build the two-pass softmax/log_softmax for rows no CTA holds on chip.

    ``ctas`` resident CTAs walk the rows in turn. The first pass reads a row in tiles of
    ``threads * accesses`` vectors and folds each thread's running ``(max, sum)`` with the
    online recurrence; the pairs then fold across the warp by shuffles and across warps
    through shared memory. The second pass reads the row again from the far end, where
    the tiles read last are likeliest still in L2, and writes the result. The first
    tiles of a row, as many whole ones as ``held_bytes`` of shared memory take, are kept
    from the first pass, so the second reads them from shared memory instead.
    """
    vec = VECTOR_ACCESS_BYTES // (DataType(dtype).bits // 8)
    while N % vec:
        vec //= 2
    neg_inf = float("-inf")

    @tilelang.jit(out_idx=[1])
    def _func(threads, accesses):
        tile = threads * accesses * vec
        tiles = -(-N // tile)
        warps = threads // WARP_LANES
        kept = min(N // tile, held_bytes // (tile * DataType(dtype).bits // 8))

        def row_scale(total):
            """What a row's result takes from its sum: its reciprocal, or its log."""
            if op_kind == "softmax":
                return 1.0 / total
            return T.log(total)

        def fold(m, s, other_m, other_s):
            """The ``(max, sum)`` of two partial pairs; two empty ones stay empty."""
            top = T.max(m, other_m)
            both = s * exp_shifted(m, top) + other_s * exp_shifted(other_m, top)
            return top, T.if_then_else(top == neg_inf, T.cast(0, "float32"), both)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M, N), out_dtype],
        ):
            with T.Kernel(ctas, threads=threads) as cta:
                # Held to the registers that leave room for every CTA an SM is given.
                T.annotate_min_blocks_per_sm(ctas_per_sm)
                tx = T.get_thread_binding()
                held = T.alloc_local([accesses * vec], dtype)
                out = T.alloc_local([vec], out_dtype)
                stat = T.alloc_local([2], "float32")  # this thread's (max, sum)
                peer = T.alloc_local([2], "float32")
                tile_max = T.alloc_local([1], "float32")
                # Two slots, so a row's pairs never overwrite the ones still being read.
                warp_stats = T.alloc_shared([2, 2, warps], "float32")
                head = T.alloc_shared([max(kept, 1) * tile], dtype)

                for step in T.serial(T.ceildiv(M - cta, ctas)):
                    row = cta + step * ctas
                    stat[0] = T.cast(neg_inf, "float32")
                    stat[1] = T.cast(0, "float32")
                    for t in T.serial(tiles):
                        for a in T.unroll(accesses):
                            col = t * tile + (a * threads + tx) * vec
                            if col < N:
                                for i in T.vectorized(vec):
                                    held[a * vec + i] = x[row, col + i]
                            else:
                                for i in T.vectorized(vec):
                                    held[a * vec + i] = T.cast(neg_inf, dtype)
                            if t < kept:
                                for i in T.vectorized(vec):
                                    head[t * tile + (a * threads + tx) * vec + i] = held[
                                        a * vec + i
                                    ]
                        tile_max[0] = T.cast(neg_inf, "float32")
                        for j in T.unroll(accesses * vec):
                            tile_max[0] = T.max(tile_max[0], T.cast(held[j], "float32"))
                        top = T.max(stat[0], tile_max[0])
                        stat[1] = T.if_then_else(
                            top == neg_inf,
                            T.cast(0, "float32"),
                            stat[1] * exp_shifted(stat[0], top),
                        )
                        stat[0] = top
                        for j in T.unroll(accesses * vec):
                            stat[1] += T.if_then_else(
                                top == neg_inf,
                                T.cast(0, "float32"),
                                exp_shifted(T.cast(held[j], "float32"), top),
                            )
                    for k in T.unroll(WARP_LANES.bit_length() - 1):
                        peer[0] = T.shfl_xor(stat[0], T.shift_left(1, k))
                        peer[1] = T.shfl_xor(stat[1], T.shift_left(1, k))
                        stat[0], stat[1] = fold(stat[0], stat[1], peer[0], peer[1])
                    if tx % WARP_LANES == 0:
                        warp_stats[step % 2, 0, tx // WARP_LANES] = stat[0]
                        warp_stats[step % 2, 1, tx // WARP_LANES] = stat[1]
                    T.sync_threads()
                    stat[0] = T.cast(neg_inf, "float32")
                    stat[1] = T.cast(0, "float32")
                    for w in T.unroll(warps):
                        stat[0], stat[1] = fold(
                            stat[0], stat[1], warp_stats[step % 2, 0, w], warp_stats[step % 2, 1, w]
                        )
                    scale = row_scale(stat[1])

                    for t_back in T.serial(tiles):
                        t = tiles - 1 - t_back
                        for a in T.unroll(accesses):
                            col = t * tile + (a * threads + tx) * vec
                            if col < N:
                                if t < kept:
                                    for i in T.vectorized(vec):
                                        held[a * vec + i] = head[col + i]
                                else:
                                    for i in T.vectorized(vec):
                                        held[a * vec + i] = x[row, col + i]
                                for i in T.unroll(vec):
                                    v = T.cast(held[a * vec + i], "float32")
                                    if op_kind == "softmax":
                                        out[i] = T.cast(exp_shifted(v, stat[0]) * scale, out_dtype)
                                    else:
                                        out[i] = T.cast(v - stat[0] - scale, out_dtype)
                                for i in T.vectorized(vec):
                                    y[row, col + i] = out[i]

        return main

    return _func


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
                        x_f32[i, j] = exp_shifted(x_f32[i, j], row_max[i])
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

    def _rows(self, x: torch.Tensor) -> torch.Tensor:
        """The rows of *x* along ``call.axis``, starting on a 16-byte vector boundary."""
        return rows_for_axes(vector_aligned(x), (self.call.axis,))

    @classmethod
    def num_buffers(cls, call: SoftmaxCall) -> int:
        """Row-sized shared buffers a row block is planned with.

        Two, and an output of another dtype adds its stage. The plan this sizes decides
        which rows one tile holds, and so where :class:`SoftmaxKernel` stops serving.
        """
        in_bytes = call.dtype.itemsize
        out_stage = 0 if call.out_dtype == call.dtype else -(-call.out_dtype.itemsize // in_bytes)
        return 2 + out_stage

    @classmethod
    def row_scratch_per_thread(cls, call: SoftmaxCall) -> int:
        """Shared bytes a thread's reduction scratch takes beside the one-tile row.

        log_softmax reads its staged row again after both reductions, so their scratch,
        one fp32 a thread, sits beside the row. softmax stages no row.
        """
        return 4 if call.op_kind == "log_softmax" else 0

    @classmethod
    def row_plan(cls, call: SoftmaxCall) -> "tuple[int, int]":
        """The row kernel's untuned ``(block_m, tile_n)``; ``tile_n == 0`` is one tile."""
        return cls._plan_rows(
            align_up(call.n, DEFAULT_ALIGNMENT),
            call.dtype.itemsize,
            call.smem_budget,
            cls.num_buffers(call),
            cls.row_scratch_per_thread(call),
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
        n_padded: int, elem_bytes: int, smem_budget: int, num_buffers: int, row_scratch: int
    ) -> "tuple[int, int]":
        """One tile keeps the *smallest* block_m: each extra row hands every thread
        another ``N_padded / threads`` registers until the fragment spills. Tiled rows
        take the block_m with strictly the fewest tiles, since fewer tiles means fewer
        global passes, and the smallest one on a tie, for occupancy."""
        planner = BlockConfigPlanner(
            n_padded,
            elem_bytes,
            smem_budget,
            num_buffers=num_buffers,
            row_scratch_per_thread=row_scratch,
        )
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
        rows = self._rows(x)
        if self.fused is not None:
            stats = torch.empty(
                2, self.call.m * self.num_segs, dtype=torch.float32, device=x.device
            )
            y = self.fused(rows, stats[0], stats[1])
        else:
            seg_max, seg_sum = self.partials(rows)
            y = self.finalize(rows, seg_max, seg_sum)
        return restore_same_shape(y, self.call.shape, (self.call.axis,))


class SoftmaxStreamingKernel(_SoftmaxKernelBase):
    """Softmax / log-softmax of rows no shared-memory tile holds, from resident CTAs.

    Resident CTAs walk the rows in turn; each row is read twice, the second time from
    the far end, except its first tiles, which wait in shared memory between the passes.
    Serves the rows no shared-memory tile holds, except those :class:`SoftmaxSplitKernel`
    splits.

    A tile is 16384 elements, four 16-byte vectors a thread: 512 threads of a 16-bit
    dtype, 1024 of fp32. Each SM is given 1024 threads, which leaves every thread 64
    registers, and its CTAs share its shared memory for the kept tiles.
    """

    @classmethod
    def applies(cls, call: SoftmaxCall) -> bool:
        return cls.row_plan(call)[1] != 0 and cls.split_seg_n(call) == 0

    def __init__(self, call: SoftmaxCall):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype
        ctas_per_sm = 1024 // self.default_config["threads"]
        self.kernel = _softmax_streaming_kernel(
            call.m,
            call.n,
            call.op_kind,
            self.dtype_str,
            self.dtype_to_str(call.out_dtype),
            min(call.m, ctas_per_sm * call.sm_count),
            ctas_per_sm,
            # A CTA keeps 1 KB for the warps' statistics rather than row tiles.
            call.smem_budget // ctas_per_sm - 1024,
        )
        self.init_config(None)

    @property
    def default_config(self) -> dict:
        accesses = 4
        vec = VECTOR_ACCESS_BYTES // self.call.dtype.itemsize
        return {"threads": 16384 // (accesses * vec), "accesses": accesses}

    @property
    def autotune_configs(self) -> list[dict]:
        """The default alone: tuning repeats one candidate back to back, which reads the
        row out of L2 and ranks a schedule this kernel never runs as."""
        return [self.default_config]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``call.axis`` of the contiguous input *x*."""
        y = self.kernel(**self.config)(self._rows(x))
        return restore_same_shape(y, self.call.shape, (self.call.axis,))


class SoftmaxKernel(RowTiledAutotuneMixin, _SoftmaxKernelBase):
    """Softmax / log-softmax of rows one shared-memory tile holds, ``block_m`` rows a block.

    The general implementation: it runs where no specialised one applies. Each row is
    normalized from its tile; non-aligned N is masked inside the kernel. Tunes
    ``block_m`` and ``threads``.

    ``forward`` takes the tensor the op declares and normalizes over ``call.axis``;
    moving that axis to the end, flattening to rows and putting the result back are
    this kernel's business.
    """

    general: bool = True
    _MAX_TILE_N_CANDIDATES = 3

    @classmethod
    def applies(cls, call: SoftmaxCall) -> bool:
        return cls.row_plan(call)[1] == 0

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
            self.N_padded,
            self._elem_bytes,
            self._smem_budget,
            num_buffers=self.num_buffers(call),
            row_scratch_per_thread=self.row_scratch_per_thread(call),
        )
        self._block_m, self._tile_n = self.row_plan(call)
        self.kernel = self._build_row_kernel(self._tile_n)
        self.init_config(None)

    @property
    def default_config(self) -> dict:
        return {"block_m": self._block_m, "threads": DEFAULT_THREADS, "tile_n": self._tile_n}

    def _build_row_kernel(self, tile_n: int):
        # ``applies`` admits one-tile rows only, so every candidate's tile_n is 0.
        return _softmax_kernel_single(
            self.M, self.N, self.call.op_kind, self.dtype_str, self.out_dtype_str
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``call.axis`` of the contiguous input *x*.

        The prim_func writes an alignment-padded row; the surplus columns are trimmed.
        """
        program = self.kernel(self.config["block_m"], self.config["threads"])
        y = program(self._rows(x))
        y = y[:, : self.N] if y.shape[1] > self.N else y
        return restore_same_shape(y, self.call.shape, (self.call.axis,))
