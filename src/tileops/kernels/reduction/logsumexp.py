"""LogSumExp forward kernels using TileLang.

  - logsumexp: y[i] = max_i + log(sum_i(exp(x[i,j] - max_i)))

Five implementations of one dispatch key, each stating the calls it serves over a
:class:`LogSumExpCall`: an edge-axis split read in the tensor's own layout, a
streaming kernel for long fp16/bf16 rows on a filled grid, a two-launch split for a
handful of long rows, a single-tile row kernel, and the general tiled row kernel.

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

from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    AUTOTUNE_THREADS,
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    VECTOR_ACCESS_BYTES,
    BlockConfigPlanner,
    RowTiledAutotuneMixin,
    align_up,
    ceildiv_int,
    restore_reduced,
    rows_for_axes,
    torch_dtype_nbytes,
)
from tileops.kernels.reduction._split_softmax import (
    edge_split_partials_kernel,
    make_block_split_fold,
    softmax_split_partials_kernel,
)
from tileops.kernels.reduction.call_spec import STREAMING_LOGSUMEXP, LogSumExpCall
from tileops.utils import WARP_LANES

__all__ = [
    "LogSumExpEdgeSplitKernel",
    "LogSumExpKernel",
    "LogSumExpSingleTileKernel",
    "LogSumExpSplitKernel",
    "LogSumExpStreamingKernel",
]


@functools.lru_cache(maxsize=64)
def _logsumexp_split_fold_kernel(M: int, N: int, dtype: str, seg_n: int):
    """Fold per-segment ``(max, sum)`` into one logsumexp per row.

    One block per row folds the row's pairs lane-parallel. The block is the
    narrowest power of two that holds ``num_segs``, since an idle lane still
    pays the block reduction; past a warp it stays one warp and each lane
    folds several pairs. Unlike softmax there is no second pass over the
    input. An all--inf row reads ``-inf + log(0)`` and a row holding +inf
    reads ``inf + log(inf)``, which are torch's ``-inf`` and ``+inf``.
    """
    num_segs = ceildiv_int(N, seg_n)
    lanes = min(WARP_LANES, 1 << (num_segs - 1).bit_length())
    fold = make_block_split_fold(num_segs, lanes, keep_inf=True)

    @tilelang.jit(out_idx=[2])
    def _func():
        @T.prim_func
        def main(
            seg_max: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            seg_sum: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(M, threads=lanes) as pid_m:
                part_max = T.alloc_fragment((1, lanes), "float32")
                part_sum = T.alloc_fragment((1, lanes), "float32")
                row_max = T.alloc_fragment((1,), "float32")
                row_sum = T.alloc_fragment((1,), "float32")

                fold(seg_max, seg_sum, pid_m * num_segs, part_max, part_sum, row_max, row_sum)
                y[pid_m] = T.cast(row_max[0] + T.log(row_sum[0]), dtype)

        return main

    return _func


# Single-tile kernel (N fits in shared memory) -- original fast path


@functools.lru_cache(maxsize=64)
def _logsumexp_kernel_single(M: int, N: int, dtype: str):
    """Build a single-tile logsumexp kernel (N fits in smem).

    Accepts an ``(M, N)`` input tensor.  When ``N`` is not a multiple of
    ``DEFAULT_ALIGNMENT``, the kernel uses element-wise ``T.if_then_else``
    loads that substitute ``-inf`` for out-of-bounds columns (kernel-side
    boundary handling).  When ``N`` is already aligned, the fast ``T.copy``
    path is used.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    _neg_inf = float("-inf")

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                row_max = T.alloc_fragment((block_m,), "float32")
                row_shift = T.alloc_fragment((block_m,), "float32")
                row_sum = T.alloc_fragment((block_m,), "float32")

                if _needs_pad:
                    # Kernel-side boundary handling: element-wise load
                    # with T.if_then_else masking for padding columns
                    # and row-tail safety (M % block_m != 0).
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_neg_inf, "float32"),
                            )
                else:
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(x_local[i, j], "float32")

                T.fill(row_max, -T.infinity("float32"))
                T.reduce_max(x_f32, row_max, dim=1, clear=False)
                # torch.logsumexp's shift: zero where the max is infinite, so an all -inf
                # row sums to 0 and a row holding +inf to +inf instead of NaN.
                for i in T.Parallel(block_m):
                    row_shift[i] = T.if_then_else(
                        T.abs(row_max[i]) == T.infinity("float32"),
                        T.cast(0.0, "float32"),
                        row_max[i],
                    )

                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        x_f32[i, j] = T.exp(x_f32[i, j] - row_shift[i])
                T.reduce_sum(x_f32, row_sum, dim=1)

                out_local = T.alloc_fragment((block_m,), dtype)
                for i in T.Parallel(block_m):
                    out_local[i] = row_shift[i] + T.log(row_sum[i])

                T.copy(out_local, y[pid_m * block_m])

        return main

    return _func


# Multi-tile kernel (N tiled over shared memory)


@functools.lru_cache(maxsize=64)
def _logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int):
    """Build a multi-tile logsumexp kernel.

    Uses online softmax recurrence across N-tiles:
      Single pass: compute running max and rescaled running sum.
      Then: logsumexp = max + log(sum).

    The input tensor has the raw shape $[M \\times N]$ (no host-side padding).
    Boundary handling for the last tile is performed via masked loads.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    total_cols = num_tiles * tile_n
    _needs_mask = total_cols > N
    _neg_inf = float("-inf")

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                tile_local = T.alloc_fragment((block_m, tile_n), dtype)
                tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")

                row_max = T.alloc_fragment((block_m,), "float32")
                row_sum = T.alloc_fragment((block_m,), "float32")
                prev_max = T.alloc_fragment((block_m,), "float32")
                row_shift = T.alloc_fragment((block_m,), "float32")
                tile_max = T.alloc_fragment((block_m,), "float32")
                tile_sum = T.alloc_fragment((block_m,), "float32")

                T.fill(row_max, -T.infinity("float32"))
                T.fill(row_shift, 0.0)
                T.fill(row_sum, 0.0)

                for t in T.Serial(num_tiles):
                    if _needs_mask:
                        # Only the last tile may have out-of-bounds columns.
                        # Use fast vectorized T.copy for all earlier tiles,
                        # and element-wise T.if_then_else only for the last.
                        with T.If(t < num_tiles - 1):
                            with T.Then():
                                T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                T.copy(shared_buf, tile_local)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        tile_f32[i, j] = T.cast(tile_local[i, j], "float32")
                            with T.Else():
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        tile_f32[i, j] = T.if_then_else(
                                            T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                                            T.cast(
                                                x[pid_m * block_m + i, t * tile_n + j], "float32"
                                            ),
                                            T.cast(_neg_inf, "float32"),
                                        )
                    else:
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        T.copy(shared_buf, tile_local)
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                tile_f32[i, j] = T.cast(tile_local[i, j], "float32")

                    T.fill(tile_max, -T.infinity("float32"))
                    T.reduce_max(tile_f32, tile_max, dim=1, clear=False)

                    for i in T.Parallel(block_m):
                        prev_max[i] = row_max[i]
                        row_max[i] = T.max(row_max[i], tile_max[i])
                        row_shift[i] = T.if_then_else(
                            T.abs(row_max[i]) == T.infinity("float32"),
                            T.cast(0.0, "float32"),
                            row_max[i],
                        )

                    for i in T.serial(block_m):
                        for j in T.Parallel(tile_n):
                            tile_f32[i, j] = T.exp(tile_f32[i, j] - row_shift[i])
                    T.reduce_sum(tile_f32, tile_sum, dim=1)

                    # Rescaled by the maxima, not the shifts: exp(0 - shift) of a
                    # row that has seen only -inf can overflow. An unchanged max
                    # scales by one, where exp(inf - inf) would be NaN.
                    for i in T.Parallel(block_m):
                        row_sum[i] = (
                            row_sum[i]
                            * T.exp(
                                T.if_then_else(
                                    prev_max[i] == row_max[i],
                                    T.cast(0.0, "float32"),
                                    prev_max[i] - row_max[i],
                                )
                            )
                            + tile_sum[i]
                        )

                out_local = T.alloc_fragment((block_m,), dtype)
                for i in T.Parallel(block_m):
                    out_local[i] = row_shift[i] + T.log(row_sum[i])

                T.copy(out_local, y[pid_m * block_m])

        return main

    return _func


# Streaming kernel (one block per row, direct vectorized loads)


@functools.lru_cache(maxsize=32)
def _logsumexp_kernel_streaming(M: int, N: int, dtype: str, threads: int, cols_per_thread: int):
    """Build a streaming logsumexp kernel for long rows on a filled grid.

    One block per row. Each thread vector-loads its ``cols_per_thread``
    consecutive elements per chunk straight into registers (a reduction has
    no reuse to stage through shared memory), keeps one running max and
    ``cols_per_thread`` independent fp32 sum chains rescaled once per chunk
    with the online-softmax recurrence, and merges once at the end: a warp
    shuffle tree, then one warp folding the per-warp partials.
    """
    chunk = threads * cols_per_thread
    if N % chunk:
        raise ValueError(f"streaming kernel needs N % {chunk} == 0, got N={N}")
    num_chunks = N // chunk
    num_warps = threads // WARP_LANES
    vec_elems = min(cols_per_thread, VECTOR_ACCESS_BYTES // torch_dtype_nbytes(dtype))
    vec_groups = cols_per_thread // vec_elems
    warp_stages = WARP_LANES.bit_length() - 1
    floor = STREAMING_LOGSUMEXP.max_floor
    # Clamp for exponent arguments, above every finite fp16/bf16 value: a +inf element
    # contributes exp2(+inf) = +inf and its row folds to +inf, matching torch.
    ceil = -floor

    @tilelang.jit(out_idx=[1])
    def _func():
        @T.macro
        def merge_pair(dst_m, dst_s, src_m, src_s, m_new, m_safe):
            # Exponents subtract the ceiling-clamped max, never the true one,
            # so (+inf) - (+inf) = NaN cannot form.
            m_new[0] = T.max(dst_m[0], src_m)
            m_safe[0] = T.min(m_new[0], ceil)
            dst_s[0] = dst_s[0] * T.exp2(
                (T.min(dst_m[0], ceil) - m_safe[0]) * LOG2E
            ) + src_s * T.exp2((T.min(src_m, ceil) - m_safe[0]) * LOG2E)
            dst_m[0] = m_new[0]

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                held = T.alloc_local((cols_per_thread,), dtype)
                held_f = T.alloc_local((cols_per_thread,), "float32")
                slots = T.alloc_local((cols_per_thread,), "float32")
                m_run = T.alloc_local((1,), "float32")
                m_safe = T.alloc_local((1,), "float32")
                s_run = T.alloc_local((1,), "float32")
                m_new = T.alloc_local((1,), "float32")
                scale = T.alloc_local((1,), "float32")
                other_m = T.alloc_local((1,), "float32")
                other_s = T.alloc_local((1,), "float32")
                warp_m = T.alloc_shared((num_warps,), "float32")
                warp_s = T.alloc_shared((num_warps,), "float32")

                m_run[0] = floor
                m_safe[0] = floor
                for c in T.serial(cols_per_thread):
                    slots[c] = 0.0

                for t in T.serial(num_chunks):
                    for g in T.serial(vec_groups):
                        for c in T.vectorized(vec_elems):
                            held[g * vec_elems + c] = x[
                                row, t * chunk + tx * cols_per_thread + g * vec_elems + c
                            ]
                    for c in T.serial(cols_per_thread):
                        held_f[c] = T.cast(held[c], "float32")

                    m_new[0] = m_run[0]
                    for c in T.serial(cols_per_thread):
                        m_new[0] = T.max(m_new[0], held_f[c])
                    scale[0] = T.exp2((m_safe[0] - T.min(m_new[0], ceil)) * LOG2E)
                    m_run[0] = m_new[0]
                    m_safe[0] = T.min(m_new[0], ceil)
                    for c in T.serial(cols_per_thread):
                        slots[c] = slots[c] * scale[0] + T.exp2((held_f[c] - m_safe[0]) * LOG2E)

                s_run[0] = slots[0]
                for c in T.serial(1, cols_per_thread):
                    s_run[0] = s_run[0] + slots[c]

                for stage in T.serial(warp_stages):
                    # Bound locals: a bare expression is substituted per mention.
                    other_m[0] = T.shfl_xor(
                        m_run[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES
                    )
                    other_s[0] = T.shfl_xor(
                        s_run[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES
                    )
                    merge_pair(m_run, s_run, other_m[0], other_s[0], m_new, m_safe)
                if tx % WARP_LANES == 0:
                    warp_m[tx // WARP_LANES] = m_run[0]
                    warp_s[tx // WARP_LANES] = s_run[0]
                T.sync_threads()
                if tx == 0:
                    for w in T.serial(1, num_warps):
                        merge_pair(warp_m, warp_s, warp_m[w], warp_s[w], m_new, m_safe)
                    y[row] = T.cast(warp_m[0] + T.log(warp_s[0]), dtype)

        return main

    return _func


class LogSumExpEdgeSplitKernel(Kernel):
    """LogSumExp over a leading plus a trailing axis set, read in the tensor's own layout.

    Each kept row is ``outer`` contiguous runs of ``inner`` elements. One launch writes
    per-run ``(max, sum)`` partials, a second folds each row's partials, and no permute
    runs. Serves the calls that have ``edge_view``.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: LogSumExpCall) -> bool:
        return call.edge_view is not None

    def __init__(self, call: LogSumExpCall):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype
        self.view = call.edge_view
        outer, kept, inner = self.view
        self.partials = edge_split_partials_kernel(
            outer, kept, inner, self.dtype_str, DEFAULT_THREADS
        )()
        self.fold = _logsumexp_split_fold_kernel(kept, outer * inner, self.dtype_str, inner)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce ``call.axes`` of the contiguous input *x*."""
        seg_max, seg_sum = self.partials(x.reshape(self.view))
        return restore_reduced(
            self.fold(seg_max, seg_sum), self.call.shape, self.call.axes, self.call.keepdim
        )


class LogSumExpStreamingKernel(Kernel):
    """LogSumExp of long fp16/bf16 rows on a filled grid, one block per row.

    Rows stream straight to registers at a fixed launch shape
    (``STREAMING_LOGSUMEXP``), so there is nothing to tune. Serves the calls without an
    ``edge_view`` whose rows ``stream``.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: LogSumExpCall) -> bool:
        return call.edge_view is None and call.streams

    def __init__(self, call: LogSumExpCall):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype
        self.kernel = _logsumexp_kernel_streaming(
            call.m,
            call.n,
            self.dtype_str,
            STREAMING_LOGSUMEXP.threads,
            STREAMING_LOGSUMEXP.cols_per_thread,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce ``call.axes`` of the contiguous input *x*."""
        y = self.kernel()(rows_for_axes(x, self.call.axes))
        return restore_reduced(y, self.call.shape, self.call.axes, self.call.keepdim)


class LogSumExpSplitKernel(Kernel):
    """LogSumExp of a handful of long rows, split into segments across blocks.

    One launch writes each segment's fp32 ``(max, sum)``, shared with softmax; a
    second folds each row's segments. Serves the calls without an ``edge_view`` whose
    rows do not ``stream`` and have a ``split_seg_n``.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: LogSumExpCall) -> bool:
        return call.edge_view is None and not call.streams and call.split_seg_n > 0

    def __init__(self, call: LogSumExpCall):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype
        seg_n = call.split_seg_n
        # split_seg_n's fragment cap assumes the default width.
        self.partials = softmax_split_partials_kernel(
            call.m, call.n, seg_n, self.dtype_str, DEFAULT_THREADS
        )()
        self.fold = _logsumexp_split_fold_kernel(call.m, call.n, self.dtype_str, seg_n)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce ``call.axes`` of the contiguous input *x*."""
        seg_max, seg_sum = self.partials(rows_for_axes(x, self.call.axes))
        return restore_reduced(
            self.fold(seg_max, seg_sum), self.call.shape, self.call.axes, self.call.keepdim
        )


class LogSumExpKernel(RowTiledAutotuneMixin, Kernel):
    """LogSumExp of rows tiled over shared memory, one block per ``block_m`` rows.

    The general implementation: it serves any call, and runs where no specialised one
    applies. Tiles over N with the online softmax recurrence (running max and
    rescaled sum); non-aligned N is masked inside the kernel. Tunes ``tile_n``,
    ``block_m`` and ``threads``.

    ``forward`` takes the tensor the op declares and reduces ``call.axes`` of it;
    moving those axes to the end, flattening to rows and shaping the result back are
    this kernel's business.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general: bool = True
    _MAX_TILE_N_CANDIDATES = 3

    @classmethod
    def applies(cls, call: LogSumExpCall) -> bool:
        return True

    def __init__(self, call: LogSumExpCall):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.M = call.m
        self.N = call.n
        self.dtype = call.dtype
        self.N_padded = align_up(self.N, DEFAULT_ALIGNMENT)
        self._elem_bytes = call.dtype.itemsize
        self._smem_budget = call.smem_budget
        self._planner = BlockConfigPlanner(self.N_padded, self._elem_bytes, self._smem_budget)
        self._block_m, self._tile_n = self._untuned_rows(call, self._planner)
        self.kernel = self._build_row_kernel(self._tile_n)
        self.init_config(None, call.tune)

    @staticmethod
    def _untuned_rows(call: LogSumExpCall, planner: BlockConfigPlanner) -> "tuple[int, int]":
        """``(block_m, tile_n)`` untuned; a row ``row_plan`` holds whole still tiles here."""
        block_m, tile_n = call.row_plan
        return block_m, tile_n or planner.tiled_tile_n(block_m, max(AUTOTUNE_THREADS))

    @property
    def default_config(self) -> dict:
        return {"block_m": self._block_m, "threads": DEFAULT_THREADS, "tile_n": self._tile_n}

    def _tile_n_candidates(self) -> list[int]:
        return [tn for tn in super()._tile_n_candidates() if tn] or [self._tile_n]

    def _build_row_kernel(self, tile_n: int):
        return _logsumexp_kernel_tiled(self.M, self.N, self.dtype_str, tile_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce ``call.axes`` of the contiguous input *x*."""
        program = self.kernel(self.config["block_m"], self.config["threads"])
        y = program(rows_for_axes(x, self.call.axes))
        return restore_reduced(y, self.call.shape, self.call.axes, self.call.keepdim)


class LogSumExpSingleTileKernel(LogSumExpKernel):
    """LogSumExp of rows that fit one shared-memory tile, ``block_m`` rows per block.

    Serves the calls without an ``edge_view`` whose rows do not ``stream``, have no
    ``split_seg_n`` and fit one tile in ``row_plan``. Tunes ``block_m`` and ``threads``.
    """

    general: bool = False

    @classmethod
    def applies(cls, call: LogSumExpCall) -> bool:
        return (
            call.edge_view is None
            and not call.streams
            and call.split_seg_n == 0
            and call.row_plan[1] == 0
        )

    @staticmethod
    def _untuned_rows(call: LogSumExpCall, planner: BlockConfigPlanner) -> "tuple[int, int]":
        return call.row_plan

    def _build_row_kernel(self, tile_n: int):
        return _logsumexp_kernel_single(self.M, self.N, self.dtype_str)
