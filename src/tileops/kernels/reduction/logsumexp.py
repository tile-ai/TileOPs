"""LogSumExp forward kernel using TileLang.

Implements a 2-pass online algorithm for:
  - logsumexp: y[i] = max_i + log(sum_i(exp(x[i,j] - max_i)))

Supports arbitrarily large N dimensions by tiling over N when the full
N_padded does not fit in shared memory.  Uses the online softmax recurrence
(track running max and rescaled running sum) across N-tiles.

Long fp16/bf16 rows on a filled grid take a streaming kernel instead
(see ``_logsumexp_kernel_streaming`` and ``StreamingLogSumExpPolicy``).

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.  Boundary handling for non-aligned N is performed
inside the kernel via masked loads and -inf fills, eliminating host-side
``F.pad`` from the forward path.  In the multi-tile path, only the last
tile uses element-wise masked loads; all preceding tiles use the fast
vectorized T.copy path since their columns are fully in-bounds.
"""

import functools
from dataclasses import dataclass
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    VECTOR_ACCESS_BYTES,
    BlockConfigPlanner,
    RowTiledAutotuneMixin,
    align_up,
    ceildiv_int,
    device_smem_budget,
    edge_axis_split,
    restore_reduced,
    rows_for_axes,
    torch_dtype_nbytes,
)
from tileops.kernels.reduction._split_softmax import (
    edge_split_partials_kernel,
    edge_split_view,
    make_split_fold,
    softmax_split_partials_kernel,
    split_seg_n,
    split_target_blocks,
)
from tileops.utils import WARP_LANES

# These two kernels bake tile_n in at build time and default to the wider
# thread block; AUTOTUNE_THREADS still bounds what the sweep explores.
_DEFAULT_TUNE_THREADS = 256


@dataclass(frozen=True)
class StreamingLogSumExpPolicy:
    """Launch shape and eligibility gate of the streaming kernel.

    The launch pair is fixed rather than tuned, and ``eligible`` keeps the
    kernel on the shapes that pair suits.
    """

    threads: int = 128

    cols_per_thread: int = 8

    # Enough rows to fill the device with one block per row.
    min_rows: int = 256

    # Rows long enough that the tiled kernel's staging measurably loses.
    min_cols: int = 16384

    # Seed of the running max: below every finite fp16/bf16 value, but
    # finite, so an all--inf row keeps a zero sum and folds to
    # max_floor + log(0) = -inf, matching torch.
    max_floor: float = -3.4e38

    @property
    def max_ceil(self) -> float:
        """Clamp for exponent arguments: above every finite fp16/bf16 value.

        Subtracting ``min(max, max_ceil)`` instead of the true max keeps
        (+inf) - (+inf) = NaN out of exp2: a +inf element contributes
        exp2(+inf) = +inf and its row folds to +inf, matching torch. A NaN
        element propagates through exp2, and a finite max is never clamped.
        """
        return -self.max_floor

    @property
    def chunk(self) -> int:
        return self.threads * self.cols_per_thread

    def eligible(self, M: int, N: int, dtype: torch.dtype) -> bool:
        return (
            dtype in (torch.float16, torch.bfloat16)
            and self.min_rows <= M
            and self.min_cols <= N
            and N % self.chunk == 0
        )


_STREAM_POLICY = StreamingLogSumExpPolicy()

__all__ = ["LogSumExpKernel"]


@functools.lru_cache(maxsize=64)
def _logsumexp_split_fold_kernel(M: int, N: int, dtype: str, seg_n: int):
    """Fold per-segment ``(max, sum)`` into one logsumexp per row.

    The fold is over a few hundred fp32 pairs, so one warp per row is enough;
    unlike softmax there is no second pass over the input. An all--inf row
    reads ``-inf + log(0)``, which is torch's ``-inf``.
    """
    num_segs = ceildiv_int(N, seg_n)
    fold = make_split_fold(num_segs)

    @tilelang.jit(out_idx=[2])
    def _func():
        @T.prim_func
        def main(
            seg_max: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            seg_sum: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(M, threads=WARP_LANES) as pid_m:
                tx = T.get_thread_binding()
                row_max = T.alloc_local((1,), "float32")
                row_sum = T.alloc_local((1,), "float32")
                held = T.alloc_local((1,), "float32")

                if tx == 0:
                    fold(seg_max, seg_sum, pid_m * num_segs, row_max, row_sum, held)
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

                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        x_f32[i, j] = T.exp(x_f32[i, j] - row_max[i])
                T.reduce_sum(x_f32, row_sum, dim=1)

                out_local = T.alloc_fragment((block_m,), dtype)
                for i in T.Parallel(block_m):
                    out_local[i] = row_max[i] + T.log(row_sum[i])

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
                tile_max = T.alloc_fragment((block_m,), "float32")
                tile_sum = T.alloc_fragment((block_m,), "float32")

                T.fill(row_max, -T.infinity("float32"))
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

                    for i in T.serial(block_m):
                        for j in T.Parallel(tile_n):
                            tile_f32[i, j] = T.exp(tile_f32[i, j] - row_max[i])
                    T.reduce_sum(tile_f32, tile_sum, dim=1)

                    for i in T.Parallel(block_m):
                        row_sum[i] = row_sum[i] * T.exp(prev_max[i] - row_max[i]) + tile_sum[i]

                # logsumexp = max + log(sum)
                out_local = T.alloc_fragment((block_m,), dtype)
                for i in T.Parallel(block_m):
                    out_local[i] = row_max[i] + T.log(row_sum[i])

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
    floor = _STREAM_POLICY.max_floor
    ceil = _STREAM_POLICY.max_ceil

    @tilelang.jit(out_idx=[1])
    def _func():
        @T.macro
        def merge_pair(dst_m, dst_s, src_m, src_s, m_new, m_safe):
            # Exponents subtract the ceiling-clamped max, never the true one,
            # so (+inf) - (+inf) = NaN cannot form; see _STREAM_POLICY.max_ceil.
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


# Dispatch


@functools.lru_cache(maxsize=64)
def _logsumexp_kernel(M: int, N: int, dtype: str, tile_n: int = 0):
    """Build the appropriate logsumexp kernel."""
    if tile_n == 0:
        return _logsumexp_kernel_single(M, N, dtype)
    return _logsumexp_kernel_tiled(M, N, dtype, tile_n)


class LogSumExpKernel(RowTiledAutotuneMixin, Kernel):
    """LogSumExp forward kernel.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Implements a 2-pass online algorithm.

    For large N that does not fit in shared memory, tiles over N using
    the online softmax recurrence (running max + rescaled sum).

    Boundary handling for non-aligned N is performed inside the kernel
    via masked loads and ``-inf`` fills, so no host-side ``F.pad`` is
    needed.

    ``forward`` takes the tensor the op declares and reduces *reduce_axes* of it; moving
    those axes to the end, flattening to rows and shaping the result back are this
    kernel's business, so both sides of the op/backend boundary speak the declared shape.

    Args:
        M: Rows the reduction leaves.
        N: Elements each row reduces.
        op_kind: Must be "logsumexp" (kept for API consistency with SoftmaxKernel).
        dtype: Data type (float32, float16, or bfloat16).
        reduce_axes: Non-negative axis indices, ascending, that the reduction runs over.
        keepdim: Whether a reduced axis stays as a length-1 axis.
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
        device_index: CUDA device index for shared memory budget query.
            When ``None``, ``torch.cuda.current_device()`` is used.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    _MAX_TILE_N_CANDIDATES = 3

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        reduce_axes: "tuple[int, ...]",
        keepdim: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: int | None = None,
    ):
        super().__init__(device_index=device_index)
        if op_kind != "logsumexp":
            raise ValueError(f"Unsupported op_kind '{op_kind}'. Expected 'logsumexp'.")
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.reduce_axes = tuple(reduce_axes)
        self.keepdim = keepdim
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._split_target = split_target_blocks(device_index)
        self._elem_bytes = torch_dtype_nbytes(dtype)
        self._smem_budget = device_smem_budget(device_index)
        self._planner = BlockConfigPlanner(
            self.N_padded,
            self._elem_bytes,
            self._smem_budget,
        )

        # Build self.kernel BEFORE init_config: when tune=True, init_config
        # delegates to autotune() which requires self.kernel to exist.
        #
        # tile_n is baked into the kernel at build time, so pre-compute it from
        # default_config; autotune() rebuilds once per candidate width.
        self._streaming = _STREAM_POLICY.eligible(M, N, dtype)
        self._tile_n = self.default_config["tile_n"]
        if self._streaming:
            self.kernel = _logsumexp_kernel_streaming(
                self.M,
                self.N,
                self.dtype_str,
                _STREAM_POLICY.threads,
                _STREAM_POLICY.cols_per_thread,
            )
        else:
            self.kernel = _logsumexp_kernel(
                self.M,
                self.N,
                self.dtype_str,
                self._tile_n,
            )

        self.init_config(config, tune)

        # When tune=True, autotune() already set self._tile_n and
        # self.config["tile_n"], and rebuilt the kernel.  Only apply
        # the post-init tile_n fixup for user-provided configs.
        if not tune and not self._streaming:
            # If the caller supplied an explicit tile_n (e.g. from a
            # previous autotuner result), honour it.  Only fall back to
            # the heuristic when tile_n was not provided.
            caller_tile_n = config.get("tile_n") if config is not None else None
            if caller_tile_n == 0:
                caller_tile_n = None
            if caller_tile_n is not None:
                reason = self._planner.reject_tile_n(
                    self.config["block_m"],
                    caller_tile_n,
                    self.config.get("threads", _DEFAULT_TUNE_THREADS),
                )
                if reason:
                    raise ValueError(reason)
                target_tile_n = caller_tile_n
            else:
                target_tile_n = self._tile_n_for_block_m(self.config["block_m"])
            if target_tile_n != self._tile_n:
                self._tile_n = target_tile_n
                self.kernel = _logsumexp_kernel(
                    self.M,
                    self.N,
                    self.dtype_str,
                    self._tile_n,
                )
            self.config["tile_n"] = self._tile_n

        # A config from before the split choice was recorded falls back to
        # the gate; a round-tripped tuned config keeps its recorded choice.
        self.config.setdefault(
            "split",
            bool(split_seg_n(self.M, self.N, self.config["block_m"], self._split_target)),
        )

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget.

        For the single-tile path (tile_n == 0), prefer the largest
        block_m that fits in shared memory.

        For the tiled path, prefer the block_m that minimises the
        number of N-tiles (maximises tile_n) to reduce global memory
        passes.  Among configs with equal tile count, prefer smaller
        block_m for better occupancy.
        """
        best_bm = 1
        best_tile_n = self._tile_n_for_block_m(1)

        for bm in [2, 4, 8, 16]:
            if not self._planner.layout_ok(bm, self.N_padded, _DEFAULT_TUNE_THREADS):
                continue
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn == 0:
                # Single-tile is always better: prefer larger block_m
                best_bm = bm
                best_tile_n = tn
            elif best_tile_n == 0:
                pass
            else:
                best_num = (self.N_padded + best_tile_n - 1) // best_tile_n
                curr_num = (self.N_padded + tn - 1) // tn
                if curr_num < best_num:
                    best_bm = bm
                    best_tile_n = tn

        return {
            "block_m": best_bm,
            "threads": _DEFAULT_TUNE_THREADS,
            "tile_n": best_tile_n,
            "split": bool(split_seg_n(self.M, self.N, best_bm, self._split_target)),
        }

    def _build_row_kernel(self, tile_n: int):
        return _logsumexp_kernel(self.M, self.N, self.dtype_str, tile_n)

    def _row_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._reduce_rows(x)

    def _sweep_applies(self) -> bool:
        """The streaming kernel bakes its launch shape in; nothing to vary."""
        return not self._streaming

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce *reduce_axes* of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device. Boundary
                handling for non-aligned ``N`` is performed inside the GPU kernel
                (masked loads + ``-inf`` fill), so no host-side ``F.pad`` is needed.

        Returns:
            The reduced tensor.

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        k, j = edge_axis_split(x.ndim, self.reduce_axes)
        view = edge_split_view(in_shape, k, j, _DEFAULT_TUNE_THREADS) if k else None
        if view is not None:
            y = self._reduce_edge_axes(x, view)
        else:
            y = self._reduce_rows(rows_for_axes(x, self.reduce_axes))
        return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)

    def _reduce_edge_axes(self, x: torch.Tensor, view: "tuple[int, int, int]") -> torch.Tensor:
        """Reduce edge axes in the tensor's own layout.

        Each kept row is ``outer`` contiguous runs of ``inner`` elements, so
        the split pair reads them directly -- per-run ``(max, sum)`` partials,
        then the per-row fold -- skipping the permute ``rows_for_axes`` pays.

        A layout dispatch decided from the shape alone, like the vector-norm
        edge path: the alternative pays the permute and then the same
        reduction, so there is no trade for the tuner to referee.
        ``config["split"]`` governs only the long-row split in
        ``_reduce_rows``.
        """
        outer, kept, inner = view
        seg_max, seg_sum = edge_split_partials_kernel(
            outer, kept, inner, self.dtype_str, _DEFAULT_TUNE_THREADS
        )()(x.reshape(view))
        return _logsumexp_split_fold_kernel(kept, outer * inner, self.dtype_str, inner)()(
            seg_max, seg_sum
        )

    def _reduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the trailing axis of an ``(M, N)`` buffer.

        Long rows on a filled grid stream straight to registers; a handful of
        long rows goes to the split pair: softmax's per-segment statistics,
        then a per-row fold.
        """
        if self._streaming:
            return self.kernel()(x)
        seg_n = (
            split_seg_n(self.M, self.N, self.config["block_m"], self._split_target)
            if self.config["split"]
            else 0
        )
        if seg_n:
            # split_seg_n's fragment cap assumes the default width.
            threads = _DEFAULT_TUNE_THREADS
            seg_max, seg_sum = softmax_split_partials_kernel(
                self.M, self.N, seg_n, self.dtype_str, threads
            )()(x)
            return _logsumexp_split_fold_kernel(self.M, self.N, self.dtype_str, seg_n)()(
                seg_max, seg_sum
            )
        program = _logsumexp_kernel(self.M, self.N, self.dtype_str, self._tile_n)
        return program(self.config["block_m"], self.config["threads"])(x)
