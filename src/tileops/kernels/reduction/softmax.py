"""Softmax / log-softmax forward kernel using TileLang.

Implements a 2-pass online softmax algorithm for two operations:
  - softmax:     y[i,j] = exp(x[i,j] - max_i) / sum_i(exp(x[i,j] - max_i))
  - log_softmax: y[i,j] = x[i,j] - max_i - log(sum_i(exp(x[i,j] - max_i)))

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
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    BlockConfigPlanner,
    RowTiledAutotuneMixin,
    align_up,
    ceildiv_int,
    device_busy_of,
    device_smem_budget,
    restore_same_shape,
    rows_for_axes,
)
from tileops.kernels.reduction._split_softmax import (
    make_split_fold,
    softmax_split_partials_kernel,
    split_seg_n,
    split_target_blocks,
)

# These two kernels bake tile_n in at build time and default to the wider
# thread block; AUTOTUNE_THREADS still bounds what the sweep explores.
_DEFAULT_TUNE_THREADS = 256

__all__ = ["SoftmaxKernel"]


# Single-tile kernel (N fits in shared memory) -- original fast path


@functools.lru_cache(maxsize=64)
def _softmax_kernel_single(M: int, N: int, op_kind: str, dtype: str):
    """Build a single-tile softmax/log_softmax kernel (N fits in smem).

    Accepts an ``(M, N)`` input tensor.  When ``N`` is not a multiple of
    ``DEFAULT_ALIGNMENT``, the kernel uses element-wise ``T.if_then_else``
    loads that substitute ``-inf`` for out-of-bounds columns (kernel-side
    boundary handling).  When ``N`` is already aligned, the fast ``T.copy``
    path is used.

    softmax never reads the row again once it has exponentiated, so it goes global to
    fragment and back with nothing staged in between: measured on H200 at 1024x4096,
    dropping the shared round trip takes it from 1.76 to 2.27 TB/s in fp16 and 2.75 to
    3.34 in fp32. log_softmax needs the row a second time, for ``(x - max) - log(sum)``,
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
            y: T.Tensor[(M, N_padded), dtype],
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
                        y[pid_m * block_m + i, j] = T.cast(x_f32[i, j], dtype)

        return main

    return _func


# Multi-tile kernel (N tiled over shared memory)


@functools.lru_cache(maxsize=64)
def _softmax_kernel_tiled(M: int, N: int, op_kind: str, dtype: str, tile_n: int):
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
    ``compute_tile_n``.
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
                y: T.Tensor[(M, total_cols), dtype],
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
                                p2_shared[i, j] = T.cast(p2_f32[i, j], dtype)
                        T.copy(p2_shared, y[pid_m * block_m, t * tile_n])

            return main

    else:  # log_softmax

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                y: T.Tensor[(M, total_cols), dtype],
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
                                p2_shared[i, j] = T.cast(p2_f32[i, j], dtype)
                        T.copy(p2_shared, y[pid_m * block_m, t * tile_n])

            return main

    return _func


# Dispatch: choose single-tile or multi-tile kernel based on tile_n


@functools.lru_cache(maxsize=64)
def _softmax_kernel(M: int, N: int, op_kind: str, dtype: str, tile_n: int = 0):
    """Build the appropriate softmax kernel.

    If tile_n == 0, the full N fits in shared memory and the single-tile
    kernel is used. Otherwise, the multi-tile kernel is used.
    """
    if tile_n == 0:
        return _softmax_kernel_single(M, N, op_kind, dtype)
    return _softmax_kernel_tiled(M, N, op_kind, dtype, tile_n)


@functools.lru_cache(maxsize=64)
def _softmax_split_finalize_kernel(
    M: int, N: int, op_kind: str, dtype: str, seg_n: int, threads: int
):
    """Fold per-segment ``(max, sum)`` and write one normalized segment per block.

    The fold reads ``num_segs`` fp32 pairs; staged through shared memory
    once, every thread then folds redundantly from there rather than each
    walking global memory.
    """
    num_segs = ceildiv_int(N, seg_n)
    fold = make_split_fold(num_segs)

    @tilelang.jit(out_idx=[3])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            seg_max: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            seg_sum: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(num_segs, M, threads=threads) as (pid_s, pid_m):
                stat_max = T.alloc_shared((num_segs,), "float32")
                stat_sum = T.alloc_shared((num_segs,), "float32")
                row_max = T.alloc_local((1,), "float32")
                row_sum = T.alloc_local((1,), "float32")
                held = T.alloc_local((1,), "float32")

                for s in T.Parallel(num_segs):
                    stat_max[s] = seg_max[pid_m * num_segs + s]
                    stat_sum[s] = seg_sum[pid_m * num_segs + s]
                T.sync_threads()

                fold(stat_max, stat_sum, 0, row_max, row_sum, held)

                for _, j in T.Parallel(1, seg_n):
                    col = pid_s * seg_n + j
                    with T.If(col < N):  # noqa: SIM117
                        with T.Then():
                            if op_kind == "softmax":
                                y[pid_m, col] = T.cast(
                                    T.exp(T.cast(x[pid_m, col], "float32") - row_max[0])
                                    / row_sum[0],
                                    dtype,
                                )
                            else:
                                y[pid_m, col] = T.cast(
                                    T.cast(x[pid_m, col], "float32")
                                    - row_max[0]
                                    - T.log(row_sum[0]),
                                    dtype,
                                )

        return main

    return _func


def _compute_padded_cols(N: int, tile_n: int) -> int:
    """Compute the total column count (may exceed N_padded for tiled path)."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    if tile_n == 0:
        return N_padded
    num_tiles = (N_padded + tile_n - 1) // tile_n
    return num_tiles * tile_n


# Kernel class


def _elem_bytes(dtype: torch.dtype) -> int:
    """Return bytes per element for the given dtype."""
    return torch.tensor([], dtype=dtype).element_size()


class SoftmaxKernel(RowTiledAutotuneMixin, Kernel):
    """Softmax / log-softmax forward kernel.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Implements a 2-pass online softmax algorithm.

    For large N that does not fit in shared memory, tiles over N using
    the online softmax recurrence (running max + rescaled sum).

    Boundary handling for non-aligned N is performed inside the kernel
    via masked loads and ``-inf`` fills, so no host-side ``F.pad`` is
    needed.

    ``forward`` takes the tensor the op declares and normalizes over *norm_axis*; moving
    that axis to the end, flattening to rows and putting the result back are this kernel's
    business, so both sides of the op/backend boundary speak the declared shape.

    Args:
        M: Rows the normalization runs over — the product of every axis but *norm_axis*.
        N: Length of the normalized axis.
        op_kind: One of "softmax", "log_softmax".
        dtype: Data type (float32, float16, or bfloat16).
        norm_axis: Non-negative index of the axis the normalization runs over.
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
        device_index: CUDA device index for shared memory budget query.
            When ``None``, ``torch.cuda.current_device()`` is used.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        norm_axis: int,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: int | None = None,
    ):
        super().__init__(device_index=device_index)
        if op_kind not in ("softmax", "log_softmax"):
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of 'softmax', 'log_softmax'."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.norm_axis = norm_axis
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = _elem_bytes(dtype)
        self._smem_budget = device_smem_budget(device_index)
        self._split_target = split_target_blocks(device_index)
        self._planner = BlockConfigPlanner(
            self.N_padded,
            self._elem_bytes,
            self._smem_budget,
            num_buffers=self._NUM_SHARED_BUFFERS,
        )

        # Build self.kernel BEFORE init_config: when tune=True, init_config
        # delegates to autotune() which requires self.kernel to exist.
        #
        # tile_n is baked into the kernel at build time, so pre-compute it from
        # default_config; autotune() rebuilds once per candidate width.
        self._tile_n = self.default_config["tile_n"]
        self.kernel = _softmax_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self._tile_n,
        )

        self.init_config(config, tune)

        # When tune=True, autotune() already set self._tile_n and
        # self.config["tile_n"], and rebuilt the kernel.  Only apply
        # the post-init tile_n fixup for user-provided configs.
        if not tune:
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
                self.kernel = _softmax_kernel(
                    self.M,
                    self.N,
                    self.op_kind,
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

    # Tiled softmax/log_softmax allocates 2 shared buffers (one per pass)
    # due to TileLang allocator aliasing -- see _softmax_kernel_tiled docstring.
    _NUM_SHARED_BUFFERS = 2
    _MAX_TILE_N_CANDIDATES = 3

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget.

        For the single-tile path (tile_n == 0), prefer the *smallest* block_m.
        Measured on H200 at 1024x4096 fp16, bandwidth falls monotonically as rows are
        added to a block, from 1.76 TB/s at one row to 0.19 at sixteen: each extra row
        hands every thread another ``N_padded / threads`` registers until the fragment
        spills, and the kernel then re-reads its own row through L2 on every pass.

        For the tiled path, prefer the block_m that **minimises the
        number of N-tiles** (i.e. maximises tile_n).  Fewer tiles means
        fewer global memory passes in the 2-pass algorithm, which
        dominates latency on bandwidth-bound workloads.  Among configs
        with equal tile count, prefer *smaller* block_m: the tiled
        kernel is bandwidth-bound, and smaller shared-memory footprint
        per block improves occupancy.
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
            if tn == 0 and not self._planner.frag_fits(bm, self.N_padded, _DEFAULT_TUNE_THREADS):
                continue
            if tn == 0 and best_tile_n == 0:
                # Both single-tile, and block_m=1 is where the loop starts: keep it.
                pass
            elif tn == 0 and best_tile_n != 0:
                # Switching from tiled to single-tile is always better
                best_bm = bm
                best_tile_n = tn
            elif tn != 0 and best_tile_n == 0:
                # Don't give up single-tile for tiled
                pass
            else:
                # Both tiled: prefer strictly fewer tiles only.
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

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune across tile_n candidates by rebuilding the kernel per regime.

        Groups configs by tile_n, benchmarks each group with its own kernel,
        and picks the overall best (block_m, threads, tile_n) config.
        """
        from tilelang.autotuner import autotune as tl_autotune

        default = self.default_config
        split_eligible = bool(split_seg_n(self.M, self.N, default["block_m"], self._split_target))

        configs = self.autotune_configs
        if not configs:
            self.config = default
            return

        # Group configs by tile_n
        by_tile_n: dict[int, list[dict]] = {}
        for cfg in configs:
            tn = cfg["tile_n"]
            by_tile_n.setdefault(tn, []).append(
                {"block_m": cfg["block_m"], "threads": cfg["threads"]}
            )

        best_time = float("inf")
        best_config = None

        for tile_n, group_cfgs in by_tile_n.items():
            kernel = _softmax_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_str,
                tile_n,
            )
            tunable_params = list(self._autotune_initial_kwargs(kernel, group_cfgs[0]).keys())
            autotune_kwargs: dict = dict(
                configs=group_cfgs,
                warmup=warmup,
                rep=rep,
            )
            if tunable_params:
                autotune_kwargs["do_not_specialize"] = tunable_params
            if self.autotune_supply_prog is not None:
                autotune_kwargs["supply_prog"] = self.autotune_supply_prog
            autotuned = tl_autotune(**autotune_kwargs)(kernel)
            tuned = self._call_autotuned_kernel(autotuned, kernel, group_cfgs[0])
            latency = tuned.latency
            if latency < best_time:
                best_time = latency
                best_config = {**tuned.config, "tile_n": tile_n}

        if best_config is not None:
            self.config = best_config
            # Rebuild kernel for the winning tile_n
            winning_tile_n = best_config["tile_n"]
            if winning_tile_n != self._tile_n:
                self._tile_n = winning_tile_n
                self.kernel = _softmax_kernel(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.dtype_str,
                    self._tile_n,
                )

        # The split pair has nothing for the sweep to vary, so it is timed
        # against the sweep's winner on device kernel time: paths launching
        # different kernel counts are judged by GPU work, not host-launch gaps.
        if split_eligible and best_config is not None:
            device = torch.device(
                "cuda",
                self.device_index if self.device_index is not None else torch.cuda.current_device(),
            )
            probe = torch.randn(self.M, self.N, dtype=self.dtype, device=device)
            swept_config = dict(self.config, split=False)
            self.config = swept_config
            swept_ms = device_busy_of(lambda: self._normalize_rows(probe), device)
            self.config = default
            split_ms = device_busy_of(lambda: self._normalize_rows(probe), device)
            if split_ms > swept_ms:
                self.config = swept_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize *x* over *norm_axis*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device. Boundary
                handling for non-aligned ``N`` happens inside the GPU kernel (masked
                loads + ``-inf`` fill), so no host-side ``F.pad`` is needed.

        Returns:
            A tensor shaped like *x*.

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        axes = (self.norm_axis,)
        y = self._normalize_rows(rows_for_axes(x, axes))
        return restore_same_shape(y, in_shape, axes)

    def _normalize_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the trailing axis of an ``(M, N)`` buffer.

        The prim_func writes an alignment-padded row; the surplus columns are trimmed
        here. A handful of long rows goes to the split-row pair instead, which
        writes exact columns.
        """
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
            return _softmax_split_finalize_kernel(
                self.M, self.N, self.op_kind, self.dtype_str, seg_n, threads
            )()(x, seg_max, seg_sum)
        program = _softmax_kernel(self.M, self.N, self.op_kind, self.dtype_str, self._tile_n)
        y = program(self.config["block_m"], self.config["threads"])(x)
        return y[:, : self.N] if y.shape[1] > self.N else y
