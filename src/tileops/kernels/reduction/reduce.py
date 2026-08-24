"""Reduce kernels (sum, mean, amin, amax, prod, std, var, var_mean) using TileLang.

Four kernel families, by what the reduction's shape allows:
  - _simple_reduce_kernel: row-wise single pass for sum/mean/amin/amax.
  - _welford_reduce_kernel: row-wise two-pass Welford for std/var/var_mean.
  - _leading_reduce_kernel: sum/mean/amin/amax down the leading axes, reading the
    tensor in the layout it already has instead of permuting the reduced axes last.
  - _prod_reduce_kernel: one block per row, multiplying in fp32.

The row-wise families accept raw ``(M, N)`` tensors.  Boundary handling for non-aligned
N is performed inside the kernel via masked loads with identity-element fills,
eliminating host-side ``F.pad`` from the forward path.  When ``N`` is already a multiple
of ``DEFAULT_ALIGNMENT``, the fast vectorized ``T.copy`` path is used.

A row goes to a tiled variant when it exceeds any of the three capacities
``BlockConfigPlanner`` tracks -- the vectorizer's column cap, shared memory, or the
register file -- and the tiled variant iterates over N in chunks of ``tile_n`` columns.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    BlockConfigPlanner,
    align_up,
    device_smem_budget,
    restore_reduced,
    rows_for_axes,
    tune_by_forward,
)

__all__ = ["ReduceKernel"]

_WELFORD_KINDS = {"std", "var", "var_mean"}

_WARP_LANES = 32

# Stride-halving shuffle steps that reduce one warp.
_WARP_STAGES = 5

# Tile-width fragments each kind keeps alive, which is what decides whether its tile
# fits in registers.  sum/mean/amax/amin hold the fp32 working copy alone; Welford holds
# it beside the squared deviation.  Kinds absent here take the 1 the planner defaults to,
# and prod holds none -- it has its own kernel.
_FRAG_SLOTS = {
    "std": 2,
    "var": 2,
    "var_mean": 2,
}


# Simple reduce kernel


def _pad_value_for_op(op_kind: str) -> float:
    """Return the identity element for padding columns of the given op."""
    if op_kind == "prod":
        return 1.0
    if op_kind == "amin":
        return float("inf")
    if op_kind == "amax":
        return float("-inf")
    # sum, mean, std, var, var_mean: zero padding
    return 0.0


@functools.lru_cache(maxsize=32)
def _simple_reduce_kernel(M, N, op_kind, dtype):
    """Build a simple reduce kernel for sum/mean/amax/amin.

    Accepts an ``(M, N)`` input tensor.  When ``N`` is not a multiple of
    ``DEFAULT_ALIGNMENT``, the kernel uses element-wise ``T.if_then_else``
    loads that substitute the identity element for out-of-bounds columns
    (kernel-side boundary handling).  When ``N`` is already aligned, the
    fast ``T.copy`` path is used.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    _pad_val = _pad_value_for_op(op_kind)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), dtype)

                if _needs_pad:
                    # Kernel-side boundary handling: element-wise load
                    # with T.if_then_else masking for padding columns
                    # and row-tail safety (M % block_m != 0).
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_pad_val, "float32"),
                            )
                else:
                    # Load via shared memory (fast vectorized path)
                    T.copy(x[pid_m * block_m, 0], shared_buf)

                    # Cast to fp32
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                # Reduce
                if op_kind == "sum":
                    T.reduce_sum(x_f32, acc, dim=1)
                elif op_kind == "mean":
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i] = acc[i] / float(N)
                elif op_kind == "amax":
                    T.reduce_max(x_f32, acc, dim=1)
                elif op_kind == "amin":
                    # Negate, reduce_max, negate back
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = -x_f32[i, j]
                    T.reduce_max(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i] = -acc[i]

                # Cast back to output dtype
                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(acc[i], dtype)

                # Write output
                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n):
    """Tiled simple reduce for N_padded > MAX_SINGLE_TILE_COLS.

    Iterates over N in chunks of ``tile_n`` columns, accumulating
    partial results.  The last tile uses masked loads when
    ``num_tiles * tile_n > N``.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    total_cols = num_tiles * tile_n
    _needs_mask = total_cols > N
    _pad_val = _pad_value_for_op(op_kind)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                tile_acc = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), dtype)

                # Initialize accumulator
                if op_kind in ("sum", "mean"):
                    T.fill(acc, 0.0)
                elif op_kind == "amax":
                    T.fill(acc, -T.infinity("float32"))
                elif op_kind == "amin":
                    T.fill(acc, T.infinity("float32"))

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
                                            T.And(
                                                pid_m * block_m + i < M,
                                                t * tile_n + j < N,
                                            ),
                                            T.cast(
                                                x[pid_m * block_m + i, t * tile_n + j],
                                                "float32",
                                            ),
                                            T.cast(_pad_val, "float32"),
                                        )
                    else:
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                    # Tile-local reduce
                    if op_kind in ("sum", "mean"):
                        T.reduce_sum(tile_f32, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] + tile_acc[i]
                    elif op_kind == "amax":
                        T.reduce_max(tile_f32, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.max(acc[i], tile_acc[i])
                    elif op_kind == "amin":
                        # Negate, reduce_max, negate back
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                tile_f32[i, j] = -tile_f32[i, j]
                        T.reduce_max(tile_f32, tile_acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = T.min(acc[i], -tile_acc[i])

                # Finalize
                if op_kind == "mean":
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i] / float(N), dtype)
                else:
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i], dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


#: Output columns one thread accumulates in the leading-axis kernel.  Each is an fp32
#: register held for the whole walk, and 8 covers a 128-bit access per thread for a
#: 2-byte dtype.
_LEADING_COLS_PER_THREAD: int = 8

#: Threads a leading-axis reduction runs. Its own, not the row path's: the column block
#: is ``threads * _LEADING_COLS_PER_THREAD`` wide, so a wider block means fewer column
#: blocks and more row splits to make up the grid, and more splits means more partials for
#: the second pass to read. Bandwidth measured on H200 over
#: {2048x4096 bf16, 4096x4096 fp16, 512x8192 fp16}: 1.47 / 2.20 / 1.36 TB/s at 64
#: threads against 0.90 / 1.41 / 0.75 at 256.
_LEADING_THREADS: int = 64

#: Blocks the leading-axis reduction aims to launch.  A reduction down the leading axes
#: has only as many output columns as the kept axes, which on its own leaves most of the
#: device idle -- 2048x4096 fp16 gives four blocks -- so the reduced axis is split until
#: there are about this many.  Comfortably above the 132 SMs of an H200 so the tail
#: block is not the whole tail.
_LEADING_TARGET_BLOCKS: int = 512

#: Kinds the leading-axis kernel implements.  prod carries a second accumulator for the
#: sign, and Welford's kinds two passes; neither is expressible as one running value.
_LEADING_AXIS_KINDS = frozenset({"sum", "mean", "amax", "amin"})


def leading_axis_split(shape: "tuple[int, ...]", axes: "tuple[int, ...]") -> "tuple[int, int]":
    """Split *shape* into ``(reduced, kept)`` element counts for a leading-axis reduce.

    Only valid when *axes* is ``(0, 1, ... k-1)``: the reduced axes are then the
    outermost ones, so a contiguous tensor reshapes to ``(reduced, kept)`` for free and
    the reduction runs down the rows of that view.
    """
    k = len(axes)
    reduced = 1
    for d in shape[:k]:
        reduced *= d
    kept = 1
    for d in shape[k:]:
        kept *= d
    return reduced, kept


def reduces_leading_axes(ndim: int, axes: "tuple[int, ...]") -> bool:
    """Whether *axes* is a non-empty proper prefix of the axes of an *ndim* tensor."""
    return 0 < len(axes) < ndim and tuple(axes) == tuple(range(len(axes)))


def leading_row_splits(reduced: int, kept: int, threads: int) -> int:
    """How many ways to split the reduced axis so the grid fills the device."""
    block_b = threads * _LEADING_COLS_PER_THREAD
    column_blocks = -(-kept // block_b)
    return max(1, min(reduced, -(-_LEADING_TARGET_BLOCKS // column_blocks)))


@functools.lru_cache(maxsize=32)
def _leading_reduce_kernel(
    A: int,
    B: int,
    op_kind: str,
    in_dtype: str,
    out_dtype: str,
    threads: int,
    splits: int,
    divisor: float,
):
    """Build a reduce down the leading axes of an ``(A, B)`` view.

    One accumulator per output column, walked down the rows the block owns. Adjacent
    threads take adjacent columns, so every row of the walk is one coalesced pass and
    the tensor is read once in the layout it already has. The alternative -- permuting
    the reduced axes to the end and making that contiguous -- reads and writes the whole
    tensor before the reduction has read anything, which is three passes where this is
    one.

    The grid is ``(column blocks, splits)``: a leading-axis reduction has only as many
    output columns as the axes it keeps, and one block per column tile would leave an
    H200 running four of them.

    Args:
        A: Elements the reduction consumes per output column.
        B: Output columns.
        op_kind: One of ``_LEADING_AXIS_KINDS``.
        in_dtype: TileLang dtype string of the input.
        out_dtype: TileLang dtype string of the output. A split pass writes fp32
            partials whatever it read; the pass that finishes writes the declared dtype.
        threads: Threads per block.
        splits: Row slices, each its own block row. Above 1 the output is one row of
            partials per slice, for a second call with ``splits=1`` to finish.
        divisor: What the accumulator is divided by before the output cast, or 0 for
            none. Mean's divisor is the row count of the whole reduction, which a
            second pass over partials can no longer see.
    """
    block_b = threads * _LEADING_COLS_PER_THREAD
    rows_per_split = -(-A // splits)
    exact = B % block_b == 0

    @tilelang.jit(out_idx=[1])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(A, B), in_dtype],
            out: T.Tensor[(splits * B,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(B, block_b), splits, threads=threads) as (pid_b, pid_a):
                acc = T.alloc_fragment((block_b,), "float32")
                out_local = T.alloc_fragment((block_b,), out_dtype)

                if op_kind == "amax":
                    T.fill(acc, -T.infinity("float32"))
                elif op_kind == "amin":
                    T.fill(acc, T.infinity("float32"))
                else:
                    T.fill(acc, 0.0)

                for step in T.serial(rows_per_split):
                    row = pid_a * rows_per_split + step
                    for j in T.Parallel(block_b):
                        col = pid_b * block_b + j
                        in_range = row < A if exact else T.And(row < A, col < B)
                        val = T.if_then_else(
                            in_range,
                            T.cast(x[row, col], "float32"),
                            T.cast(_pad_value_for_op(op_kind), "float32"),
                        )
                        if op_kind == "amax":
                            acc[j] = T.max(acc[j], val)
                        elif op_kind == "amin":
                            acc[j] = T.min(acc[j], val)
                        else:
                            acc[j] = acc[j] + val

                for j in T.Parallel(block_b):
                    if divisor:
                        out_local[j] = T.cast(acc[j] / divisor, out_dtype)
                    else:
                        out_local[j] = T.cast(acc[j], out_dtype)

                if exact:
                    T.copy(out_local, out[pid_a * B + pid_b * block_b])
                else:
                    for j in T.Parallel(block_b):
                        # TileLang requires T.If/T.Then as nested context managers.
                        with T.If(pid_b * block_b + j < B):  # noqa: SIM117
                            with T.Then():
                                out[pid_a * B + pid_b * block_b + j] = out_local[j]

        return main

    return _func


#: Columns one thread multiplies before the warp combines. Bandwidth measured on H200 at
#: 2048x4096 fp16: 2.18 TB/s at 8, 2.21 at 4, 1.57 at 16.
_PROD_COLS_PER_THREAD: int = 8


@functools.lru_cache(maxsize=32)
def _prod_reduce_kernel(M: int, N: int, dtype: str, threads: int):
    """Build a product reduce: one block per row, multiplying in fp32.

    Each thread multiplies a contiguous run of the row into one register, the warps
    combine those by shuffle, and the block combines the warp products in shared
    memory. Nothing of row width is held, so the row length does not bound the kernel.

    A zero, a sign and an overflow to inf each come out of the arithmetic rather than
    being reconstructed after it, so the result matches an fp32 ``torch.prod`` exactly.
    Taking a logarithm per element instead needs an epsilon to survive ``log(0)`` and
    costs a transcendental: measured on H200 at 2048x4096 fp16, 2.18 TB/s against 1.17.

    Args:
        M: Rows to reduce.
        N: Elements each row reduces.
        dtype: TileLang dtype string.
        threads: Threads per row.
    """
    chunk = threads * _PROD_COLS_PER_THREAD
    tiles = -(-N // chunk)
    exact = tiles * chunk == N
    num_warps = threads // _WARP_LANES

    @tilelang.jit(out_idx=[1])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                running = T.alloc_local((1,), "float32")
                warp_prod = T.alloc_shared((num_warps,), "float32")

                running[0] = T.cast(1.0, "float32")
                for t in T.serial(tiles):
                    for c in T.serial(_PROD_COLS_PER_THREAD):
                        col = (t * threads + tx) * _PROD_COLS_PER_THREAD + c
                        if exact:
                            running[0] = running[0] * T.cast(x[row, col], "float32")
                        else:
                            running[0] = running[0] * T.if_then_else(
                                col < N, T.cast(x[row, col], "float32"), T.cast(1.0, "float32")
                            )

                for stage in T.serial(_WARP_STAGES):
                    running[0] = running[0] * T.shfl_xor(
                        running[0], T.int32(_WARP_LANES // 2) >> stage, width=_WARP_LANES
                    )
                if tx % _WARP_LANES == 0:
                    warp_prod[tx // _WARP_LANES] = running[0]
                T.sync_threads()
                if tx == 0:
                    for w in T.serial(1, num_warps):
                        warp_prod[0] = warp_prod[0] * warp_prod[w]
                    out[row] = T.cast(warp_prod[0], dtype)

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _welford_reduce_kernel(M, N, op_kind, correction, dtype):
    """Build a Welford-based reduce kernel for std/var/var_mean.

    Accepts an ``(M, N)`` input tensor.  Padding columns are filled with
    ``0.0`` via masked loads when ``N`` is not aligned.  The padding
    correction (subtracting ``pad_count * mean^2`` from the variance sum)
    is applied analytically, so the result is exact regardless of padding.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N

    out_idx = [1, 2] if op_kind == "var_mean" else [1]

    @tilelang.jit(out_idx=out_idx)
    def _func(block_m, threads):
        if op_kind == "var_mean":

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out_var: T.Tensor[(M,), dtype],
                out_mean: T.Tensor[(M,), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    sq_diff = T.alloc_fragment((block_m, N_padded), "float32")
                    var_sum = T.alloc_fragment((block_m,), "float32")
                    out_v = T.alloc_fragment((block_m,), dtype)
                    out_m = T.alloc_fragment((block_m,), dtype)

                    if _needs_pad:
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = T.if_then_else(
                                    T.And(pid_m * block_m + i < M, j < N),
                                    T.cast(x[pid_m * block_m + i, j], "float32"),
                                    T.cast(0.0, "float32"),
                                )
                    else:
                        T.copy(x[pid_m * block_m, 0], shared_buf)

                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                    # Mean
                    T.reduce_sum(x_f32, row_sum, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = row_sum[i] / float(N)

                    # Variance: sum((x - mean)^2) / (N - correction)
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            dev = x_f32[i, j] - mean_val[i]
                            sq_diff[i, j] = dev * dev

                    T.reduce_sum(sq_diff, var_sum, dim=1)

                    # Correct for padding: padded elements contribute mean^2 each
                    pad_count = N_padded - N
                    for i in T.Parallel(block_m):
                        corrected_sum = var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                        variance = corrected_sum / float(N - correction)
                        out_v[i] = T.cast(variance, dtype)
                        out_m[i] = T.cast(mean_val[i], dtype)

                    T.copy(out_v, out_var[pid_m * block_m])
                    T.copy(out_m, out_mean[pid_m * block_m])

        else:
            # std or var (single output)
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out: T.Tensor[(M,), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    sq_diff = T.alloc_fragment((block_m, N_padded), "float32")
                    var_sum = T.alloc_fragment((block_m,), "float32")
                    out_local = T.alloc_fragment((block_m,), dtype)

                    if _needs_pad:
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = T.if_then_else(
                                    T.And(pid_m * block_m + i < M, j < N),
                                    T.cast(x[pid_m * block_m + i, j], "float32"),
                                    T.cast(0.0, "float32"),
                                )
                    else:
                        T.copy(x[pid_m * block_m, 0], shared_buf)

                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                    # Mean
                    T.reduce_sum(x_f32, row_sum, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = row_sum[i] / float(N)

                    # Variance
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            dev = x_f32[i, j] - mean_val[i]
                            sq_diff[i, j] = dev * dev

                    T.reduce_sum(sq_diff, var_sum, dim=1)

                    pad_count = N_padded - N
                    if op_kind == "var":
                        for i in T.Parallel(block_m):
                            corrected_sum = (
                                var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            )
                            out_local[i] = T.cast(corrected_sum / float(N - correction), dtype)
                    else:  # std
                        for i in T.Parallel(block_m):
                            corrected_sum = (
                                var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            )
                            out_local[i] = T.cast(
                                T.sqrt(corrected_sum / float(N - correction)), dtype
                            )

                    T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _welford_reduce_kernel_tiled(M, N, op_kind, correction, dtype, tile_n):
    """Tiled Welford reduce for N_padded > MAX_SINGLE_TILE_COLS.

    Two-pass approach over N tiles:
      Pass 1: accumulate row sum for mean computation.
      Pass 2: accumulate sum of squared deviations from the mean.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    total_cols = num_tiles * tile_n
    _needs_mask = total_cols > N

    out_idx = [1, 2] if op_kind == "var_mean" else [1]

    @tilelang.jit(out_idx=out_idx)
    def _func(block_m, threads):
        if op_kind == "var_mean":

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out_var: T.Tensor[(M,), dtype],
                out_mean: T.Tensor[(M,), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                    tile_sum = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    sq_diff = T.alloc_fragment((block_m, tile_n), "float32")
                    tile_sq = T.alloc_fragment((block_m,), "float32")
                    var_sum = T.alloc_fragment((block_m,), "float32")
                    out_v = T.alloc_fragment((block_m,), dtype)
                    out_m = T.alloc_fragment((block_m,), dtype)

                    T.fill(row_sum, 0.0)

                    # Pass 1: compute row sums for mean
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
                                                T.And(
                                                    pid_m * block_m + i < M,
                                                    t * tile_n + j < N,
                                                ),
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                ),
                                                0.0,
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.reduce_sum(tile_f32, tile_sum, dim=1)
                        for i in T.Parallel(block_m):
                            row_sum[i] = row_sum[i] + tile_sum[i]

                    for i in T.Parallel(block_m):
                        mean_val[i] = row_sum[i] / float(N)

                    # Pass 2: dedicated buffers to avoid TileLang aliasing
                    p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                    p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                    T.fill(var_sum, 0.0)

                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")
                                with T.Else():
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            p2_f32[i, j] = T.if_then_else(
                                                T.And(
                                                    pid_m * block_m + i < M,
                                                    t * tile_n + j < N,
                                                ),
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                ),
                                                0.0,
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")

                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                sq_diff[i, j] = (p2_f32[i, j] - mean_val[i]) * (
                                    p2_f32[i, j] - mean_val[i]
                                )
                        T.reduce_sum(sq_diff, tile_sq, dim=1)
                        for i in T.Parallel(block_m):
                            var_sum[i] = var_sum[i] + tile_sq[i]

                    # Correct for padding: out-of-bound elements were filled
                    # with 0.0, so each contributes mean^2 to the sq_diff sum.
                    pad_count = total_cols - N
                    for i in T.Parallel(block_m):
                        corrected = var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                        out_v[i] = T.cast(corrected / float(N - correction), dtype)
                        out_m[i] = T.cast(mean_val[i], dtype)

                    T.copy(out_v, out_var[pid_m * block_m])
                    T.copy(out_m, out_mean[pid_m * block_m])

        else:
            # std or var (single output)
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out: T.Tensor[(M,), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                    tile_sum = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    sq_diff = T.alloc_fragment((block_m, tile_n), "float32")
                    tile_sq = T.alloc_fragment((block_m,), "float32")
                    var_sum = T.alloc_fragment((block_m,), "float32")
                    out_local = T.alloc_fragment((block_m,), dtype)

                    T.fill(row_sum, 0.0)

                    # Pass 1: compute row sums for mean
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
                                                T.And(
                                                    pid_m * block_m + i < M,
                                                    t * tile_n + j < N,
                                                ),
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                ),
                                                0.0,
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.reduce_sum(tile_f32, tile_sum, dim=1)
                        for i in T.Parallel(block_m):
                            row_sum[i] = row_sum[i] + tile_sum[i]

                    for i in T.Parallel(block_m):
                        mean_val[i] = row_sum[i] / float(N)

                    # Pass 2: dedicated buffers
                    p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                    p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                    T.fill(var_sum, 0.0)

                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")
                                with T.Else():
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            p2_f32[i, j] = T.if_then_else(
                                                T.And(
                                                    pid_m * block_m + i < M,
                                                    t * tile_n + j < N,
                                                ),
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                ),
                                                0.0,
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")

                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                sq_diff[i, j] = (p2_f32[i, j] - mean_val[i]) * (
                                    p2_f32[i, j] - mean_val[i]
                                )
                        T.reduce_sum(sq_diff, tile_sq, dim=1)
                        for i in T.Parallel(block_m):
                            var_sum[i] = var_sum[i] + tile_sq[i]

                    pad_count = total_cols - N
                    if op_kind == "var":
                        for i in T.Parallel(block_m):
                            corrected = var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            out_local[i] = T.cast(corrected / float(N - correction), dtype)
                    else:  # std
                        for i in T.Parallel(block_m):
                            corrected = var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            out_local[i] = T.cast(
                                T.sqrt(corrected / float(N - correction)),
                                dtype,
                            )

                    T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


# ReduceKernel class


class ReduceKernel(Kernel):
    """Unified reduce kernel supporting sum/mean/amin/amax/prod/std/var/var_mean.

    Supports SM80+ architectures. Uses 256-element alignment for shared memory
    copies. Dispatches to simple or Welford kernel based on op_kind.

    When ``N_padded`` exceeds ``MAX_SINGLE_TILE_COLS``, tiled kernel variants
    are used that iterate over N in chunks of ``tile_n`` columns, avoiding the
    TileLang vectorizer limit at the 32768-column boundary.

    Boundary handling for non-aligned N is performed inside the kernel via
    masked loads with identity-element fills, so no host-side ``F.pad`` is
    needed.

    ``forward`` takes the tensor the op declares and reduces *reduce_axes* of it: moving
    those axes to the end, flattening to ``(M, N)`` and shaping the result back are this
    kernel's business, so both sides of the op/backend boundary speak the declared shape.

    Args:
        M: Rows the reduction leaves — the product of the axes it keeps.
        N: Elements each row reduces — the product of *reduce_axes*.
        op_kind: One of sum, mean, amin, amax, prod, std, var, var_mean.
        dtype: Element type of the input.
        reduce_axes: Non-negative axis indices, ascending, that the reduction runs over.
        keepdim: Whether a reduced axis stays as a length-1 axis.
        correction: Bessel's correction, for the Welford kinds.
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
        device_index: CUDA device the input lives on, for the shared-memory budget.
            ``None`` reads the current device.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        reduce_axes: "tuple[int, ...]",
        keepdim: bool = False,
        correction: int = 1,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.reduce_axes = tuple(reduce_axes)
        self.keepdim = keepdim
        self.correction = correction
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._is_welford = op_kind in _WELFORD_KINDS
        self._is_prod = op_kind == "prod"
        # Whether the leading-axis kernel could serve this reduction. The axes are known
        # now; whether they are a proper prefix depends on the input's rank, so forward
        # settles it.
        self._leading_axis_kind = op_kind in _LEADING_AXIS_KINDS and self.reduce_axes == tuple(
            range(len(self.reduce_axes))
        )
        self._elem_bytes = torch.tensor([], dtype=dtype).element_size()
        self._smem_budget = device_smem_budget(device_index)
        self._planner = BlockConfigPlanner(
            self.N_padded,
            self._elem_bytes,
            self._smem_budget,
            num_buffers=2 if self._is_welford else 1,
            frag_slots=_FRAG_SLOTS.get(op_kind, 1),
        )
        self._needs_tiling = self._planner.needs_tiling

        if self._is_prod:
            self.kernel = _prod_reduce_kernel(self.M, self.N, self.dtype_str, DEFAULT_THREADS)
        elif not self._needs_tiling:
            if self._is_welford:
                self.kernel = _welford_reduce_kernel(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.correction,
                    self.dtype_str,
                )
            else:
                self.kernel = _simple_reduce_kernel(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.dtype_str,
                )
        # For tiled path, kernel is built lazily using tile_n from config.
        # Tiled kernels use wrapped dispatch functions (not a single self.kernel),
        # so standard autotune via self.kernel is not applicable -- see autotune().
        self.init_config(config, tune)

        # After init_config, ensure tile_n is consistent with the chosen block_m.
        # A caller-provided config may have block_m without tile_n.
        if self._needs_tiling and not tune:
            bm = self.config.get("block_m", 1)
            threads = self.config.get("threads", DEFAULT_THREADS)
            if "tile_n" not in self.config or self.config["tile_n"] == 0:
                self.config["tile_n"] = self._planner.tile_n_for(bm, threads)
            reason = self._planner.reject_tile_n(bm, self.config["tile_n"], threads)
            if reason:
                raise ValueError(reason)

    @property
    def default_config(self) -> dict:
        return self._planner.default_config()

    @property
    def autotune_configs(self) -> list[dict]:
        return self._planner.autotune_configs()

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune the reduce kernel by benchmarking candidate configs."""
        if not self._needs_tiling:
            return super().autotune(warmup=warmup, rep=rep)
        x = torch.randn(self.M, self.N, dtype=self.dtype, device=torch.cuda.current_device())
        tune_by_forward(self, x, warmup=warmup, rep=rep, forward=self._reduce_rows)

    def forward(self, x: torch.Tensor) -> object:
        """Reduce *reduce_axes* of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device.

        Returns:
            The reduced tensor, or ``(var, mean)`` for ``op_kind="var_mean"``.

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        if self._leading_axis_kind and reduces_leading_axes(x.ndim, self.reduce_axes):
            columns = self._reduce_leading_axes(x)
            return restore_reduced(columns, in_shape, self.reduce_axes, self.keepdim)
        rows = rows_for_axes(x, self.reduce_axes)
        result = self._reduce_rows(rows)
        if self.op_kind == "var_mean":
            var, mean = result
            return (
                restore_reduced(var, in_shape, self.reduce_axes, self.keepdim),
                restore_reduced(mean, in_shape, self.reduce_axes, self.keepdim),
            )
        return restore_reduced(result, in_shape, self.reduce_axes, self.keepdim)

    def _reduce_leading_axes(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the leading axes of *x* down to one value per kept column.

        Splitting the reduced axis is what fills the grid, and each slice leaves an
        fp32 partial row; a second call over those rows finishes the op. The partials
        are a few thousand values against the millions the first pass reads, so the
        second call costs about nothing.
        """
        reduced, kept = leading_axis_split(tuple(x.shape), self.reduce_axes)
        flat = x.reshape(reduced, kept)
        splits = leading_row_splits(reduced, kept, _LEADING_THREADS)
        divisor = float(reduced) if self.op_kind == "mean" else 0.0
        if splits == 1:
            single = _leading_reduce_kernel(
                reduced,
                kept,
                self.op_kind,
                self.dtype_str,
                self.dtype_str,
                _LEADING_THREADS,
                1,
                divisor,
            )
            return single()(flat)
        partials = _leading_reduce_kernel(
            reduced,
            kept,
            self.op_kind,
            self.dtype_str,
            "float32",
            _LEADING_THREADS,
            splits,
            0.0,
        )()(flat)
        # Partials are summed whatever the kind was averaging, and the divisor is the
        # whole reduction's row count rather than this pass's.
        finish = _leading_reduce_kernel(
            splits,
            kept,
            "sum" if self.op_kind == "mean" else self.op_kind,
            "float32",
            self.dtype_str,
            _LEADING_THREADS,
            1,
            divisor,
        )
        return finish()(partials.reshape(splits, kept))

    def _reduce_rows(self, x: torch.Tensor) -> object:
        """Reduce the trailing axis of an ``(M, N)`` buffer."""
        if self._is_prod:
            return _prod_reduce_kernel(self.M, self.N, self.dtype_str, DEFAULT_THREADS)()(x)
        block_m, threads = self.config["block_m"], self.config["threads"]
        if self._is_welford:
            if self._needs_tiling:
                program = _welford_reduce_kernel_tiled(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.correction,
                    self.dtype_str,
                    self.config["tile_n"],
                )
            else:
                program = _welford_reduce_kernel(
                    self.M, self.N, self.op_kind, self.correction, self.dtype_str
                )
            results = program(block_m, threads)(x)
            if self.op_kind == "var_mean":
                return results[0], results[1]
            return results
        if self._needs_tiling:
            program = _simple_reduce_kernel_tiled(
                self.M, self.N, self.op_kind, self.dtype_str, self.config["tile_n"]
            )
        else:
            program = _simple_reduce_kernel(self.M, self.N, self.op_kind, self.dtype_str)
        return program(block_m, threads)(x)
