"""Reduce kernels (sum, mean, amin, amax, prod, std, var, var_mean)."""

import functools
from dataclasses import dataclass
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
    ceildiv_int,
    device_smem_budget,
    restore_reduced,
    rows_for_axes,
    torch_dtype_nbytes,
    tune_by_forward,
)

__all__ = ["ReduceKernel"]

_WELFORD_KINDS = {"std", "var", "var_mean"}

_WARP_LANES = 32

# Stride-halving shuffle steps that reduce one warp.
_WARP_STAGES = 5

_FRAG_SLOTS = {
    "std": 2,
    "var": 2,
    "var_mean": 2,
}


@dataclass(frozen=True)
class LeadingAxisReducePolicy:
    """Launch heuristics for reductions down contiguous leading axes."""

    cols_per_thread: int = 8

    threads: int = 64

    target_blocks: int = 512


@dataclass(frozen=True)
class ProductReducePolicy:
    """Launch heuristics for product reductions."""

    cols_per_thread: int = 8


_LEADING_POLICY = LeadingAxisReducePolicy()
_PROD_POLICY = ProductReducePolicy()


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
    block_b = threads * _LEADING_POLICY.cols_per_thread
    column_blocks = ceildiv_int(kept, block_b)
    return max(1, min(reduced, ceildiv_int(_LEADING_POLICY.target_blocks, column_blocks)))


def _make_leading_reduce_ops(op_kind: str, divisor: float, out_dtype: str):
    """Create the per-op macros used by the leading-axis reduction."""

    @T.macro
    def init(acc):
        if op_kind == "amax":
            T.fill(acc, -T.infinity("float32"))
        elif op_kind == "amin":
            T.fill(acc, T.infinity("float32"))
        else:
            T.fill(acc, 0.0)

    @T.macro
    def combine(acc, slot, value):
        if op_kind == "amax":
            acc[slot] = T.max(acc[slot], value)
        elif op_kind == "amin":
            acc[slot] = T.min(acc[slot], value)
        else:
            acc[slot] = acc[slot] + value

    @T.macro
    def finish(out_local, slot, accumulated):
        if divisor:
            out_local[slot] = T.cast(accumulated / divisor, out_dtype)
        else:
            out_local[slot] = T.cast(accumulated, out_dtype)

    return init, combine, finish


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
    block_b = threads * _LEADING_POLICY.cols_per_thread
    rows_per_split = ceildiv_int(A, splits)
    exact = B % block_b == 0
    init_acc, combine, finish = _make_leading_reduce_ops(op_kind, divisor, out_dtype)

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

                init_acc(acc)

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
                        combine(acc, j, val)

                for j in T.Parallel(block_b):
                    finish(out_local, j, acc[j])

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


@functools.lru_cache(maxsize=32)
def _prod_reduce_kernel(M: int, N: int, dtype: str, threads: int):
    """Build a product reduce: one block per row, multiplying in fp32."""
    chunk = threads * _PROD_POLICY.cols_per_thread
    tiles = ceildiv_int(N, chunk)
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
                staged = T.alloc_shared((chunk,), dtype)

                running[0] = T.cast(1.0, "float32")
                for t in T.serial(tiles):
                    for i in T.Parallel(chunk):
                        staged[i] = x[row, t * chunk + i]
                    T.sync_threads()
                    for c in T.serial(_PROD_POLICY.cols_per_thread):
                        held = staged[tx * _PROD_POLICY.cols_per_thread + c]
                        if exact:
                            running[0] = running[0] * T.cast(held, "float32")
                        else:
                            col = (t * threads + tx) * _PROD_POLICY.cols_per_thread + c
                            running[0] = running[0] * T.if_then_else(
                                col < N, T.cast(held, "float32"), T.cast(1.0, "float32")
                            )
                    T.sync_threads()

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

    A row that exceeds any of the capacities ``BlockConfigPlanner`` tracks goes to a
    tiled variant, which iterates over N in chunks of ``tile_n`` columns.

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
        self._elem_bytes = torch_dtype_nbytes(dtype)
        self._smem_budget = device_smem_budget(device_index)
        self._planner = BlockConfigPlanner(
            self.N_padded,
            self._elem_bytes,
            self._smem_budget,
            num_buffers=2 if self._is_welford else 1,
            frag_slots=_FRAG_SLOTS.get(op_kind, 1),
        )
        self._needs_tiling = self._planner.needs_tiling
        self.strategy = self._select_strategy()

        if self.strategy == "prod":
            self.kernel = _prod_reduce_kernel(self.M, self.N, self.dtype_str, DEFAULT_THREADS)
        elif self.strategy == "welford":
            self.kernel = _welford_reduce_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.correction,
                self.dtype_str,
            )
        elif self.strategy == "simple":
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

    def _select_strategy(self) -> str:
        """Return the row-wise reduction backend selected by op kind and shape."""
        if self._is_prod:
            return "prod"
        if self._is_welford:
            return "welford_tiled" if self._needs_tiling else "welford"
        return "simple_tiled" if self._needs_tiling else "simple"

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
        splits = leading_row_splits(reduced, kept, _LEADING_POLICY.threads)
        divisor = float(reduced) if self.op_kind == "mean" else 0.0
        if splits == 1:
            single = _leading_reduce_kernel(
                reduced,
                kept,
                self.op_kind,
                self.dtype_str,
                self.dtype_str,
                _LEADING_POLICY.threads,
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
            _LEADING_POLICY.threads,
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
            _LEADING_POLICY.threads,
            1,
            divisor,
        )
        return finish()(partials.reshape(splits, kept))

    def _reduce_rows(self, x: torch.Tensor) -> object:
        """Reduce the trailing axis of an ``(M, N)`` buffer."""
        if self.strategy == "prod":
            return self.kernel()(x)
        block_m, threads = self.config["block_m"], self.config["threads"]
        if self.strategy in {"welford", "welford_tiled"}:
            if self.strategy == "welford_tiled":
                program = _welford_reduce_kernel_tiled(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.correction,
                    self.dtype_str,
                    self.config["tile_n"],
                )
            else:
                program = self.kernel
            results = program(block_m, threads)(x)
            if self.op_kind == "var_mean":
                return results[0], results[1]
            return results
        if self.strategy == "simple_tiled":
            program = _simple_reduce_kernel_tiled(
                self.M, self.N, self.op_kind, self.dtype_str, self.config["tile_n"]
            )
        else:
            program = self.kernel
        return program(block_m, threads)(x)
