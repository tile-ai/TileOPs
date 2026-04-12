"""Reduce kernels (sum, mean, amin, amax, prod, std, var, var_mean) using TileLang.

Two kernel families:
  - _simple_reduce_kernel: single-pass reduce for sum/mean/amin/amax/prod.
  - _welford_reduce_kernel: two-pass Welford for std/var/var_mean.

Both accept raw ``(M, N)`` tensors.  Boundary handling for non-aligned N
is performed inside the kernel via masked loads with identity-element fills,
eliminating host-side ``F.pad`` from the forward path.  When ``N`` is already
a multiple of ``DEFAULT_ALIGNMENT``, the fast vectorized ``T.copy`` path is
used.

When ``N_padded`` exceeds ``MAX_SINGLE_TILE_COLS`` (TileLang's vectorizer
limit at the 32768-column boundary), tiled variants iterate over N in
chunks of ``tile_n`` columns, accumulating partial results across tiles.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.
"""

import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    MAX_SINGLE_TILE_COLS,
    align_up,
    compute_tile_n,
    device_smem_budget,
)

__all__ = ["ReduceKernel"]

# Supported simple op kinds and their T.reduce_* mapping
_SIMPLE_REDUCE_MAP = {
    "sum": "reduce_sum",
    "mean": "reduce_sum",
    "amax": "reduce_max",
    "amin": "reduce_min",
    "prod": "reduce_sum",  # prod uses log-sum-exp trick
}

_WELFORD_KINDS = {"std", "var", "var_mean"}


# ---------------------------------------------------------------------------
# Simple reduce kernel
# ---------------------------------------------------------------------------


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
    """Build a simple reduce kernel for sum/mean/amax/amin/prod.

    Accepts an ``(M, N)`` input tensor.  When ``N`` is not a multiple of
    ``DEFAULT_ALIGNMENT``, the kernel uses element-wise ``T.if_then_else``
    loads that substitute the identity element for out-of-bounds columns
    (kernel-side boundary handling).  When ``N`` is already aligned, the
    fast ``T.copy`` path is used.

    For prod, we compute exp(sum(log(abs(x)))) * sign, which is numerically
    more stable than direct T.reduce_prod for large N.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)

    if op_kind == "prod":
        return _prod_reduce_kernel(M, N, dtype)

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
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < N),
                            T.cast(x[pid_m * block_m + i, j], "float32"),
                            T.cast(_pad_val, "float32"),
                        )
                else:
                    # Load via shared memory (fast vectorized path)
                    T.copy(x[pid_m * block_m, 0], shared_buf)

                    # Cast to fp32
                    for i, j in T.Parallel(block_m, N_padded):
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
                    for i, j in T.Parallel(block_m, N_padded):
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

    if op_kind == "prod":
        return _prod_reduce_kernel_tiled(M, N, dtype, tile_n)

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
                                for i, j in T.Parallel(block_m, tile_n):
                                    tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                            with T.Else():
                                for i, j in T.Parallel(block_m, tile_n):
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
                        for i, j in T.Parallel(block_m, tile_n):
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
                        for i, j in T.Parallel(block_m, tile_n):
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


@functools.lru_cache(maxsize=32)
def _prod_reduce_kernel(M, N, dtype):
    """Product reduce via log-sum-exp: exp(sum(log(|x|))) * sign.

    Accepts an ``(M, N)`` input tensor.  Padding columns are filled with
    ``1.0`` (the identity for product) via ``T.if_then_else`` when ``N``
    is not a multiple of ``DEFAULT_ALIGNMENT``.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N

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
                log_abs = T.alloc_fragment((block_m, N_padded), "float32")
                sign_neg = T.alloc_fragment((block_m, N_padded), "float32")
                acc_log = T.alloc_fragment((block_m,), "float32")
                acc_sign = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), dtype)

                if _needs_pad:
                    # Kernel-side boundary handling: fill out-of-bounds
                    # columns with 1.0 (identity for product).
                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.if_then_else(
                            T.And(pid_m * block_m + i < M, j < N),
                            T.cast(x[pid_m * block_m + i, j], "float32"),
                            T.cast(1.0, "float32"),
                        )
                else:
                    T.copy(x[pid_m * block_m, 0], shared_buf)

                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                # Compute log(|x|) and count negatives
                # Padded positions are 1.0, so log(1.0)=0 (neutral for sum)
                # and sign_neg=0 (non-negative).
                for i, j in T.Parallel(block_m, N_padded):
                    abs_val = T.abs(x_f32[i, j])
                    # Use a small epsilon to avoid log(0)
                    log_abs[i, j] = T.log(T.max(abs_val, 1e-38))
                    # 1 if negative, 0 if non-negative
                    sign_neg[i, j] = T.if_then_else(x_f32[i, j] < 0.0, 1.0, 0.0)

                T.reduce_sum(log_abs, acc_log, dim=1)
                T.reduce_sum(sign_neg, acc_sign, dim=1)

                for i in T.Parallel(block_m):
                    prod_val = T.exp(acc_log[i])
                    # If odd number of negatives, negate
                    # Use fmod to check parity
                    neg_count_mod2 = acc_sign[i] - T.floor(acc_sign[i] / 2.0) * 2.0
                    prod_val = T.if_then_else(neg_count_mod2 > 0.5, -prod_val, prod_val)
                    out_local[i] = T.cast(prod_val, dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _prod_reduce_kernel_tiled(M, N, dtype, tile_n):
    """Tiled product reduce via log-sum-exp for N_padded > MAX_SINGLE_TILE_COLS.

    Iterates over N in chunks of ``tile_n``, accumulating log(|x|) sums
    and negative-element counts across tiles.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    total_cols = num_tiles * tile_n
    _needs_mask = total_cols > N

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
                log_abs = T.alloc_fragment((block_m, tile_n), "float32")
                sign_neg = T.alloc_fragment((block_m, tile_n), "float32")
                tile_log = T.alloc_fragment((block_m,), "float32")
                tile_sign = T.alloc_fragment((block_m,), "float32")
                acc_log = T.alloc_fragment((block_m,), "float32")
                acc_sign = T.alloc_fragment((block_m,), "float32")
                out_local = T.alloc_fragment((block_m,), dtype)

                T.fill(acc_log, 0.0)
                T.fill(acc_sign, 0.0)

                for t in T.Serial(num_tiles):
                    if _needs_mask:
                        with T.If(t < num_tiles - 1):
                            with T.Then():
                                T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                for i, j in T.Parallel(block_m, tile_n):
                                    tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                            with T.Else():
                                for i, j in T.Parallel(block_m, tile_n):
                                    tile_f32[i, j] = T.if_then_else(
                                        T.And(
                                            pid_m * block_m + i < M,
                                            t * tile_n + j < N,
                                        ),
                                        T.cast(
                                            x[pid_m * block_m + i, t * tile_n + j],
                                            "float32",
                                        ),
                                        T.cast(1.0, "float32"),
                                    )
                    else:
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        for i, j in T.Parallel(block_m, tile_n):
                            tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                    for i, j in T.Parallel(block_m, tile_n):
                        abs_val = T.abs(tile_f32[i, j])
                        log_abs[i, j] = T.log(T.max(abs_val, 1e-38))
                        sign_neg[i, j] = T.if_then_else(
                            tile_f32[i, j] < 0.0, 1.0, 0.0,
                        )

                    T.reduce_sum(log_abs, tile_log, dim=1)
                    T.reduce_sum(sign_neg, tile_sign, dim=1)

                    for i in T.Parallel(block_m):
                        acc_log[i] = acc_log[i] + tile_log[i]
                        acc_sign[i] = acc_sign[i] + tile_sign[i]

                for i in T.Parallel(block_m):
                    prod_val = T.exp(acc_log[i])
                    neg_count_mod2 = acc_sign[i] - T.floor(acc_sign[i] / 2.0) * 2.0
                    prod_val = T.if_then_else(neg_count_mod2 > 0.5, -prod_val, prod_val)
                    out_local[i] = T.cast(prod_val, dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Welford reduce kernel (std, var, var_mean)
# ---------------------------------------------------------------------------


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
                        for i, j in T.Parallel(block_m, N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(0.0, "float32"),
                            )
                    else:
                        T.copy(x[pid_m * block_m, 0], shared_buf)

                        for i, j in T.Parallel(block_m, N_padded):
                            x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                    # Mean
                    T.reduce_sum(x_f32, row_sum, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = row_sum[i] / float(N)

                    # Variance: sum((x - mean)^2) / (N - correction)
                    for i, j in T.Parallel(block_m, N_padded):
                        sq_diff[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

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
                        for i, j in T.Parallel(block_m, N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(0.0, "float32"),
                            )
                    else:
                        T.copy(x[pid_m * block_m, 0], shared_buf)

                        for i, j in T.Parallel(block_m, N_padded):
                            x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                    # Mean
                    T.reduce_sum(x_f32, row_sum, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = row_sum[i] / float(N)

                    # Variance
                    for i, j in T.Parallel(block_m, N_padded):
                        sq_diff[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

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
                                    for i, j in T.Parallel(block_m, tile_n):
                                        tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                with T.Else():
                                    for i, j in T.Parallel(block_m, tile_n):
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
                            for i, j in T.Parallel(block_m, tile_n):
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
                                    for i, j in T.Parallel(block_m, tile_n):
                                        p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")
                                with T.Else():
                                    for i, j in T.Parallel(block_m, tile_n):
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
                            for i, j in T.Parallel(block_m, tile_n):
                                p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")

                        for i, j in T.Parallel(block_m, tile_n):
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
                                    for i, j in T.Parallel(block_m, tile_n):
                                        tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                with T.Else():
                                    for i, j in T.Parallel(block_m, tile_n):
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
                            for i, j in T.Parallel(block_m, tile_n):
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
                                    for i, j in T.Parallel(block_m, tile_n):
                                        p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")
                                with T.Else():
                                    for i, j in T.Parallel(block_m, tile_n):
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
                            for i, j in T.Parallel(block_m, tile_n):
                                p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")

                        for i, j in T.Parallel(block_m, tile_n):
                            sq_diff[i, j] = (p2_f32[i, j] - mean_val[i]) * (
                                p2_f32[i, j] - mean_val[i]
                            )
                        T.reduce_sum(sq_diff, tile_sq, dim=1)
                        for i in T.Parallel(block_m):
                            var_sum[i] = var_sum[i] + tile_sq[i]

                    pad_count = total_cols - N
                    if op_kind == "var":
                        for i in T.Parallel(block_m):
                            corrected = (
                                var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            )
                            out_local[i] = T.cast(corrected / float(N - correction), dtype)
                    else:  # std
                        for i in T.Parallel(block_m):
                            corrected = (
                                var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            )
                            out_local[i] = T.cast(
                                T.sqrt(corrected / float(N - correction)), dtype,
                            )

                    T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# custom_op wrappers
# ---------------------------------------------------------------------------


@torch.library.custom_op("top::reduce_simple_fwd", mutates_args=())
def _reduce_simple_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _simple_reduce_kernel(M, N, op_kind, dtype_str)(block_m, threads)(x)


@_reduce_simple_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, threads, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


@torch.library.custom_op("top::reduce_simple_tiled_fwd", mutates_args=())
def _reduce_simple_tiled_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    tile_n: int,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _simple_reduce_kernel_tiled(M, N, op_kind, dtype_str, tile_n)(block_m, threads)(x)


@_reduce_simple_tiled_wrapped.register_fake
def _(M, N, op_kind, dtype_str, tile_n, block_m, threads, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


@torch.library.custom_op("top::reduce_welford_fwd", mutates_args=())
def _reduce_welford_wrapped(
    M: int,
    N: int,
    op_kind: str,
    correction: int,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    kernel = _welford_reduce_kernel(M, N, op_kind, correction, dtype_str)
    result = kernel(block_m, threads)(x)
    if op_kind == "var_mean":
        # Returns (var, mean)
        return [result[0], result[1]]
    else:
        return [result]


@_reduce_welford_wrapped.register_fake
def _(M, N, op_kind, correction, dtype_str, block_m, threads, x):
    if op_kind == "var_mean":
        return [
            torch.empty((M,), dtype=x.dtype, device=x.device),
            torch.empty((M,), dtype=x.dtype, device=x.device),
        ]
    return [torch.empty((M,), dtype=x.dtype, device=x.device)]


@torch.library.custom_op("top::reduce_welford_tiled_fwd", mutates_args=())
def _reduce_welford_tiled_wrapped(
    M: int,
    N: int,
    op_kind: str,
    correction: int,
    dtype_str: str,
    tile_n: int,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    kernel = _welford_reduce_kernel_tiled(M, N, op_kind, correction, dtype_str, tile_n)
    result = kernel(block_m, threads)(x)
    if op_kind == "var_mean":
        return [result[0], result[1]]
    else:
        return [result]


@_reduce_welford_tiled_wrapped.register_fake
def _(M, N, op_kind, correction, dtype_str, tile_n, block_m, threads, x):
    if op_kind == "var_mean":
        return [
            torch.empty((M,), dtype=x.dtype, device=x.device),
            torch.empty((M,), dtype=x.dtype, device=x.device),
        ]
    return [torch.empty((M,), dtype=x.dtype, device=x.device)]


# ---------------------------------------------------------------------------
# ReduceKernel class
# ---------------------------------------------------------------------------


class ReduceKernel(Kernel):
    """Unified reduce kernel supporting sum/mean/amin/amax/prod/std/var/var_mean.

    Supports SM80+ architectures. Uses 256-element alignment for shared memory
    copies. Dispatches to simple or Welford kernel based on op_kind.

    When ``N_padded`` exceeds ``MAX_SINGLE_TILE_COLS``, tiled kernel variants
    are used that iterate over N in chunks of ``tile_n`` columns, avoiding the
    TileLang vectorizer limit at the 32768-column boundary.

    Boundary handling for non-aligned N is performed inside the kernel via
    masked loads with identity-element fills, so no host-side ``F.pad`` is
    needed.  The ``forward()`` method accepts raw ``(M, N)`` tensors.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        correction: int = 1,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.correction = correction
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._is_welford = op_kind in _WELFORD_KINDS
        self._elem_bytes = torch.tensor([], dtype=dtype).element_size()
        self._smem_budget = device_smem_budget()

        # Determine whether tiling is needed
        self._needs_tiling = self.N_padded > MAX_SINGLE_TILE_COLS

        if not self._needs_tiling:
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

    def _tile_n_for_block_m(self, block_m: int) -> int:
        """Return tile_n for a given block_m (0 means no tiling needed).

        Welford kernels allocate 2 shared buffers per pass, so
        ``num_buffers=2`` is used to ensure both fit in shared memory.
        """
        budget = self._smem_budget
        if self.N_padded <= MAX_SINGLE_TILE_COLS:
            single = compute_tile_n(
                block_m, self._elem_bytes, self.N_padded, budget=budget,
            )
            if single == self.N_padded:
                return 0

        num_buffers = 2 if self._is_welford else 1
        col_budget = MAX_SINGLE_TILE_COLS * num_buffers * block_m * self._elem_bytes
        effective_budget = min(budget, col_budget)
        return compute_tile_n(
            block_m, self._elem_bytes, self.N_padded,
            num_buffers=num_buffers,
            budget=effective_budget,
        )

    @property
    def default_config(self) -> dict:
        if not self._needs_tiling:
            smem_per_row = self.N_padded * self._elem_bytes
            max_block_m = (48 * 1024) // smem_per_row
            block_m = 1
            for bm in [1, 2, 4, 8]:
                if bm <= max_block_m:
                    block_m = bm
            return {"block_m": block_m, "threads": 128}

        # Tiled path: pick block_m that minimizes tile count
        best_bm = 1
        best_tile_n = self._tile_n_for_block_m(1)

        for bm in [2, 4, 8]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn == 0 and best_tile_n == 0 or tn == 0 and best_tile_n != 0:
                best_bm = bm
                best_tile_n = tn
            elif tn != 0 and best_tile_n == 0:
                pass
            else:
                best_num = (self.N_padded + best_tile_n - 1) // best_tile_n
                curr_num = (self.N_padded + tn - 1) // tn
                if curr_num < best_num:
                    best_bm = bm
                    best_tile_n = tn

        return {"block_m": best_bm, "threads": 128, "tile_n": best_tile_n}

    @property
    def autotune_configs(self) -> list[dict]:
        if not self._needs_tiling:
            smem_per_row = self.N_padded * self._elem_bytes
            max_block_m = (48 * 1024) // smem_per_row
            block_ms = [bm for bm in [1, 2, 4, 8] if bm <= max_block_m]
            threads_list = [128, 256]
            configs = list(itertools.product(block_ms, threads_list))
            return [{"block_m": bm, "threads": t} for bm, t in configs]

        # Tiled path
        configs = []
        for bm in [1, 2, 4, 8]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn == 0:
                continue
            for t in [128, 256]:
                configs.append({"block_m": bm, "threads": t, "tile_n": tn})
        return configs if configs else [self.default_config]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune the reduce kernel by benchmarking candidate configs.

        The non-tiled path delegates to the base ``Kernel.autotune`` which
        applies TileLang's ``autotune`` decorator to ``self.kernel``.

        The tiled path has no single ``self.kernel`` object -- it dispatches
        through wrapped helper functions -- so we benchmark each candidate
        config via :meth:`forward` directly and pick the fastest.
        """
        if not self._needs_tiling:
            return super().autotune(warmup=warmup, rep=rep)

        configs = self.autotune_configs
        if not configs:
            self.config = self.default_config
            return

        print(f'Start autotuning {self.__class__.__name__} (tiled path)...')

        device = torch.cuda.current_device()
        x = torch.randn(self.M, self.N, dtype=self.dtype, device=device)

        best_config = configs[0]
        best_time = float('inf')

        for cfg in configs:
            self.config = cfg
            # Warmup
            for _ in range(warmup):
                self.forward(x)
            torch.cuda.synchronize()

            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(rep):
                self.forward(x)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / rep

            if elapsed < best_time:
                best_time = elapsed
                best_config = cfg

        self.config = best_config
        print(f'Best config: {self.config}')

    def forward(self, x: torch.Tensor) -> object:
        if self._is_welford:
            if self._needs_tiling:
                results = _reduce_welford_tiled_wrapped(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.correction,
                    self.dtype_str,
                    self.config["tile_n"],
                    self.config["block_m"],
                    self.config["threads"],
                    x,
                )
            else:
                results = _reduce_welford_wrapped(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.correction,
                    self.dtype_str,
                    self.config["block_m"],
                    self.config["threads"],
                    x,
                )
            if self.op_kind == "var_mean":
                return results[0], results[1]
            return results[0]
        else:
            if self._needs_tiling:
                return _reduce_simple_tiled_wrapped(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.dtype_str,
                    self.config["tile_n"],
                    self.config["block_m"],
                    self.config["threads"],
                    x,
                )
            else:
                return _reduce_simple_wrapped(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.dtype_str,
                    self.config["block_m"],
                    self.config["threads"],
                    x,
                )
