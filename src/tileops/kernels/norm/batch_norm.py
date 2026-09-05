"""Batch Normalization kernels (training forward, inference forward, backward).

Reference: Ioffe & Szegedy (2015) https://arxiv.org/abs/1502.03167

C is the channel count, L = N * prod(spatial) the reduction length of one
channel, and S = prod(spatial) its contiguous run within one batch item.
Training forward and inference forward index the caller's (N, C, *spatial)
layout directly. Backward reduces over a channel on a (C, L) copy and moves
the caller's tensor into that layout itself.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel

__all__ = [
    "BatchNormBwdKernel",
    "BatchNormFwdInferKernel",
    "BatchNormFwdTrainKernel",
]

# Length at or below which one block holds a whole channel: x_shared costs
# L * sizeof(dtype), 16 KB at L=8192 in fp16, and backward holds two of them.
_PERSISTENT_MAX_L = 8192

# TileLang's AllReduce template needs a power-of-two thread count.
_REDUCE_THREADS = (256, 128, 64, 32)

# Widest tile one block takes, bounding register pressure.
_MAX_BLOCK_L = 512


def _vector_elements(dtype: torch.dtype) -> int:
    """Elements one thread accesses at once for a 128-bit vector in *dtype*."""
    return VECTOR_ACCESS_BYTES // dtype.itemsize


def _widths_down_to_one(widest: int) -> tuple[int, ...]:
    """*widest* and every halving of it, down to one element."""
    return tuple(widest >> k for k in range(widest.bit_length()))


def _tiled_configs(L: int) -> list[dict]:
    """The (block_l, threads) pairs the tiled builders accept for *L*, best first.

    Element zero is what an untuned kernel launches. The training forward and the
    backward kernel both read this: different prim_funcs, same reduction, same two
    parameters.
    """
    configs: list[dict] = []

    # One tile per channel: the whole channel is the tile.
    if L <= _PERSISTENT_MAX_L:
        configs += [{"block_l": L, "threads": t} for t in _REDUCE_THREADS if L % t == 0]

    # Several tiles per channel. block_l need not be a power of two, only the
    # thread count must, so 448 is available at L=3136 where 512 is not. No pair
    # repeats: threads separates these, and the entry above has block_l == L,
    # which this loop excludes.
    for threads in _REDUCE_THREADS:
        for k in range(_MAX_BLOCK_L // threads, 0, -1):
            block_l = threads * k
            if block_l < L and L % block_l == 0:
                configs.append({"block_l": block_l, "threads": threads})

    if configs:
        return configs

    # No reduce width divides L. Below the threshold the channel is still one
    # tile, at the narrowest width; above it, the widest tile that does divide L.
    if L <= _PERSISTENT_MAX_L:
        return [{"block_l": L, "threads": _REDUCE_THREADS[-1]}]
    for block_l in (512, 256, 128, 64, 32, 16):
        if L % block_l == 0:
            return [{"block_l": block_l, "threads": min(256, block_l)}]
    raise ValueError(
        f"L={L} is not divisible by any supported block_l. "
        "L must be divisible by at least 16 for the current kernel implementation."
    )


@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_train_kernel(
    C: int,
    L: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
) -> Callable:
    """Return the JIT-compiled training-forward kernel factory.

    Kernel computes, per channel:
      1. mean   = sum(x) / L
      2. var    = sum(x^2) / L  -  mean^2
      3. rstd   = 1 / sqrt(var + eps)
      4. y      = weight * (x - mean) * rstd + bias
      5. running_mean/var updated with *momentum*.

    Saved mean and rstd are needed by the backward pass.

    A channel is *S* elements contiguous at a time, once per batch item, so
    element *l* of channel *c* lives at ``(l // S) * C * S + c * S + l % S``.
    Reading through that index needs no transposed copy.

    Persistent path (block_l >= L): after pass 1 loads all L elements into
    x_shared, pass 2 normalizes directly from x_shared — no second global read.

    Non-persistent path (block_l < L): two global reads (classic two-pass BN).

    Requirements: L must be divisible by block_l; threads must divide block_l.
    """
    accum_dtype = "float32"
    plane = C * S

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_train_func(block_l: int, threads: int) -> Callable:
        @T.prim_func
        def _bn_fwd_train(
            x: T.Tensor([C * L], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_out: T.Tensor([C], accum_dtype),
            rstd_out: T.Tensor([C], accum_dtype),
            y: T.Tensor([C * L], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                x_shared = T.alloc_shared([block_l], dtype)

                # Per-element accumulators: each thread owns block_l/threads elements.
                # Accumulated across L/block_l tiles before the cross-thread reduce.
                xsum_frag = T.alloc_fragment([1, block_l], accum_dtype)
                xsq_frag = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(xsum_frag)
                T.clear(xsq_frag)

                # Pass 1 – accumulate sum(x) and sum(x^2) over all tiles.
                if block_l >= L:
                    # Persistent path has exactly one tile, so a pipelined loop
                    # cannot overlap producer/consumer work.
                    for _i, j in T.Parallel(1, block_l):
                        x_shared[j] = x[(j // S) * plane + bc * S + j % S]
                    for _i, j in T.Parallel(1, block_l):
                        xval = T.cast(x_shared[j], accum_dtype)
                        xsum_frag[_i, j] += xval
                        xsq_frag[_i, j] += xval * xval
                else:
                    # Read global memory directly: T.copy inside T.Pipelined
                    # races with the async copy.
                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            xval = T.cast(x[(l // S) * plane + bc * S + l % S], accum_dtype)
                            xsum_frag[_i, j] += xval
                            xsq_frag[_i, j] += xval * xval

                # Cross-thread reduction along block_l dimension.
                sum_result = T.alloc_fragment([1], accum_dtype)
                sq_result = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(xsum_frag, sum_result, dim=1)
                T.reduce_sum(xsq_frag, sq_result, dim=1)

                mean_val = sum_result[0] / T.cast(L, accum_dtype)
                var_val = sq_result[0] / T.cast(L, accum_dtype) - mean_val * mean_val
                rstd_val = T.cast(1.0, accum_dtype) / T.sqrt(var_val + T.cast(eps, accum_dtype))

                mean_out[bc] = mean_val
                rstd_out[bc] = rstd_val

                # Update running statistics.
                # running_var follows PyTorch convention: updated with unbiased variance
                # (Bessel's correction: biased_var * L / (L - 1)).
                mom = T.cast(momentum, accum_dtype)
                unbiased_var = (
                    var_val
                    * T.cast(L, accum_dtype)
                    / (T.cast(L, accum_dtype) - T.cast(1.0, accum_dtype))
                )
                # One writer per block: this running-stat RMW races if every thread runs it.
                if T.get_thread_binding() == 0:
                    running_mean[bc] = (T.cast(1.0, accum_dtype) - mom) * running_mean[
                        bc
                    ] + mom * mean_val
                    running_var[bc] = (T.cast(1.0, accum_dtype) - mom) * running_var[
                        bc
                    ] + mom * unbiased_var

                # Pass 2 – normalize.
                if block_l >= L:
                    # Persistent path: x_shared still holds all L elements from pass 1.
                    # No second global read — read directly from shared memory.
                    for _i, j in T.Parallel(1, block_l):
                        xval = T.cast(x_shared[j], accum_dtype)
                        y[(j // S) * plane + bc * S + j % S] = T.cast(
                            weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype
                        )
                else:
                    # Read global memory directly: T.copy inside T.Pipelined
                    # races with the async copy.
                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            flat = (l // S) * plane + bc * S + l % S
                            xval = T.cast(x[flat], accum_dtype)
                            y[flat] = T.cast(
                                weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype
                            )

        return _bn_fwd_train

    return _bn_fwd_train_func


@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_train_split_kernel(
    C: int,
    L: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
) -> Callable:
    """Return the three-stage training-forward factories for a long channel.

    The grid is over elements, not channels: *splits* blocks sum each channel,
    one block merges the partial sums into a per-channel scale and shift, and a
    flat map applies them. A shape with fewer channels than SMs still fills the
    device.

    Returns:
        A ``(stats, finalize, apply)`` triple of JIT factories.
    """
    accum_dtype = "float32"
    plane = C * S

    @tilelang.jit
    def _stats_func(splits: int, threads: int, num_per_thread: int) -> Callable:
        chunk = T.ceildiv(L, splits)

        @T.prim_func
        def _bn_train_stats(
            x: T.Tensor([C * L], dtype),
            partial_sum: T.Tensor([C, splits], accum_dtype),
            partial_sq: T.Tensor([C, splits], accum_dtype),
        ):
            with T.Kernel(C * splits, threads=threads) as bx:
                bc = bx // splits
                start = (bx % splits) * chunk
                # One accumulator per thread, merged by a fixed reduction
                # tree. Batch norm writes running statistics back, so the
                # merge order must be deterministic.
                sums = T.alloc_fragment([1, threads], accum_dtype)
                sqs = T.alloc_fragment([1, threads], accum_dtype)
                T.clear(sums)
                T.clear(sqs)
                for _i, j in T.Parallel(1, threads):
                    for step in T.serial(T.ceildiv(chunk, threads * num_per_thread)):
                        for i in T.serial(num_per_thread):
                            l = start + (step * threads + j) * num_per_thread + i
                            if l < L:
                                v = T.cast(x[(l // S) * plane + bc * S + l % S], accum_dtype)
                                sums[_i, j] += v
                                sqs[_i, j] += v * v
                sum_result = T.alloc_fragment([1], accum_dtype)
                sq_result = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(sums, sum_result, dim=1)
                T.reduce_sum(sqs, sq_result, dim=1)
                if T.get_thread_binding() == 0:
                    partial_sum[bc, bx % splits] = sum_result[0]
                    partial_sq[bc, bx % splits] = sq_result[0]

        return _bn_train_stats

    @tilelang.jit
    def _finalize_func(splits: int, threads: int) -> Callable:
        @T.prim_func
        def _bn_train_finalize(
            partial_sum: T.Tensor([C, splits], accum_dtype),
            partial_sq: T.Tensor([C, splits], accum_dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_out: T.Tensor([C], accum_dtype),
            rstd_out: T.Tensor([C], accum_dtype),
            scale_out: T.Tensor([C], accum_dtype),
            shift_out: T.Tensor([C], accum_dtype),
        ):
            with T.Kernel(1, threads=threads) as _:
                tx = T.get_thread_binding()
                for step in T.serial(T.ceildiv(C, threads)):
                    bc = step * threads + tx
                    if bc < C:
                        total = T.alloc_local([1], accum_dtype)
                        total_sq = T.alloc_local([1], accum_dtype)
                        total[0] = T.cast(0, accum_dtype)
                        total_sq[0] = T.cast(0, accum_dtype)
                        for k in T.serial(splits):
                            total[0] += partial_sum[bc, k]
                            total_sq[0] += partial_sq[bc, k]
                        n = T.cast(L, accum_dtype)
                        mean_val = total[0] / n
                        var_val = total_sq[0] / n - mean_val * mean_val
                        rstd_val = T.cast(1.0, accum_dtype) / T.sqrt(
                            var_val + T.cast(eps, accum_dtype)
                        )
                        mean_out[bc] = mean_val
                        rstd_out[bc] = rstd_val
                        # Folding the affine into the normalisation leaves the
                        # map that follows two numbers per channel to read
                        # instead of four.
                        scale_out[bc] = weight[bc] * rstd_val
                        shift_out[bc] = bias[bc] - mean_val * weight[bc] * rstd_val
                        # running_var follows PyTorch convention: updated with
                        # unbiased variance (Bessel's correction).
                        mom = T.cast(momentum, accum_dtype)
                        one = T.cast(1.0, accum_dtype)
                        running_mean[bc] = (one - mom) * running_mean[bc] + mom * mean_val
                        running_var[bc] = (one - mom) * running_var[bc] + mom * (
                            var_val * n / (n - one)
                        )

        return _bn_train_finalize

    @tilelang.jit(out_idx=[-1])
    def _apply_func(blocks: int, threads: int, num_per_thread: int) -> Callable:
        vector_holds_one_channel = S % num_per_thread == 0
        span = threads * num_per_thread
        total = C * L

        @T.prim_func
        def _bn_train_apply(
            x: T.Tensor([total], dtype),
            scale: T.Tensor([C], accum_dtype),
            shift: T.Tensor([C], accum_dtype),
            y: T.Tensor([total], dtype),
        ):
            with T.Kernel(blocks, threads=threads) as bx:
                tx = T.get_thread_binding()
                v = T.alloc_local([num_per_thread], dtype)
                o = T.alloc_local([num_per_thread], dtype)
                base = bx * span + tx * num_per_thread
                if base + num_per_thread <= total:
                    for i in T.vectorized(num_per_thread):
                        v[i] = x[base + i]
                    if vector_holds_one_channel:
                        ch = (base // S) % C
                        for i in T.serial(num_per_thread):
                            o[i] = T.cast(T.cast(v[i], accum_dtype) * scale[ch] + shift[ch], dtype)
                    else:
                        for i in T.serial(num_per_thread):
                            ch = ((base + i) // S) % C
                            o[i] = T.cast(T.cast(v[i], accum_dtype) * scale[ch] + shift[ch], dtype)
                    for i in T.vectorized(num_per_thread):
                        y[base + i] = o[i]
                else:
                    for i in T.serial(num_per_thread):
                        if base + i < total:
                            ch = ((base + i) // S) % C
                            y[base + i] = T.cast(
                                T.cast(x[base + i], accum_dtype) * scale[ch] + shift[ch],
                                dtype,
                            )

        return _bn_train_apply

    return _stats_func, _finalize_func, _apply_func


@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_train_wide_kernel(
    C: int,
    L: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
) -> Callable:
    """Return the JIT-compiled training-forward factory for a register-held channel.

    A channel short enough to fit in one block's registers is read once into
    registers and normalised from there, with no shared-memory staging and no
    second global read: global traffic is one read and one write of the tensor.

    The two sums are merged by a shuffle tree within each warp, then by one
    serial pass over the per-warp totals. That order is fixed, so the running
    statistics this kernel writes back are the same on every run.

    Requirements: ``num_per_thread`` divides *S*, so a thread's vector never
    straddles two batch items and its address stays affine; ``threads`` is a
    multiple of the warp width.
    """
    accum_dtype = "float32"
    plane = C * S
    lanes = 32

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_train_wide_func(threads: int, num_per_thread: int) -> Callable:
        steps = (L + threads * num_per_thread - 1) // (threads * num_per_thread)
        exact = steps * threads * num_per_thread == L
        n_warps = max(threads // lanes, 1)

        @T.prim_func
        def _bn_fwd_train_wide(
            x: T.Tensor([C * L], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_out: T.Tensor([C], accum_dtype),
            rstd_out: T.Tensor([C], accum_dtype),
            y: T.Tensor([C * L], dtype),
        ):
            with T.Kernel(C, threads=threads) as bc:
                tx = T.get_thread_binding()
                # Read before the sums so the latency overlaps the element loads.
                params = T.alloc_local([4], accum_dtype)
                params[0] = weight[bc]
                params[1] = bias[bc]
                params[2] = running_mean[bc]
                params[3] = running_var[bc]
                held = T.alloc_local([steps * num_per_thread], dtype)
                out = T.alloc_local([num_per_thread], dtype)
                acc = T.alloc_local([1], accum_dtype)
                sq = T.alloc_local([1], accum_dtype)
                acc[0] = T.cast(0, accum_dtype)
                sq[0] = T.cast(0, accum_dtype)

                for k in T.serial(steps):
                    head = (k * threads + tx) * num_per_thread
                    if exact or head + num_per_thread <= L:
                        start = (head // S) * plane + bc * S + head % S
                        for i in T.vectorized(num_per_thread):
                            held[k * num_per_thread + i] = x[start + i]
                        for i in T.serial(num_per_thread):
                            v = T.cast(held[k * num_per_thread + i], accum_dtype)
                            acc[0] += v
                            sq[0] += v * v
                    else:
                        for i in T.serial(num_per_thread):
                            l = head + i
                            if l < L:
                                held[k * num_per_thread + i] = x[(l // S) * plane + bc * S + l % S]
                                v = T.cast(held[k * num_per_thread + i], accum_dtype)
                                acc[0] += v
                                sq[0] += v * v

                for step in T.serial(5):
                    acc[0] += T.shfl_xor(acc[0], T.shift_left(1, step))
                    sq[0] += T.shfl_xor(sq[0], T.shift_left(1, step))

                warp_sum = T.alloc_shared([n_warps], accum_dtype)
                warp_sq = T.alloc_shared([n_warps], accum_dtype)
                if tx % lanes == 0:
                    warp_sum[tx // lanes] = acc[0]
                    warp_sq[tx // lanes] = sq[0]
                T.sync_threads()
                acc[0] = T.cast(0, accum_dtype)
                sq[0] = T.cast(0, accum_dtype)
                for w in T.serial(n_warps):
                    acc[0] += warp_sum[w]
                    sq[0] += warp_sq[w]

                n = T.cast(L, accum_dtype)
                mean_val = acc[0] / n
                var_val = sq[0] / n - mean_val * mean_val
                rstd_val = T.cast(1.0, accum_dtype) / T.sqrt(var_val + T.cast(eps, accum_dtype))
                # The affine and the normalisation fold into one multiply-add.
                scale_val = params[0] * rstd_val
                shift_val = params[1] - mean_val * scale_val

                one = T.cast(1.0, accum_dtype)
                mom = T.cast(momentum, accum_dtype)
                # One writer per block: this running-stat RMW races if every thread runs it.
                if tx == 0:
                    mean_out[bc] = mean_val
                    rstd_out[bc] = rstd_val
                    running_mean[bc] = (one - mom) * params[2] + mom * mean_val
                    # running_var follows PyTorch convention: updated with
                    # unbiased variance (Bessel's correction).
                    running_var[bc] = (one - mom) * params[3] + mom * (var_val * n / (n - one))

                for k in T.serial(steps):
                    head = (k * threads + tx) * num_per_thread
                    if exact or head + num_per_thread <= L:
                        start = (head // S) * plane + bc * S + head % S
                        for i in T.serial(num_per_thread):
                            out[i] = T.cast(
                                T.cast(held[k * num_per_thread + i], accum_dtype) * scale_val
                                + shift_val,
                                dtype,
                            )
                        for i in T.vectorized(num_per_thread):
                            y[start + i] = out[i]
                    else:
                        for i in T.serial(num_per_thread):
                            l = head + i
                            if l < L:
                                y[(l // S) * plane + bc * S + l % S] = T.cast(
                                    T.cast(held[k * num_per_thread + i], accum_dtype) * scale_val
                                    + shift_val,
                                    dtype,
                                )

        return _bn_fwd_train_wide

    return _bn_fwd_train_wide_func


@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_train_whole_kernel(
    C: int,
    L: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
) -> Callable:
    """Return the JIT-compiled training-forward factory for a channel per thread.

    A channel is *S* elements contiguous at a time, so a short spatial extent
    — ``S == 1`` for an ``(N, C)`` input — leaves a block that owns one channel
    one useful element per cache line, whatever the tile size. One channel per
    *thread* instead puts neighbouring channels in neighbouring threads, so a
    warp's reads fall in the same lines. A channel that fits in a thread's
    registers needs no cross-thread reduction, no shared memory and no barrier.
    """
    accum_dtype = "float32"
    plane = C * S

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_train_whole_func(threads: int) -> Callable:
        blocks = (C + threads - 1) // threads

        @T.prim_func
        def _bn_fwd_train_whole(
            x: T.Tensor([C * L], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_out: T.Tensor([C], accum_dtype),
            rstd_out: T.Tensor([C], accum_dtype),
            y: T.Tensor([C * L], dtype),
        ):
            with T.Kernel(blocks, threads=threads) as bx:
                c = bx * threads + T.get_thread_binding()
                # Read the channel's four parameters before the sums so their
                # latency overlaps the element loads.
                params = T.alloc_local([4], accum_dtype)
                if c < C:
                    params[0] = weight[c]
                    params[1] = bias[c]
                    params[2] = running_mean[c]
                    params[3] = running_var[c]
                held = T.alloc_local([L], dtype)
                acc = T.alloc_local([1], accum_dtype)
                sq = T.alloc_local([1], accum_dtype)
                acc[0] = T.cast(0, accum_dtype)
                sq[0] = T.cast(0, accum_dtype)
                if c < C:
                    for l in T.serial(L):
                        held[l] = x[(l // S) * plane + c * S + l % S]
                        v = T.cast(held[l], accum_dtype)
                        acc[0] += v
                        sq[0] += v * v

                    n = T.cast(L, accum_dtype)
                    mean_val = acc[0] / n
                    var_val = sq[0] / n - mean_val * mean_val
                    rstd_val = T.cast(1.0, accum_dtype) / T.sqrt(var_val + T.cast(eps, accum_dtype))
                    mean_out[c] = mean_val
                    rstd_out[c] = rstd_val
                    # The affine and the normalisation fold into one multiply-add.
                    scale_val = params[0] * rstd_val
                    shift_val = params[1] - mean_val * scale_val

                    mom = T.cast(momentum, accum_dtype)
                    one = T.cast(1.0, accum_dtype)
                    running_mean[c] = (one - mom) * params[2] + mom * mean_val
                    # running_var follows PyTorch convention: updated with
                    # unbiased variance (Bessel's correction).
                    running_var[c] = (one - mom) * params[3] + mom * (var_val * n / (n - one))

                    for l in T.serial(L):
                        y[(l // S) * plane + c * S + l % S] = T.cast(
                            T.cast(held[l], accum_dtype) * scale_val + shift_val, dtype
                        )

        return _bn_fwd_train_whole

    return _bn_fwd_train_whole_func


class BatchNormFwdTrainKernel(Kernel):
    """Training-mode batch normalization forward kernel.

    Args:
        C: Number of channels.
        L: Total reduction length = N * H * W * ... (must be divisible by block_l).
        dtype: Input/output data type.
        eps: Numerical stability constant.
        momentum: Running-stat update momentum.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
        S: Elements per channel in one batch item, ``product(spatial)``.
            Defaults to *L*, correct when the batch size is one.
    """

    supported_archs: list[int] = [80, 89, 90]

    # Block widths the split path's sums are tuned over. Powers of two only:
    # T.reduce_sum lowers to an XOR butterfly.
    _SPLIT_SUM_THREADS = (128, 256, 512, 1024)

    # The thresholds below are measured crossovers, not derived bounds.

    # Blocks the split path aims for, as a multiple of the channel count. Above
    # this the runtime is flat.
    _SPLIT_TARGET_BLOCKS = 1024

    # Per-channel length above which one block per channel is too narrow a grid,
    # whatever the tile size, and the split path takes over.
    _SPLIT_MIN_L = 1 << 16

    # A channel goes to one thread only where it is one element per batch item.
    # Past that the channel-per-block path wins at every channel count.
    _WHOLE_MAX_S = 1

    # Longest channel one thread holds, and the width of the block holding them.
    # That block is wider than the channels it covers when there are few, so it
    # still launches enough warps to cover load latency.
    _WHOLE_MAX_L = 32
    _WHOLE_BLOCK_THREADS = 256

    # The register-held path's block width, and the most elements one thread may
    # hold across all its steps. Past this the spills cost more than the second
    # global read the other paths pay.
    _WIDE_BLOCK_THREADS = 256
    _WIDE_MAX_HELD = 256

    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        momentum: float = 0.1,
        config: Optional[dict] = None,
        tune: bool = False,
        S: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.S = L if S is None else S
        self.dtype = dtype
        self.eps = eps
        self.momentum = momentum
        self.path, self.launch = self._select_path(C, L, self.S, dtype)
        if self.path == "whole":
            self.whole_kernel = _batch_norm_fwd_train_whole_kernel(
                C, L, self.S, self.dtype_str, eps, momentum
            )
        elif self.path == "wide":
            self.wide_kernel = _batch_norm_fwd_train_wide_kernel(
                C, L, self.S, self.dtype_str, eps, momentum
            )
        elif self.path == "split":
            self.stages = _batch_norm_fwd_train_split_kernel(
                C, L, self.S, self.dtype_str, eps, momentum
            )
        self.kernel = _batch_norm_fwd_train_kernel(C, L, self.S, self.dtype_str, eps, momentum)
        self.init_config(config, tune)

    @classmethod
    def _select_path(cls, C: int, L: int, S: int, dtype: torch.dtype) -> tuple[str, object]:
        """Which launch serves this shape, and the sizing it needs.

        The four differ in what owns a channel, which is what a short channel,
        a long one, and a short spatial extent each need:

        ==========  ====================================  ==================
        path        a channel belongs to                  chosen when
        ==========  ====================================  ==================
        ``whole``   one thread, held in its registers     *S* is 1
        ``wide``    one block, held in its registers      *L* fits registers
        ``split``   several blocks, summed and merged     *L* is long
        ``tiled``   one block, streamed through shared    otherwise
        ==========  ====================================  ==================
        """
        if S <= cls._WHOLE_MAX_S and L <= cls._WHOLE_MAX_L:
            return "whole", cls._WHOLE_BLOCK_THREADS
        wide = cls._wide_launch(L, S, dtype)
        if wide is not None:
            return "wide", wide
        if C < cls._SPLIT_TARGET_BLOCKS and L >= cls._SPLIT_MIN_L:
            return "split", max(1, min(L, -(-cls._SPLIT_TARGET_BLOCKS // C)))
        return "tiled", None

    @classmethod
    def _wide_launch(cls, L: int, S: int, dtype: torch.dtype) -> Optional[tuple[int, int]]:
        """The ``(threads, num_per_thread)`` a register-held channel needs, or None.

        The vector must not straddle two batch items, and the channel must fit
        in the widest block the device allows.
        """
        for num_per_thread in _widths_down_to_one(_vector_elements(dtype)):
            if S % num_per_thread:
                continue
            # Halve the block width while the channel would leave half of it
            # empty; the length the width does not cover becomes steps, each
            # holding its own vector, so _WIDE_MAX_HELD caps them.
            threads = cls._WIDE_BLOCK_THREADS
            while threads > 32 and threads * num_per_thread >= L * 2:
                threads //= 2
            steps = -(-L // (threads * num_per_thread))
            if steps * num_per_thread <= cls._WIDE_MAX_HELD:
                return threads, num_per_thread
        return None

    @property
    def default_config(self) -> dict:
        return _tiled_configs(self.L)[0]

    @property
    def autotune_configs(self) -> list[dict]:
        return _tiled_configs(self.L)

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        """Tune the kernel this shape's path actually launches.

        The base class tunes ``self.kernel``, the tiled builder. The whole and
        wide paths take their launch from the shape and read nothing off the
        config. The split path reads only the block width, in the sums.
        """
        if self.path in ("whole", "wide"):
            self.config = self.default_config
            return
        if self.path == "split":
            print(f"Start autotuning {type(self).__name__} (split sums)...")
            num_per_thread = _vector_elements(self.dtype)
            configs = [
                {"splits": self.launch, "threads": t, "num_per_thread": num_per_thread}
                for t in self._SPLIT_SUM_THREADS
            ]
            # A candidate seeds it, not default_config: that describes the
            # tiled kernel and names none of the sums builder's parameters.
            tuned = self.tune_jit_kernel(
                self.stages[0], configs, warmup=warmup, rep=rep, seed_config=configs[0]
            )
            self.config = dict(self.default_config, threads=tuned.config["threads"])
            print(f"Best config: {self.config}")
            return
        super().autotune(warmup=warmup, rep=rep)

    def _forward_split(
        self,
        flat: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        mean_out: torch.Tensor,
        rstd_out: torch.Tensor,
    ) -> torch.Tensor:
        """Sum, merge, then map -- three launches over an element-wide grid."""
        stats, finalize, apply_ = self.stages
        splits = self.launch
        threads = self.config["threads"]
        num_per_thread = _vector_elements(self.dtype)
        empty = functools.partial(torch.empty, device=flat.device, dtype=torch.float32)
        partial_sum = empty((self.C, splits))
        partial_sq = empty((self.C, splits))
        stats(splits, threads, num_per_thread)(flat, partial_sum, partial_sq)
        scale, shift = empty(self.C), empty(self.C)
        finalize(splits, min(256, self.C))(
            partial_sum,
            partial_sq,
            weight,
            bias,
            running_mean,
            running_var,
            mean_out,
            rstd_out,
            scale,
            shift,
        )
        span = threads * num_per_thread
        blocks = (flat.numel() + span - 1) // span
        return apply_(blocks, threads, num_per_thread)(flat, scale, shift)

    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        """Run training forward pass on an ``(N, C, *spatial)`` input.

        Returns:
            y: Normalized output, shaped like *x*.
            mean_out: Per-channel batch mean (saved for backward).
            rstd_out: Per-channel reciprocal std (saved for backward).

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_cuda(
            x=x,
            weight=weight,
            bias=bias,
            running_mean=running_mean,
            running_var=running_var,
        )
        mean_out = torch.empty(self.C, device=x.device, dtype=torch.float32)
        rstd_out = torch.empty(self.C, device=x.device, dtype=torch.float32)
        flat = x.contiguous().reshape(-1)
        if self.path == "whole":
            y = self.whole_kernel(self.launch)(
                flat, weight, bias, running_mean, running_var, mean_out, rstd_out
            )
            return y.reshape(x.shape), mean_out, rstd_out
        if self.path == "wide":
            y = self.wide_kernel(*self.launch)(
                flat, weight, bias, running_mean, running_var, mean_out, rstd_out
            )
            return y.reshape(x.shape), mean_out, rstd_out
        if self.path == "split":
            y = self._forward_split(
                flat, running_mean, running_var, weight, bias, mean_out, rstd_out
            )
            return y.reshape(x.shape), mean_out, rstd_out
        y = self.kernel(
            self.config["block_l"],
            self.config["threads"],
        )(
            flat,
            weight,
            bias,
            running_mean,
            running_var,
            mean_out,
            rstd_out,
        )
        return y.reshape(x.shape), mean_out, rstd_out


# Inference forward


@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_infer_kernel(
    total: int,
    C: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
) -> Callable:
    """Return the JIT-compiled inference-forward kernel factory.

    Inference reads no statistic off the input, so the whole op is one map:
    ``y = x * scale + shift`` with a scale and shift the channel picks. The
    grid is over elements, not channels, and the channel of element ``i`` is
    ``(i // S) % C`` -- which holds for the ``(N, C, *spatial)`` the caller
    already has, so nothing is transposed on the way in or out.

    Args:
        total: Element count of the whole tensor.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: Input/output data type.
        eps: Numerical stability constant.
    """
    accum_dtype = "float32"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_infer_func(threads: int, num_per_thread: int, steps: int) -> Callable:
        # A vector of num_per_thread elements sits in one channel only when a
        # channel's run divides it; otherwise each element picks its own.
        vector_holds_one_channel = S % num_per_thread == 0
        span = threads * num_per_thread * steps
        # Derived, not a parameter: the autotuner binds by name, and every
        # parameter of this builder must therefore be a config key.
        blocks = -(-total // span)

        @T.prim_func
        def _bn_fwd_infer(
            x: T.Tensor([total], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            y: T.Tensor([total], dtype),
        ):
            with T.Kernel(blocks, threads=threads) as bx:
                tx = T.get_thread_binding()
                # Fusing weight and bias into one scale and shift per channel
                # turns the body into a single multiply-add. The table is
                # per-channel, so the block builds it once rather than every
                # element recomputing a square root and a divide.
                scale = T.alloc_shared([C], accum_dtype)
                shift = T.alloc_shared([C], accum_dtype)
                for c in T.serial(T.ceildiv(C, threads)):
                    ch = c * threads + tx
                    if ch < C:
                        sc = weight[ch] / T.sqrt(running_var[ch] + T.cast(eps, accum_dtype))
                        scale[ch] = sc
                        shift[ch] = bias[ch] - running_mean[ch] * sc
                T.sync_threads()

                v = T.alloc_local([num_per_thread], dtype)
                o = T.alloc_local([num_per_thread], dtype)
                # A block walks its span in *steps* vectors per thread, each
                # step contiguous across the block.
                for k in T.serial(steps):
                    base = bx * span + (k * threads + tx) * num_per_thread
                    if base + num_per_thread <= total:
                        for i in T.vectorized(num_per_thread):
                            v[i] = x[base + i]
                        if vector_holds_one_channel:
                            ch = (base // S) % C
                            for i in T.serial(num_per_thread):
                                o[i] = T.cast(
                                    T.cast(v[i], accum_dtype) * scale[ch] + shift[ch], dtype
                                )
                        else:
                            for i in T.serial(num_per_thread):
                                ch = ((base + i) // S) % C
                                o[i] = T.cast(
                                    T.cast(v[i], accum_dtype) * scale[ch] + shift[ch], dtype
                                )
                        for i in T.vectorized(num_per_thread):
                            y[base + i] = o[i]
                    else:
                        for i in T.serial(num_per_thread):
                            if base + i < total:
                                ch = ((base + i) // S) % C
                                y[base + i] = T.cast(
                                    T.cast(x[base + i], accum_dtype) * scale[ch] + shift[ch], dtype
                                )

        return _bn_fwd_infer

    return _bn_fwd_infer_func


class BatchNormFwdInferKernel(Kernel):
    """Inference-mode batch normalization forward kernel.

    Args:
        C: Number of channels.
        L: Total reduction length = N * H * W * ... (kept for the op's signature).
        dtype: Input/output data type.
        eps: Numerical stability constant.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
        S: Elements per channel in one batch item, ``product(spatial)``.
            Defaults to *L*, correct when the batch size is one.
    """

    supported_archs: list[int] = [80, 89, 90]

    # Elements one block covers, and its width. Each block builds the whole
    # per-channel table first, so that prologue is paid per block; fixing the
    # span fixes the block count, and the step count absorbs whatever the
    # per-thread vector gives up to stay a single 128-bit access.
    _BLOCK_SPAN = 2048
    _BLOCK_THREADS = 256

    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        config: Optional[dict] = None,
        tune: bool = False,
        S: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.S = L if S is None else S
        self.total = C * L
        self.dtype = dtype
        self.eps = eps
        self.kernel = _batch_norm_fwd_infer_kernel(self.total, C, self.S, self.dtype_str, eps)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # The widest vector the element count divides evenly, with the step
        # count taking whatever keeps the block span at _BLOCK_SPAN.
        threads = self._BLOCK_THREADS
        for num_per_thread in _widths_down_to_one(_vector_elements(self.dtype)):
            if self.total % num_per_thread == 0:
                steps = max(1, self._BLOCK_SPAN // (threads * num_per_thread))
                return {
                    "threads": threads,
                    "num_per_thread": num_per_thread,
                    "steps": steps,
                }
        return {"threads": threads, "num_per_thread": 1, "steps": self._BLOCK_SPAN // threads}

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for threads in (128, 256, 512, 1024):
            for num_per_thread in _widths_down_to_one(_vector_elements(self.dtype)):
                if self.total % num_per_thread:
                    continue
                for steps in (1, 2, 4):
                    configs.append(
                        {
                            "threads": threads,
                            "num_per_thread": num_per_thread,
                            "steps": steps,
                        }
                    )
        return configs if configs else [self.default_config]

    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference forward pass on an ``(N, C, *spatial)`` input.

        Returns:
            Normalized output, shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_cuda(
            x=x,
            weight=weight,
            bias=bias,
            running_mean=running_mean,
            running_var=running_var,
        )
        y = self.kernel(
            self.config["threads"], self.config["num_per_thread"], self.config["steps"]
        )(x.contiguous().reshape(-1), weight, bias, running_mean, running_var)
        return y.reshape(x.shape)


# Backward


def _to_cl(t: torch.Tensor) -> torch.Tensor:
    """Move (N, C, *spatial) into the (C, L) layout the backward prim_func reads."""
    channels = t.shape[1]
    return t.permute(1, 0, *range(2, t.ndim)).reshape(channels, -1).contiguous()


def _from_cl(t: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Move a (C, L) backward result back to the caller's shape."""
    batch, channels, *spatial = original_shape
    restored = t.reshape(channels, batch, *spatial)
    return restored.permute(1, 0, *range(2, restored.ndim)).contiguous()


@functools.lru_cache(maxsize=32)
def _batch_norm_bwd_kernel(
    C: int,
    L: int,
    dtype: str = "float16",
) -> Callable:
    """Return the JIT-compiled backward kernel factory.

    Given saved mean and rstd from the training forward pass, computes:
      grad_bias[c]   = sum_i( grad_out[c, i] )
      grad_weight[c] = sum_i( grad_out[c, i] * x_hat[c, i] )
      grad_x[c, i]   = weight[c] * rstd[c] / L
                       * ( L * grad_out[c, i]
                           - grad_bias[c]
                           - x_hat[c, i] * grad_weight[c] )

    where x_hat[c, i] = (x[c, i] - mean[c]) * rstd[c].

    Persistent path (block_l >= L): after pass 1 accumulates grad_bias /
    grad_weight while loading grad_out and x into shared memory, pass 2 computes
    grad_x directly from shared memory — eliminates the second global read.

    Non-persistent path (block_l < L): two global reads (classic two-pass BN bwd).

    Requirements: L must be divisible by block_l.
    """
    accum_dtype = "float32"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_bwd_func(block_l: int, threads: int) -> Callable:
        @T.prim_func
        def _bn_bwd(
            grad_out: T.Tensor([C, L], dtype),
            x: T.Tensor([C, L], dtype),
            weight: T.Tensor([C], accum_dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
            grad_x: T.Tensor([C, L], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                go_shared = T.alloc_shared([block_l], dtype)
                x_shared = T.alloc_shared([block_l], dtype)

                mean_val = mean[bc]
                rstd_val = rstd[bc]
                w_val = weight[bc]

                # Accumulators for sum(grad_out) and sum(grad_out * x_hat).
                do_frag = T.alloc_fragment([1, block_l], accum_dtype)
                do_xhat_frag = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(do_frag)
                T.clear(do_xhat_frag)

                # Pass 1 – accumulate grad_bias and grad_weight contributions.
                if block_l >= L:
                    # Persistent path has exactly one tile, so a pipelined loop
                    # cannot overlap producer/consumer work.
                    T.copy(grad_out[bc, 0:block_l], go_shared)
                    T.copy(x[bc, 0:block_l], x_shared)
                    for _i, j in T.Parallel(1, block_l):
                        go_val = T.cast(go_shared[j], accum_dtype)
                        x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                        do_frag[_i, j] += go_val
                        do_xhat_frag[_i, j] += go_val * x_hat
                else:
                    # Read global memory directly: T.copy inside T.Pipelined
                    # races with the async copy.
                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            go_val = T.cast(grad_out[bc, l_tile * block_l + j], accum_dtype)
                            x_hat = (
                                T.cast(x[bc, l_tile * block_l + j], accum_dtype) - mean_val
                            ) * rstd_val
                            do_frag[_i, j] += go_val
                            do_xhat_frag[_i, j] += go_val * x_hat

                sum_do = T.alloc_fragment([1], accum_dtype)
                sum_do_xhat = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(do_frag, sum_do, dim=1)
                T.reduce_sum(do_xhat_frag, sum_do_xhat, dim=1)

                grad_bias[bc] = sum_do[0]
                grad_weight[bc] = sum_do_xhat[0]

                # Precompute per-channel constant.
                w_rstd_over_L = w_val * rstd_val / T.cast(L, accum_dtype)

                # Pass 2 – compute grad_x.
                if block_l >= L:
                    # Persistent path: go_shared and x_shared hold all L elements.
                    # No second global read needed.
                    for _i, j in T.Parallel(1, block_l):
                        go_val = T.cast(go_shared[j], accum_dtype)
                        x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                        gx = w_rstd_over_L * (
                            T.cast(L, accum_dtype) * go_val - sum_do[0] - x_hat * sum_do_xhat[0]
                        )
                        grad_x[bc, j] = T.cast(gx, dtype)
                else:
                    # Read global memory directly: T.copy inside T.Pipelined
                    # races with the async copy.
                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            go_val = T.cast(grad_out[bc, l_tile * block_l + j], accum_dtype)
                            x_hat = (
                                T.cast(x[bc, l_tile * block_l + j], accum_dtype) - mean_val
                            ) * rstd_val
                            gx = w_rstd_over_L * (
                                T.cast(L, accum_dtype) * go_val - sum_do[0] - x_hat * sum_do_xhat[0]
                            )
                            grad_x[bc, l_tile * block_l + j] = T.cast(gx, dtype)

        return _bn_bwd

    return _bn_bwd_func


class BatchNormBwdKernel(Kernel):
    """Batch normalization backward kernel.

    Args:
        C: Number of channels.
        L: Total reduction length = N * H * W * ... (must be divisible by block_l).
        dtype: grad_out/x/grad_x data type.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.dtype = dtype
        self.kernel = _batch_norm_bwd_kernel(C, L, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return _tiled_configs(self.L)[0]

    @property
    def autotune_configs(self) -> list[dict]:
        return _tiled_configs(self.L)

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ):
        """Run the backward pass on ``(N, C, *spatial)`` inputs.

        Moves the inputs into the $[C \\times L]$ layout and ``grad_x`` back.

        Returns:
            grad_x: Gradient w.r.t. the input, shaped like *x*.
            grad_weight: Gradient w.r.t. affine scale (gamma).
            grad_bias: Gradient w.r.t. affine shift (beta).

        Raises:
            ValueError: An input is not on a CUDA device.
        """
        self._require_cuda(grad_out=grad_out, x=x, weight=weight, mean=mean, rstd=rstd)
        grad_weight = torch.empty(self.C, device=grad_out.device, dtype=torch.float32)
        grad_bias = torch.empty(self.C, device=grad_out.device, dtype=torch.float32)
        grad_x = self.kernel(
            self.config["block_l"],
            self.config["threads"],
        )(_to_cl(grad_out), _to_cl(x), weight, mean, rstd, grad_weight, grad_bias)
        return _from_cl(grad_x, x.shape), grad_weight, grad_bias
