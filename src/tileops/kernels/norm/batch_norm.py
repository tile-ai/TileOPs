"""Batch Normalization kernels (training forward, inference forward, backward).

Reference: Ioffe & Szegedy (2015) https://arxiv.org/abs/1502.03167

Training forward and backward reduce over a channel, so they work on (C, L)
where C is the channel count and L = N * H * W * ... , and move the caller's
(N, C, *spatial) tensor into that layout themselves. Inference reduces over
nothing and reads the caller's layout as it is.

Performance notes:
  - Persistent path (block_l >= L): loads all L elements into shared memory once
    and normalizes from there — single global read, eliminates the second pass.
    Active when L <= _PERSISTENT_THRESHOLD (8192).
  - Non-power-of-2 block_l: _find_best_block_l() searches thread counts
    [256, 128, 64, 32] to find the largest valid block_l,
    fixing poor occupancy for L values like 3136 = 2^6 * 7^2.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = [
    "BatchNormBwdKernel",
    "BatchNormFwdInferKernel",
    "BatchNormFwdTrainKernel",
]

# Blocks the split training path aims to launch, as a multiple of the channel
# count: enough to cover the device several times over so no SM sits idle.
_SPLIT_TARGET_BLOCKS = 1024

# Per-channel length above which one block per channel is too narrow a grid,
# whatever the tile size, and the split path takes over.
_SPLIT_MIN_L = 1 << 16

# L threshold for the persistent (single global read) training path.
# x_shared uses L * sizeof(dtype) bytes per block:
#   L=8192, fp16 → 16 KB — well within H100 shared memory limits.
# Spatial extent below which a channel-per-block read cannot fill a cache line,
# and the longest channel a thread that owns one can keep in its registers.
_WHOLE_MAX_S = 16
_WHOLE_MAX_L = 32

# Block width for that path. It is wider than the channels a block covers when
# there are few of them, which measured faster than a block sized to fit: the
# extra warps give the scheduler something to issue while the loads land.
_WHOLE_BLOCK_THREADS = 256

# The register-held path's block width, its widest per-thread vector, and the
# most elements one thread may hold across all its steps before the register
# file is the binding constraint.
_WIDE_BLOCK_THREADS = 256
_WIDE_MAX_NUM_PER_THREAD = 8
_WIDE_MAX_HELD = 32

_PERSISTENT_THRESHOLD = 8192


def _find_best_threads(L: int) -> int:
    """Largest power-of-2 t in [256, 128, 64, 32] that evenly divides L.

    TileLang's AllReduce template requires a power-of-2 thread count.
    """
    for t in [256, 128, 64, 32]:
        if L % t == 0:
            return t
    return 32  # fallback


def _find_best_block_l(L: int) -> dict:
    """Find best non-persistent block_l config for given L.

    Uses power-of-2 thread counts only (required by TileLang's AllReduce).
    Block_l can be any multiple of `threads` that divides L — including
    non-power-of-2 values such as 448 for L=3136 — giving more tiles per
    channel and better GPU utilization than the strict power-of-2 search.
    block_l is capped at 512 to limit register pressure.
    """
    for threads in [256, 128, 64, 32]:
        for k in range(512 // threads, 0, -1):
            bl = threads * k
            if bl >= L:
                continue
            if L % bl == 0:
                return {"block_l": bl, "num_stages": 0, "threads": threads}
    # Fallback (should rarely be reached).
    for bl in [512, 256, 128, 64, 32, 16]:
        if L % bl == 0:
            return {"block_l": bl, "num_stages": 0, "threads": min(256, bl)}
    raise ValueError(
        f"L={L} is not divisible by any supported block_l. "
        "L must be divisible by at least 16 for the current kernel implementation."
    )


# Training forward


def _to_cl(t: torch.Tensor) -> torch.Tensor:
    """Move (N, C, *spatial) into the (C, L) layout the prim_funcs read."""
    channels = t.shape[1]
    return t.permute(1, 0, *range(2, t.ndim)).reshape(channels, -1).contiguous()


def _from_cl(t: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Move a (C, L) result back to the shape the caller handed over."""
    batch, channels, *spatial = original_shape
    restored = t.reshape(channels, batch, *spatial)
    return restored.permute(1, 0, *range(2, restored.ndim)).contiguous()


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

    A channel is not contiguous in the ``(N, C, *spatial)`` the caller holds,
    but it is *S* elements contiguous at a time, once per batch item, which
    coalesces as well as a transposed copy would and costs no pass to build.
    Element *l* of channel *c* therefore lives at ``(l // S) * C * S + c * S +
    l % S``.

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
                    # Non-persistent path: direct global memory access avoids async-copy
                    # data race that occurs when T.copy is used inside T.Pipelined.
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
                    # Non-persistent path: direct global memory access avoids async-copy
                    # data race that occurs when T.copy is used inside T.Pipelined.
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

    One block per channel leaves most of the device idle whenever there are
    fewer channels than there are SMs, and no tile size fixes that: the grid
    is the channel count. These stages put the grid over elements instead —
    the sums are taken by *splits* blocks per channel and merged, and the
    normalisation is a flat map — so the launch is as wide as the tensor.

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
                # One accumulator per thread, merged by a fixed reduction tree.
                # Batch norm writes running statistics back, so a run has to
                # land on the same bits every time an atomic order would not.
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
                        # The affine and the normalisation fold into one
                        # multiply-add, so the map that follows reads two
                        # numbers per channel instead of four.
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

    A channel short enough to sit in one block's registers is read once, wide,
    and never written to shared memory: the same values that fed the sums are
    the ones the normalisation writes back. The traffic is then exactly a copy
    of the tensor, which is the floor for an op that must read and write it.

    The two sums are merged by a shuffle tree inside each warp and one pass
    over the per-warp totals, which is a fixed order and so gives the same
    bits every run -- batch norm writes its statistics back, and a run that
    lands on different bits is a run that cannot be reproduced.

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

    A channel is *S* elements contiguous at a time, so where the spatial extent
    is short — an ``(N, C)`` input has ``S == 1`` — a block that owns one
    channel reads one element per cache line, whatever the tile size. Giving a
    channel to a *thread* instead turns that around: neighbouring threads then
    read neighbouring channels, which is one line between them, and a channel
    short enough to sit in a thread's registers needs no reduction across
    threads, no shared memory and no barrier.
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
                # The channel's four parameters are read before the sums, not
                # after: their latency then overlaps the element loads instead
                # of standing alone at the end, and with one channel to a
                # thread there is no other warp to hide it behind.
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
            Defaults to *L*, which is right when the batch is one.
    """

    supported_archs: list[int] = [80, 89, 90]

    # Wide enough for a 16-byte access in every supported dtype.
    _SPLIT_NUM_PER_THREAD = 8

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
        self.path, self.launch = self._select_path(C, L, self.S)
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
    def _select_path(cls, C: int, L: int, S: int) -> tuple[str, object]:
        """Which launch serves this shape, and the sizing it needs.

        The four differ in what owns a channel, which is what a short channel,
        a long one, and a short spatial extent each need:

        ==========  ====================================  ==================
        path        a channel belongs to                  chosen when
        ==========  ====================================  ==================
        ``whole``   one thread, held in its registers     *S* is short
        ``wide``    one block, held in its registers      *L* fits registers
        ``split``   several blocks, summed and merged     *L* is long
        ``tiled``   one block, streamed through shared    otherwise
        ==========  ====================================  ==================
        """
        if S <= _WHOLE_MAX_S and L <= _WHOLE_MAX_L:
            return "whole", _WHOLE_BLOCK_THREADS
        wide = cls._wide_launch(L, S)
        if wide is not None:
            return "wide", wide
        if C < _SPLIT_TARGET_BLOCKS and L >= _SPLIT_MIN_L:
            return "split", max(1, min(L, -(-_SPLIT_TARGET_BLOCKS // C)))
        return "tiled", None

    @staticmethod
    def _wide_launch(L: int, S: int) -> Optional[tuple[int, int]]:
        """The ``(threads, num_per_thread)`` a register-held channel needs, or None.

        The vector must not straddle two batch items, and the channel must fit
        in the widest block the device allows.
        """
        for num_per_thread in (_WIDE_MAX_NUM_PER_THREAD, 4, 2, 1):
            if S % num_per_thread:
                continue
            # A block no wider than the warp merge stays cheap, with the step
            # count taking whatever the width does not cover. What caps the
            # steps is the register file: every step holds its own vector.
            threads = _WIDE_BLOCK_THREADS
            while threads > 32 and threads * num_per_thread >= L * 2:
                threads //= 2
            steps = -(-L // (threads * num_per_thread))
            if steps * num_per_thread <= _WIDE_MAX_HELD:
                return threads, num_per_thread
        return None

    @property
    def default_config(self) -> dict:
        if self.L <= _PERSISTENT_THRESHOLD:
            # Persistent path: block_l = L, single global read.
            t = _find_best_threads(self.L)
            return {"block_l": self.L, "threads": t}
        # Non-persistent path: find best block_l with non-power-of-2 thread counts.
        cfg = _find_best_block_l(self.L)
        return {"block_l": cfg["block_l"], "threads": cfg["threads"]}

    @property
    def autotune_configs(self) -> list[dict]:
        seen: set = set()
        configs = []

        def _add(cfg: dict) -> None:
            key = (cfg["block_l"], cfg["threads"])
            if key not in seen:
                seen.add(key)
                configs.append(cfg)

        # Persistent configs (block_l = L); power-of-2 threads only.
        if self.L <= _PERSISTENT_THRESHOLD:
            for t in [256, 128, 64, 32]:
                if self.L % t == 0:
                    _add({"block_l": self.L, "threads": t})

        # Non-persistent configs: power-of-2 threads, block_l can be non-power-of-2.
        # num_stages=0 disables T.Pipelined's async prefetch, which is required for
        # correctness in multi-tile loops (pipelining causes x_shared data shift).
        for threads in [256, 128, 64, 32]:
            for k in range(512 // threads, 0, -1):
                bl = threads * k
                if bl >= self.L or self.L % bl != 0:
                    continue
                _add({"block_l": bl, "threads": threads})

        return configs if configs else [self.default_config]

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
        num_per_thread = self._SPLIT_NUM_PER_THREAD
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
    def _bn_fwd_infer_func(blocks: int, threads: int, num_per_thread: int) -> Callable:
        # A vector of num_per_thread elements sits in one channel only when a
        # channel's run divides it; otherwise each element picks its own.
        vector_holds_one_channel = S % num_per_thread == 0
        span = threads * num_per_thread

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
            Defaults to *L*, which is right when the batch is one.
    """

    supported_archs: list[int] = [80, 89, 90]

    # Wide enough for a 16-byte access in every supported dtype.
    _MAX_NUM_PER_THREAD = 8

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

    def _blocks(self, threads: int, num_per_thread: int) -> int:
        span = threads * num_per_thread
        return (self.total + span - 1) // span

    @property
    def default_config(self) -> dict:
        # The widest access the element count will carry, so the grid is as
        # short as it can be and every load is one instruction.
        for num_per_thread in (self._MAX_NUM_PER_THREAD, 4, 2, 1):
            if self.total % num_per_thread == 0:
                return {"threads": 256, "num_per_thread": num_per_thread}
        return {"threads": 256, "num_per_thread": 1}

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for threads in (128, 256, 512, 1024):
            for num_per_thread in (8, 4, 2, 1):
                if self.total % num_per_thread:
                    continue
                configs.append({"threads": threads, "num_per_thread": num_per_thread})
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
        threads = self.config["threads"]
        num_per_thread = self.config["num_per_thread"]
        y = self.kernel(self._blocks(threads, num_per_thread), threads, num_per_thread)(
            x.contiguous().reshape(-1), weight, bias, running_mean, running_var
        )
        return y.reshape(x.shape)


# Backward


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
                    # Non-persistent path: direct global memory access avoids async-copy
                    # data race that occurs when T.copy is used inside T.Pipelined.
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
                    # Non-persistent path: direct global memory access avoids async-copy
                    # data race that occurs when T.copy is used inside T.Pipelined.
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
        if self.L <= _PERSISTENT_THRESHOLD:
            # Persistent path: block_l = L, single global read.
            # go_shared and x_shared together use 2 * L * sizeof(dtype) SMEM.
            t = _find_best_threads(self.L)
            return {"block_l": self.L, "threads": t}
        cfg = _find_best_block_l(self.L)
        return {"block_l": cfg["block_l"], "threads": cfg["threads"]}

    @property
    def autotune_configs(self) -> list[dict]:
        seen: set = set()
        configs = []

        def _add(cfg: dict) -> None:
            key = (cfg["block_l"], cfg["threads"])
            if key not in seen:
                seen.add(key)
                configs.append(cfg)

        # Persistent configs (block_l = L); power-of-2 threads only.
        if self.L <= _PERSISTENT_THRESHOLD:
            for t in [256, 128, 64, 32]:
                if self.L % t == 0:
                    _add({"block_l": self.L, "threads": t})

        # Non-persistent configs: power-of-2 threads, block_l can be non-power-of-2.
        # num_stages=0 disables pipelining for correctness in multi-tile loops.
        for threads in [256, 128, 64, 32]:
            for k in range(512 // threads, 0, -1):
                bl = threads * k
                if bl >= self.L or self.L % bl != 0:
                    continue
                _add({"block_l": bl, "threads": threads})

        return configs if configs else [self.default_config]

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ):
        """Run the backward pass on ``(N, C, *spatial)`` inputs.

        Moving the inputs into the $[C \\times L]$ layout, and ``grad_x`` back, happens here.

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
