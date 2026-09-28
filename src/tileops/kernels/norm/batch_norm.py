"""Batch Normalization kernels (training forward, inference forward, backward).

Reference: Ioffe & Szegedy (2015) https://arxiv.org/abs/1502.03167

Every kernel takes the ``(N, C, S)`` view of the ``(N, C, *spatial)`` input, with S the
product of the spatial axes. C is the channel count and L = N * S the reduction length of
one channel, whose element *l* sits at ``[l // S, c, l % S]``.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel

from .call_spec import BatchNormCall

__all__ = [
    "BatchNormBwdKernel",
    "BatchNormBwdSplitKernel",
    "BatchNormBwdWideKernel",
    "BatchNormFwdInferKernel",
    "BatchNormFwdTrainKernel",
    "BatchNormFwdTrainSplitKernel",
    "BatchNormFwdTrainWholeKernel",
    "BatchNormFwdTrainWideKernel",
]


def _vector_elements(dtype: torch.dtype) -> int:
    """Elements one thread accesses at once for a 128-bit vector in *dtype*."""
    return VECTOR_ACCESS_BYTES // dtype.itemsize


class _TiledPath:
    """A channel per block, streamed through shared memory.

    Chosen when no other path serves the shape. The training forward and the
    backward kernel both read this space: different prim_funcs, same reduction,
    same two parameters. Every bound below holds for those two only.
    """

    # Length at or below which one block holds a whole channel: x_shared costs
    # L * sizeof(dtype), 16 KB at L=8192 in fp16, and backward holds two of them.
    PERSISTENT_MAX_L = 8192

    # TileLang's AllReduce template needs a power-of-two thread count.
    REDUCE_THREADS = (256, 128, 64, 32)

    # Widest tile one block takes, bounding register pressure.
    MAX_BLOCK_L = 512

    @classmethod
    def for_length(cls, L: int) -> list[dict]:
        """The pairs accepted for *L*, best first; element zero runs untuned."""
        configs: list[dict] = []

        # One tile per channel: the whole channel is the tile.
        if L <= cls.PERSISTENT_MAX_L:
            configs += [{"block_l": L, "threads": t} for t in cls.REDUCE_THREADS if L % t == 0]

        # Several tiles per channel. block_l need not be a power of two, only
        # the thread count must, so 448 is available at L=3136 where 512 is not.
        # No pair repeats: threads separates these, and the entry above has
        # block_l == L, which this loop excludes.
        for threads in cls.REDUCE_THREADS:
            for k in range(cls.MAX_BLOCK_L // threads, 0, -1):
                block_l = threads * k
                if block_l < L and L % block_l == 0:
                    configs.append({"block_l": block_l, "threads": threads})

        if configs:
            return configs

        # No tile divides L: the last tile is masked. Below the threshold one
        # block still holds the channel, rounded up to a whole thread count.
        threads = max((t for t in cls.REDUCE_THREADS if t <= L), default=cls.REDUCE_THREADS[-1])
        block_l = -(-L // threads) * threads
        if block_l <= cls.PERSISTENT_MAX_L:
            return [{"block_l": block_l, "threads": threads}]
        return [{"block_l": cls.MAX_BLOCK_L, "threads": cls.REDUCE_THREADS[0]}]


@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_train_kernel(
    N: int,
    C: int,
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

    Element *l* of channel *c* is ``x[l // S, c, l % S]``, so the channel is read
    in place with no transposed copy.

    Persistent path (block_l >= L): after pass 1 loads all L elements into
    x_shared, pass 2 normalizes directly from x_shared — no second global read.

    Non-persistent path (block_l < L): two global reads (classic two-pass BN).

    A *block_l* that does not divide L masks the last tile; threads must divide block_l.
    """
    accum_dtype = "float32"
    L = N * S

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_train_func(block_l: int, threads: int) -> Callable:
        # A divisible length keeps the unguarded body: a predicated load costs a read.
        ragged = L % block_l != 0
        tiles = -(-L // block_l)

        @T.prim_func
        def _bn_fwd_train(
            x: T.Tensor([N, C, S], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_out: T.Tensor([C], accum_dtype),
            rstd_out: T.Tensor([C], accum_dtype),
            y: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                x_shared = T.alloc_shared([block_l], dtype)

                # One accumulator per element a thread owns, summed across tiles.
                xsum_frag = T.alloc_fragment([1, block_l], accum_dtype)
                xsq_frag = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(xsum_frag)
                T.clear(xsq_frag)

                # Pass 1 – accumulate sum(x) and sum(x^2) over all tiles. A masked
                # element loads zero, which adds nothing to either sum.
                if block_l >= L:
                    # One tile: a pipelined loop has nothing to overlap.
                    if ragged:
                        for _i, j in T.Parallel(1, block_l):
                            x_shared[j] = T.if_then_else(
                                j < L, x[j // S, bc, j % S], T.cast(0, dtype)
                            )
                    else:
                        for _i, j in T.Parallel(1, block_l):
                            x_shared[j] = x[j // S, bc, j % S]
                    for _i, j in T.Parallel(1, block_l):
                        xval = T.cast(x_shared[j], accum_dtype)
                        xsum_frag[_i, j] += xval
                        xsq_frag[_i, j] += xval * xval
                elif ragged:
                    for l_tile in T.Pipelined(tiles, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            xval = T.if_then_else(
                                l < L,
                                T.cast(x[l // S, bc, l % S], accum_dtype),
                                T.cast(0, accum_dtype),
                            )
                            xsum_frag[_i, j] += xval
                            xsq_frag[_i, j] += xval * xval
                else:
                    # T.copy inside T.Pipelined races with the async copy.
                    for l_tile in T.Pipelined(tiles, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            xval = T.cast(x[l // S, bc, l % S], accum_dtype)
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
                # running_var takes the unbiased variance, as PyTorch does.
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
                if block_l >= L and ragged:
                    # x_shared still holds the channel: no second global read.
                    for _i, j in T.Parallel(1, block_l):
                        if j < L:
                            xval = T.cast(x_shared[j], accum_dtype)
                            y[j // S, bc, j % S] = T.cast(
                                weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype
                            )
                elif block_l >= L:
                    for _i, j in T.Parallel(1, block_l):
                        xval = T.cast(x_shared[j], accum_dtype)
                        y[j // S, bc, j % S] = T.cast(
                            weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype
                        )
                elif ragged:
                    for l_tile in T.Pipelined(tiles, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            if l < L:
                                xval = T.cast(x[l // S, bc, l % S], accum_dtype)
                                y[l // S, bc, l % S] = T.cast(
                                    weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype
                                )
                else:
                    # T.copy inside T.Pipelined races with the async copy.
                    for l_tile in T.Pipelined(tiles, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            xval = T.cast(x[l // S, bc, l % S], accum_dtype)
                            y[l // S, bc, l % S] = T.cast(
                                weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype
                            )

        return _bn_fwd_train

    return _bn_fwd_train_func


@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_train_split_kernel(
    N: int,
    C: int,
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
    L = N * S

    @tilelang.jit
    def _stats_func(splits: int, threads: int, num_per_thread: int) -> Callable:
        chunk = T.ceildiv(L, splits)

        @T.prim_func
        def _bn_train_stats(
            x: T.Tensor([N, C, S], dtype),
            partial_sum: T.Tensor([C, splits], accum_dtype),
            partial_sq: T.Tensor([C, splits], accum_dtype),
        ):
            with T.Kernel(C * splits, threads=threads) as bx:
                bc = bx // splits
                start = (bx % splits) * chunk
                # The last vector step can run past this block's chunk into the next one's.
                end = T.min(start + chunk, L)
                # A fixed reduction tree: the running statistics this writes
                # back must not depend on a merge order.
                sums = T.alloc_fragment([1, threads], accum_dtype)
                sqs = T.alloc_fragment([1, threads], accum_dtype)
                T.clear(sums)
                T.clear(sqs)
                for _i, j in T.Parallel(1, threads):
                    for step in T.serial(T.ceildiv(chunk, threads * num_per_thread)):
                        for i in T.serial(num_per_thread):
                            l = start + (step * threads + j) * num_per_thread + i
                            if l < end:
                                v = T.cast(x[l // S, bc, l % S], accum_dtype)
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
                        # Folded here, the map pass reads two numbers per
                        # channel instead of four.
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
            x_ncs: T.Tensor([N, C, S], dtype),
            scale: T.Tensor([C], accum_dtype),
            shift: T.Tensor([C], accum_dtype),
            y_ncs: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(blocks, threads=threads) as bx:
                # T.reshape's size check multiplies in int32 and overflows on large tensors.
                x = T.Tensor([total], dtype, x_ncs.data)
                y = T.Tensor([total], dtype, y_ncs.data)
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
    N: int,
    C: int,
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
    L = N * S
    lanes = 32

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_train_wide_func(threads: int, num_per_thread: int) -> Callable:
        steps = (L + threads * num_per_thread - 1) // (threads * num_per_thread)
        exact = steps * threads * num_per_thread == L
        n_warps = max(threads // lanes, 1)
        # One XOR step per bit of the lane index.
        butterfly_depth = lanes.bit_length() - 1

        @T.prim_func
        def _bn_fwd_train_wide(
            x: T.Tensor([N, C, S], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_out: T.Tensor([C], accum_dtype),
            rstd_out: T.Tensor([C], accum_dtype),
            y: T.Tensor([N, C, S], dtype),
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
                        for i in T.vectorized(num_per_thread):
                            held[k * num_per_thread + i] = x[head // S, bc, head % S + i]
                        for i in T.serial(num_per_thread):
                            v = T.cast(held[k * num_per_thread + i], accum_dtype)
                            acc[0] += v
                            sq[0] += v * v
                    else:
                        for i in T.serial(num_per_thread):
                            l = head + i
                            if l < L:
                                held[k * num_per_thread + i] = x[l // S, bc, l % S]
                                v = T.cast(held[k * num_per_thread + i], accum_dtype)
                                acc[0] += v
                                sq[0] += v * v

                for step in T.serial(butterfly_depth):
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
                    # running_var takes the unbiased variance, as PyTorch does.
                    running_var[bc] = (one - mom) * params[3] + mom * (var_val * n / (n - one))

                for k in T.serial(steps):
                    head = (k * threads + tx) * num_per_thread
                    if exact or head + num_per_thread <= L:
                        for i in T.serial(num_per_thread):
                            out[i] = T.cast(
                                T.cast(held[k * num_per_thread + i], accum_dtype) * scale_val
                                + shift_val,
                                dtype,
                            )
                        for i in T.vectorized(num_per_thread):
                            y[head // S, bc, head % S + i] = out[i]
                    else:
                        for i in T.serial(num_per_thread):
                            l = head + i
                            if l < L:
                                y[l // S, bc, l % S] = T.cast(
                                    T.cast(held[k * num_per_thread + i], accum_dtype) * scale_val
                                    + shift_val,
                                    dtype,
                                )

        return _bn_fwd_train_wide

    return _bn_fwd_train_wide_func


@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_train_whole_kernel(
    N: int,
    C: int,
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
    L = N * S

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_train_whole_func(threads: int) -> Callable:
        blocks = (C + threads - 1) // threads

        @T.prim_func
        def _bn_fwd_train_whole(
            x: T.Tensor([N, C, S], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_out: T.Tensor([C], accum_dtype),
            rstd_out: T.Tensor([C], accum_dtype),
            y: T.Tensor([N, C, S], dtype),
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
                        held[l] = x[l // S, c, l % S]
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
                    # running_var takes the unbiased variance, as PyTorch does.
                    running_var[c] = (one - mom) * params[3] + mom * (var_val * n / (n - one))

                    for l in T.serial(L):
                        y[l // S, c, l % S] = T.cast(
                            T.cast(held[l], accum_dtype) * scale_val + shift_val, dtype
                        )

        return _bn_fwd_train_whole

    return _bn_fwd_train_whole_func


class _BatchNormKernel(Kernel):
    """What the BatchNorm candidates share: the policy that decides what holds one channel."""

    supported_archs: list[int] = [80, 89, 90]

    # The longest channel of one element per batch item that one thread holds.
    _THREAD_MAX_L = 32

    # Elements one thread of a register-holding block keeps, over every tensor it holds.
    _BLOCK_MAX_HELD = 256
    _HOLDING_BLOCK_THREADS = 256
    _BLOCK_MAX_THREADS = 1024
    # The widest block once the grid alone covers every SM.
    _BLOCK_MAX_THREADS_FULL_GRID = 512

    # At or above this many channels one block per channel already fills the device.
    _SPLIT_MAX_C = 1024
    _SPLIT_MIN_L = 1 << 16
    # Blocks a split grid aims for before tuning.
    _SPLIT_TARGET_BLOCKS = 512

    @classmethod
    def _block_launch(cls, call: BatchNormCall, held_tensors: int) -> Optional[tuple[int, int]]:
        """The ``(threads, num_per_thread)`` of a block holding one channel in registers.

        ``None`` where the channel does not fit. A thread holds *held_tensors* elements per
        channel element, and its vector never straddles two batch items.
        """
        L = call.n * call.spatial
        full_grid = call.sm_count <= call.c
        widest = cls._BLOCK_MAX_THREADS_FULL_GRID if full_grid else cls._BLOCK_MAX_THREADS
        vector = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        for num_per_thread in (vector >> k for k in range(vector.bit_length())):
            if call.spatial % num_per_thread:
                continue
            # Halve the block while the channel would leave half of it empty.
            threads = cls._HOLDING_BLOCK_THREADS
            while threads > 32 and threads * num_per_thread >= L * 2:
                threads //= 2
            steps = -(-L // (threads * num_per_thread))
            # Widen while a step is left partly empty and a wider block takes fewer steps;
            # a one-element thread stays, since a wider block only scatters more requests.
            while (
                num_per_thread > 1
                and threads < widest
                and steps > 1
                and steps * threads * num_per_thread != L
            ):
                threads *= 2
                steps = -(-L // (threads * num_per_thread))
            if steps * num_per_thread * held_tensors <= cls._BLOCK_MAX_HELD:
                return threads, num_per_thread
        return None

    @classmethod
    def _holder(cls, call: BatchNormCall, held_tensors: int) -> str:
        """What holds one channel of *held_tensors* tensors: the smallest of ``"thread"``,
        ``"block"``, ``"split"`` (across blocks) and ``"tiled"`` that fits it."""
        if call.spatial <= 1 and call.n * call.spatial <= cls._THREAD_MAX_L:
            return "thread"
        if cls._block_launch(call, held_tensors) is not None:
            return "block"
        if call.c < cls._SPLIT_MAX_C and call.n * call.spatial >= cls._SPLIT_MIN_L:
            return "split"
        return "tiled"

    @classmethod
    def _split_seed(cls, call: BatchNormCall) -> int:
        """Pieces a split channel is cut into before anything is measured."""
        return max(1, min(call.n * call.spatial, -(-cls._SPLIT_TARGET_BLOCKS // call.c)))


class _BatchNormFwdTrainHeldKernel(_BatchNormKernel):
    """Training forward with a channel held in registers, launched with ``self.launch``."""

    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        """Normalize an ``(N, C, S)`` input by its batch statistics.

        Returns:
            ``(y, mean, rstd)``: the ``(N, C, S)`` output and the per-channel batch
            mean and reciprocal std the backward pass reads.
        """
        self._require_cuda(
            x=x, weight=weight, bias=bias, running_mean=running_mean, running_var=running_var
        )
        mean_out = torch.empty(self.C, device=x.device, dtype=torch.float32)
        rstd_out = torch.empty_like(mean_out)
        y = self.kernel(*self.launch)(
            x, weight, bias, running_mean, running_var, mean_out, rstd_out
        )
        return y, mean_out, rstd_out


class BatchNormFwdTrainWholeKernel(_BatchNormFwdTrainHeldKernel):
    """Training forward with one channel per thread, held in its registers.

    Serves a channel one thread holds. It stays apart from the Wide program: channels map
    to threads with no cross-thread reduction, where Wide maps a channel to a block.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: Input/output data type.
        eps: Numerical stability constant.
        momentum: Running-stat update momentum.
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    # Wider than the channels it covers when there are few, so the block still has
    # enough warps to cover load latency.
    _BLOCK_THREADS = 256

    @classmethod
    def applies(cls, call: BatchNormCall) -> bool:
        return cls._holder(call, 1) == "thread"

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (call.n, call.c, call.spatial, call.dtype, call.eps, call.momentum)
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.C = C
        self.dtype = dtype
        self.launch = (self._BLOCK_THREADS,)
        self.kernel = _batch_norm_fwd_train_whole_kernel(N, C, S, self.dtype_str, eps, momentum)


class BatchNormFwdTrainWideKernel(_BatchNormFwdTrainHeldKernel):
    """Training forward with one channel per block, held in the block's registers.

    Serves a channel one block holds and one thread does not.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: Input/output data type.
        eps: Numerical stability constant.
        momentum: Running-stat update momentum.
        launch: ``(threads, num_per_thread)`` of the block.
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    @classmethod
    def applies(cls, call: BatchNormCall) -> bool:
        return cls._holder(call, 1) == "block"

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (
            call.n,
            call.c,
            call.spatial,
            call.dtype,
            call.eps,
            call.momentum,
            cls._block_launch(call, 1),
        )
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype,
        eps: float,
        momentum: float,
        launch: tuple[int, int],
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.C = C
        self.dtype = dtype
        self.launch = launch
        self.kernel = _batch_norm_fwd_train_wide_kernel(N, C, S, self.dtype_str, eps, momentum)


class BatchNormFwdTrainSplitKernel(_BatchNormKernel):
    """Training forward with a channel across several blocks: sum, merge, then map.

    Serves a channel one block does not hold and that is long enough, among few enough
    channels, to cut across blocks.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: Input/output data type.
        eps: Numerical stability constant.
        momentum: Running-stat update momentum.
        splits: Untuned pieces a channel is cut into.
        config: Optional ``{"splits", "threads"}``.
        tune: If True, time the split count and block width together.
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    # How far either side of the seed the split count is offered to the tuner.
    _SEARCH_REACH = 4
    # Relative spread within which two timings tie.
    _TIE_BAND = 0.02
    # Powers of two only: T.reduce_sum lowers to an XOR butterfly.
    _SUM_THREADS = (128, 256, 512, 1024)

    @classmethod
    def applies(cls, call: BatchNormCall) -> bool:
        return cls._holder(call, 1) == "split"

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (
            call.n,
            call.c,
            call.spatial,
            call.dtype,
            call.eps,
            call.momentum,
            cls._split_seed(call),
        )
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, tune=call.tune, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype,
        eps: float,
        momentum: float,
        splits: int,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.N = N
        self.C = C
        self.S = S
        self.L = N * S
        self.splits = splits
        self.dtype = dtype
        self.stages = _batch_norm_fwd_train_split_kernel(N, C, S, self.dtype_str, eps, momentum)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # The untuned block width is the tiled path's untuned one.
        return {
            "splits": self.splits,
            "threads": _TiledPath.for_length(self.L)[0]["threads"],
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # Powers of two either side of the seed, and the seed itself, so tuning that
        # builds no candidate has nothing to fall back to and raises.
        widest = min(self.L, self.splits * self._SEARCH_REACH)
        counts = {1 << k for k in range(widest.bit_length())} | {self.splits}
        return [
            {"splits": splits, "threads": threads}
            for splits in sorted(c for c in counts if c <= self.L)
            for threads in self._SUM_THREADS
        ]

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        """Time sum, merge and map together per ``(splits, threads)`` and keep the fastest.

        The three launches share the pair, so they are timed as one. Runs on tensors of
        its own and settles ``config`` before returning.
        """
        print(f"Start autotuning {type(self).__name__} (three launches)...")
        device = torch.cuda.current_device()
        # A caller's next random number must not depend on whether its kernel
        # was tuned.
        seed = torch.Generator(device=device)
        seed.manual_seed(0)
        x = torch.randn(self.N, self.C, self.S, device=device, dtype=self.dtype, generator=seed)
        weight = torch.ones(self.C, device=device, dtype=torch.float32)
        bias = torch.zeros(self.C, device=device, dtype=torch.float32)
        stat = functools.partial(torch.empty, self.C, device=device, dtype=torch.float32)
        timed: list[tuple[float, int, int]] = []
        refused: list[str] = []
        args: tuple = ()
        try:
            for candidate in self.autotune_configs:
                splits, threads = candidate["splits"], candidate["threads"]
                args = (x, stat(), stat(), weight, bias, stat(), stat(), splits, threads)
                try:
                    for _ in range(warmup):
                        self._run(*args)
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(rep):
                        self._run(*args)
                    end.record()
                    torch.cuda.synchronize()
                except Exception as exc:  # a pair this shape's layout refuses
                    refused.append(f"splits={splits} threads={threads}: {exc}")
                    continue
                timed.append((start.elapsed_time(end) / rep, splits, threads))
        finally:
            # The argument tuple holds the input and the last candidate's
            # scratch, so it goes too or the cache reclaims neither.
            del args, x, weight, bias
            torch.cuda.empty_cache()
        if not timed:
            raise RuntimeError(
                f"{type(self).__name__} tuning built no candidate for "
                f"C={self.C} L={self.L}: " + "; ".join(refused)
            )
        # A tie inside the guard band breaks by distance from the seed, then
        # the narrower block, then the smaller count: a total order, so a seed
        # equidistant from two counts still resolves the same way.
        floor = min(timed)[0]
        best = min(
            (c for c in timed if c[0] <= floor * (1.0 + self._TIE_BAND)),
            key=lambda c: (abs(c[1] - self.splits), c[2], c[1]),
        )
        self.config = {"splits": best[1], "threads": best[2]}
        print(f"Best config: {self.config} ({best[0]:.4f} ms)")

    def _run(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        mean_out: torch.Tensor,
        rstd_out: torch.Tensor,
        splits: int,
        threads: int,
    ) -> torch.Tensor:
        """Sum, merge, then map -- three launches over an element-wide grid."""
        stats, finalize, apply_ = self.stages
        num_per_thread = _vector_elements(self.dtype)
        empty = functools.partial(torch.empty, device=x.device, dtype=torch.float32)
        partial_sum = empty((self.C, splits))
        partial_sq = empty((self.C, splits))
        stats(splits, threads, num_per_thread)(x, partial_sum, partial_sq)
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
        blocks = (x.numel() + span - 1) // span
        return apply_(blocks, threads, num_per_thread)(x, scale, shift)

    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        """Normalize an ``(N, C, S)`` input by its batch statistics.

        Returns:
            ``(y, mean, rstd)``: the ``(N, C, S)`` output and the per-channel batch
            mean and reciprocal std the backward pass reads.
        """
        self._require_cuda(
            x=x, weight=weight, bias=bias, running_mean=running_mean, running_var=running_var
        )
        mean_out = torch.empty(self.C, device=x.device, dtype=torch.float32)
        rstd_out = torch.empty_like(mean_out)
        y = self._run(
            x,
            running_mean,
            running_var,
            weight,
            bias,
            mean_out,
            rstd_out,
            self.config["splits"],
            self.config["threads"],
        )
        return y, mean_out, rstd_out


class BatchNormFwdTrainKernel(Kernel):
    """Training forward with one channel per block, streamed through shared memory.

    The general implementation: it serves every shape the specialised ones do not.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: Input/output data type.
        eps: Numerical stability constant.
        momentum: Running-stat update momentum.
        config: Optional ``{"block_l", "threads"}``.
        tune: If True, autotune the tile config.
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    supported_archs: list[int] = [80, 89, 90]
    general: bool = True

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (call.n, call.c, call.spatial, call.dtype, call.eps, call.momentum)
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, tune=call.tune, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        momentum: float = 0.1,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.C = C
        self.L = N * S
        self.dtype = dtype
        self.kernel = _batch_norm_fwd_train_kernel(N, C, S, self.dtype_str, eps, momentum)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return _TiledPath.for_length(self.L)[0]

    @property
    def autotune_configs(self) -> list[dict]:
        return _TiledPath.for_length(self.L)

    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        """Normalize an ``(N, C, S)`` input by its batch statistics.

        Returns:
            ``(y, mean, rstd)``: the ``(N, C, S)`` output and the per-channel batch
            mean and reciprocal std the backward pass reads.
        """
        self._require_cuda(
            x=x, weight=weight, bias=bias, running_mean=running_mean, running_var=running_var
        )
        mean_out = torch.empty(self.C, device=x.device, dtype=torch.float32)
        rstd_out = torch.empty_like(mean_out)
        y = self.kernel(self.config["block_l"], self.config["threads"])(
            x, weight, bias, running_mean, running_var, mean_out, rstd_out
        )
        return y, mean_out, rstd_out


# Inference forward


@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_infer_kernel(
    N: int,
    C: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    input_dtype_params: bool = False,
    has_weight: bool = True,
    has_bias: bool = True,
) -> Callable:
    """Return the JIT-compiled inference-forward kernel factory.

    Inference reads no statistic off the input, so the whole op is one map:
    ``y = x * scale + shift`` with a scale and shift the channel picks. The
    grid is over elements, not channels, and the channel of element ``i`` is
    ``(i // S) % C`` of the flat ``(N, C, S)`` layout, so nothing is transposed
    on the way in or out.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: Input/output data type.
        eps: Numerical stability constant.
        input_dtype_params: Whether ``weight`` and ``bias`` are in *dtype* and the running
            statistics are read rounded to it, as ``instance_norm`` reads them.
        has_weight: With *input_dtype_params*, whether ``weight`` is read; else the scale is one.
        has_bias: With *input_dtype_params*, whether ``bias`` is read; else the shift is zero.
    """
    accum_dtype = "float32"
    affine_dtype = dtype if input_dtype_params else accum_dtype
    total = N * C * S

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_infer_func(threads: int, num_per_thread: int, steps: int) -> Callable:
        # A vector stays inside one channel only where the channel's run
        # divides it; otherwise each element picks its own.
        vector_holds_one_channel = S % num_per_thread == 0
        span = threads * num_per_thread * steps
        # Derived, not a parameter: the autotuner binds by name, and every
        # parameter of this builder must therefore be a config key.
        blocks = -(-total // span)

        @T.prim_func
        def _bn_fwd_infer(
            x_ncs: T.Tensor([N, C, S], dtype),
            weight: T.Tensor([C], affine_dtype),
            bias: T.Tensor([C], affine_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            y_ncs: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(blocks, threads=threads) as bx:
                # T.reshape's size check multiplies in int32 and overflows on large tensors.
                x = T.Tensor([total], dtype, x_ncs.data)
                y = T.Tensor([total], dtype, y_ncs.data)
                tx = T.get_thread_binding()
                # One scale and shift per channel turns the body into a single
                # multiply-add, and the block builds that table once rather than
                # every element recomputing a root and a divide.
                scale = T.alloc_shared([C], accum_dtype)
                shift = T.alloc_shared([C], accum_dtype)
                for c in T.serial(T.ceildiv(C, threads)):
                    ch = c * threads + tx
                    if ch < C:
                        if input_dtype_params:
                            mean_c = T.cast(T.cast(running_mean[ch], dtype), accum_dtype)
                            var_c = T.cast(T.cast(running_var[ch], dtype), accum_dtype)
                            weight_c = (
                                T.cast(weight[ch], accum_dtype) if has_weight else T.float32(1.0)
                            )
                            bias_c = T.cast(bias[ch], accum_dtype) if has_bias else T.float32(0.0)
                            sc = T.rsqrt(var_c + T.cast(eps, accum_dtype)) * weight_c
                            scale[ch] = sc
                            shift[ch] = bias_c - mean_c * sc
                        else:
                            sc = weight[ch] / T.sqrt(running_var[ch] + T.cast(eps, accum_dtype))
                            scale[ch] = sc
                            shift[ch] = bias[ch] - running_mean[ch] * sc
                T.sync_threads()

                v = T.alloc_local([num_per_thread], dtype)
                o = T.alloc_local([num_per_thread], dtype)
                # Each step is contiguous across the block.
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
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: Input/output data type.
        eps: Numerical stability constant.
        input_dtype_params: See `_batch_norm_fwd_infer_kernel`.
        has_weight: See `_batch_norm_fwd_infer_kernel`.
        has_bias: See `_batch_norm_fwd_infer_kernel`.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    supported_archs: list[int] = [80, 89, 90]
    general: bool = True

    # Elements one block covers, and its width. Each block pays the whole
    # per-channel table as a prologue, so the span fixes the block count and the
    # step count absorbs what the per-thread vector gives up.
    _BLOCK_SPAN = 2048
    _BLOCK_THREADS = 256

    # What autotune sweeps around that default: block widths, and how many
    # per-thread vectors one block walks.
    _TUNE_THREADS = (128, 256, 512, 1024)
    _TUNE_STEPS = (1, 2, 4)

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (
            call.n,
            call.c,
            call.spatial,
            call.dtype,
            call.eps,
            call.input_dtype_params,
            call.has_weight,
            call.has_bias,
        )
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, tune=call.tune, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        input_dtype_params: bool = False,
        has_weight: bool = True,
        has_bias: bool = True,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.total = N * C * S
        self.dtype = dtype
        self.kernel = _batch_norm_fwd_infer_kernel(
            N, C, S, self.dtype_str, eps, input_dtype_params, has_weight, has_bias
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # The widest vector the element count divides, with steps keeping the
        # block span at _BLOCK_SPAN.
        threads = self._BLOCK_THREADS
        vector = _vector_elements(self.dtype)
        for num_per_thread in (vector >> k for k in range(vector.bit_length())):
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
        vector = _vector_elements(self.dtype)
        for threads in self._TUNE_THREADS:
            for num_per_thread in (vector >> k for k in range(vector.bit_length())):
                if self.total % num_per_thread:
                    continue
                for steps in self._TUNE_STEPS:
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
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run inference forward pass on an ``(N, C, S)`` input.

        An absent ``weight`` or ``bias`` is handed as a placeholder the program never reads.

        Returns:
            The ``(N, C, S)`` normalized output.

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
        if weight is None or bias is None:
            placeholder = torch.empty(x.shape[1], dtype=x.dtype, device=x.device)
            weight = placeholder if weight is None else weight
            bias = placeholder if bias is None else bias
        return self.kernel(
            self.config["threads"], self.config["num_per_thread"], self.config["steps"]
        )(x, weight, bias, running_mean, running_var)


# Backward


@functools.lru_cache(maxsize=32)
def _batch_norm_bwd_kernel(
    N: int,
    C: int,
    S: int,
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

    Element *l* of channel *c* is ``[l // S, c, l % S]`` of every tensor.

    Persistent path (block_l >= L): after pass 1 accumulates grad_bias /
    grad_weight while loading grad_out and x into shared memory, pass 2 computes
    grad_x directly from shared memory — eliminates the second global read.

    Non-persistent path (block_l < L): two global reads (classic two-pass BN bwd).

    A *block_l* that does not divide L masks the last tile.
    """
    accum_dtype = "float32"
    L = N * S

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_bwd_func(block_l: int, threads: int) -> Callable:
        # A divisible length keeps the unguarded body: a predicated load costs a read.
        ragged = L % block_l != 0
        tiles = -(-L // block_l)

        @T.prim_func
        def _bn_bwd(
            grad_out: T.Tensor([N, C, S], dtype),
            x: T.Tensor([N, C, S], dtype),
            weight: T.Tensor([C], accum_dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
            grad_x: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                go_shared = T.alloc_shared([block_l], dtype)
                x_shared = T.alloc_shared([block_l], dtype)

                mean_val = mean[bc]
                rstd_val = rstd[bc]
                w_val = weight[bc]

                # Accumulators for sum(grad_out) and sum(grad_out * x_hat). A masked
                # element loads a zero gradient, which adds nothing to either sum.
                do_frag = T.alloc_fragment([1, block_l], accum_dtype)
                do_xhat_frag = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(do_frag)
                T.clear(do_xhat_frag)

                # Pass 1 – accumulate grad_bias and grad_weight contributions.
                if block_l >= L:
                    # One tile: a pipelined loop has nothing to overlap.
                    if ragged:
                        for _i, j in T.Parallel(1, block_l):
                            go_shared[j] = T.if_then_else(
                                j < L, grad_out[j // S, bc, j % S], T.cast(0, dtype)
                            )
                            x_shared[j] = T.if_then_else(
                                j < L, x[j // S, bc, j % S], T.cast(0, dtype)
                            )
                        # A padded lane adds nothing, whatever its normalized value would be.
                        for _i, j in T.Parallel(1, block_l):
                            go_val = T.cast(go_shared[j], accum_dtype)
                            x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                            zero = T.cast(0, accum_dtype)
                            do_frag[_i, j] += go_val
                            do_xhat_frag[_i, j] += T.if_then_else(j < L, go_val * x_hat, zero)
                    else:
                        for _i, j in T.Parallel(1, block_l):
                            go_shared[j] = grad_out[j // S, bc, j % S]
                            x_shared[j] = x[j // S, bc, j % S]
                        for _i, j in T.Parallel(1, block_l):
                            go_val = T.cast(go_shared[j], accum_dtype)
                            x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                            do_frag[_i, j] += go_val
                            do_xhat_frag[_i, j] += go_val * x_hat
                else:
                    # T.copy inside T.Pipelined races with the async copy.
                    for l_tile in T.Pipelined(tiles, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            if ragged:
                                go_val = T.if_then_else(
                                    l < L,
                                    T.cast(grad_out[l // S, bc, l % S], accum_dtype),
                                    T.cast(0, accum_dtype),
                                )
                                x_val = T.if_then_else(
                                    l < L, T.cast(x[l // S, bc, l % S], accum_dtype), mean_val
                                )
                            else:
                                go_val = T.cast(grad_out[l // S, bc, l % S], accum_dtype)
                                x_val = T.cast(x[l // S, bc, l % S], accum_dtype)
                            x_hat = (x_val - mean_val) * rstd_val
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
                    # Both shared buffers still hold the channel: no second read.
                    for _i, j in T.Parallel(1, block_l):
                        go_val = T.cast(go_shared[j], accum_dtype)
                        x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                        gx = w_rstd_over_L * (
                            T.cast(L, accum_dtype) * go_val - sum_do[0] - x_hat * sum_do_xhat[0]
                        )
                        if ragged:
                            if j < L:
                                grad_x[j // S, bc, j % S] = T.cast(gx, dtype)
                        else:
                            grad_x[j // S, bc, j % S] = T.cast(gx, dtype)
                else:
                    # T.copy inside T.Pipelined races with the async copy.
                    for l_tile in T.Pipelined(tiles, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            if ragged:
                                if l < L:
                                    go_val = T.cast(grad_out[l // S, bc, l % S], accum_dtype)
                                    x_hat = (
                                        T.cast(x[l // S, bc, l % S], accum_dtype) - mean_val
                                    ) * rstd_val
                                    gx = w_rstd_over_L * (
                                        T.cast(L, accum_dtype) * go_val
                                        - sum_do[0]
                                        - x_hat * sum_do_xhat[0]
                                    )
                                    grad_x[l // S, bc, l % S] = T.cast(gx, dtype)
                            else:
                                go_val = T.cast(grad_out[l // S, bc, l % S], accum_dtype)
                                x_hat = (
                                    T.cast(x[l // S, bc, l % S], accum_dtype) - mean_val
                                ) * rstd_val
                                gx = w_rstd_over_L * (
                                    T.cast(L, accum_dtype) * go_val
                                    - sum_do[0]
                                    - x_hat * sum_do_xhat[0]
                                )
                                grad_x[l // S, bc, l % S] = T.cast(gx, dtype)

        return _bn_bwd

    return _bn_bwd_func


@functools.lru_cache(maxsize=32)
def _batch_norm_bwd_wide_kernel(
    N: int,
    C: int,
    S: int,
    dtype: str = "float16",
) -> Callable:
    """Return the JIT-compiled backward factory for a register-held channel.

    The sums merge by a fixed shuffle tree, so ``grad_weight`` and ``grad_bias``
    are the same on every run. ``num_per_thread`` must divide *S*, so a vector
    never straddles two batch items; ``threads`` is a multiple of the warp width.
    """
    accum_dtype = "float32"
    L = N * S
    lanes = 32

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_bwd_wide_func(threads: int, num_per_thread: int) -> Callable:
        steps = (L + threads * num_per_thread - 1) // (threads * num_per_thread)
        exact = steps * threads * num_per_thread == L
        n_warps = max(threads // lanes, 1)
        butterfly_depth = lanes.bit_length() - 1

        @T.prim_func
        def _bn_bwd_wide(
            grad_out: T.Tensor([N, C, S], dtype),
            x: T.Tensor([N, C, S], dtype),
            weight: T.Tensor([C], accum_dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
            grad_x: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(C, threads=threads) as bc:
                tx = T.get_thread_binding()
                # Read before the sums so the latency overlaps the element loads.
                params = T.alloc_local([3], accum_dtype)
                params[0] = weight[bc]
                params[1] = mean[bc]
                params[2] = rstd[bc]
                go_held = T.alloc_local([steps * num_per_thread], dtype)
                x_held = T.alloc_local([steps * num_per_thread], dtype)
                out = T.alloc_local([num_per_thread], dtype)
                acc = T.alloc_local([1], accum_dtype)
                acc_xhat = T.alloc_local([1], accum_dtype)
                acc[0] = T.cast(0, accum_dtype)
                acc_xhat[0] = T.cast(0, accum_dtype)

                for k in T.serial(steps):
                    head = (k * threads + tx) * num_per_thread
                    if exact or head + num_per_thread <= L:
                        for i in T.vectorized(num_per_thread):
                            go_held[k * num_per_thread + i] = grad_out[head // S, bc, head % S + i]
                        for i in T.vectorized(num_per_thread):
                            x_held[k * num_per_thread + i] = x[head // S, bc, head % S + i]
                        for i in T.serial(num_per_thread):
                            g = T.cast(go_held[k * num_per_thread + i], accum_dtype)
                            x_hat = (
                                T.cast(x_held[k * num_per_thread + i], accum_dtype) - params[1]
                            ) * params[2]
                            acc[0] += g
                            acc_xhat[0] += g * x_hat
                    else:
                        for i in T.serial(num_per_thread):
                            l = head + i
                            if l < L:
                                go_held[k * num_per_thread + i] = grad_out[l // S, bc, l % S]
                                x_held[k * num_per_thread + i] = x[l // S, bc, l % S]
                                g = T.cast(go_held[k * num_per_thread + i], accum_dtype)
                                x_hat = (
                                    T.cast(x_held[k * num_per_thread + i], accum_dtype) - params[1]
                                ) * params[2]
                                acc[0] += g
                                acc_xhat[0] += g * x_hat

                for step in T.serial(butterfly_depth):
                    acc[0] += T.shfl_xor(acc[0], T.shift_left(1, step))
                    acc_xhat[0] += T.shfl_xor(acc_xhat[0], T.shift_left(1, step))

                warp_sum = T.alloc_shared([n_warps], accum_dtype)
                warp_sum_xhat = T.alloc_shared([n_warps], accum_dtype)
                if tx % lanes == 0:
                    warp_sum[tx // lanes] = acc[0]
                    warp_sum_xhat[tx // lanes] = acc_xhat[0]
                T.sync_threads()
                acc[0] = T.cast(0, accum_dtype)
                acc_xhat[0] = T.cast(0, accum_dtype)
                for w in T.serial(n_warps):
                    acc[0] += warp_sum[w]
                    acc_xhat[0] += warp_sum_xhat[w]

                if tx == 0:
                    grad_bias[bc] = acc[0]
                    grad_weight[bc] = acc_xhat[0]

                # grad_x = w * rstd * grad_out - (w * rstd / L) * (sum + x_hat * sum_xhat).
                scale_val = params[0] * params[2]
                per_elem = scale_val / T.cast(L, accum_dtype)
                shift_val = -per_elem * acc[0]
                xhat_coef = per_elem * acc_xhat[0]

                for k in T.serial(steps):
                    head = (k * threads + tx) * num_per_thread
                    if exact or head + num_per_thread <= L:
                        for i in T.serial(num_per_thread):
                            x_hat = (
                                T.cast(x_held[k * num_per_thread + i], accum_dtype) - params[1]
                            ) * params[2]
                            out[i] = T.cast(
                                scale_val * T.cast(go_held[k * num_per_thread + i], accum_dtype)
                                + shift_val
                                - xhat_coef * x_hat,
                                dtype,
                            )
                        for i in T.vectorized(num_per_thread):
                            grad_x[head // S, bc, head % S + i] = out[i]
                    else:
                        for i in T.serial(num_per_thread):
                            l = head + i
                            if l < L:
                                x_hat = (
                                    T.cast(x_held[k * num_per_thread + i], accum_dtype) - params[1]
                                ) * params[2]
                                grad_x[l // S, bc, l % S] = T.cast(
                                    scale_val * T.cast(go_held[k * num_per_thread + i], accum_dtype)
                                    + shift_val
                                    - xhat_coef * x_hat,
                                    dtype,
                                )

        return _bn_bwd_wide

    return _bn_bwd_wide_func


@functools.lru_cache(maxsize=32)
def _batch_norm_bwd_split_kernel(
    N: int,
    C: int,
    S: int,
    dtype: str = "float16",
) -> Callable:
    """Return the three-stage backward factories for a long channel.

    *splits* blocks per channel sum, one block merges the partial sums into the
    channel gradients and ``grad_x`` coefficients, and a flat map applies them.

    Returns:
        A ``(stats, finalize, apply)`` triple of JIT factories.
    """
    accum_dtype = "float32"
    L = N * S

    @tilelang.jit
    def _stats_func(splits: int, threads: int, num_per_thread: int) -> Callable:
        chunk = T.ceildiv(L, splits)

        @T.prim_func
        def _bn_bwd_stats(
            grad_out: T.Tensor([N, C, S], dtype),
            x: T.Tensor([N, C, S], dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            partial_sum: T.Tensor([C, splits], accum_dtype),
            partial_sum_xhat: T.Tensor([C, splits], accum_dtype),
        ):
            with T.Kernel(C * splits, threads=threads) as bx:
                bc = bx // splits
                start = (bx % splits) * chunk
                # The last vector step can run past this block's chunk into the next one's.
                end = T.min(start + chunk, L)
                mean_val = mean[bc]
                rstd_val = rstd[bc]
                # A fixed reduction tree: the channel gradients must not depend
                # on a merge order.
                sums = T.alloc_fragment([1, threads], accum_dtype)
                sums_xhat = T.alloc_fragment([1, threads], accum_dtype)
                T.clear(sums)
                T.clear(sums_xhat)
                for _i, j in T.Parallel(1, threads):
                    for step in T.serial(T.ceildiv(chunk, threads * num_per_thread)):
                        for i in T.serial(num_per_thread):
                            l = start + (step * threads + j) * num_per_thread + i
                            if l < end:
                                g = T.cast(grad_out[l // S, bc, l % S], accum_dtype)
                                x_hat = (
                                    T.cast(x[l // S, bc, l % S], accum_dtype) - mean_val
                                ) * rstd_val
                                sums[_i, j] += g
                                sums_xhat[_i, j] += g * x_hat
                sum_result = T.alloc_fragment([1], accum_dtype)
                sum_xhat_result = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(sums, sum_result, dim=1)
                T.reduce_sum(sums_xhat, sum_xhat_result, dim=1)
                if T.get_thread_binding() == 0:
                    partial_sum[bc, bx % splits] = sum_result[0]
                    partial_sum_xhat[bc, bx % splits] = sum_xhat_result[0]

        return _bn_bwd_stats

    @tilelang.jit
    def _finalize_func(splits: int, threads: int) -> Callable:
        @T.prim_func
        def _bn_bwd_finalize(
            partial_sum: T.Tensor([C, splits], accum_dtype),
            partial_sum_xhat: T.Tensor([C, splits], accum_dtype),
            weight: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
            scale_out: T.Tensor([C], accum_dtype),
            shift_out: T.Tensor([C], accum_dtype),
            centered_coef_out: T.Tensor([C], accum_dtype),
        ):
            with T.Kernel(1, threads=threads) as _:
                tx = T.get_thread_binding()
                for step in T.serial(T.ceildiv(C, threads)):
                    bc = step * threads + tx
                    if bc < C:
                        total = T.alloc_local([1], accum_dtype)
                        total_xhat = T.alloc_local([1], accum_dtype)
                        total[0] = T.cast(0, accum_dtype)
                        total_xhat[0] = T.cast(0, accum_dtype)
                        for k in T.serial(splits):
                            total[0] += partial_sum[bc, k]
                            total_xhat[0] += partial_sum_xhat[bc, k]
                        grad_bias[bc] = total[0]
                        grad_weight[bc] = total_xhat[0]
                        # grad_x = scale * grad_out + shift - centered_coef * (x - mean).
                        rstd_val = rstd[bc]
                        scale_val = weight[bc] * rstd_val
                        per_elem = scale_val / T.cast(L, accum_dtype)
                        scale_out[bc] = scale_val
                        shift_out[bc] = -per_elem * total[0]
                        centered_coef_out[bc] = per_elem * total_xhat[0] * rstd_val

        return _bn_bwd_finalize

    @tilelang.jit(out_idx=[-1])
    def _apply_func(blocks: int, threads: int, num_per_thread: int) -> Callable:
        vector_holds_one_channel = S % num_per_thread == 0
        span = threads * num_per_thread
        total = C * L

        @T.prim_func
        def _bn_bwd_apply(
            grad_out_ncs: T.Tensor([N, C, S], dtype),
            x_ncs: T.Tensor([N, C, S], dtype),
            mean: T.Tensor([C], accum_dtype),
            scale: T.Tensor([C], accum_dtype),
            shift: T.Tensor([C], accum_dtype),
            centered_coef: T.Tensor([C], accum_dtype),
            grad_x_ncs: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(blocks, threads=threads) as bx:
                # T.reshape's size check multiplies in int32 and overflows on large tensors.
                grad_out = T.Tensor([total], dtype, grad_out_ncs.data)
                x = T.Tensor([total], dtype, x_ncs.data)
                grad_x = T.Tensor([total], dtype, grad_x_ncs.data)
                tx = T.get_thread_binding()
                g = T.alloc_local([num_per_thread], dtype)
                v = T.alloc_local([num_per_thread], dtype)
                o = T.alloc_local([num_per_thread], dtype)
                base = bx * span + tx * num_per_thread
                if base + num_per_thread <= total:
                    for i in T.vectorized(num_per_thread):
                        g[i] = grad_out[base + i]
                    for i in T.vectorized(num_per_thread):
                        v[i] = x[base + i]
                    if vector_holds_one_channel:
                        ch = (base // S) % C
                        for i in T.serial(num_per_thread):
                            o[i] = T.cast(
                                scale[ch] * T.cast(g[i], accum_dtype)
                                + shift[ch]
                                - centered_coef[ch] * (T.cast(v[i], accum_dtype) - mean[ch]),
                                dtype,
                            )
                    else:
                        for i in T.serial(num_per_thread):
                            ch = ((base + i) // S) % C
                            o[i] = T.cast(
                                scale[ch] * T.cast(g[i], accum_dtype)
                                + shift[ch]
                                - centered_coef[ch] * (T.cast(v[i], accum_dtype) - mean[ch]),
                                dtype,
                            )
                    for i in T.vectorized(num_per_thread):
                        grad_x[base + i] = o[i]
                else:
                    for i in T.serial(num_per_thread):
                        if base + i < total:
                            ch = ((base + i) // S) % C
                            grad_x[base + i] = T.cast(
                                scale[ch] * T.cast(grad_out[base + i], accum_dtype)
                                + shift[ch]
                                - centered_coef[ch] * (T.cast(x[base + i], accum_dtype) - mean[ch]),
                                dtype,
                            )

        return _bn_bwd_apply

    return _stats_func, _finalize_func, _apply_func


class BatchNormBwdWideKernel(_BatchNormKernel):
    """Backward with one channel per block, ``grad_out`` and ``x`` held in its registers.

    Serves a channel whose two tensors one block holds, a thread-held one included.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: grad_out/x/grad_x data type.
        launch: ``(threads, num_per_thread)`` of the block.
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    @classmethod
    def applies(cls, call: BatchNormCall) -> bool:
        return cls._holder(call, 2) in ("thread", "block")

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (call.n, call.c, call.spatial, call.dtype, cls._block_launch(call, 2))
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype,
        launch: tuple[int, int],
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.C = C
        self.dtype = dtype
        self.launch = launch
        self.kernel = _batch_norm_bwd_wide_kernel(N, C, S, self.dtype_str)

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ):
        """Run the backward pass on ``(N, C, S)`` inputs.

        Returns:
            ``(grad_x, grad_weight, grad_bias)``, ``grad_x`` in ``(N, C, S)``.
        """
        self._require_cuda(grad_out=grad_out, x=x, weight=weight, mean=mean, rstd=rstd)
        grad_weight = torch.empty(self.C, device=x.device, dtype=torch.float32)
        grad_bias = torch.empty_like(grad_weight)
        grad_x = self.kernel(*self.launch)(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)
        return grad_x, grad_weight, grad_bias


class BatchNormBwdSplitKernel(_BatchNormKernel):
    """Backward with a channel across several blocks: sum, merge, then map.

    Serves a channel whose two tensors one block does not hold and that is long enough,
    among few enough channels, to cut across blocks. The split count and block width
    follow from the shape.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: grad_out/x/grad_x data type.
        splits: Pieces a channel is cut into.
        config: Optional ``{"splits", "threads"}``.
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    @classmethod
    def applies(cls, call: BatchNormCall) -> bool:
        return cls._holder(call, 2) == "split"

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (call.n, call.c, call.spatial, call.dtype, cls._split_seed(call))
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype,
        splits: int,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.C = C
        self.L = N * S
        self.splits = splits
        self.dtype = dtype
        self.stages = _batch_norm_bwd_split_kernel(N, C, S, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # The block width is the tiled path's untuned one.
        return {
            "splits": self.splits,
            "threads": _TiledPath.for_length(self.L)[0]["threads"],
        }

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ):
        """Run the backward pass on ``(N, C, S)`` inputs.

        Returns:
            ``(grad_x, grad_weight, grad_bias)``, ``grad_x`` in ``(N, C, S)``.
        """
        self._require_cuda(grad_out=grad_out, x=x, weight=weight, mean=mean, rstd=rstd)
        grad_weight = torch.empty(self.C, device=x.device, dtype=torch.float32)
        grad_bias = torch.empty_like(grad_weight)
        stats, finalize, apply_ = self.stages
        splits, threads = self.config["splits"], self.config["threads"]
        num_per_thread = _vector_elements(self.dtype)
        empty = functools.partial(torch.empty, device=x.device, dtype=torch.float32)
        partial_sum = empty((self.C, splits))
        partial_sum_xhat = empty((self.C, splits))
        stats(splits, threads, num_per_thread)(
            grad_out, x, mean, rstd, partial_sum, partial_sum_xhat
        )
        scale, shift, centered_coef = empty(self.C), empty(self.C), empty(self.C)
        finalize(splits, min(256, self.C))(
            partial_sum,
            partial_sum_xhat,
            weight,
            rstd,
            grad_weight,
            grad_bias,
            scale,
            shift,
            centered_coef,
        )
        span = threads * num_per_thread
        blocks = (x.numel() + span - 1) // span
        grad_x = apply_(blocks, threads, num_per_thread)(
            grad_out, x, mean, scale, shift, centered_coef
        )
        return grad_x, grad_weight, grad_bias


class BatchNormBwdKernel(Kernel):
    """Backward with one channel per block, streamed through shared memory.

    The general implementation: it serves every shape the specialised ones do not.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: grad_out/x/grad_x data type.
        config: Optional ``{"block_l", "threads"}``.
        tune: If True, autotune the tile config.
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    supported_archs: list[int] = [80, 89, 90]
    general: bool = True

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (call.n, call.c, call.spatial, call.dtype)
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, tune=call.tune, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.C = C
        self.L = N * S
        self.dtype = dtype
        self.kernel = _batch_norm_bwd_kernel(N, C, S, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return _TiledPath.for_length(self.L)[0]

    @property
    def autotune_configs(self) -> list[dict]:
        return _TiledPath.for_length(self.L)

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ):
        """Run the backward pass on ``(N, C, S)`` inputs.

        Returns:
            ``(grad_x, grad_weight, grad_bias)``, ``grad_x`` in ``(N, C, S)``.
        """
        self._require_cuda(grad_out=grad_out, x=x, weight=weight, mean=mean, rstd=rstd)
        grad_weight = torch.empty(self.C, device=x.device, dtype=torch.float32)
        grad_bias = torch.empty_like(grad_weight)
        program = self.kernel(self.config["block_l"], self.config["threads"])
        grad_x = program(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)
        return grad_x, grad_weight, grad_bias
