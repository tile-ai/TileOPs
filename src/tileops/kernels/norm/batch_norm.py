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
from tileops.utils import get_sm_count

__all__ = [
    "BatchNormBwdKernel",
    "BatchNormFwdInferKernel",
    "BatchNormFwdTrainKernel",
]


def _vector_elements(dtype: torch.dtype) -> int:
    """Elements one thread accesses at once for a 128-bit vector in *dtype*."""
    return VECTOR_ACCESS_BYTES // dtype.itemsize


def _widths_down_to_one(widest: int) -> tuple[int, ...]:
    """*widest* and every halving of it, down to one element."""
    return tuple(widest >> k for k in range(widest.bit_length()))


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

    # Tile widths tried when no reduce width divides the channel, widest first.
    FALLBACK_BLOCK_L = (512, 256, 128, 64, 32, 16)

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

        # No reduce width divides L. Below the threshold the channel is still
        # one tile, at the narrowest width; above it, the widest tile that does
        # divide L.
        if L <= cls.PERSISTENT_MAX_L:
            return [{"block_l": L, "threads": cls.REDUCE_THREADS[-1]}]
        for block_l in cls.FALLBACK_BLOCK_L:
            if L % block_l == 0:
                return [{"block_l": block_l, "threads": min(cls.REDUCE_THREADS[0], block_l)}]
        raise ValueError(
            f"L={L} is not divisible by any supported block_l. L must be divisible "
            f"by at least {cls.FALLBACK_BLOCK_L[-1]}."
        )


class _WholePath:
    """A channel per thread, held in its registers.

    Chosen where a channel is one element per batch item: a block owning one
    channel would then get one useful element per cache line, whatever the tile
    size. The block is wider than the channels it covers when there are few, so
    it still launches enough warps to cover load latency. Bounds are measured
    crossovers, not derived.
    """

    MAX_S = 1
    MAX_L = 32
    BLOCK_THREADS = 256

    # Channel counts and channel lengths from which the narrow block wins. The
    # two bounds hold for different reasons. A narrow block wastes half of every
    # transaction -- a warp covers 32 channels, 64 bytes of a 128-byte line --
    # and that costs the same at every channel count while what it buys grows:
    # 768 channels fill three blocks of the default width against twelve narrow
    # ones. The length bound is the register footprint, threads * L elements per
    # block: only from eight elements per channel does it cap how many blocks of
    # the default width an SM hosts.
    SPREAD_MIN_C = 768
    SPREAD_MIN_L = 8
    SPREAD_BLOCK_THREADS = 64

    @classmethod
    def launch(cls, L: int, S: int, C: int) -> Optional[int]:
        """The block width this path needs, or None where it does not serve."""
        if S > cls.MAX_S or L > cls.MAX_L:
            return None
        if C >= cls.SPREAD_MIN_C and L >= cls.SPREAD_MIN_L:
            return cls.SPREAD_BLOCK_THREADS
        return cls.BLOCK_THREADS


class _WidePath:
    """A channel per block, held in the block's registers across its steps.

    Chosen where the channel fits those registers. Past MAX_HELD the spills cost
    more than the second global read the other paths pay. Bounds are measured
    crossovers, not derived.
    """

    BLOCK_THREADS = 256
    MAX_HELD = 256
    MAX_BLOCK_THREADS = 1024

    # A grid that already covers the device stops at the narrower block: past it
    # a block's warps compete for one SM instead of filling an idle one.
    MAX_BLOCK_THREADS_FULL_GRID = 512

    @classmethod
    def launch(cls, C: int, L: int, S: int, dtype: torch.dtype) -> Optional[tuple[int, int]]:
        """The ``(threads, num_per_thread)`` this path needs, or None.

        The vector must not straddle two batch items, and the channel must fit
        in the widest block *C* allows.
        """
        widest = cls.MAX_BLOCK_THREADS_FULL_GRID if get_sm_count() <= C else cls.MAX_BLOCK_THREADS
        for num_per_thread in _widths_down_to_one(_vector_elements(dtype)):
            if S % num_per_thread:
                continue
            # Halve the width while the channel would leave half the block empty;
            # what the width does not cover becomes steps, which MAX_HELD caps.
            threads = cls.BLOCK_THREADS
            while threads > 32 and threads * num_per_thread >= L * 2:
                threads //= 2
            steps = -(-L // (threads * num_per_thread))
            # A step the channel does not fill still costs a whole pass of the
            # load loop and of the store loop. Widen while such a step is left
            # and a wider block takes fewer of them. A thread reading one
            # element at a time is left alone: a wider block then issues more
            # scattered requests rather than more vectors in flight.
            while (
                num_per_thread > 1
                and threads < widest
                and steps > 1
                and steps * threads * num_per_thread != L
            ):
                threads *= 2
                steps = -(-L // (threads * num_per_thread))
            if steps * num_per_thread <= cls.MAX_HELD:
                return threads, num_per_thread
        return None


class _SplitPath:
    """A channel across several blocks, summed and merged by three launches.

    Chosen for a channel long enough that one block per channel is too narrow a
    grid, whatever the tile size. Bounds are measured crossovers, not derived.
    """

    # Above MAX_C one block per channel already fills the device and the split
    # has nothing to add.
    MAX_C = 1024
    MIN_L = 1 << 16

    # Blocks the path aims for: a grid this size still covers the device several
    # times over, and the longer piece each block walks reads faster.
    TARGET_BLOCKS = 512

    # How far either side of the seed the split count is offered to the tuner.
    SEARCH_REACH = 4

    # Spread within which two candidates are one answer: the tuner times the
    # launches back to back while the benchmark clears L2 between them.
    TIE_BAND = 0.02

    # Block widths the sums are tuned over, powers of two only: T.reduce_sum
    # lowers to an XOR butterfly.
    SUM_THREADS = (128, 256, 512, 1024)

    @classmethod
    def admits(cls, C: int, L: int) -> bool:
        """Whether this path serves the shape."""
        return C < cls.MAX_C and L >= cls.MIN_L

    @classmethod
    def seed(cls, C: int, L: int) -> int:
        """Pieces a channel is cut into before anything is measured."""
        return max(1, min(L, -(-cls.TARGET_BLOCKS // C)))

    @classmethod
    def candidates(cls, C: int, L: int) -> list[int]:
        """Split counts to offer the tuner: powers of two either side of the seed.

        The seed is always a member, which is why tuning raises rather than
        falling back when every candidate refuses: the count path selection
        chose was among them, so nothing is left to fall back to.
        """
        widest = min(L, cls.seed(C, L) * cls.SEARCH_REACH)
        counts, count = [], 1
        while count <= widest:
            counts.append(count)
            count *= 2
        counts.append(cls.seed(C, L))
        return sorted({c for c in counts if c <= L})


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

                # One accumulator per element a thread owns, summed across tiles.
                xsum_frag = T.alloc_fragment([1, block_l], accum_dtype)
                xsq_frag = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(xsum_frag)
                T.clear(xsq_frag)

                # Pass 1 – accumulate sum(x) and sum(x^2) over all tiles.
                if block_l >= L:
                    # One tile: a pipelined loop has nothing to overlap.
                    for _i, j in T.Parallel(1, block_l):
                        x_shared[j] = x[(j // S) * plane + bc * S + j % S]
                    for _i, j in T.Parallel(1, block_l):
                        xval = T.cast(x_shared[j], accum_dtype)
                        xsum_frag[_i, j] += xval
                        xsq_frag[_i, j] += xval * xval
                else:
                    # T.copy inside T.Pipelined races with the async copy.
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
                if block_l >= L:
                    # x_shared still holds the channel: no second global read.
                    for _i, j in T.Parallel(1, block_l):
                        xval = T.cast(x_shared[j], accum_dtype)
                        y[(j // S) * plane + bc * S + j % S] = T.cast(
                            weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype
                        )
                else:
                    # T.copy inside T.Pipelined races with the async copy.
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
        # One XOR step per bit of the lane index.
        butterfly_depth = lanes.bit_length() - 1

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
                    # running_var takes the unbiased variance, as PyTorch does.
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
        if L == 1:
            # Every path folds Bessel's correction, whose divisor is L - 1.
            raise ValueError(
                "BatchNormFwdTrainKernel needs more than one value per channel, got L=1"
            )
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

    @staticmethod
    def _select_path(C: int, L: int, S: int, dtype: torch.dtype) -> tuple[str, object]:
        """Which launch serves this shape, and the sizing it needs.

        Each path class states what owns a channel there and when it is chosen.
        """
        whole = _WholePath.launch(L, S, C)
        if whole is not None:
            return "whole", whole
        wide = _WidePath.launch(C, L, S, dtype)
        if wide is not None:
            return "wide", wide
        if _SplitPath.admits(C, L):
            return "split", _SplitPath.seed(C, L)
        return "tiled", None

    @property
    def default_config(self) -> dict:
        return _TiledPath.for_length(self.L)[0]

    @property
    def autotune_configs(self) -> list[dict]:
        return _TiledPath.for_length(self.L)

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        """Tune the kernel this shape's path actually launches.

        The base class tunes ``self.kernel``, the tiled builder. The whole and
        wide paths take their launch from the shape and read nothing off the
        config. The split path reads both the split count and the block width,
        and both decide three launches, so all three are timed together.
        """
        if self.path in ("whole", "wide"):
            self.config = self.default_config
            return
        if self.path == "split":
            self._tune_split(warmup=warmup, rep=rep)
            return
        super().autotune(warmup=warmup, rep=rep)

    def _tune_split(self, warmup: int, rep: int) -> None:
        """Pick the split count and block width by timing sum, merge and map.

        All three read the same pair, and they do not want the same one: the
        width that reads a channel fastest costs the map pass more than it saves,
        because the map writes the tensor the sums only read. Timing one stage
        settles the other two against it, so the three are timed together.

        Runs on tensors of its own, so nothing a caller holds is touched, and
        settles the config before returning -- a caller that asks for tuning and
        then reads ``config`` sees what will run.
        """
        print(f"Start autotuning {type(self).__name__} (split, three launches)...")
        device = torch.cuda.current_device()
        # A caller's next random number must not depend on whether its kernel
        # was tuned.
        seed = torch.Generator(device=device)
        seed.manual_seed(0)
        flat = torch.randn(self.C * self.L, device=device, dtype=self.dtype, generator=seed)
        weight = torch.ones(self.C, device=device, dtype=torch.float32)
        bias = torch.zeros(self.C, device=device, dtype=torch.float32)
        stat = functools.partial(torch.empty, self.C, device=device, dtype=torch.float32)
        timed: list[tuple[float, int, int]] = []
        refused: list[str] = []
        args: tuple = ()
        try:
            for splits in _SplitPath.candidates(self.C, self.L):
                for threads in _SplitPath.SUM_THREADS:
                    args = (flat, stat(), stat(), weight, bias, stat(), stat(), splits, threads)
                    try:
                        for _ in range(warmup):
                            self._run_split(*args)
                        torch.cuda.synchronize()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(rep):
                            self._run_split(*args)
                        end.record()
                        torch.cuda.synchronize()
                    except Exception as exc:  # a pair this shape's layout refuses
                        refused.append(f"splits={splits} threads={threads}: {exc}")
                        continue
                    timed.append((start.elapsed_time(end) / rep, splits, threads))
        finally:
            # The argument tuple holds the input and the last candidate's
            # scratch, so it goes too or the cache reclaims neither.
            del args, flat, weight, bias
            torch.cuda.empty_cache()
        if not timed:
            raise RuntimeError(
                f"{type(self).__name__} split tuning built no candidate for "
                f"C={self.C} L={self.L}: " + "; ".join(refused)
            )
        # A tie inside the guard band breaks by distance from the seed, then
        # the narrower block, then the smaller count: a total order, so a seed
        # equidistant from two counts still resolves the same way.
        floor = min(timed)[0]
        seed_splits = _SplitPath.seed(self.C, self.L)
        best = min(
            (c for c in timed if c[0] <= floor * (1.0 + _SplitPath.TIE_BAND)),
            key=lambda c: (abs(c[1] - seed_splits), c[2], c[1]),
        )
        self.launch = best[1]
        self.config = dict(self.default_config, threads=best[2])
        print(f"Best config: {self.config} splits={self.launch} ({best[0]:.4f} ms)")

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
        """Sum, merge, then map, at the split count and width tuning settled."""
        return self._run_split(
            flat,
            running_mean,
            running_var,
            weight,
            bias,
            mean_out,
            rstd_out,
            self.launch,
            self.config["threads"],
        )

    def _run_split(
        self,
        flat: torch.Tensor,
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
        # A vector stays inside one channel only where the channel's run
        # divides it; otherwise each element picks its own.
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
                # One scale and shift per channel turns the body into a single
                # multiply-add, and the block builds that table once rather than
                # every element recomputing a root and a divide.
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

    # Elements one block covers, and its width. Each block pays the whole
    # per-channel table as a prologue, so the span fixes the block count and the
    # step count absorbs what the per-thread vector gives up.
    _BLOCK_SPAN = 2048
    _BLOCK_THREADS = 256

    # What autotune sweeps around that default: block widths, and how many
    # per-thread vectors one block walks.
    _TUNE_THREADS = (128, 256, 512, 1024)
    _TUNE_STEPS = (1, 2, 4)

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
        # The widest vector the element count divides, with steps keeping the
        # block span at _BLOCK_SPAN.
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
        for threads in self._TUNE_THREADS:
            for num_per_thread in _widths_down_to_one(_vector_elements(self.dtype)):
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
                    # One tile: a pipelined loop has nothing to overlap.
                    T.copy(grad_out[bc, 0:block_l], go_shared)
                    T.copy(x[bc, 0:block_l], x_shared)
                    for _i, j in T.Parallel(1, block_l):
                        go_val = T.cast(go_shared[j], accum_dtype)
                        x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                        do_frag[_i, j] += go_val
                        do_xhat_frag[_i, j] += go_val * x_hat
                else:
                    # T.copy inside T.Pipelined races with the async copy.
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
                    # Both shared buffers still hold the channel: no second read.
                    for _i, j in T.Parallel(1, block_l):
                        go_val = T.cast(go_shared[j], accum_dtype)
                        x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                        gx = w_rstd_over_L * (
                            T.cast(L, accum_dtype) * go_val - sum_do[0] - x_hat * sum_do_xhat[0]
                        )
                        grad_x[bc, j] = T.cast(gx, dtype)
                else:
                    # T.copy inside T.Pipelined races with the async copy.
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

    @staticmethod
    def _to_channel_major(t: torch.Tensor) -> torch.Tensor:
        """Move (N, C, *spatial) into the (C, L) layout this prim_func reads."""
        channels = t.shape[1]
        return t.permute(1, 0, *range(2, t.ndim)).reshape(channels, -1).contiguous()

    @staticmethod
    def _from_channel_major(t: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """Move a (C, L) result back to the caller's shape."""
        batch, channels, *spatial = original_shape
        restored = t.reshape(channels, batch, *spatial)
        return restored.permute(1, 0, *range(2, restored.ndim)).contiguous()

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
        )(
            self._to_channel_major(grad_out),
            self._to_channel_major(x),
            weight,
            mean,
            rstd,
            grad_weight,
            grad_bias,
        )
        return self._from_channel_major(grad_x, x.shape), grad_weight, grad_bias
