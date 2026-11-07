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

from tileops._kernels.autotune import TUNING_FATAL, candidate_busy_ms
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._kernels.norm._config import pairwise_sum, raw_sums_ratio
from tileops._kernels.norm.call_spec import (
    BatchNormBwdInterface,
    BatchNormCall,
    BatchNormInferFwdInterface,
    BatchNormTrainFwdInterface,
    InstanceNormInferFwdInterface,
)

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


def _param(tensor, index, present: bool, absent: float):
    """``tensor[index]``, or the constant *absent* where the caller passed no tensor."""
    return tensor[index] if present else T.float32(absent)


def _fill_absent(x: torch.Tensor, channels: int, *tensors: Optional[torch.Tensor]) -> tuple:
    """*tensors*, each absent one replaced by an unwritten ``(channels,)`` float32 tensor.

    A program built without a tensor never reads or writes it, so the stand-in costs an
    allocation and no kernel launch.
    """
    stand_in = None
    out = []
    for tensor in tensors:
        if tensor is None:
            if stand_in is None:
                stand_in = torch.empty(channels, dtype=torch.float32, device=x.device)
            tensor = stand_in
        out.append(tensor)
    return tuple(out)


@T.macro
def read_channel_lead(x, c, lead):
    """``lead[0]``: channel *c*'s first element in fp32, read as `make_read_leads` reads one."""
    for j in T.unroll(2):
        if j == 0:
            lead[0] = T.cast(x[0, c, j], "float32")


@T.macro
def unit_of(msq, unit):
    """``unit[0:2]``: an exact power of two ``u`` with ``msq * u^2`` in ``[1, 8)``, and ``1 / u``;
    a mean square past fp32's range picks no unit (see `unit_of_spread`)."""
    half = T.max(
        T.min(T.shift_right(T.shift_right(T.reinterpret(msq, "int32"), 23) - 127, 1), 63), -64
    )
    unit[0] = T.reinterpret(T.shift_left(127 - half, 23), "float32")
    unit[1] = T.reinterpret(T.shift_left(127 + half, 23), "float32")


@T.macro
def unit_of_spread(spread, unit):
    """``unit[0:2]``: an exact power of two ``u`` with ``spread * u`` in ``[1, 2)``, clamped to
    ``2^-126 .. 2^126``, and ``1 / u``."""
    e = T.max(T.min(T.shift_right(T.reinterpret(spread, "int32"), 23) - 127, 126), -126)
    unit[0] = T.reinterpret(T.shift_left(127 - e, 23), "float32")
    unit[1] = T.reinterpret(T.shift_left(127 + e, 23), "float32")


def rstd_in_unit(var_u, unit, eps: float):
    """``1 / sqrt(var + eps)`` for *var_u*, a variance in units of ``unit[0]^2``: taken in that
    unit where the sum there is a normal number, else in plain units."""
    shifted = var_u + T.float32(eps) * unit[0] * unit[0]
    return T.if_then_else(
        T.And(shifted >= 2.0**-126, shifted <= 3.0e38),
        T.rsqrt(shifted) * unit[0],
        T.rsqrt(var_u * unit[1] * unit[1] + T.float32(eps)),
    )


def _past_range(value):
    """Whether *value* is past ``3e38`` in magnitude or NaN, read from its bits: the compiler may
    negate a comparison into its converse, which a NaN fails."""
    return T.bitwise_and(T.reinterpret(value, "int32"), 0x7FFFFFFF) > 0x7F61B1E6


def _ratio(num, den):
    """``num / den`` for powers of two in fp32's normal range, exact by their exponents; zero
    below that range."""
    return T.reinterpret(
        T.max(
            T.reinterpret(T.float32(num), "int32") - T.reinterpret(den, "int32") + (127 << 23), 0
        ),
        "float32",
    )


def _vector_stats(held, lead, n: int, unit) -> tuple:
    """The sum of ``held[:n]`` less *lead* in units of ``unit[0]``, and their square sum around
    their mean in units of ``unit[0]^2``, taken less the first element: Samuelson's inequality
    keeps the cancellation within ``n`` units of rounding."""
    first = T.cast(held[0], "float32")
    e = [T.cast(held[i], "float32") * unit[0] - first * unit[0] for i in range(1, n)]
    ways = min(4, len(e))
    chains = [e[c] * e[c] for c in range(ways)]
    for i in range(ways, len(e)):
        chains[i % ways] = chains[i % ways] + e[i] * e[i]
    total = pairwise_sum(e)
    # The sum times the mean, not the sum squared, which can pass fp32's range.
    m2 = pairwise_sum(chains) - total * (total * (1.0 / n))
    return float(n) * (first * unit[0] - lead * unit[0]) + total, m2


class _Offset:
    """``buffer[base:]``."""

    def __init__(self, buffer, base):
        self.buffer, self.base = buffer, base

    def __getitem__(self, i):
        return self.buffer[self.base + i]


@T.macro
def split_sum(a, b, pair):
    """``pair[0] + pair[1] == a + b`` exactly (TwoSum), so ``x - pair[0]`` centres an element to
    one rounding of ``(x - a) - b``."""
    pair[0] = a + b
    pair[1] = (a - (pair[0] - (pair[0] - a))) + (b - (pair[0] - a))


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


@target_cache(maxsize=32)
def _batch_norm_fwd_train_kernel(
    N: int,
    C: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
    has_weight: bool = True,
    has_bias: bool = True,
    has_stats: bool = True,
) -> Callable:
    """Return the JIT-compiled training-forward kernel factory.

    Kernel computes, per channel, with ``K`` its first element:
      1. mean   = K + sum(x - K) / L
      2. var    = sum((x - mean)^2) / L, from each slot's square sum around its own mean
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

        def element(x, x_shared, bc, j, l):
            """Element *l* of channel *bc*, from shared memory where one tile holds the channel."""
            return x_shared[j] if block_l >= L else x[l // S, bc, l % S]

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
                lead = T.alloc_local([1], accum_dtype)
                read_channel_lead(x, bc, lead)

                # Each slot's Welford statistics over the tiles, less a reference that moves to its
                # running mean after every tile; a ragged last tile's masked slots see one tile
                # fewer.
                slot_lead = T.alloc_fragment([1, block_l], accum_dtype)
                slot_sum = T.alloc_fragment([1, block_l], accum_dtype)
                slot_m2 = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(slot_sum)
                T.clear(slot_m2)

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
                        slot_lead[_i, j] = T.cast(x_shared[j], accum_dtype)
                else:
                    for _i, j in T.Parallel(1, block_l):
                        slot_lead[_i, j] = T.cast(x[j // S, bc, j % S], accum_dtype)
                    # T.copy inside T.Pipelined races with the async copy.
                    for l_tile in T.Pipelined(tiles - 1, num_stages=0):
                        seen = T.cast(l_tile + 1, accum_dtype)
                        inverse_seen = 1.0 / seen
                        grown = seen / (seen + 1.0)
                        inverse_grown = 1.0 / (seen + 1.0)
                        for _i, j in T.Parallel(1, block_l):
                            l = (l_tile + 1) * block_l + j
                            if (not ragged) or l < L:
                                d = T.cast(x[l // S, bc, l % S], accum_dtype) - slot_lead[_i, j]
                                delta = d - slot_sum[_i, j] * inverse_seen
                                slot_m2[_i, j] += grown * delta * delta
                                summed = slot_sum[_i, j] + d
                                moved = slot_lead[_i, j] + summed * inverse_grown
                                slot_sum[_i, j] = summed - (seen + 1.0) * (moved - slot_lead[_i, j])
                                slot_lead[_i, j] = moved

                # Chan's merge of the slots, less a reference near the channel's mean.
                last_full = L - (tiles - 1) * block_l
                merged = T.alloc_fragment([1, block_l], accum_dtype)
                total = T.alloc_fragment([1], accum_dtype)
                for _i, j in T.Parallel(1, block_l):
                    seen = T.if_then_else(j < last_full, float(tiles), float(tiles - 1))
                    merged[_i, j] = slot_sum[_i, j] + seen * (slot_lead[_i, j] - lead[0])
                T.reduce_sum(merged, total, dim=1)
                reference = lead[0] + total[0] * (1.0 / L)
                for _i, j in T.Parallel(1, block_l):
                    seen = T.if_then_else(j < last_full, float(tiles), float(tiles - 1))
                    slot_sum[_i, j] = slot_sum[_i, j] + seen * (slot_lead[_i, j] - reference)
                T.reduce_sum(slot_sum, total, dim=1)
                mean_val = total[0] * (1.0 / L)
                for _i, j in T.Parallel(1, block_l):
                    seen = T.if_then_else(j < last_full, float(tiles), float(tiles - 1))
                    merged[_i, j] = slot_m2[_i, j] + T.if_then_else(
                        seen > 0,
                        (slot_sum[_i, j] / seen - mean_val) * (slot_sum[_i, j] - seen * mean_val),
                        0.0,
                    )
                T.reduce_sum(merged, total, dim=1)
                # (reference, mean less it).
                mean = T.alloc_local([2], accum_dtype)
                mean[0] = reference
                mean[1] = mean_val
                # (variance in a power-of-two unit, rstd, the unit's reciprocal).
                stat = T.alloc_local([3], accum_dtype)
                stat[0] = total[0] * (1.0 / L)
                stat[1] = T.rsqrt(stat[0] + T.cast(eps, accum_dtype))
                stat[2] = T.cast(1.0, accum_dtype)
                unit = T.alloc_local([2], accum_dtype)
                # A mean past fp32's range: read the channel again for its largest magnitude, whose
                # unit the sum is taken in.
                if T.Or(_past_range(mean[0]), _past_range(mean[1])):
                    T.clear(slot_sum)
                    for l_tile in T.serial(1 if block_l >= L else tiles):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            if (not ragged) or l < L:
                                slot_sum[_i, j] = T.max(
                                    slot_sum[_i, j],
                                    T.abs(T.cast(element(x, x_shared, bc, j, l), accum_dtype)),
                                )
                    T.reduce_max(slot_sum, total, dim=1)
                    unit_of_spread(total[0], unit)
                    T.clear(slot_sum)
                    for l_tile in T.serial(1 if block_l >= L else tiles):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            if (not ragged) or l < L:
                                slot_sum[_i, j] += (
                                    T.cast(element(x, x_shared, bc, j, l), accum_dtype) * unit[0]
                                )
                    T.reduce_sum(slot_sum, total, dim=1)
                    mean[0] = total[0] * (1.0 / L) * unit[1]
                    mean[1] = T.cast(0, accum_dtype)
                    stat[0] = T.cast(0, accum_dtype)
                # A variance among the subnormals or past fp32's range: read the channel again for
                # its mean distance from the mean, whose unit the sums and rsqrt are taken in.
                if T.Or(stat[0] < 2.0**-100, _past_range(stat[0])):
                    T.clear(slot_sum)
                    for l_tile in T.serial(1 if block_l >= L else tiles):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            if (not ragged) or l < L:
                                slot_sum[_i, j] += T.abs(
                                    (T.cast(element(x, x_shared, bc, j, l), accum_dtype) - mean[0])
                                    - mean[1]
                                )
                    T.reduce_sum(slot_sum, total, dim=1)
                    unit_of_spread(total[0] * (1.0 / L), unit)
                    T.clear(slot_sum)
                    T.clear(slot_m2)
                    for l_tile in T.serial(1 if block_l >= L else tiles):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            if (not ragged) or l < L:
                                d = (
                                    (T.cast(element(x, x_shared, bc, j, l), accum_dtype) - mean[0])
                                    - mean[1]
                                ) * unit[0]
                                slot_sum[_i, j] += d
                                slot_m2[_i, j] += d * d
                    T.reduce_sum(slot_sum, total, dim=1)
                    shift = total[0] * (1.0 / L)
                    T.reduce_sum(slot_m2, total, dim=1)
                    stat[0] = total[0] * (1.0 / L) - shift * shift
                    stat[1] = rstd_in_unit(stat[0], unit, eps)
                    stat[2] = unit[1]
                rstd_val = stat[1]

                mean_out[bc] = mean[0] + mean[1]
                rstd_out[bc] = rstd_val

                # Update running statistics, where the caller passed them.
                if has_stats:
                    # running_var takes the unbiased variance, as PyTorch does, in the unit
                    # until the momentum has weighted it: an update in range stays in range.
                    mom = T.cast(momentum, accum_dtype)
                    unbiased_var = (
                        stat[0]
                        * T.cast(L, accum_dtype)
                        / (T.cast(L, accum_dtype) - T.cast(1.0, accum_dtype))
                    )
                    # One writer per block: this running-stat RMW races if every thread runs it.
                    if T.get_thread_binding() == 0:
                        running_mean[bc] = (T.cast(1.0, accum_dtype) - mom) * running_mean[
                            bc
                        ] + mom * (mean[0] + mean[1])
                        running_var[bc] = (T.cast(1.0, accum_dtype) - mom) * running_var[
                            bc
                        ] + mom * unbiased_var * stat[2] * stat[2]

                # The output centres on the mean's exact pair; the low part joins the shift.
                centre = T.alloc_local([2], accum_dtype)
                split_sum(mean[0], mean[1], centre)
                scale_val = _param(weight, bc, has_weight, 1.0) * rstd_val
                shift_val = _param(bias, bc, has_bias, 0.0) - centre[1] * scale_val

                # Pass 2 – normalize.
                if block_l >= L:
                    # x_shared still holds the channel: no second global read.
                    for _i, j in T.Parallel(1, block_l):
                        if (not ragged) or j < L:
                            xval = T.cast(x_shared[j], accum_dtype)
                            y[j // S, bc, j % S] = T.cast(
                                (xval - centre[0]) * scale_val + shift_val, dtype
                            )
                else:
                    # T.copy inside T.Pipelined races with the async copy.
                    for l_tile in T.Pipelined(tiles, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            l = l_tile * block_l + j
                            if (not ragged) or l < L:
                                xval = T.cast(x[l // S, bc, l % S], accum_dtype)
                                y[l // S, bc, l % S] = T.cast(
                                    (xval - centre[0]) * scale_val + shift_val, dtype
                                )

        return _bn_fwd_train

    return _bn_fwd_train_func


@target_cache(maxsize=32)
def _batch_norm_fwd_train_split_kernel(
    N: int,
    C: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
    has_weight: bool = True,
    has_bias: bool = True,
    has_stats: bool = True,
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
        chunk = -(-L // splits)
        # A vector never straddles two batch items, so its address stays affine.
        vector_holds_one_item = S % num_per_thread == 0 and chunk % num_per_thread == 0
        # Whole pieces of an even step count: no bound test, and a thread reads ahead.
        steps = chunk // (threads * num_per_thread)
        whole_steps = (
            vector_holds_one_item and L % chunk == 0 and chunk % (2 * threads * num_per_thread) == 0
        )
        first_thread = sum(
            min(num_per_thread, chunk - step * threads * num_per_thread)
            for step in range(-(-chunk // (threads * num_per_thread)))
        )
        raw_ratio = raw_sums_ratio(
            dtype,
            (
                (num_per_thread - 1).bit_length() + -(-chunk // (threads * num_per_thread))
                if vector_holds_one_item
                else first_thread
            )
            + (threads - 1).bit_length()
            + 2,
        )

        def vector_sums(vector):
            """The sum of *vector* and of its squares, each pairwise."""
            values = [T.cast(vector[i], accum_dtype) for i in range(num_per_thread)]
            return pairwise_sum(values), pairwise_sum([v * v for v in values])

        @T.macro
        def take_vector(vector, step, ref, total, m2, count, _i, j, unit):
            """Chan's update of thread *j*'s statistics by its *step*-th full vector; ``total`` is
            in units of ``unit[0]``, ``m2`` of ``unit[0]^2``."""
            vector_sum, vector_m2 = _vector_stats(vector, ref[_i, j], num_per_thread, unit)
            seen = T.cast(step * num_per_thread, accum_dtype)
            delta = vector_sum * (1.0 / num_per_thread) - total[_i, j] * T.fast_rcp(
                T.max(seen, 1.0)
            )
            m2[_i, j] += (
                vector_m2 + seen * T.fast_rcp(seen * (1.0 / num_per_thread) + 1.0) * delta * delta
            )
            total[_i, j] += vector_sum
            count[_i, j] += float(num_per_thread)

        @T.macro
        def move_reference(ref, total, count, _i, j, unit):
            """Thread *j*'s reference moved to its running mean, its sum keeping what the move
            leaves."""
            moved = ref[_i, j] + total[_i, j] * T.fast_rcp(T.max(count[_i, j], 1.0)) * unit[1]
            total[_i, j] -= count[_i, j] * (moved * unit[0] - ref[_i, j] * unit[0])
            ref[_i, j] = moved

        @T.prim_func
        def _bn_train_stats(
            x: T.Tensor([N, C, S], dtype),
            partial_lead: T.Tensor([C, splits], accum_dtype),
            partial_sum: T.Tensor([C, splits], accum_dtype),
            partial_m2: T.Tensor([C, splits], accum_dtype),
            partial_unit: T.Tensor([C, splits], accum_dtype),
        ):
            with T.Kernel(C * splits, threads=threads) as bx:
                bc = bx // splits
                start = (bx % splits) * chunk
                # The last vector step can run past this block's chunk into the next one's.
                end = T.min(start + chunk, L)
                # Raw sums, kept where `raw_sums_ratio` holds; the block shares one verdict.
                raw_sum = T.alloc_fragment([1, threads], accum_dtype)
                raw_sq = T.alloc_fragment([1, threads], accum_dtype)
                raw = T.alloc_fragment([2, threads], accum_dtype)
                raw_total = T.alloc_fragment([2], accum_dtype)
                T.clear(raw_sum)
                T.clear(raw_sq)
                if whole_steps:
                    ahead = T.alloc_local([2 * num_per_thread], dtype)
                    for _i, j in T.Parallel(1, threads):
                        for i in T.vectorized(num_per_thread):
                            ahead[i] = x[
                                (start + j * num_per_thread) // S,
                                bc,
                                (start + j * num_per_thread) % S + i,
                            ]
                        for pair in T.serial(steps // 2):
                            for u in T.unroll(2):
                                step = pair * 2 + u
                                head = start + (T.min(step + 1, steps - 1) * threads + j) * (
                                    num_per_thread
                                )
                                for i in T.vectorized(num_per_thread):
                                    ahead[(1 - u) * num_per_thread + i] = x[
                                        head // S, bc, head % S + i
                                    ]
                                raw_sum[_i, j] += vector_sums(_Offset(ahead, u * num_per_thread))[0]
                                raw_sq[_i, j] += vector_sums(_Offset(ahead, u * num_per_thread))[1]
                vector = T.alloc_local([num_per_thread], dtype)
                for _i, j in T.Parallel(1, threads):
                    for step in T.serial(
                        0 if whole_steps else T.ceildiv(chunk, threads * num_per_thread)
                    ):
                        head = start + (step * threads + j) * num_per_thread
                        if head + num_per_thread <= end and vector_holds_one_item:
                            for i in T.vectorized(num_per_thread):
                                vector[i] = x[head // S, bc, head % S + i]
                            raw_sum[_i, j] += vector_sums(vector)[0]
                            raw_sq[_i, j] += vector_sums(vector)[1]
                        else:
                            for i in T.serial(num_per_thread):
                                if head + i < end:
                                    raw_sum[_i, j] += T.cast(
                                        x[(head + i) // S, bc, (head + i) % S], accum_dtype
                                    )
                                    raw_sq[_i, j] += T.cast(
                                        x[(head + i) // S, bc, (head + i) % S], accum_dtype
                                    ) * T.cast(x[(head + i) // S, bc, (head + i) % S], accum_dtype)
                for r, j in T.Parallel(2, threads):
                    raw[r, j] = T.if_then_else(r == 0, raw_sum[0, j], raw_sq[0, j])
                T.reduce_sum(raw, raw_total, dim=1)
                held_inverse = 1.0 / T.cast(end - start, accum_dtype)
                raw_mean = raw_total[0] * held_inverse
                raw_square = raw_total[1] * held_inverse
                raw_var = raw_square - raw_mean * raw_mean
                # A NaN fails every comparison, so a sum past fp32's range takes the shifted rule.
                # A shifted piece writes its sums in a power-of-two unit, its square sum negated.
                if T.And(
                    raw_square <= 3.0e38,
                    T.And(raw_var >= 2.0**-100, raw_square <= raw_ratio * raw_var),
                ):
                    if T.get_thread_binding() == 0:
                        partial_lead[bc, bx % splits] = raw_mean
                        partial_sum[bc, bx % splits] = (
                            raw_total[0] - T.cast(end - start, accum_dtype) * raw_mean
                        )
                        partial_m2[bc, bx % splits] = T.cast(end - start, accum_dtype) * raw_var
                else:
                    # The piece's first element, in a two-column loop (tile-ai/tilelang#3453).
                    lead = T.alloc_local([1], accum_dtype)
                    first = T.min(start, L - 1)
                    for j in T.unroll(2):
                        if j == 0:
                            lead[0] = T.cast(x[first // S, bc, first % S + j], accum_dtype)
                    # Each thread's Welford/Chan statistics less a reference that moves to its
                    # running mean after every step. A fixed reduction tree: the running statistics
                    # must not depend on a merge order.
                    count = T.alloc_fragment([1, threads], accum_dtype)
                    ref = T.alloc_fragment([1, threads], accum_dtype)
                    total = T.alloc_fragment([1, threads], accum_dtype)
                    m2 = T.alloc_fragment([1, threads], accum_dtype)
                    T.clear(count)
                    T.clear(total)
                    T.clear(m2)
                    unit = T.alloc_local([2], accum_dtype)
                    unit_of(raw_square, unit)
                    for _i, j in T.Parallel(1, threads):
                        ref[_i, j] = lead[0]
                    # Raw sums past fp32's range pick no unit; 2^-80 keeps every scaled square
                    # normal and in range, as an element below 2^128 with a square sum past it has a
                    # spread of 2^26 or more.
                    if _past_range(raw_square):
                        unit[0] = T.float32(2.0**-80)
                        unit[1] = T.float32(2.0**80)
                    if whole_steps:
                        held = T.alloc_local([2 * num_per_thread], dtype)
                        for _i, j in T.Parallel(1, threads):
                            for i in T.vectorized(num_per_thread):
                                held[i] = x[
                                    (start + j * num_per_thread) // S,
                                    bc,
                                    (start + j * num_per_thread) % S + i,
                                ]
                            for pair in T.serial(steps // 2):
                                for u in T.unroll(2):
                                    step = pair * 2 + u
                                    ahead = start + (T.min(step + 1, steps - 1) * threads + j) * (
                                        num_per_thread
                                    )
                                    for i in T.vectorized(num_per_thread):
                                        held[(1 - u) * num_per_thread + i] = x[
                                            ahead // S, bc, ahead % S + i
                                        ]
                                    take_vector(
                                        _Offset(held, u * num_per_thread),
                                        step,
                                        ref,
                                        total,
                                        m2,
                                        count,
                                        _i,
                                        j,
                                        unit,
                                    )
                                    move_reference(ref, total, count, _i, j, unit)
                    else:
                        held = T.alloc_local([num_per_thread], dtype)
                        for _i, j in T.Parallel(1, threads):
                            for step in T.serial(T.ceildiv(chunk, threads * num_per_thread)):
                                head = start + (step * threads + j) * num_per_thread
                                # The chunk bound is tested once a vector, so the vector loads in
                                # one access.
                                if head + num_per_thread <= end and vector_holds_one_item:
                                    for i in T.vectorized(num_per_thread):
                                        held[i] = x[head // S, bc, head % S + i]
                                    take_vector(held, step, ref, total, m2, count, _i, j, unit)
                                else:
                                    for i in T.serial(num_per_thread):
                                        l = head + i
                                        if l < end:
                                            v = (
                                                T.cast(x[l // S, bc, l % S], accum_dtype) * unit[0]
                                                - ref[_i, j] * unit[0]
                                            )
                                            delta = v - T.if_then_else(
                                                count[_i, j] > 0,
                                                total[_i, j] / count[_i, j],
                                                0.0,
                                            )
                                            m2[_i, j] += (
                                                count[_i, j] / (count[_i, j] + 1.0) * delta * delta
                                            )
                                            total[_i, j] += v
                                            count[_i, j] += 1.0
                                move_reference(ref, total, count, _i, j, unit)
                    # The threads' merge, less a reference near the block's mean.
                    merged = T.alloc_fragment([1, threads], accum_dtype)
                    block = T.alloc_fragment([1], accum_dtype)
                    inverse_held = 1.0 / T.cast(end - start, accum_dtype)
                    for _i, j in T.Parallel(1, threads):
                        merged[_i, j] = total[_i, j] + count[_i, j] * (
                            ref[_i, j] * unit[0] - lead[0] * unit[0]
                        )
                    T.reduce_sum(merged, block, dim=1)
                    reference = lead[0] + block[0] * inverse_held * unit[1]
                    pairs = T.alloc_fragment([2, threads], accum_dtype)
                    sums = T.alloc_fragment([2], accum_dtype)
                    for r, j in T.Parallel(2, threads):
                        pairs[r, j] = T.if_then_else(
                            r == 0,
                            total[0, j] + count[0, j] * (ref[0, j] * unit[0] - reference * unit[0]),
                            m2[0, j]
                            + T.if_then_else(
                                count[0, j] > 0,
                                (
                                    total[0, j]
                                    + count[0, j] * (ref[0, j] * unit[0] - reference * unit[0])
                                )
                                * (
                                    (
                                        total[0, j]
                                        + count[0, j] * (ref[0, j] * unit[0] - reference * unit[0])
                                    )
                                    / count[0, j]
                                ),
                                0.0,
                            ),
                        )
                    T.reduce_sum(pairs, sums, dim=1)
                    if T.get_thread_binding() == 0:
                        partial_lead[bc, bx % splits] = reference
                        partial_sum[bc, bx % splits] = sums[0]
                        # Negated, so the sign bit marks a piece in a unit, even at zero.
                        partial_m2[bc, bx % splits] = -T.max(
                            sums[1] - sums[0] * (sums[0] * inverse_held), 0.0
                        )
                        partial_unit[bc, bx % splits] = unit[1]

        return _bn_train_stats

    @tilelang.jit
    def _finalize_func(splits: int, threads: int) -> Callable:
        chunk = -(-L // splits)

        @T.prim_func
        def _bn_train_finalize(
            partial_lead: T.Tensor([C, splits], accum_dtype),
            partial_sum: T.Tensor([C, splits], accum_dtype),
            partial_m2: T.Tensor([C, splits], accum_dtype),
            partial_unit: T.Tensor([C, splits], accum_dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_out: T.Tensor([C], accum_dtype),
            rstd_out: T.Tensor([C], accum_dtype),
            coef_out: T.Tensor([C, 4], accum_dtype),
        ):
            with T.Kernel(1, threads=threads) as _:
                tx = T.get_thread_binding()
                for step in T.serial(T.ceildiv(C, threads)):
                    bc = step * threads + tx
                    if bc < C:
                        # Chan's merge of the pieces, keeping whichever reference lies nearer the
                        # merged mean.
                        lead = T.alloc_local([1], accum_dtype)
                        total = T.alloc_local([3], accum_dtype)
                        params = T.alloc_local([4], accum_dtype)
                        params[0] = _param(weight, bc, has_weight, 1.0)
                        params[1] = _param(bias, bc, has_bias, 0.0)
                        if has_stats:
                            params[2] = running_mean[bc]
                            params[3] = running_var[bc]
                        signs = T.alloc_local([1], "int32")
                        lead[0] = partial_lead[bc, 0]
                        signs[0] = 0
                        total[0] = T.cast(0, accum_dtype)
                        total[1] = T.cast(0, accum_dtype)
                        total[2] = T.cast(0, accum_dtype)
                        for k in T.serial(splits):
                            piece = T.cast(T.max(T.min(chunk, L - k * chunk), 0), accum_dtype)
                            if piece > 0:
                                signs[0] = T.bitwise_or(
                                    signs[0], T.reinterpret(partial_m2[bc, k], "int32")
                                )
                                # By reciprocals: a divide's slow path sits on the block's serial
                                # path.
                                apart = partial_lead[bc, k] - lead[0]
                                grown = total[0] + piece
                                inverse_grown = T.fast_rcp(grown)
                                delta = (apart + partial_sum[bc, k] * T.fast_rcp(piece)) - total[
                                    1
                                ] * T.fast_rcp(T.max(total[0], 1.0))
                                total[2] += (
                                    partial_m2[bc, k]
                                    + total[0] * piece * inverse_grown * delta * delta
                                )
                                merged = total[1] + partial_sum[bc, k] + piece * apart
                                if T.abs(merged * inverse_grown - apart) < T.abs(
                                    merged * inverse_grown
                                ):
                                    total[1] = merged - grown * apart
                                    lead[0] = partial_lead[bc, k]
                                else:
                                    total[1] = merged
                                total[0] = grown
                        n = T.cast(L, accum_dtype)
                        total[1] = total[1] * (1.0 / L)
                        # (variance in a power-of-two unit, rstd, the unit's reciprocal).
                        stat = T.alloc_local([3], accum_dtype)
                        stat[0] = total[2] * (1.0 / L)
                        stat[1] = T.rsqrt(stat[0] + T.cast(eps, accum_dtype))
                        stat[2] = T.cast(1.0, accum_dtype)
                        # A piece in a unit, or a variance or mean out of range: merge again in the
                        # coarsest unit among the pieces.
                        if T.Or(
                            signs[0] < 0,
                            T.Or(
                                T.Or(stat[0] < 2.0**-100, _past_range(stat[0])),
                                _past_range(total[1]),
                            ),
                        ):
                            unit = T.alloc_local([2], accum_dtype)
                            own = T.alloc_local([2], accum_dtype)
                            lead[0] = partial_lead[bc, 0]
                            unit[1] = T.cast(0, accum_dtype)
                            total[0] = T.cast(0, accum_dtype)
                            total[1] = T.cast(0, accum_dtype)
                            total[2] = T.cast(0, accum_dtype)
                            for k in T.serial(splits):
                                piece = T.cast(T.max(T.min(chunk, L - k * chunk), 0), accum_dtype)
                                if piece > 0:
                                    in_unit = T.reinterpret(partial_m2[bc, k], "int32") < 0
                                    if T.Not(in_unit):
                                        unit_of(
                                            (
                                                partial_lead[bc, k]
                                                + partial_sum[bc, k] * T.fast_rcp(piece)
                                            )
                                            * (
                                                partial_lead[bc, k]
                                                + partial_sum[bc, k] * T.fast_rcp(piece)
                                            )
                                            + partial_m2[bc, k] * T.fast_rcp(piece),
                                            own,
                                        )
                                    else:
                                        own[0] = _ratio(1.0, partial_unit[bc, k])
                                        own[1] = partial_unit[bc, k]
                                    piece_unit = T.if_then_else(in_unit, 1.0, own[0])
                                    coarser = T.max(unit[1], own[1])
                                    total[1] = total[1] * _ratio(unit[1], coarser)
                                    total[2] = (
                                        total[2]
                                        * _ratio(unit[1], coarser)
                                        * _ratio(unit[1], coarser)
                                    )
                                    unit[1] = coarser
                                    unit[0] = _ratio(1.0, coarser)
                                    scale = _ratio(own[1], coarser)
                                    piece_sum = partial_sum[bc, k] * piece_unit * scale
                                    # Scaled before the difference, which can pass fp32's range.
                                    apart = partial_lead[bc, k] * unit[0] - lead[0] * unit[0]
                                    grown = total[0] + piece
                                    inverse_grown = T.fast_rcp(grown)
                                    delta = (apart + piece_sum * T.fast_rcp(piece)) - total[
                                        1
                                    ] * T.fast_rcp(T.max(total[0], 1.0))
                                    total[2] += (
                                        T.abs(partial_m2[bc, k])
                                        * (piece_unit * scale)
                                        * (piece_unit * scale)
                                        + total[0] * piece * inverse_grown * delta * delta
                                    )
                                    merged = total[1] + piece_sum + piece * apart
                                    if T.abs(merged * inverse_grown - apart) < T.abs(
                                        merged * inverse_grown
                                    ):
                                        total[1] = merged - grown * apart
                                        lead[0] = partial_lead[bc, k]
                                    else:
                                        total[1] = merged
                                    total[0] = grown
                            total[1] = total[1] * (1.0 / L) * unit[1]
                            stat[0] = total[2] * (1.0 / L)
                            stat[1] = rstd_in_unit(stat[0], unit, eps)
                            stat[2] = unit[1]
                        rstd_val = stat[1]
                        mean_out[bc] = lead[0] + total[1]
                        rstd_out[bc] = rstd_val
                        # Folded here, the map pass reads three numbers per channel.
                        centre = T.alloc_local([2], accum_dtype)
                        split_sum(lead[0], total[1], centre)
                        scale_val = params[0] * rstd_val
                        coef_out[bc, 0] = centre[0]
                        coef_out[bc, 1] = scale_val
                        coef_out[bc, 2] = params[1] - centre[1] * scale_val
                        # running_var follows PyTorch convention: updated with
                        # unbiased variance (Bessel's correction), in the unit until the
                        # momentum has weighted it.
                        mom = T.cast(momentum, accum_dtype)
                        one = T.cast(1.0, accum_dtype)
                        if has_stats:
                            running_mean[bc] = (one - mom) * params[2] + mom * (lead[0] + total[1])
                            running_var[bc] = (one - mom) * params[3] + mom * (
                                stat[0] * n / (n - one)
                            ) * stat[2] * stat[2]

        return _bn_train_finalize

    @tilelang.jit(out_idx=[-1])
    def _apply_func(blocks: int, threads: int, num_per_thread: int) -> Callable:
        vector_holds_one_channel = S % num_per_thread == 0
        span = threads * num_per_thread
        total = C * L

        @T.prim_func
        def _bn_train_apply(
            x_ncs: T.Tensor([N, C, S], dtype),
            coef: T.Tensor([C, 4], accum_dtype),
            y_ncs: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(blocks, threads=threads) as bx:
                # T.reshape's size check multiplies in int32 and overflows on large tensors.
                x = T.Tensor([total], dtype, x_ncs.data)
                y = T.Tensor([total], dtype, y_ncs.data)
                tx = T.get_thread_binding()
                v = T.alloc_local([num_per_thread], dtype)
                o = T.alloc_local([num_per_thread], dtype)
                held = T.alloc_local([4], accum_dtype)
                base = bx * span + tx * num_per_thread
                if base + num_per_thread <= total:
                    for i in T.vectorized(num_per_thread):
                        v[i] = x[base + i]
                    if vector_holds_one_channel:
                        ch = (base // S) % C
                        for k in T.vectorized(4):
                            held[k] = coef[ch, k]
                        for i in T.serial(num_per_thread):
                            o[i] = T.cast(
                                (T.cast(v[i], accum_dtype) - held[0]) * held[1] + held[2], dtype
                            )
                    else:
                        for i in T.serial(num_per_thread):
                            ch = ((base + i) // S) % C
                            o[i] = T.cast(
                                (T.cast(v[i], accum_dtype) - coef[ch, 0]) * coef[ch, 1]
                                + coef[ch, 2],
                                dtype,
                            )
                    for i in T.vectorized(num_per_thread):
                        y[base + i] = o[i]
                else:
                    for i in T.serial(num_per_thread):
                        if base + i < total:
                            ch = ((base + i) // S) % C
                            y[base + i] = T.cast(
                                (T.cast(x[base + i], accum_dtype) - coef[ch, 0]) * coef[ch, 1]
                                + coef[ch, 2],
                                dtype,
                            )

        return _bn_train_apply

    return _stats_func, _finalize_func, _apply_func


@target_cache(maxsize=32)
def _batch_norm_fwd_train_wide_kernel(
    N: int,
    C: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
    has_weight: bool = True,
    has_bias: bool = True,
    has_stats: bool = True,
) -> Callable:
    """Return the JIT-compiled training-forward factory for a register-held channel.

    A channel short enough to fit in one block's registers is read once into
    registers and normalised from there, with no shared-memory staging and no
    second global read: global traffic is one read and one write of the tensor.

    Each thread takes its elements' sum and square sum around its own mean; a warp merges
    its threads' in one shuffle round, and the block merges the warps' after one exchange.
    That order is fixed, so the running statistics this kernel writes back are the same on
    every run.

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
        held_n = steps * num_per_thread
        n_warps = max(threads // lanes, 1)
        # One XOR step per bit of the lane index.
        butterfly_depth = lanes.bit_length() - 1
        full_steps, extra = divmod(L // num_per_thread, threads)

        def held_by(first, size):
            """Elements threads ``first`` to ``first + size - 1`` hold."""
            return num_per_thread * (full_steps * size + T.max(T.min(extra - first, size), 0))

        def warp_count(w: int) -> int:
            """Elements warp *w* holds."""
            return num_per_thread * (full_steps * lanes + max(min(extra - w * lanes, lanes), 0))

        # Steps a group of at most 32 of a thread's elements spans.
        group_steps = max(1, 32 // num_per_thread)
        groups = -(-steps // group_steps)

        def by_thread(value):
            """``value(steps)`` for this thread: a select between the counts below and past
            `extra`."""
            if not extra:
                return value(full_steps)
            return T.if_then_else(
                T.get_thread_binding() < extra, value(full_steps + 1), value(full_steps)
            )

        def held_before(g, steps):
            """Elements before group *g* a thread holding *steps* vectors holds, in fp32."""
            return T.cast(num_per_thread * T.min(steps, g * group_steps), accum_dtype)

        def inverse_of(count):
            """1 / *count*, zero for none."""
            return T.if_then_else(count > 0, 1.0 / T.max(count, 1.0), 0.0)

        def group_count(g, steps):
            """Elements of group *g* a thread holding *steps* vectors holds, in fp32."""
            return held_before(g + 1, steps) - held_before(g, steps)

        def group_inverse(g):
            """1 / the elements of group *g* this thread holds, zero for none."""
            return by_thread(lambda v: inverse_of(group_count(g, v)))

        @T.macro
        def merge_group(total, part, group_ref, g, unit):
            """Chan's update of ``total`` (sum less the reference ``total[2]``, square sum), in
            units of ``unit[0]``, by group *g*; the reference then moves to the merged mean."""
            # Constants a thread selects, where a divide would sit on its path.
            count = by_thread(lambda v: group_count(g, v))
            seen = by_thread(lambda v: held_before(g + 1, v))
            inverse_seen = by_thread(lambda v: inverse_of(held_before(g + 1, v)))
            inverse_before = by_thread(lambda v: inverse_of(held_before(g, v)))
            weight = by_thread(
                lambda v: held_before(g, v) * group_count(g, v) * inverse_of(held_before(g + 1, v))
            )
            # Scaled before the difference, which can pass fp32's range.
            apart = group_ref[g] * unit[0] - total[2] * unit[0]
            group_sum = (part[g * 8] + part[g * 8 + 1]) + (part[g * 8 + 2] + part[g * 8 + 3])
            group_m2 = (
                (part[g * 8 + 4] + part[g * 8 + 5]) + (part[g * 8 + 6] + part[g * 8 + 7])
            ) - (group_sum * (group_sum * group_inverse(g)))
            delta = apart + group_sum * group_inverse(g) - total[0] * inverse_before
            if count > 0:
                total[1] = total[1] + group_m2 + weight * delta * delta
                merged = total[0] + group_sum + count * apart
                moved = total[2] + merged * inverse_seen * unit[1]
                total[0] = merged - seen * (moved * unit[0] - total[2] * unit[0])
                total[2] = moved

        # Warps past which lanes merge them by shuffles.
        lanes_merge = 8
        warp_depth = (n_warps - 1).bit_length()

        def warp_terms(slots, unit):
            """The warps' shares times their means less warp 0's, and times their mean squares
            about it."""
            terms = [
                (
                    warp_count(w),
                    (slots[0, w] * unit[0] - slots[0, 0] * unit[0])
                    + (slots[1, w] - slots[1, 0]) * unit[0],
                    w,
                )
                for w in range(n_warps)
                if warp_count(w)
            ]
            return (
                pairwise_sum([(n / L) * b for n, b, _ in terms]),
                pairwise_sum([(n / L) * slots[2, w] + (n / L) * b * b for n, b, w in terms]),
            )

        def raw_sums(slots, j):
            """The warps' sums of row *j* of *slots*, pairwise."""
            return pairwise_sum([slots[j, w] for w in range(n_warps) if warp_count(w)])

        raw_ratio = raw_sums_ratio(
            dtype, -(-held_n // 4) + 2 + butterfly_depth + max(warp_depth, 1) + 1
        )

        @T.macro
        def write_output(held, x, y, bc, tx, scale_val, shift_val, centre):
            for k in T.unroll(steps):
                head = (k * threads + tx) * num_per_thread
                if exact or head < L:
                    out = T.alloc_local([num_per_thread], dtype)
                    for i in T.unroll(num_per_thread):
                        if centre is None:
                            out[i] = T.cast(
                                T.cast(held[k * num_per_thread + i], accum_dtype) * scale_val
                                + shift_val,
                                dtype,
                            )
                        else:
                            out[i] = T.cast(
                                (T.cast(held[k * num_per_thread + i], accum_dtype) - centre)
                                * scale_val
                                + shift_val,
                                dtype,
                            )
                    for i in T.vectorized(num_per_thread):
                        y[head // S, bc, head % S + i] = out[i]

        @T.macro
        def finish(params, result, held, x, y, bc, tx, centre, outs, unit):
            """Write the channel's statistics from ``result`` (mean, variance, centre, the mean's
            low part, the variance in ``unit``) and the output, centred on ``centre`` unless
            ``None``."""
            if unit is None:
                rstd_val = T.rsqrt(result[1] + T.cast(eps, accum_dtype))
            else:
                rstd_val = rstd_in_unit(result[4], unit, eps)
            scale_val = params[0] * rstd_val
            shift_val = params[1] - result[3] * scale_val
            one = T.cast(1.0, accum_dtype)
            mom = T.cast(momentum, accum_dtype)
            # One writer per block: this running-stat RMW races if every thread runs it.
            if tx == 0:
                outs[0][bc] = result[0]
                outs[1][bc] = rstd_val
                if has_stats:
                    outs[2][bc] = (one - mom) * params[2] + mom * result[0]
                    # running_var takes the unbiased variance, as PyTorch does.
                    if unit is None:
                        outs[3][bc] = (one - mom) * params[3] + mom * (result[1] * (L / (L - 1.0)))
                    else:
                        outs[3][bc] = (one - mom) * params[3] + mom * (
                            result[4] * (L / (L - 1.0))
                        ) * unit[1] * unit[1]
            write_output(held, x, y, bc, tx, scale_val, shift_val, centre)

        def warp_inverse(w):
            """1 / the elements warp *w* holds, zero for none: a select over constants."""
            inverse = T.float32(0)
            for v in range(n_warps):
                if warp_count(v):
                    inverse = T.if_then_else(w == v, T.float32(1.0 / warp_count(v)), inverse)
            return inverse

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
                params[0] = _param(weight, bc, has_weight, 1.0)
                params[1] = _param(bias, bc, has_bias, 0.0)
                if has_stats:
                    params[2] = running_mean[bc]
                    params[3] = running_var[bc]
                held = T.alloc_local([held_n], dtype)
                part = T.alloc_local([8 * groups], accum_dtype)
                group_ref = T.alloc_local([groups], accum_dtype)
                stats = T.alloc_local([2], accum_dtype)
                # One allocation for both exchanges, so neither waits on the other's reads.
                slots = T.alloc_shared([5, n_warps], accum_dtype)

                for k in T.unroll(steps):
                    head = (k * threads + tx) * num_per_thread
                    if exact or head < L:
                        for i in T.vectorized(num_per_thread):
                            held[k * num_per_thread + i] = x[head // S, bc, head % S + i]
                    else:
                        for i in T.vectorized(num_per_thread):
                            held[k * num_per_thread + i] = T.cast(0, dtype)
                count = num_per_thread * full_steps + T.if_then_else(tx < extra, num_per_thread, 0)
                inverse = T.if_then_else(
                    tx < extra,
                    1.0 / (num_per_thread * (full_steps + 1)),
                    (1.0 / (num_per_thread * full_steps)) if full_steps else 0.0,
                )

                # Raw sums, kept where `raw_sums_ratio` holds; the block shares one verdict.
                raw = T.alloc_local([8], accum_dtype)
                for j in T.unroll(8):
                    raw[j] = T.cast(0, accum_dtype)
                for k in T.unroll(steps):
                    head = (k * threads + tx) * num_per_thread
                    if exact or head < L:
                        for i in T.unroll(num_per_thread):
                            raw[(k * num_per_thread + i) % 4] += T.cast(
                                held[k * num_per_thread + i], accum_dtype
                            )
                            raw[4 + (k * num_per_thread + i) % 4] += T.cast(
                                held[k * num_per_thread + i], accum_dtype
                            ) * T.cast(held[k * num_per_thread + i], accum_dtype)
                stats[0] = (raw[0] + raw[1]) + (raw[2] + raw[3])
                stats[1] = (raw[4] + raw[5]) + (raw[6] + raw[7])
                for st in T.unroll(butterfly_depth):
                    for j in T.unroll(2):
                        stats[j] += T.shfl_xor(stats[j], T.shift_left(1, st))
                if tx % lanes == 0:
                    slots[3, tx // lanes] = stats[0]
                    slots[4, tx // lanes] = stats[1]
                T.sync_threads()
                if n_warps > lanes_merge:
                    lane = tx % (1 << warp_depth)
                    for j in T.unroll(2):
                        stats[j] = T.if_then_else(
                            lane < n_warps, slots[3 + j, T.min(lane, n_warps - 1)], 0.0
                        )
                    for st in T.unroll(warp_depth):
                        for j in T.unroll(2):
                            stats[j] += T.shfl_xor(stats[j], T.shift_left(1, st))
                else:
                    stats[0] = raw_sums(slots, 3)
                    stats[1] = raw_sums(slots, 4)
                outs = (mean_out, rstd_out, running_mean, running_var)
                # (mean, variance, centre, the mean's low part, the shifted rule's variance in its
                # unit).
                result = T.alloc_local([5], accum_dtype)
                result[0] = stats[0] * (1.0 / L)
                result[1] = stats[1] * (1.0 / L) - result[0] * result[0]
                result[2] = T.cast(0, accum_dtype)
                result[3] = result[0]
                raw_holds = T.alloc_local([1], "int32")
                # A NaN fails every comparison, so a sum past fp32's range takes the shifted rule.
                raw_holds[0] = T.if_then_else(
                    T.And(
                        stats[1] * (1.0 / L) <= 3.0e38,
                        T.And(
                            result[1] >= 2.0**-100,
                            stats[1] * (1.0 / L) <= raw_ratio * result[1],
                        ),
                    ),
                    1,
                    0,
                )
                if raw_holds[0] == 1:
                    finish(params, result, held, x, y, bc, tx, None, outs, None)
                else:
                    # Groups of at most 32 elements less their first: Samuelson's inequality keeps a
                    # group's mean square less its squared mean within its count in units of
                    # rounding. The differences are in a power-of-two unit the mean square picks.
                    unit = T.alloc_local([2], accum_dtype)
                    unit_of(result[1] + result[0] * result[0], unit)
                    # Raw sums past fp32's range: the largest magnitude picks the unit.
                    if _past_range(result[1] + result[0] * result[0]):
                        far = T.alloc_local([1], accum_dtype)
                        far[0] = T.cast(0, accum_dtype)
                        for k in T.unroll(held_n):
                            far[0] = T.max(far[0], T.abs(T.cast(held[k], accum_dtype)))
                        for st in T.unroll(butterfly_depth):
                            far[0] = T.max(far[0], T.shfl_xor(far[0], T.shift_left(1, st)))
                        if tx % lanes == 0:
                            slots[0, tx // lanes] = far[0]
                        T.sync_threads()
                        for w in T.unroll(n_warps):
                            far[0] = T.max(far[0], slots[0, w])
                        T.sync_threads()
                        unit_of_spread(far[0], unit)
                    for g in T.unroll(groups):
                        group_ref[g] = T.cast(held[g * group_steps * num_per_thread], accum_dtype)
                    for j in T.unroll(8 * groups):
                        part[j] = T.cast(0, accum_dtype)
                    for k in T.unroll(steps):
                        head = (k * threads + tx) * num_per_thread
                        if exact or head < L:
                            for i in T.unroll(num_per_thread):
                                part[k // group_steps * 8 + (k * num_per_thread + i) % 4] += (
                                    T.cast(held[k * num_per_thread + i], accum_dtype) * unit[0]
                                    - group_ref[k // group_steps] * unit[0]
                                )
                                part[k // group_steps * 8 + 4 + (k * num_per_thread + i) % 4] += (
                                    T.cast(held[k * num_per_thread + i], accum_dtype) * unit[0]
                                    - group_ref[k // group_steps] * unit[0]
                                ) * (
                                    T.cast(held[k * num_per_thread + i], accum_dtype) * unit[0]
                                    - group_ref[k // group_steps] * unit[0]
                                )
                    group_total = T.alloc_local([3], accum_dtype)
                    group_total[2] = group_ref[0]
                    group_total[0] = (part[0] + part[1]) + (part[2] + part[3])
                    group_total[1] = ((part[4] + part[5]) + (part[6] + part[7])) - group_total[
                        0
                    ] * (group_total[0] * (inverse if groups == 1 else group_inverse(0)))
                    for g in T.unroll(1, groups):
                        merge_group(group_total, part, group_ref, g, unit)
                    # Exact mean pairs: two pairs' difference is as small as the means' spread.
                    mean_pair = T.alloc_local([2], accum_dtype)
                    split_sum(group_total[2], group_total[0] * inverse * unit[1], mean_pair)
                    reference_hi = T.shfl_sync(mean_pair[0], 0)
                    reference_lo = T.shfl_sync(mean_pair[1], 0)
                    # Shares rather than counts: a count times a squared spread can pass fp32's
                    # range; lane 0's mean as reference bounds the cancellation as within a thread.
                    offset = T.if_then_else(
                        count > 0,
                        (mean_pair[0] * unit[0] - reference_hi * unit[0])
                        + (mean_pair[1] - reference_lo) * unit[0],
                        0.0,
                    )
                    share = T.cast(count, accum_dtype) * warp_inverse(tx // lanes)
                    stats[0] = share * offset
                    stats[1] = share * (group_total[1] * inverse) + share * offset * offset
                    for st in T.unroll(butterfly_depth):
                        for j in T.unroll(2):
                            stats[j] += T.shfl_xor(stats[j], T.shift_left(1, st))
                    if tx % lanes == 0:
                        w = tx // lanes
                        slots[0, w] = reference_hi
                        slots[1, w] = reference_lo + stats[0] * unit[1]
                        slots[2, w] = stats[1] - stats[0] * stats[0]
                    T.sync_threads()
                    if n_warps > lanes_merge:
                        lane = tx % (1 << warp_depth)
                        lane_count = T.cast(held_by(lane * lanes, lanes), accum_dtype)
                        # A warp holding nothing has no mean: it adds nothing.
                        spread = T.if_then_else(
                            T.And(lane < n_warps, lane_count > 0),
                            (slots[0, T.min(lane, n_warps - 1)] * unit[0] - slots[0, 0] * unit[0])
                            + (slots[1, T.min(lane, n_warps - 1)] - slots[1, 0]) * unit[0],
                            0.0,
                        )
                        stats[0] = lane_count * (1.0 / L) * spread
                        stats[1] = (
                            lane_count
                            * (1.0 / L)
                            * T.if_then_else(
                                lane < n_warps, slots[2, T.min(lane, n_warps - 1)], 0.0
                            )
                            + lane_count * (1.0 / L) * spread * spread
                        )
                        for st in T.unroll(warp_depth):
                            for j in T.unroll(2):
                                stats[j] += T.shfl_xor(stats[j], T.shift_left(1, st))
                    else:
                        stats[0] = warp_terms(slots, unit)[0]
                        stats[1] = warp_terms(slots, unit)[1]
                    result[4] = stats[1] - stats[0] * stats[0]
                    centre = T.alloc_local([2], accum_dtype)
                    split_sum(slots[0, 0], slots[1, 0] + stats[0] * unit[1], centre)
                    result[0] = centre[0] + centre[1]
                    result[2] = centre[0]
                    result[3] = centre[1]

                    finish(params, result, held, x, y, bc, tx, result[2], outs, unit)

        return _bn_fwd_train_wide

    return _bn_fwd_train_wide_func


@target_cache(maxsize=32)
def _batch_norm_fwd_train_whole_kernel(
    N: int,
    C: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
    has_weight: bool = True,
    has_bias: bool = True,
    has_stats: bool = True,
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
                    params[0] = _param(weight, c, has_weight, 1.0)
                    params[1] = _param(bias, c, has_bias, 0.0)
                    if has_stats:
                        params[2] = running_mean[c]
                        params[3] = running_var[c]
                lead = T.alloc_local([1], accum_dtype)
                held = T.alloc_local([L], accum_dtype)
                part = T.alloc_local([8], accum_dtype)
                if c < C:
                    read_channel_lead(x, c, lead)
                    # Corrected two-pass, four chains a sum.
                    for j in T.unroll(8):
                        part[j] = T.cast(0, accum_dtype)
                    for l in T.unroll(L):
                        held[l] = T.cast(x[l // S, c, l % S], accum_dtype)
                        part[l % 4] += held[l] - lead[0]
                    centre = T.alloc_local([2], accum_dtype)
                    unit = T.alloc_local([2], accum_dtype)
                    centre[0] = lead[0] + ((part[0] + part[1]) + (part[2] + part[3])) * (1.0 / L)
                    # A mean past fp32's range: formed again in the unit the largest magnitude
                    # picks.
                    if _past_range(centre[0]):
                        unit[0] = T.cast(0, accum_dtype)
                        for l in T.unroll(L):
                            unit[0] = T.max(unit[0], T.abs(held[l]))
                        unit_of_spread(unit[0], unit)
                        for j in T.unroll(4):
                            part[j] = T.cast(0, accum_dtype)
                        for l in T.unroll(L):
                            part[l % 4] += held[l] * unit[0]
                        centre[0] = (
                            ((part[0] + part[1]) + (part[2] + part[3])) * (1.0 / L) * unit[1]
                        )
                    for j in T.unroll(4):
                        part[j] = T.cast(0, accum_dtype)
                    for l in T.unroll(L):
                        held[l] = held[l] - centre[0]
                        part[4 + l % 4] += held[l]
                        part[l % 4] += held[l] * held[l]
                    centre[1] = ((part[4] + part[5]) + (part[6] + part[7])) * (1.0 / L)
                    # (variance in a power-of-two unit, rstd, the unit's reciprocal).
                    stat = T.alloc_local([3], accum_dtype)
                    stat[0] = ((part[0] + part[1]) + (part[2] + part[3])) * (1.0 / L) - centre[
                        1
                    ] * centre[1]
                    stat[1] = T.rsqrt(stat[0] + T.cast(eps, accum_dtype))
                    stat[2] = T.cast(1.0, accum_dtype)
                    # A variance among the subnormals or past fp32's range: taken again in the unit
                    # the mean distance from the mean picks.
                    if T.Or(stat[0] < 2.0**-100, _past_range(stat[0])):
                        for j in T.unroll(4):
                            part[j] = T.cast(0, accum_dtype)
                        for l in T.unroll(L):
                            part[l % 4] += T.abs(held[l])
                        unit_of_spread(
                            ((part[0] + part[1]) + (part[2] + part[3])) * (1.0 / L), unit
                        )
                        for j in T.unroll(8):
                            part[j] = T.cast(0, accum_dtype)
                        for l in T.unroll(L):
                            part[4 + l % 4] += held[l] * unit[0]
                            part[l % 4] += (held[l] * unit[0]) * (held[l] * unit[0])
                        shift = ((part[4] + part[5]) + (part[6] + part[7])) * (1.0 / L)
                        stat[0] = ((part[0] + part[1]) + (part[2] + part[3])) * (
                            1.0 / L
                        ) - shift * shift
                        stat[1] = rstd_in_unit(stat[0], unit, eps)
                        stat[2] = unit[1]
                        centre[1] = shift * unit[1]
                    rstd_val = stat[1]
                    mean_out[c] = centre[0] + centre[1]
                    rstd_out[c] = rstd_val
                    # The affine and the normalisation fold into one multiply-add.
                    scale_val = params[0] * rstd_val
                    shift_val = params[1] - centre[1] * scale_val

                    mom = T.cast(momentum, accum_dtype)
                    one = T.cast(1.0, accum_dtype)
                    if has_stats:
                        running_mean[c] = (one - mom) * params[2] + mom * (centre[0] + centre[1])
                        # running_var takes the unbiased variance, as PyTorch does. By a
                        # constant factor: around a divide's slow path the compiler reads
                        # running_var after the sums, its latency then on every thread's path.
                        running_var[c] = (one - mom) * params[3] + mom * (
                            stat[0] * (L / (L - 1.0))
                        ) * stat[2] * stat[2]

                    for l in T.unroll(L):
                        y[l // S, c, l % S] = T.cast(held[l] * scale_val + shift_val, dtype)

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


class _BatchNormFwdTrainHeldKernel(_BatchNormKernel, BatchNormTrainFwdInterface):
    """Training forward with a channel held in registers, launched with ``self.launch``."""

    def forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor],
        running_var: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ):
        """Normalize an ``(N, C, S)`` input by its batch statistics.

        Returns:
            ``(y, mean, rstd)``: the ``(N, C, S)`` output and the per-channel batch
            mean and reciprocal std the backward pass reads.
        """
        self._require_devices(
            x=x, weight=weight, bias=bias, running_mean=running_mean, running_var=running_var
        )
        running_mean, running_var, weight, bias = _fill_absent(
            x, self.C, running_mean, running_var, weight, bias
        )
        mean_out = torch.empty(self.C, device=x.device, dtype=torch.float32)
        rstd_out = torch.empty_like(mean_out)
        y = self.program(self._compile)(
            x, weight, bias, running_mean, running_var, mean_out, rstd_out
        )
        return y, mean_out, rstd_out

    def _compile(self, cfg: dict):
        """The program, launched as ``self.launch`` says."""
        return self.kernel(*self.launch)


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
    def refusal(cls, call: BatchNormCall) -> "str | None":
        holder = cls._holder(call, 1)
        if holder != "thread":
            return f"a channel of this call is held by a {holder}, not a thread"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (call.n, call.c, call.spatial, call.dtype, call.eps, call.momentum)
        index = call.device.index if call.device is not None else None
        presence = (call.has_weight, call.has_bias, call.has_running_stats)
        return (*args, presence, index), lambda: cls(*args, presence=presence, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        momentum: float = 0.1,
        presence: tuple[bool, bool, bool] = (True, True, True),
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.C = C
        self.dtype = dtype
        self.launch = (self._BLOCK_THREADS,)
        self.kernel = _batch_norm_fwd_train_whole_kernel(
            N, C, S, self.dtype_str, eps, momentum, *presence
        )


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
    def refusal(cls, call: BatchNormCall) -> "str | None":
        holder = cls._holder(call, 1)
        if holder != "block":
            return f"a channel of this call is held by a {holder}, not a block"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (
            call.n,
            call.c,
            call.spatial,
            call.dtype,
            call.eps,
            call.momentum,
            cls._trimmed_launch(call),
        )
        index = call.device.index if call.device is not None else None
        presence = (call.has_weight, call.has_bias, call.has_running_stats)
        return (*args, presence, index), lambda: cls(*args, presence=presence, device_index=index)

    @classmethod
    def _trimmed_launch(cls, call: BatchNormCall) -> tuple[int, int]:
        """The ``(threads, num_per_thread)`` of the block: the fewest whole warps holding the
        channel at one group of 32 elements a thread, and at least `_HOLDING_BLOCK_THREADS` threads
        on each SM the grid fills."""
        _, num_per_thread = cls._block_launch(call, 1)
        vectors = call.n * call.spatial // num_per_thread
        lanes = 32
        per_sm = -(-call.c // call.sm_count)
        grouped = -(-vectors // max(1, lanes // num_per_thread))
        threads = min(max(grouped, cls._HOLDING_BLOCK_THREADS // per_sm), cls._BLOCK_MAX_THREADS)
        steps = -(-vectors // threads)
        return -(-vectors // (steps * lanes)) * lanes, num_per_thread

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype,
        eps: float,
        momentum: float,
        launch: tuple[int, int],
        presence: tuple[bool, bool, bool] = (True, True, True),
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.C = C
        self.dtype = dtype
        self.launch = launch
        self.kernel = _batch_norm_fwd_train_wide_kernel(
            N, C, S, self.dtype_str, eps, momentum, *presence
        )


class BatchNormFwdTrainSplitKernel(_BatchNormKernel, BatchNormTrainFwdInterface):
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
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    # How far either side of the seed the split count is offered to the tuner.
    _SEARCH_REACH = 4
    # Relative spread within which two timings tie.
    _TIE_BAND = 0.02
    # Powers of two only: T.reduce_sum lowers to an XOR butterfly.
    _SUM_THREADS = (128, 256, 512, 1024)

    @classmethod
    def refusal(cls, call: BatchNormCall) -> "str | None":
        holder = cls._holder(call, 1)
        if holder != "split":
            return f"a channel of this call is held by a {holder}, not a split"
        return super().refusal(call)

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
        presence = (call.has_weight, call.has_bias, call.has_running_stats)
        return (*args, presence, index), lambda: cls(*args, presence=presence, device_index=index)

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
        presence: tuple[bool, bool, bool] = (True, True, True),
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.N = N
        self.C = C
        self.S = S
        self.L = N * S
        self.splits = splits
        self.dtype = dtype
        self.stages = _batch_norm_fwd_train_split_kernel(
            N, C, S, self.dtype_str, eps, momentum, *presence
        )
        self.init_config(config)

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
        """Read sum, merge and map together per ``(splits, threads)`` and keep the fastest.

        The three launches share the pair, so they are read as one call, by the tuner's
        device-busy reading over *warmup* and then *rep* calls. Runs on tensors of its own
        and settles ``config`` before returning.
        """
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
            for candidate in self.tuning_candidates(self.autotune_configs):
                splits, threads = candidate["splits"], candidate["threads"]
                args = (x, stat(), stat(), weight, bias, stat(), stat())
                try:
                    run = self._launch_of(splits, threads)
                    elapsed = candidate_busy_ms(lambda run=run, args=args: run(*args), warmup, rep)
                except TUNING_FATAL:
                    raise
                except Exception as exc:  # a pair this shape's layout refuses
                    refused.append(f"splits={splits} threads={threads}: {exc}")
                    continue
                timed.append((elapsed, splits, threads))
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
        self._publish_tuned({"splits": best[1], "threads": best[2]})

    def _launch_of(self, splits: int, threads: int) -> Callable:
        """Sum, merge, then map -- three launches over an element-wide grid, run on
        ``(x, running_mean, running_var, weight, bias, mean_out, rstd_out)``."""
        stats, finalize, apply_ = self.stages
        num_per_thread = _vector_elements(self.dtype)
        span = threads * num_per_thread
        blocks = (self.C * self.L + span - 1) // span
        # Two vectors a step, at twice the map's threads: the Chan update is paid once for both.
        stats = CompiledProgram(stats(splits, min(2 * threads, 1024), 2 * num_per_thread))
        finalize = CompiledProgram(finalize(splits, min(256, self.C)))
        apply_ = CompiledProgram(apply_(blocks, threads, num_per_thread))
        C = self.C

        def launch(x, running_mean, running_var, weight, bias, mean_out, rstd_out):
            empty = functools.partial(torch.empty, device=x.device, dtype=torch.float32)
            partial_lead = empty((C, splits))
            partial_sum = empty((C, splits))
            partial_m2 = empty((C, splits))
            partial_unit = empty((C, splits))
            stats(x, partial_lead, partial_sum, partial_m2, partial_unit)
            coef = empty((C, 4))
            finalize(
                partial_lead,
                partial_sum,
                partial_m2,
                partial_unit,
                weight,
                bias,
                running_mean,
                running_var,
                mean_out,
                rstd_out,
                coef,
            )
            return apply_(x, coef)

        return launch

    def _launcher(self, cfg: dict) -> Callable:
        """What one launch under *cfg* runs; see :meth:`_launch_of`."""
        return self._launch_of(cfg["splits"], cfg["threads"])

    def forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor],
        running_var: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ):
        """Normalize an ``(N, C, S)`` input by its batch statistics.

        Returns:
            ``(y, mean, rstd)``: the ``(N, C, S)`` output and the per-channel batch
            mean and reciprocal std the backward pass reads.
        """
        self._require_devices(
            x=x, weight=weight, bias=bias, running_mean=running_mean, running_var=running_var
        )
        running_mean, running_var, weight, bias = _fill_absent(
            x, self.C, running_mean, running_var, weight, bias
        )
        mean_out = torch.empty(self.C, device=x.device, dtype=torch.float32)
        rstd_out = torch.empty_like(mean_out)
        y = self.program(self._launcher)(
            x, running_mean, running_var, weight, bias, mean_out, rstd_out
        )
        return y, mean_out, rstd_out


class BatchNormFwdTrainKernel(Kernel, BatchNormTrainFwdInterface):
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
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    supported_archs: list[int] = [80, 89, 90]
    general: bool = True

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (call.n, call.c, call.spatial, call.dtype, call.eps, call.momentum)
        index = call.device.index if call.device is not None else None
        presence = (call.has_weight, call.has_bias, call.has_running_stats)
        return (*args, presence, index), lambda: cls(*args, presence=presence, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        momentum: float = 0.1,
        config: Optional[dict] = None,
        presence: tuple[bool, bool, bool] = (True, True, True),
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.C = C
        self.L = N * S
        self.dtype = dtype
        self.kernel = _batch_norm_fwd_train_kernel(
            N, C, S, self.dtype_str, eps, momentum, *presence
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return _TiledPath.for_length(self.L)[0]

    @property
    def autotune_configs(self) -> list[dict]:
        return _TiledPath.for_length(self.L)

    def forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor],
        running_var: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ):
        """Normalize an ``(N, C, S)`` input by its batch statistics.

        Returns:
            ``(y, mean, rstd)``: the ``(N, C, S)`` output and the per-channel batch
            mean and reciprocal std the backward pass reads.
        """
        self._require_devices(
            x=x, weight=weight, bias=bias, running_mean=running_mean, running_var=running_var
        )
        running_mean, running_var, weight, bias = _fill_absent(
            x, self.C, running_mean, running_var, weight, bias
        )
        mean_out = torch.empty(self.C, device=x.device, dtype=torch.float32)
        rstd_out = torch.empty_like(mean_out)
        y = self.program(self._compile)(
            x, weight, bias, running_mean, running_var, mean_out, rstd_out
        )
        return y, mean_out, rstd_out

    def _compile(self, cfg: dict):
        """The program *cfg* tiles."""
        return self.kernel(cfg["block_l"], cfg["threads"])


# Inference forward


@target_cache(maxsize=32)
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

        @T.macro
        def _channel_affine(weight, bias, running_mean, running_var, ch, sc, sh):
            """Write channel *ch*'s scale into ``sc[0]`` and its shift into ``sh[0]``."""
            if input_dtype_params:
                mean_c = T.cast(T.cast(running_mean[ch], dtype), accum_dtype)
                var_c = T.cast(T.cast(running_var[ch], dtype), accum_dtype)
                weight_c = T.cast(weight[ch], accum_dtype) if has_weight else T.float32(1.0)
                bias_c = T.cast(bias[ch], accum_dtype) if has_bias else T.float32(0.0)
                sc[0] = T.rsqrt(var_c + T.cast(eps, accum_dtype)) * weight_c
                sh[0] = bias_c - mean_c * sc[0]
            else:
                sc[0] = weight[ch] / T.sqrt(running_var[ch] + T.cast(eps, accum_dtype))
                sh[0] = bias[ch] - running_mean[ch] * sc[0]

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
                if vector_holds_one_channel:
                    # A thread's vectors each stay in one channel, whose scale and shift it
                    # computes itself after issuing the read: no table and no barrier
                    # stand between the block's start and its loads.
                    v = T.alloc_local([steps * num_per_thread], dtype)
                    o = T.alloc_local([num_per_thread], dtype)
                    sc = T.alloc_local([1], accum_dtype)
                    sh = T.alloc_local([1], accum_dtype)
                    for k in T.unroll(steps):
                        base = bx * span + (k * threads + tx) * num_per_thread
                        if base + num_per_thread <= total:
                            for i in T.vectorized(num_per_thread):
                                v[k * num_per_thread + i] = x[base + i]
                    for k in T.unroll(steps):
                        base = bx * span + (k * threads + tx) * num_per_thread
                        if base + num_per_thread <= total:
                            ch = (base // S) % C
                            _channel_affine(weight, bias, running_mean, running_var, ch, sc, sh)
                            for i in T.unroll(num_per_thread):
                                o[i] = T.cast(
                                    T.cast(v[k * num_per_thread + i], accum_dtype) * sc[0] + sh[0],
                                    dtype,
                                )
                            for i in T.vectorized(num_per_thread):
                                y[base + i] = o[i]
                else:
                    # One scale and shift per channel turns the body into a single
                    # multiply-add, and the block builds that table once rather than
                    # every element recomputing a root and a divide.
                    scale = T.alloc_shared([C], accum_dtype)
                    shift = T.alloc_shared([C], accum_dtype)
                    sc = T.alloc_local([1], accum_dtype)
                    sh = T.alloc_local([1], accum_dtype)
                    for c in T.serial(T.ceildiv(C, threads)):
                        ch = c * threads + tx
                        if ch < C:
                            _channel_affine(weight, bias, running_mean, running_var, ch, sc, sh)
                            scale[ch] = sc[0]
                            shift[ch] = sh[0]
                    T.sync_threads()

                    v = T.alloc_local([num_per_thread], dtype)
                    o = T.alloc_local([num_per_thread], dtype)
                    # Each step is contiguous across the block.
                    for k in T.serial(steps):
                        base = bx * span + (k * threads + tx) * num_per_thread
                        if base + num_per_thread <= total:
                            for i in T.vectorized(num_per_thread):
                                v[i] = x[base + i]
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
                                        T.cast(x[base + i], accum_dtype) * scale[ch] + shift[ch],
                                        dtype,
                                    )

        return _bn_fwd_infer

    return _bn_fwd_infer_func


class BatchNormFwdInferKernel(Kernel, BatchNormInferFwdInterface, InstanceNormInferFwdInterface):
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
        return (*args, index), lambda: cls(*args, device_index=index)

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
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.total = N * C * S
        self.dtype = dtype
        self.kernel = _batch_norm_fwd_infer_kernel(
            N, C, S, self.dtype_str, eps, input_dtype_params, has_weight, has_bias
        )
        self.init_config(config)

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
        self._require_devices(
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
        return self.program(self._compile)(x, weight, bias, running_mean, running_var)

    def _compile(self, cfg: dict):
        """The program *cfg* tiles."""
        return self.kernel(cfg["threads"], cfg["num_per_thread"], cfg["steps"])


# Backward


@target_cache(maxsize=32)
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


@target_cache(maxsize=32)
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


@target_cache(maxsize=32)
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
        # A vector never straddles two batch items, so its address stays affine.
        vector_holds_one_item = S % num_per_thread == 0 and -(-L // splits) % num_per_thread == 0

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
                held_g = T.alloc_local([num_per_thread], dtype)
                held_x = T.alloc_local([num_per_thread], dtype)
                for _i, j in T.Parallel(1, threads):
                    for step in T.serial(T.ceildiv(chunk, threads * num_per_thread)):
                        head = start + (step * threads + j) * num_per_thread
                        # The chunk bound is tested once a vector, so the vector loads in one
                        # access.
                        if head + num_per_thread <= end and vector_holds_one_item:
                            for i in T.vectorized(num_per_thread):
                                held_g[i] = grad_out[head // S, bc, head % S + i]
                            for i in T.vectorized(num_per_thread):
                                held_x[i] = x[head // S, bc, head % S + i]
                            for i in T.serial(num_per_thread):
                                g = T.cast(held_g[i], accum_dtype)
                                x_hat = (T.cast(held_x[i], accum_dtype) - mean_val) * rstd_val
                                sums[_i, j] += g
                                sums_xhat[_i, j] += g * x_hat
                        else:
                            for i in T.serial(num_per_thread):
                                l = head + i
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


class BatchNormBwdWideKernel(_BatchNormKernel, BatchNormBwdInterface):
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
    def refusal(cls, call: BatchNormCall) -> "str | None":
        holder = cls._holder(call, 2)
        if holder not in ("thread", "block"):
            return f"a channel of this call is held by a {holder}, not a thread or a block"
        return super().refusal(call)

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
        self._require_devices(grad_out=grad_out, x=x, weight=weight, mean=mean, rstd=rstd)
        grad_weight = torch.empty(self.C, device=x.device, dtype=torch.float32)
        grad_bias = torch.empty_like(grad_weight)
        grad_x = self.program(self._compile)(
            grad_out, x, weight, mean, rstd, grad_weight, grad_bias
        )
        return grad_x, grad_weight, grad_bias

    def _compile(self, cfg: dict):
        """The program, launched as ``self.launch`` says."""
        return self.kernel(*self.launch)


class BatchNormBwdSplitKernel(_BatchNormKernel, BatchNormBwdInterface):
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
    def refusal(cls, call: BatchNormCall) -> "str | None":
        holder = cls._holder(call, 2)
        if holder != "split":
            return f"a channel of this call is held by a {holder}, not a split"
        return super().refusal(call)

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
        self._require_devices(grad_out=grad_out, x=x, weight=weight, mean=mean, rstd=rstd)
        grad_weight = torch.empty(self.C, device=x.device, dtype=torch.float32)
        grad_bias = torch.empty_like(grad_weight)
        grad_x = self.program(self._launcher)(
            grad_out, x, weight, mean, rstd, grad_weight, grad_bias
        )
        return grad_x, grad_weight, grad_bias

    def _launcher(self, cfg: dict) -> Callable:
        """Sum, merge, then map under *cfg* -- three launches over an element-wide grid, run
        on ``(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)``."""
        stats, finalize, apply_ = self.stages
        splits, threads = cfg["splits"], cfg["threads"]
        num_per_thread = _vector_elements(self.dtype)
        span = threads * num_per_thread
        blocks = (self.C * self.L + span - 1) // span
        stats = CompiledProgram(stats(splits, threads, num_per_thread))
        finalize = CompiledProgram(finalize(splits, min(256, self.C)))
        apply_ = CompiledProgram(apply_(blocks, threads, num_per_thread))
        C = self.C

        def launch(grad_out, x, weight, mean, rstd, grad_weight, grad_bias):
            empty = functools.partial(torch.empty, device=x.device, dtype=torch.float32)
            partial_sum = empty((C, splits))
            partial_sum_xhat = empty((C, splits))
            stats(grad_out, x, mean, rstd, partial_sum, partial_sum_xhat)
            scale, shift, centered_coef = empty(C), empty(C), empty(C)
            finalize(
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
            return apply_(grad_out, x, mean, scale, shift, centered_coef)

        return launch


class BatchNormBwdKernel(Kernel, BatchNormBwdInterface):
    """Backward with one channel per block, streamed through shared memory.

    The general implementation: it serves every shape the specialised ones do not.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Elements per channel in one batch item, ``product(spatial)``.
        dtype: grad_out/x/grad_x data type.
        config: Optional ``{"block_l", "threads"}``.
        device_index: CUDA device the kernel runs on; ``None`` is the current one.
    """

    supported_archs: list[int] = [80, 89, 90]
    general: bool = True

    @classmethod
    def entry_for(cls, call: BatchNormCall) -> Entry:
        args = (call.n, call.c, call.spatial, call.dtype)
        index = call.device.index if call.device is not None else None
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.C = C
        self.L = N * S
        self.dtype = dtype
        self.kernel = _batch_norm_bwd_kernel(N, C, S, self.dtype_str)
        self.init_config(config)

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
        self._require_devices(grad_out=grad_out, x=x, weight=weight, mean=mean, rstd=rstd)
        grad_weight = torch.empty(self.C, device=x.device, dtype=torch.float32)
        grad_bias = torch.empty_like(grad_weight)
        program = self.program(self._compile)
        grad_x = program(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)
        return grad_x, grad_weight, grad_bias

    def _compile(self, cfg: dict):
        """The program *cfg* tiles."""
        return self.kernel(cfg["block_l"], cfg["threads"])
