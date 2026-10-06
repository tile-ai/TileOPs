"""Per-row top-k logit mask: a cluster of CTAs holds each row in registers and selects its
k-th largest value by a most-significant-digit radix select over the key bits."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel, vector_aligned
from tileops.kernels.sampling.call_spec import SamplingCall, TopKMaskFwdInterface
from tileops.kernels.sampling.radix_select import (
    BRACKET_SIGMAS,
    BRACKET_SLACK,
    cluster_limit,
    cluster_plan,
    merge_counts,
    rank_in_bins,
    widest_row,
)
from tileops.utils import WARP_LANES

__all__ = ["TopKMaskFwdKernel"]


@functools.lru_cache(maxsize=32)
def _top_k_mask_kernel(batch: int, vocab: int, dtype: str):
    """Build the top-k mask of ``batch`` rows of ``vocab`` logits.

    A value is held as its key: its bits with the sign bit flipped when it is non-negative
    and every bit flipped when it is negative, so keys order as the values do, and every
    NaN takes the largest key, as ``sort`` places NaN first. A row's cluster pads its last
    vectors with -inf, which no rank below ``vocab`` reaches.
    """
    n = batch * vocab
    itemsize = torch.empty((), dtype=getattr(torch, dtype)).element_size()
    vec = VECTOR_ACCESS_BYTES // itemsize
    words = VECTOR_ACCESS_BYTES // 4
    per_word = 4 // itemsize
    packed = per_word == 2
    aligned = vocab % vec == 0
    bits = 8 * itemsize
    sign = 1 << (bits - 1)
    inf_bits = {"float16": 0x7C00, "bfloat16": 0x7F80, "float32": 0x7F800000}[dtype]
    neg_inf = sign | inf_bits
    # Each 16-bit half of a word, or the whole word.
    halves = 0x00010001 if packed else 1
    word_neg_inf = neg_inf * halves
    # One pass of the select settles one digit of the key, over this many bins.
    radix_bits = 8
    bins = 1 << radix_bits
    digits = bits // radix_bits
    # What the cluster reduces per pass: one count per digit value, plus the keys whose
    # digit is above the ones the pass counts.
    slabs = bins + 1

    @tilelang.jit(
        compile_flags=[
            "-include",
            csrc_path("streaming_load.h"),
            "-include",
            csrc_path("top_k_mask_helper.h"),
        ]
    )
    def _top_k_mask_func(threads: int, cluster: int, slots: int):
        chunk = slots * threads * vec
        assert cluster * chunk >= vocab
        held = slots * words
        warps = threads // WARP_LANES
        assert threads % WARP_LANES == 0 and bins % WARP_LANES == 0
        # The threads whose first element is inside the row are the ones that sample it, and
        # their count is the population a sample rank stands for.
        if aligned:
            in_row = [max(0, min(threads, -(-(vocab - c * chunk) // vec))) for c in range(cluster)]
        else:
            in_row = [max(0, min(threads, vocab - c * chunk)) for c in range(cluster)]
        sampled = sum(in_row)
        # Every digit, plus one retry of the first for a bracket that missed.
        rounds = digits + 1

        def element(j, c, h):
            """The offset in the CTA's chunk of half ``h`` of word ``c`` of slot ``j``."""
            if aligned:
                return (j * threads + T.get_thread_binding()) * vec + c * per_word + h
            return ((j * words + c) * per_word + h) * threads + T.get_thread_binding()

        @T.macro
        def store_slot(word, j, out_of_row, dst, at):
            if aligned:
                if element(j, 0, 0) < out_of_row:
                    T.call_extern(
                        "handle",
                        "tl::tileops_store16",
                        T.address_of(dst[at + element(j, 0, 0)]),
                        T.address_of(word[0]),
                    )
            else:
                for c in T.unroll(words):
                    for h in T.unroll(per_word):
                        if element(j, c, h) < out_of_row:
                            if packed:
                                dst[at + element(j, c, h)] = T.reinterpret(
                                    T.cast(
                                        (word[c] >> T.uint32(16 * h)) & T.uint32(0xFFFF), "uint16"
                                    ),
                                    dtype,
                                )
                            else:
                                dst[at + element(j, c, h)] = T.reinterpret(word[c], dtype)

        @T.prim_func
        def _top_k_mask_main(
            x: T.Tensor((n,), dtype),
            k: T.Tensor((batch,), "int32"),
            out: T.Tensor((n,), dtype),
        ):
            with (
                T.Kernel(batch, threads=threads)
                if cluster == 1
                else T.ClusterKernel(batch * cluster, cluster_dims=cluster, threads=threads)
            ) as bx:
                tx = T.get_thread_binding()
                vals = T.alloc_local((held,), "uint32")
                outw = T.alloc_local((words,), "uint32")
                acc = T.alloc_local((4,), "int32")
                sum32 = T.alloc_local((1,), "uint32")
                above = T.alloc_local((1,), "int32")
                # The pass's shift and settled bits, and the digit values it counts, read
                # once: every mention of a shared element is a load of it, and the loop over
                # the held keys mentions each of them per key.
                digit_shift = T.alloc_local((1,), "uint32")
                settled = T.alloc_local((1,), "uint32")
                counted = T.alloc_local((2,), "int32")
                hist = T.alloc_shared((2 * slabs,), "uint32")
                total = T.alloc_shared((slabs,), "uint32")
                red = T.alloc_shared((warps,), "int32")
                # The key bits the select has settled.
                pre = T.alloc_shared((1,), "uint32")
                # 0 the rank still to find inside the settled bits, 1 whether the select has
                # finished, 2 the winning bin and 3 the rank inside it, 4 whether this pass
                # settled the threshold, 5 the digit being settled, 6 and 7 the lowest and
                # highest digit value the pass counts, 8 whether the pass found the rank.
                # Every CTA of a cluster reduces the same counts, so all of them finish on
                # the same pass.
                st = T.alloc_shared((9,), "int32")

                row = bx if cluster == 1 else bx // cluster
                rank = 0 if cluster == 1 else T.block_rank_in_cluster()
                base = row * vocab + rank * chunk
                out_of_row = vocab - rank * chunk

                for j in T.unroll(slots):
                    if aligned:
                        if element(j, 0, 0) < out_of_row:
                            T.call_extern(
                                "handle",
                                "tl::tileops_load16_evict_first",
                                T.address_of(vals[j * words]),
                                T.address_of(x[base + element(j, 0, 0)]),
                            )
                        else:
                            for c in T.unroll(words):
                                vals[j * words + c] = T.uint32(word_neg_inf)
                    else:
                        for c in T.unroll(words):
                            vals[j * words + c] = T.uint32(word_neg_inf)
                            for h in T.unroll(per_word):
                                if element(j, c, h) < out_of_row:
                                    if packed:
                                        piece = T.cast(
                                            T.reinterpret(x[base + element(j, c, h)], "uint16"),
                                            "uint32",
                                        )
                                        vals[j * words + c] = (
                                            vals[j * words + c]
                                            & T.uint32(0xFFFF0000 if h == 0 else 0x0000FFFF)
                                        ) | (piece << T.uint32(16 * h))
                                    else:
                                        vals[j * words + c] = T.reinterpret(
                                            x[base + element(j, c, h)], "uint32"
                                        )
                kk = k[row]
                if kk >= vocab:
                    for j in T.unroll(slots):
                        for c in T.unroll(words):
                            outw[c] = vals[j * words + c]
                        store_slot(outw, j, out_of_row, out, base)
                else:
                    # The key of each value: its bits with the sign flipped when it is
                    # non-negative and every bit flipped when it is negative, so keys order
                    # as the values do, and every NaN at the largest key.
                    for i in T.unroll(held):
                        if packed:
                            flip = ((vals[i] >> T.uint32(15)) & T.uint32(halves)) * T.uint32(0x7FFF)
                            nan = T.call_extern(
                                "uint32",
                                "__vcmpgtu2",
                                vals[i] & T.uint32(0x7FFF7FFF),
                                T.uint32(inf_bits * halves),
                            )
                            vals[i] = (vals[i] ^ (flip | T.uint32(0x80008000))) | nan
                        else:
                            vals[i] = T.if_then_else(
                                (vals[i] & T.uint32(0x7FFFFFFF)) > T.uint32(inf_bits),
                                T.uint32(0xFFFFFFFF),
                                vals[i]
                                ^ T.if_then_else(
                                    (vals[i] >> T.uint32(31)) != T.uint32(0),
                                    T.uint32(0xFFFFFFFF),
                                    T.uint32(0x80000000),
                                ),
                            )

                    # The digit values that can hold the k-th key, from one sample per thread.
                    for i in T.serial(-(-slabs // threads)):
                        if i * threads + tx < slabs:
                            hist[i * threads + tx] = T.uint32(0)
                    if tx == 0:
                        st[6] = 0
                        st[7] = bins - 1
                    T.sync_threads()
                    if element(0, 0, 0) < out_of_row:
                        sample = (vals[0] & T.uint32(0xFFFF)) if packed else vals[0]
                        T.atomic_add(
                            hist[T.cast(sample >> T.uint32(bits - radix_bits), "int32")],
                            T.uint32(1),
                        )
                    merge_counts(hist, total, sum32, 0, tx, cluster, slabs, threads)
                    if tx < WARP_LANES:
                        expected = T.cast(kk, "float32") * T.float32(sampled / vocab)
                        spread = T.float32(BRACKET_SIGMAS) * T.sqrt(expected) + T.float32(
                            BRACKET_SLACK
                        )
                        rank_in_bins(
                            total,
                            T.cast(T.floor(expected - spread), "int32"),
                            acc,
                            tx,
                            bins // WARP_LANES,
                        )
                        if acc[0] >= 0:
                            st[7] = acc[0]
                        rank_in_bins(
                            total,
                            T.cast(T.ceil(expected + spread), "int32"),
                            acc,
                            tx,
                            bins // WARP_LANES,
                        )
                        if acc[0] >= 0:
                            st[6] = acc[0]
                    if tx == 0:
                        pre[0] = T.uint32(0)
                        st[0] = kk
                        st[1] = 0
                        st[5] = 0
                    T.sync_threads()

                    for p in T.serial(rounds):
                        if st[1] == 0:
                            # The digit this pass settles starts here; the bits below it are
                            # what a pass that settles the threshold leaves unset.
                            digit_shift[0] = T.uint32(digits - 1 - st[5]) * T.uint32(radix_bits)
                            settled[0] = pre[0]
                            counted[0] = st[6]
                            counted[1] = st[7]
                            # ``hist`` alternates, so a peer still reading the counts of the
                            # phase before never sees this pass zero them. The samples hold
                            # slab 0, so the first pass takes slab 1.
                            q = (p + 1) % 2
                            for i in T.serial(-(-slabs // threads)):
                                if i * threads + tx < slabs:
                                    hist[q * slabs + i * threads + tx] = T.uint32(0)
                            if tx == 0:
                                st[4] = 0
                                st[8] = 0
                            T.sync_threads()
                            above[0] = 0
                            for i in T.unroll(held):
                                for h in T.unroll(per_word):
                                    if packed:
                                        key = (vals[i] >> T.uint32(16 * h)) & T.uint32(0xFFFF)
                                    else:
                                        key = vals[i]
                                    # Two shifts, so the first digit does not shift by the
                                    # whole key width.
                                    moved = key >> digit_shift[0]
                                    if (moved >> T.uint32(radix_bits)) == settled[0]:
                                        digit = T.cast(moved & T.uint32(bins - 1), "int32")
                                        if digit > counted[1]:
                                            above[0] = above[0] + 1
                                        elif digit >= counted[0]:
                                            T.atomic_add(hist[q * slabs + digit], T.uint32(1))
                            for stage in T.unroll(WARP_LANES.bit_length() - 1):
                                above[0] = above[0] + T.shfl_xor(
                                    above[0], 1 << stage, width=WARP_LANES
                                )
                            if tx % WARP_LANES == 0:
                                red[tx // WARP_LANES] = above[0]
                            T.sync_threads()
                            if tx == 0:
                                acc[0] = 0
                                for w in T.serial(warps):
                                    acc[0] = acc[0] + red[w]
                                hist[q * slabs + bins] = T.cast(acc[0], "uint32")
                            merge_counts(hist, total, sum32, q, tx, cluster, slabs, threads)
                            if tx < WARP_LANES:
                                # The keys above the counted digits already outrank the
                                # target, so the counted bins hold what is left of it.
                                acc[0] = -1
                                if T.cast(total[bins], "int32") < st[0]:
                                    rank_in_bins(
                                        total,
                                        st[0] - T.cast(total[bins], "int32"),
                                        acc,
                                        tx,
                                        bins // WARP_LANES,
                                    )
                                if acc[0] >= 0:
                                    st[2] = acc[0]
                                    st[3] = acc[1]
                                    st[8] = 1
                                    # One key carries this prefix, so its remaining bits
                                    # cannot change which keys the mask keeps.
                                    if acc[3] == 1:
                                        st[4] = 1
                            T.sync_threads()
                            if tx == 0:
                                # A bracket that did not hold the rank counts every digit on
                                # the retry, which always holds it.
                                st[6] = 0
                                st[7] = bins - 1
                                if st[8] == 1:
                                    # A pass that settled the threshold leaves the bits below
                                    # its digit zero, the smallest key carrying it.
                                    pre[0] = (
                                        (pre[0] << T.uint32(radix_bits)) | T.cast(st[2], "uint32")
                                    ) << T.if_then_else(st[4] == 1, digit_shift[0], T.uint32(0))
                                    st[0] = st[3]
                                    st[5] = st[5] + 1
                                # Set last: the parser re-reads the guard of this pass's own
                                # region after every barrier inside it.
                                st[1] = T.if_then_else((st[4] == 1) | (st[5] == digits), 1, 0)
                            T.sync_threads()
                    # A threshold on either zero keeps both zeros, as a float compare does.
                    thr = T.if_then_else(
                        (pre[0] == T.uint32(sign)) | (pre[0] == T.uint32(sign - 1)),
                        T.uint32(sign - 1),
                        pre[0],
                    ) * T.uint32(halves)
                    # Each value at least ``thr`` keeps its bits; the rest, and every NaN,
                    # become -inf.
                    for j in T.unroll(slots):
                        for c in T.unroll(words):
                            key = vals[j * words + c]
                            if packed:
                                back = T.uint32(0xFFFFFFFF) - (
                                    (key >> T.uint32(15)) & T.uint32(halves)
                                ) * T.uint32(0x7FFF)
                                keep = T.call_extern(
                                    "uint32", "__vcmpgeu2", key, thr
                                ) & ~T.call_extern("uint32", "__vcmpeq2", key, T.uint32(0xFFFFFFFF))
                                outw[c] = ((key ^ back) & keep) | (T.uint32(word_neg_inf) & ~keep)
                            else:
                                outw[c] = T.if_then_else(
                                    (key >= thr) & (key != T.uint32(0xFFFFFFFF)),
                                    key
                                    ^ T.if_then_else(
                                        (key >> T.uint32(31)) != T.uint32(0),
                                        T.uint32(0x80000000),
                                        T.uint32(0xFFFFFFFF),
                                    ),
                                    T.uint32(word_neg_inf),
                                )
                        store_slot(outw, j, out_of_row, out, base)
                if cluster > 1:
                    # Peers read this CTA's counts, so it stays alive to the last read.
                    T.cluster_arrive()
                    T.cluster_wait()

        return _top_k_mask_main

    return _top_k_mask_func


class TopKMaskFwdKernel(Kernel, TopKMaskFwdInterface):
    """Mask each row of logits to its ``k`` largest, reading and writing each row once.

    A cluster of CTAs holds a row in registers and selects its k-th largest key one 8-bit
    digit at a time. One sample per thread brackets the digit values the k-th key can take,
    so a pass counts only the keys near it; a bracket that misses costs one more pass and
    changes no result. The select is exact for every input, ties and NaN included.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``threads``, ``cluster`` and ``slots``.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general = True

    # Launch policy, fitted on the manifest rows with the repo benchmark. Re-fit by timing
    # threads in {128, 256, 512, 1024} against the cluster widths a row admits.
    _THREADS: ClassVar[int] = 1024
    # Most 16-byte vectors a thread holds before its registers cost the SM a resident block.
    _MAX_SLOTS: ClassVar[int] = 16
    # Slots a thread holds where `cluster_limit` keeps a row in one CTA: the fewest that hold
    # llama 3's float32 vocabulary, 128256 values, at `_THREADS`. Those past the registers
    # spill to local memory, which every pass then reads back.
    _MAX_SLOTS_ONE_CTA: ClassVar[int] = 32

    @classmethod
    def refusal(cls, call: SamplingCall) -> Optional[str]:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.batch * call.vocab > 2**31 - 1:
            return f"indexes elements with int32, and B * V = {call.batch * call.vocab}"
        slots = cls._MAX_SLOTS if cluster_limit(call.arch) > 1 else cls._MAX_SLOTS_ONE_CTA
        widest = widest_row(call.dtype, cls._THREADS, slots, call.arch)
        if call.vocab > widest:
            return f"supports rows of at most {widest} values, and V = {call.vocab}"
        return None

    def __init__(self, call: SamplingCall, config: Optional[dict] = None, tune: bool = False):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self.kernel = _top_k_mask_kernel(call.batch, call.vocab, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return cluster_plan(self.call, self._THREADS, self._MAX_SLOTS)

    def forward(self, logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        self._require_cuda(logits=logits, k=k)
        out = torch.empty_like(logits)
        if logits.numel() == 0:
            return out
        # A row that is a whole number of 16-byte vectors is read as vectors, from the
        # start of the storage; any other row is read element by element.
        vec = VECTOR_ACCESS_BYTES // self.call.dtype.itemsize
        if self.call.vocab % vec == 0:
            logits = vector_aligned(logits)
        self.kernel(**self.config)(logits.view(-1), k, out.view(-1))
        return out
