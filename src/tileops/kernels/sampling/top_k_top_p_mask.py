"""Per-row top-k then top-p logit mask on one kernel.

The top-k select is ``top_k_mask.py``'s, copied rather than shared: its pieces are regions of
one prim_func, and the two files hold them at different points of the same kernel body.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import MAX_PORTABLE_CLUSTER_BLOCKS, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import SamplingCall, TopKTopPMaskFwdInterface
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["TopKTopPMaskFwdKernel"]


@functools.lru_cache(maxsize=32)
def _top_k_top_p_mask_kernel(batch: int, vocab: int, dtype: str):
    """Build the top-k then top-p mask of ``batch`` rows of ``vocab`` logits.

    A row's cluster pads its last vectors with -inf, which no rank below ``vocab`` reaches.
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
    # The key a NaN takes, above every number's.
    nan_key = (1 << bits) - 1
    # One pass of either select settles one digit of the key, over this many bins.
    radix_bits = 8
    bins = 1 << radix_bits
    digits = bits // radix_bits
    # What the cluster reduces per counting pass: one count per digit value, plus the keys
    # whose digit is above the ones the pass counts.
    slabs = bins + 1
    # Bins one level of the nucleus walk splits its key range into. A range narrower than
    # this settles the bound in one level, which is what a row whose k leaves a handful of
    # values gives; a wider table trades its own cost against the levels it saves. Re-fit by
    # timing the manifest rows: at [256, 128256] float16 the walk takes 262.2 us over 256
    # bins, 205.7 over 512 and 235.0 over 1024, and at bfloat16 249.2, 262.6 and 321.7.
    walk_bins = 512
    walk_bits = walk_bins.bit_length() - 1
    # How far the bracket taken from the samples reaches around the expected sample rank of
    # the k-th key: this many standard deviations of a binomial count, plus a constant that
    # covers the ranks whose expected count is a handful. A bracket that misses costs one more
    # pass and changes no result, so the two trade retries against keys counted. Re-fit by
    # timing the manifest rows over sigmas in {2, 3, 4, 6} and slack in {1, 3, 8}.
    sigmas, slack = 4.0, 3.0

    @tilelang.jit(
        compile_flags=[
            "-include",
            csrc_path("streaming_load.h"),
            "-include",
            csrc_path("top_k_mask_helper.h"),
        ]
    )
    def _top_k_top_p_mask_func(threads: int, cluster: int, slots: int):
        chunk = slots * threads * vec
        assert cluster * chunk >= vocab
        held = slots * words
        warps = threads // WARP_LANES
        assert threads % WARP_LANES == 0 and bins % WARP_LANES == 0
        span = bins // WARP_LANES
        walk_span = walk_bins // WARP_LANES
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

        def key_of(vals, i, h):
            """Key ``h`` of word ``i`` of the held keys."""
            if packed:
                return (vals[i] >> T.uint32(16 * h)) & T.uint32(0xFFFF)
            return vals[i]

        def value_of(key):
            """The float32 value a key stands for: the key transform is its own inverse."""
            back = T.if_then_else(
                (key & T.uint32(sign)) != T.uint32(0), T.uint32(sign), T.uint32(nan_key)
            )
            if packed:
                return T.cast(T.reinterpret(T.cast(key ^ back, "uint16"), dtype), "float32")
            return T.reinterpret(key ^ back, "float32")

        def keeps_both_zeros(key):
            """``key``, lowered to the key of -0.0 when it is the key of either zero, so that
            a bound on one zero keeps the other, as a float comparison does."""
            return T.if_then_else(
                (key == T.uint32(sign)) | (key == T.uint32(sign - 1)),
                T.uint32(sign - 1),
                key,
            )

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

        @T.macro
        def merge(hist, total, sum32, q, tx):
            """Sum slab ``q`` of every CTA of the cluster into ``total``."""
            if cluster == 1:
                T.sync_threads()
                for i in T.serial(-(-slabs // threads)):
                    if i * threads + tx < slabs:
                        total[i * threads + tx] = hist[q * slabs + i * threads + tx]
            else:
                T.cluster_sync()
                for i in T.serial(-(-slabs // threads)):
                    if i * threads + tx < slabs:
                        sum32[0] = T.uint32(0)
                        for c in T.serial(cluster):
                            sum32[0] = sum32[0] + T.call_extern(
                                "uint32",
                                "tl::tileops_cluster_load_u32",
                                T.address_of(hist[q * slabs + i * threads + tx]),
                                c,
                            )
                        total[i * threads + tx] = sum32[0]
            T.sync_threads()

        @T.macro
        def merge_weights(wbins, wtot, weight, q, tx):
            """Sum bin table ``q`` of every CTA of the cluster into ``wtot``, in float32: a
            remote word arrives as its bits and is added as the weight it stands for."""
            if cluster == 1:
                T.sync_threads()
                for i in T.serial(-(-walk_bins // threads)):
                    if i * threads + tx < walk_bins:
                        wtot[i * threads + tx] = wbins[q * walk_bins + i * threads + tx]
            else:
                T.cluster_sync()
                for i in T.serial(-(-walk_bins // threads)):
                    if i * threads + tx < walk_bins:
                        weight[0] = T.float32(0)
                        for c in T.serial(cluster):
                            weight[0] = weight[0] + T.reinterpret(
                                T.call_extern(
                                    "uint32",
                                    "tl::tileops_cluster_load_u32",
                                    T.address_of(wbins[q * walk_bins + i * threads + tx]),
                                    c,
                                ),
                                "float32",
                            )
                        wtot[i * threads + tx] = weight[0]
            T.sync_threads()

        @T.macro
        def rank_in_bins(total, target, acc, tx):
            """Warp 0: into ``acc``, the bin of ``total`` holding rank ``target`` counted from
            the largest, that rank inside it, and the bin's count.

            ``acc[0]`` stays -1 when the bins hold fewer than ``target`` keys.
            """
            acc[0] = -1
            acc[3] = 0
            for i in T.serial(span):
                acc[3] = acc[3] + T.cast(total[tx * span + i], "int32")
            acc[1] = acc[3]
            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                up = T.shfl_down(acc[1], 1 << stage, width=WARP_LANES)
                if tx + (1 << stage) < WARP_LANES:
                    acc[1] = acc[1] + up
            # This lane's bins hold the ranks in (acc[1] - acc[3], acc[1]].
            acc[2] = acc[1] - acc[3]
            if (acc[2] < target) & (target <= acc[1]):
                for i in T.serial(span):
                    b = tx * span + span - 1 - i
                    count = T.cast(total[b], "int32")
                    if (acc[2] < target) & (target <= acc[2] + count):
                        acc[0] = b
                        acc[1] = target - acc[2]
                        acc[3] = count
                    acc[2] = acc[2] + count

        @T.macro
        def bin_holding_weight(wtot, target, hit, outside, lane, upward, running, tx):
            """Warp 0: into ``hit[0]`` the highest bin of ``wtot`` that its own weight and the
            weight above it reach ``target``, and into ``outside[0]`` the weight above that
            bin. ``hit[0]`` stays -1 when the bins never reach ``target``.
            """
            hit[0] = -1
            lane[0] = T.float32(0)
            for i in T.serial(walk_span):
                lane[0] = lane[0] + wtot[tx * walk_span + i]
            upward[0] = lane[0]
            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                up = T.shfl_down(upward[0], 1 << stage, width=WARP_LANES)
                if tx + (1 << stage) < WARP_LANES:
                    upward[0] = upward[0] + up
            running[0] = upward[0] - lane[0]
            if (running[0] < target) & (target <= upward[0]):
                for i in T.serial(walk_span):
                    b = tx * walk_span + walk_span - 1 - i
                    if hit[0] < 0:
                        outside[0] = running[0]
                        running[0] = running[0] + wtot[b]
                        if running[0] >= target:
                            hit[0] = b

        @T.prim_func
        def _top_k_top_p_mask_main(
            x: T.Tensor((n,), dtype),
            k: T.Tensor((batch,), "int32"),
            p: T.Tensor((batch,), "float32"),
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
                weight = T.alloc_local((1,), "float32")
                seen_any = T.alloc_local((1,), "uint32")
                seen_number = T.alloc_local((1,), "uint32")
                merged = T.alloc_local((1,), "uint32")
                # The pass's shift and settled bits, and the digit values it counts, read
                # once: every mention of a shared element is a load of it, and the loop over
                # the held keys mentions each of them per key.
                digit_shift = T.alloc_local((1,), "uint32")
                settled = T.alloc_local((1,), "uint32")
                counted = T.alloc_local((2,), "int32")
                bound = T.alloc_local((1,), "uint32")
                drops_nan = T.alloc_local((1,), "uint32")
                high = T.alloc_local((1,), "float32")
                hist = T.alloc_shared((2 * slabs,), "uint32")
                total = T.alloc_shared((slabs,), "uint32")
                wbins = T.alloc_shared((2 * walk_bins,), "float32")
                wtot = T.alloc_shared((walk_bins,), "float32")
                red = T.alloc_shared((warps,), "int32")
                wred = T.alloc_shared((warps,), "float32")
                kred = T.alloc_shared((2 * warps,), "uint32")
                # The maximum over every key of the row and the maximum over the keys that
                # are not NaN, packed so that a row reads the one its own ``k`` asks for by
                # index and two threads write both.
                rowmax = T.alloc_shared((2,), "uint32")
                pre = T.alloc_shared((1,), "uint32")
                lowest = T.alloc_shared((1,), "uint32")
                highest = T.alloc_shared((1,), "uint32")
                nucleus = T.alloc_shared((1,), "uint32")
                above_weight = T.alloc_shared((1,), "float32")
                goal = T.alloc_shared((1,), "float32")
                chosen = T.alloc_shared((1,), "int32")
                settled_bound = T.alloc_shared((1,), "int32")
                hit = T.alloc_local((1,), "int32")
                outside = T.alloc_local((1,), "float32")
                lane = T.alloc_local((1,), "float32")
                upward = T.alloc_local((1,), "float32")
                running = T.alloc_local((1,), "float32")
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

                # The key of each value: its bits with the sign flipped when it is
                # non-negative and every bit flipped when it is negative, so keys order as
                # the values do, and every NaN at the largest key.
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

                # The row's largest key and its largest key that is not NaN. The padding a
                # cluster leaves is -inf, whose key is below every value the row holds.
                seen_any[0] = T.uint32(0)
                seen_number[0] = T.uint32(0)
                for i in T.unroll(held):
                    for h in T.unroll(per_word):
                        seen_any[0] = T.max(seen_any[0], key_of(vals, i, h))
                        if key_of(vals, i, h) != T.uint32(nan_key):
                            seen_number[0] = T.max(seen_number[0], key_of(vals, i, h))
                for stage in T.unroll(WARP_SHUFFLE_STAGES):
                    reach = T.int32(WARP_LANES // 2) >> stage
                    seen_any[0] = T.max(
                        seen_any[0], T.shfl_xor(seen_any[0], reach, width=WARP_LANES)
                    )
                    seen_number[0] = T.max(
                        seen_number[0], T.shfl_xor(seen_number[0], reach, width=WARP_LANES)
                    )
                if tx % WARP_LANES == 0:
                    kred[2 * (tx // WARP_LANES)] = seen_any[0]
                    kred[2 * (tx // WARP_LANES) + 1] = seen_number[0]
                T.sync_threads()
                if tx < 2:
                    merged[0] = T.uint32(0)
                    for w in T.serial(warps):
                        merged[0] = T.max(merged[0], kred[2 * w + tx])
                    rowmax[tx] = merged[0]

                kk = k[row]
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
                merge(hist, total, sum32, 0, tx)
                if tx < WARP_LANES:
                    expected = T.cast(kk, "float32") * T.float32(sampled / vocab)
                    spread = T.float32(sigmas) * T.sqrt(expected) + T.float32(slack)
                    rank_in_bins(total, T.cast(T.floor(expected - spread), "int32"), acc, tx)
                    if acc[0] >= 0:
                        st[7] = acc[0]
                    rank_in_bins(total, T.cast(T.ceil(expected + spread), "int32"), acc, tx)
                    if acc[0] >= 0:
                        st[6] = acc[0]
                if tx == 0:
                    pre[0] = T.uint32(0)
                    st[0] = kk
                    # A row k leaves whole keeps every key, NaN included, and takes no select.
                    st[1] = T.if_then_else(kk >= vocab, 1, 0)
                    st[5] = 0
                T.sync_threads()

                for q in T.serial(rounds):
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
                        slab = (q + 1) % 2
                        for i in T.serial(-(-slabs // threads)):
                            if i * threads + tx < slabs:
                                hist[slab * slabs + i * threads + tx] = T.uint32(0)
                        if tx == 0:
                            st[4] = 0
                            st[8] = 0
                        T.sync_threads()
                        above[0] = 0
                        for i in T.unroll(held):
                            for h in T.unroll(per_word):
                                key = key_of(vals, i, h)
                                # Two shifts, so the first digit does not shift by the
                                # whole key width.
                                moved = key >> digit_shift[0]
                                if (moved >> T.uint32(radix_bits)) == settled[0]:
                                    digit = T.cast(moved & T.uint32(bins - 1), "int32")
                                    if digit > counted[1]:
                                        above[0] = above[0] + 1
                                    elif digit >= counted[0]:
                                        T.atomic_add(hist[slab * slabs + digit], T.uint32(1))
                        for stage in T.unroll(WARP_SHUFFLE_STAGES):
                            above[0] = above[0] + T.shfl_xor(above[0], 1 << stage, width=WARP_LANES)
                        if tx % WARP_LANES == 0:
                            red[tx // WARP_LANES] = above[0]
                        T.sync_threads()
                        if tx == 0:
                            acc[0] = 0
                            for w in T.serial(warps):
                                acc[0] = acc[0] + red[w]
                            hist[slab * slabs + bins] = T.cast(acc[0], "uint32")
                        merge(hist, total, sum32, slab, tx)
                        if tx < WARP_LANES:
                            # The keys above the counted digits already outrank the
                            # target, so the counted bins hold what is left of it.
                            acc[0] = -1
                            if T.cast(total[bins], "int32") < st[0]:
                                rank_in_bins(total, st[0] - T.cast(total[bins], "int32"), acc, tx)
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

                if cluster > 1:
                    T.cluster_sync()
                    if tx < 2:
                        merged[0] = T.uint32(0)
                        for c in T.serial(cluster):
                            merged[0] = T.max(
                                merged[0],
                                T.call_extern(
                                    "uint32",
                                    "tl::tileops_cluster_load_u32",
                                    T.address_of(rowmax[tx]),
                                    c,
                                ),
                            )
                        seen_any[0] = merged[0]
                    T.sync_threads()
                    if tx < 2:
                        rowmax[tx] = seen_any[0]
                    T.sync_threads()
                if tx == 0:
                    lowest[0] = keeps_both_zeros(pre[0])
                    highest[0] = rowmax[T.if_then_else(kk >= vocab, 0, 1)]
                    above_weight[0] = T.float32(0)
                    goal[0] = T.float32(0)
                    settled_bound[0] = 0
                T.sync_threads()

                # The walk weighs each surviving key by the softmax of the values the top-k
                # filter left, taken from their largest. A row whose largest is not a number,
                # +inf or -inf weighs every key NaN, so no bin ever reaches the target, the
                # walk narrows nothing and the top-k mask stands: that is what the
                # reference's softmax of such a row leaves.
                high[0] = value_of(rowmax[T.if_then_else(kk >= vocab, 0, 1)])
                for level in T.serial(digits):
                    if settled_bound[0] == 0:
                        slab = level % 2
                        settled[0] = lowest[0]
                        bound[0] = highest[0]
                        # The bins cover the range in ``walk_bins`` steps of this width, so
                        # a range narrower than that settles the bound in this one level.
                        digit_shift[0] = T.cast(
                            T.max(
                                0,
                                32
                                - walk_bits
                                - T.call_extern(
                                    "int32",
                                    "__clz",
                                    T.cast(bound[0] - settled[0], "int32"),
                                ),
                            ),
                            "uint32",
                        )
                        for i in T.serial(-(-walk_bins // threads)):
                            if i * threads + tx < walk_bins:
                                wbins[slab * walk_bins + i * threads + tx] = T.float32(0)
                        T.sync_threads()
                        for i in T.unroll(held):
                            for h in T.unroll(per_word):
                                key = key_of(vals, i, h)
                                if (key >= settled[0]) & (key <= bound[0]):
                                    T.atomic_add(
                                        wbins[
                                            slab * walk_bins
                                            + T.cast((key - settled[0]) >> digit_shift[0], "int32")
                                        ],
                                        T.exp(value_of(key) - high[0]),
                                    )
                        # The atomic accumulation has to settle before a peer reads this table.
                        T.sync_threads()
                        merge_weights(wbins, wtot, weight, slab, tx)
                        if level == 0:
                            # The whole surviving weight: what p is a fraction of.
                            weight[0] = T.float32(0)
                            for i in T.serial(-(-walk_bins // threads)):
                                if i * threads + tx < walk_bins:
                                    weight[0] = weight[0] + wtot[i * threads + tx]
                            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                                weight[0] = weight[0] + T.shfl_xor(
                                    weight[0], 1 << stage, width=WARP_LANES
                                )
                            if tx % WARP_LANES == 0:
                                wred[tx // WARP_LANES] = weight[0]
                            T.sync_threads()
                            if tx == 0:
                                lane[0] = T.float32(0)
                                for w in T.serial(warps):
                                    lane[0] = lane[0] + wred[w]
                                goal[0] = p[row] * lane[0]
                            T.sync_threads()
                        if tx == 0:
                            chosen[0] = -1
                        T.sync_threads()
                        if tx < WARP_LANES:
                            bin_holding_weight(
                                wtot,
                                goal[0] - above_weight[0],
                                hit,
                                outside,
                                lane,
                                upward,
                                running,
                                tx,
                            )
                            if hit[0] >= 0:
                                chosen[0] = hit[0]
                                above_weight[0] = above_weight[0] + outside[0]
                        T.sync_threads()
                        if tx == 0:
                            # No bin reaching the target leaves every key of the range in the
                            # nucleus, which is its smallest key and settles the bound.
                            if chosen[0] >= 0:
                                lowest[0] = settled[0] + (
                                    T.cast(chosen[0], "uint32") << digit_shift[0]
                                )
                                highest[0] = T.min(
                                    bound[0],
                                    lowest[0] + ((T.uint32(1) << digit_shift[0]) - T.uint32(1)),
                                )
                            settled_bound[0] = T.if_then_else(
                                (chosen[0] < 0) | (digit_shift[0] == T.uint32(0)), 1, 0
                            )
                        T.sync_threads()
                if tx == 0:
                    nucleus[0] = keeps_both_zeros(lowest[0])
                T.sync_threads()

                bound[0] = nucleus[0] * T.uint32(halves)
                # A row k leaves whole keeps its NaNs, as the reference's softmax of a row
                # holding one removes nothing; any other row masks them.
                drops_nan[0] = T.if_then_else(kk < vocab, T.uint32(0xFFFFFFFF), T.uint32(0))
                # Each value at least the bound keeps its bits; the rest become -inf.
                for j in T.unroll(slots):
                    for c in T.unroll(words):
                        key = vals[j * words + c]
                        if packed:
                            back = T.uint32(0xFFFFFFFF) - (
                                (key >> T.uint32(15)) & T.uint32(halves)
                            ) * T.uint32(0x7FFF)
                            keep = T.call_extern("uint32", "__vcmpgeu2", key, bound[0]) & ~(
                                T.call_extern("uint32", "__vcmpeq2", key, T.uint32(0xFFFFFFFF))
                                & drops_nan[0]
                            )
                            outw[c] = ((key ^ back) & keep) | (T.uint32(word_neg_inf) & ~keep)
                        else:
                            outw[c] = T.if_then_else(
                                (key < bound[0])
                                | ((key == T.uint32(nan_key)) & (drops_nan[0] != T.uint32(0))),
                                T.uint32(word_neg_inf),
                                key
                                ^ T.if_then_else(
                                    (key >> T.uint32(31)) != T.uint32(0),
                                    T.uint32(0x80000000),
                                    T.uint32(0xFFFFFFFF),
                                ),
                            )
                    store_slot(outw, j, out_of_row, out, base)
                if cluster > 1:
                    # Peers read this CTA's counts, so it stays alive to the last read.
                    T.cluster_arrive()
                    T.cluster_wait()

        return _top_k_top_p_mask_main

    return _top_k_top_p_mask_func


class TopKTopPMaskFwdKernel(Kernel, TopKTopPMaskFwdInterface):
    """Mask each row of logits to its top ``k``, then to the nucleus ``p`` of those, reading
    and writing each row once.

    A cluster of CTAs holds a row in registers. It selects the row's k-th largest key one
    8-bit digit at a time, with one sample per thread bracketing the digit values the k-th
    key can take, then narrows the key range above that threshold, weighted by the softmax
    of the keys the filter left, until the smallest key the nucleus keeps is settled. Both
    bounds are a key, so the mask is one comparison per element.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``threads``, ``cluster`` and ``slots``.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [90]
    general = True

    # Launch policy, fitted on the manifest rows with the repo benchmark. Re-fit by timing
    # threads in {128, 256, 512, 1024} against the cluster widths a row admits.
    _THREADS: ClassVar[int] = 1024
    # Most 16-byte vectors a thread holds before its registers cost the SM a resident block;
    # a row that needs more spreads over more CTAs instead. Re-fit by sweeping (threads,
    # cluster, slots) on the manifest rows: at [256, 128256] bfloat16, 4 slots on 4 CTAs run
    # 258.9 us, 8 slots on 2 CTAs 372.1 and 16 slots on 1 CTA 750.7.
    _MAX_SLOTS: ClassVar[int] = 4

    @classmethod
    def refusal(cls, call: SamplingCall) -> Optional[str]:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.batch * call.vocab > 2**31 - 1:
            return f"indexes elements with int32, and B * V = {call.batch * call.vocab}"
        widest = cls._widest_row(call.dtype)
        if call.vocab > widest:
            return f"holds a row of at most {widest} values in registers, and V = {call.vocab}"
        return None

    @classmethod
    def _widest_row(cls, dtype: torch.dtype) -> int:
        """The longest row the launch policy holds without spilling a thread's registers."""
        return (
            MAX_PORTABLE_CLUSTER_BLOCKS
            * cls._THREADS
            * (VECTOR_ACCESS_BYTES // dtype.itemsize)
            * cls._MAX_SLOTS
        )

    def __init__(self, call: SamplingCall, config: Optional[dict] = None, tune: bool = False):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self.kernel = _top_k_top_p_mask_kernel(call.batch, call.vocab, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """As many CTAs per row as it takes to fill the device, each holding an equal run.

        A cluster costs a barrier across its CTAs on every pass, so a batch that already
        fills the device puts one CTA on each row and only a batch short of it spreads a row
        wider; a row too long to hold in ``_MAX_SLOTS`` per thread spreads wider still. The
        run is then exactly the row's share, since a slot no key lands in is a register the
        passes scan for nothing.
        """
        vec = VECTOR_ACCESS_BYTES // self.call.dtype.itemsize
        cluster = 1
        while (
            cluster < MAX_PORTABLE_CLUSTER_BLOCKS
            and 2 * cluster * self.call.batch <= self.call.sm_count
        ):
            cluster *= 2
        slots = -(-self.call.vocab // (cluster * self._THREADS * vec))
        while cluster < MAX_PORTABLE_CLUSTER_BLOCKS and slots > self._MAX_SLOTS:
            cluster *= 2
            slots = -(-self.call.vocab // (cluster * self._THREADS * vec))
        return {"threads": self._THREADS, "cluster": cluster, "slots": slots}

    def forward(self, logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        self._require_cuda(logits=logits, k=k, p=p)
        out = torch.empty_like(logits)
        if logits.numel() == 0:
            return out
        # A row that is a whole number of 16-byte vectors is read as vectors, from the
        # start of the storage; any other row is read element by element.
        vec = VECTOR_ACCESS_BYTES // self.call.dtype.itemsize
        if self.call.vocab % vec == 0 and logits.data_ptr() % VECTOR_ACCESS_BYTES:
            logits = logits.clone()
        self.kernel(**self.config)(logits.view(-1), k, p, out.view(-1))
        return out
