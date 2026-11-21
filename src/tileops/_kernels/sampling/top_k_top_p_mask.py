"""Per-row top-k then top-p logit mask on one kernel."""

from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sampling.call_spec import SamplingCall, TopKTopPMaskFwdInterface
from tileops._kernels.sampling.radix_select import cluster_plan, key_select, row_refusal
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["TopKTopPMaskFwdKernel"]


@target_cache(maxsize=32)
def _top_k_top_p_mask_kernel(batch: int, vocab: int, dtype: str):
    """Build the top-k then top-p mask of ``batch`` rows of ``vocab`` logits."""
    n = batch * vocab
    # Bins one level of the nucleus walk splits its key range into. A range narrower than
    # this settles the bound in one level, which is what a row whose k leaves a handful of
    # values gives; a wider table trades its own cost against the levels it saves.
    walk_bins = 512
    walk_bits = walk_bins.bit_length() - 1

    @tilelang.jit(compile_flags=csrc_include("streaming_load.h", "top_k_mask_helper.h"))
    def _top_k_top_p_mask_func(threads: int, cluster: int, slots: int):
        rows = key_select(vocab, dtype, threads, cluster, slots)
        words, held, per_word, packed = rows.words, rows.held, rows.per_word, rows.packed
        bits, sign, halves, word_neg_inf = rows.bits, rows.sign, rows.halves, rows.word_neg_inf
        warps = rows.warps
        walk_span = walk_bins // WARP_LANES
        # The key a NaN takes, above every number's.
        nan_key = (1 << bits) - 1

        def value_of(key):
            """The float32 value a key stands for: the key transform is its own inverse."""
            back = T.if_then_else(
                (key & T.uint32(sign)) != T.uint32(0), T.uint32(sign), T.uint32(nan_key)
            )
            if packed:
                return T.cast(T.reinterpret(T.cast(key ^ back, "uint16"), dtype), "float32")
            return T.reinterpret(key ^ back, "float32")

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
                                    "tileops::cluster_load_u32",
                                    T.address_of(wbins[q * walk_bins + i * threads + tx]),
                                    c,
                                ),
                                "float32",
                            )
                        wtot[i * threads + tx] = weight[0]
            T.sync_threads()

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
                digit_shift = T.alloc_local((1,), "uint32")
                settled = T.alloc_local((1,), "uint32")
                counted = T.alloc_local((2,), "int32")
                bound = T.alloc_local((1,), "uint32")
                drops_nan = T.alloc_local((1,), "uint32")
                high = T.alloc_local((1,), "float32")
                hist = T.alloc_shared((2 * rows.slabs,), "uint32")
                total = T.alloc_shared((rows.slabs,), "uint32")
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
                st = T.alloc_shared((9,), "int32")

                row = bx if cluster == 1 else bx // cluster
                rank = 0 if cluster == 1 else T.block_rank_in_cluster()
                base = row * vocab + rank * rows.chunk
                out_of_row = vocab - rank * rows.chunk

                rows.load_row(vals, x, base, out_of_row)
                rows.to_keys(vals)

                # The row's largest key and its largest key that is not NaN. The padding a
                # cluster leaves is -inf, whose key is below every value the row holds.
                seen_any[0] = T.uint32(0)
                seen_number[0] = T.uint32(0)
                for i in T.unroll(held):
                    for h in T.unroll(per_word):
                        seen_any[0] = T.max(seen_any[0], rows.key_of(vals, i, h))
                        if rows.key_of(vals, i, h) != T.uint32(nan_key):
                            seen_number[0] = T.max(seen_number[0], rows.key_of(vals, i, h))
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
                rows.select_kth(
                    vals,
                    kk,
                    # A row k leaves whole keeps every key, NaN included, and takes no select.
                    T.if_then_else(kk >= vocab, 1, 0),
                    out_of_row,
                    hist,
                    total,
                    red,
                    pre,
                    st,
                    acc,
                    sum32,
                    above,
                    digit_shift,
                    settled,
                    counted,
                )

                if cluster > 1:
                    T.cluster_sync()
                    if tx < 2:
                        merged[0] = T.uint32(0)
                        for c in T.serial(cluster):
                            merged[0] = T.max(
                                merged[0],
                                T.call_extern(
                                    "uint32",
                                    "tileops::cluster_load_u32",
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
                    lowest[0] = rows.keeps_both_zeros(pre[0])
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
                for level in T.serial(rows.digits):
                    if settled_bound[0] == 0:
                        slab = level % 2
                        settled[0] = lowest[0]
                        bound[0] = highest[0]
                        # The bins cover the range in ``walk_bins`` steps of this width, so
                        # a range narrower than that settles the bound in this one level.
                        digit_shift[0] = T.cast(
                            T.max(
                                0,
                                32 - walk_bits - T.clz(T.cast(bound[0] - settled[0], "int32")),
                            ),
                            "uint32",
                        )
                        for i in T.serial(-(-walk_bins // threads)):
                            if i * threads + tx < walk_bins:
                                wbins[slab * walk_bins + i * threads + tx] = T.float32(0)
                        T.sync_threads()
                        for i in T.unroll(held):
                            for h in T.unroll(per_word):
                                key = rows.key_of(vals, i, h)
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
                    nucleus[0] = rows.keeps_both_zeros(lowest[0])
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
                    rows.store_slot(outw, j, out_of_row, out, base)
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
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general = True

    # Block width for the supported cluster sizes.
    _THREADS: ClassVar[int] = 1024
    # Cap register-held vectors; larger rows spread over more CTAs to preserve occupancy.
    _MAX_SLOTS: ClassVar[int] = 4
    # Slots a thread holds where `cluster_limit` keeps a row in one CTA: the fewest that hold
    # llama 3's float32 vocabulary, 128256 values, at `_THREADS`. Those past the registers
    # spill to local memory, which every pass then reads back.
    _MAX_SLOTS_ONE_CTA: ClassVar[int] = 32

    @classmethod
    def refusal(cls, call: SamplingCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        return row_refusal(call, cls._THREADS, cls._MAX_SLOTS, cls._MAX_SLOTS_ONE_CTA)

    def __init__(self, call: SamplingCall, config: dict | None = None):
        super().__init__(device_index=call.device_index)
        self.call = call
        self.dtype = call.dtype
        self.kernel = _top_k_top_p_mask_kernel(call.batch, call.vocab, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return cluster_plan(self.call, self._THREADS, self._MAX_SLOTS)

    def forward(self, logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        self._require_devices(logits=logits, k=k, p=p)
        out = torch.empty_like(logits)
        if logits.numel() == 0:
            return out
        self.program(self._compile)(logits.view(-1), k, p, out.view(-1))
        return out

    def _compile(self, cfg: dict):
        """The program of config *cfg*."""
        return self.kernel(**cfg)
