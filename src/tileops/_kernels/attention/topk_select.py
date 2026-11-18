import dataclasses
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface

__all__ = ["TopKSelectCall", "TopKSelectFwdInterface", "TopKSelectKernel"]


def shared_add(dst, value):
    """``atomicAdd`` on a shared-memory element, returning its old value."""
    return T.call_extern(  # lint: extern-target allow: T.atomic_add's return is device-scope
        "int32", "atomicAdd", T.address_of(dst), value
    )


def shared_min(dst, value):
    """``atomicMin`` on a shared-memory element."""
    return T.call_extern(  # lint: extern-target allow: T.atomic_min is device-scope
        "uint32", "atomicMin", T.address_of(dst), value
    )


def shared_max(dst, value):
    """``atomicMax`` on a shared-memory element."""
    return T.call_extern(  # lint: extern-target allow: T.atomic_max is device-scope
        "uint32", "atomicMax", T.address_of(dst), value
    )


def half_key(x):
    """A score's fp16 value as an order-preserving int32, -32768 .. 32767: its bits read as a
    signed integer, with the magnitude flipped below zero."""
    bits = T.Cast(T.int32, T.reinterpret(T.Cast(T.float16, x), T.int16))
    return bits ^ ((bits >> 31) & 0x7FFF)


def float_key(x):
    """A score's float32 value as an order-preserving uint32."""
    bits = T.reinterpret(T.Cast(T.float32, x), T.uint32)
    return T.if_then_else(
        (bits & T.Cast(T.uint32, 0x80000000)) != T.Cast(T.uint32, 0),
        ~bits,
        bits | T.Cast(T.uint32, 0x80000000),
    )


def warp_sum(value):
    """The sum of an int32 over the warp's 32 lanes, on every lane."""
    return T.call_extern(  # lint: extern-target allow: one redux.sync, not five shuffles
        "int32", "__reduce_add_sync", T.Cast(T.uint32, 0xFFFFFFFF), value
    )


@dataclasses.dataclass(frozen=True)
class TopKSelectCall(CallSpec):
    """One windowed top-k selection over a $[B \\times S \\times S\\_kv \\times G]$ score tensor."""

    batch: int = 0
    seq_len: int = 0
    seq_len_kv: int = 0
    kv_group: int = 0
    topk: int = 0
    fill_value: int | None = None
    dtype: torch.dtype | None = None
    out_dtype: torch.dtype | None = None


class TopKSelectFwdInterface(KernelInterface):
    """The highest-scoring key positions of each query row's own window."""

    request = TopKSelectCall

    @abstractmethod
    def forward(
        self, index_score: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        """Select each row's top ``call.topk`` keys; nothing is written in place.

        Row ``(b, s, g)`` selects from ``index_score[b, s, starts[b, s]:ends[b, s], g]``. The
        positions come back in no particular order along the ``topk`` axis, and a window
        holding fewer than ``call.topk`` positions fills the rest with ``call.fill_value``,
        ``call.seq_len_kv`` when that is ``None``. ``call.topk`` may exceed
        ``call.seq_len_kv``. Nothing is written in place, and the output aliases no input.

        Args:
            index_score: The scores,
                ``(call.batch, call.seq_len, call.seq_len_kv, call.kv_group)`` in
                ``call.dtype`` on ``call.device``, contiguous in that axis order.
            starts: The first key of each window, ``int32``
                ``(call.batch, call.seq_len)`` on ``call.device``, contiguous.
            ends: One past the last key of each window, in the same shape, dtype, device
                and layout.

        Returns:
            A new contiguous ``(call.batch, call.seq_len, call.kv_group, call.topk)`` tensor
            in ``call.out_dtype`` on ``call.device``.
        """


@target_cache(maxsize=32)
def _topk_select_kernel(
    batch,
    seq_len,
    seq_len_kv,
    kv_group,
    topk,
    fill_value,
    in_dtype,
    out_dtype,
    bin_cache=65536,
    stage_size=2048,
):
    # Keys a thread loads at once, adjacent.
    VEC = 4 if kv_group == 1 and seq_len_kv % 4 == 0 else 1

    @tilelang.jit(
        out_idx=[1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
    )
    def topk_select_fwd_func(BLOCK_SIZE=1024):
        def bin8(x):
            # The fp16 value's top byte as an order-preserving bin, 0 .. 255.
            return (half_key(x) >> 8) + 128

        def any_at_least(word, thr):
            # Bit 7 of each byte of word that is at least thr, 0 .. 255: b >= thr when
            # b + (256 - thr) carries out of the byte.
            c4 = T.Cast(T.uint32, 256 - thr) * T.Cast(T.uint32, 0x01010101)
            low = (word & T.Cast(T.uint32, 0x7F7F7F7F)) + (c4 & T.Cast(T.uint32, 0x7F7F7F7F))
            return T.if_then_else(
                thr == 0,
                T.Cast(T.uint32, 0x80808080),
                ((word & c4) | ((word | c4) & low)) & T.Cast(T.uint32, 0x80808080),
            )

        batch = T.dynamic("batch")
        # Dynamic: a static extent lets index arithmetic over 2^31 elements fold into int32.
        seq_len_kv = T.dynamic("seq_len_kv")
        # The index a slot no candidate fills holds.
        FILL = seq_len_kv if fill_value is None else fill_value
        RADIX = 1 << 8
        # Threshold-bin candidates a round stages, each with its 32-bit key so the next
        # round reads shared memory, not the scores; more re-read the window instead.
        SMEM_INPUT_SIZE = stage_size
        # Loads a thread issues before it bins any of them: one window of global latency
        # covers ITEMS keys instead of each key paying its own.
        ITEMS = 8
        # Keys ``0 .. bin_cache - 1`` keep their stage-1 bin in shared memory, one byte
        # each, four adjacent keys' to a word: the collect pass re-reads those bins instead
        # of the scores, and reads the scores of any key past them again.
        BIN_CACHE = bin_cache

        @T.prim_func
        def _topk_select_kernel_main(
            index_score: T.Tensor[(batch, seq_len, seq_len_kv, kv_group), in_dtype],
            index: T.Tensor[(batch, seq_len, kv_group, topk), out_dtype],
            starts: T.Tensor[(batch, seq_len), out_dtype],
            ends: T.Tensor[(batch, seq_len), out_dtype],
        ):
            # Parallelize over seq rows by assigning one block per (batch, seq_row, kv_group).
            with T.Kernel(batch, seq_len, kv_group, threads=BLOCK_SIZE) as (bx, by, g):
                # A full SM of threads, which holds only at 32 registers a thread.
                T.annotate_min_blocks_per_sm(2048 // BLOCK_SIZE)
                if VEC == 4:
                    # Rows start 16-byte aligned: the stage-1 loads go four keys at a time.
                    T.assume(seq_len_kv % 4 == 0)
                tx = T.get_thread_binding()
                # by is the seq row index (one block per row; no m_i loop)
                seq_row = by

                s_threshold_bin_id = T.alloc_shared([1], T.int32)
                # Two histograms: a level's suffix sums are its output position counters
                # while its pass builds the next level's counts in the other.
                # Bin RADIX + 1 counts the keys outside the window.
                s_hist = T.alloc_shared([2, RADIX + 2], T.int32)
                s_warp_sum = T.alloc_shared([RADIX // 32], T.int32)
                s_num_input = T.alloc_shared([2], T.int32)
                # The slots a round has filled: keys above its threshold bin, and ties.
                s_cnt = T.alloc_shared([2], T.int32)
                # The candidates' smallest and largest keys, the smallest the rounds measure
                # from, and the bits past it they differ in.
                s_kminmax = T.alloc_shared([2], T.uint32)
                s_kmin = T.alloc_shared([1], T.uint32)
                s_bits = T.alloc_shared([1], T.int32)
                s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)
                s_input_key = T.alloc_shared([2, SMEM_INPUT_SIZE], T.uint32)
                l_keep = T.alloc_var(T.int32)
                # The stage-1 threshold bucket, and the key bytes the refining rounds
                # have settled. Held in shared memory: a register each would cost the
                # block its second residency on an SM.
                s_refine = T.alloc_shared([2], T.uint32)
                s_bin = T.alloc_shared([BIN_CACHE // 4], T.uint32)
                l_scores = T.alloc_local([ITEMS], T.float32)

                l_threshold_bin_id = T.alloc_var(T.int32)
                l_new_topk = T.alloc_var(T.int32)
                l_num_input = T.alloc_var(T.int32)
                l_bin_id32 = T.alloc_var(T.int32)
                l_val = T.alloc_var(T.int32)
                l_shfl = T.alloc_var(T.int32)
                l_start_pos = T.alloc_var(T.int32)
                l_start_idx = T.alloc_var(T.int32)
                l_end_idx = T.alloc_var(T.int32)
                l_out_pos = T.alloc_var(T.int32)
                l_pos = T.alloc_var(T.int32)
                l_score = T.alloc_var(T.float32)
                l_key = T.alloc_var(T.uint32)
                l_cand = T.alloc_var(T.int32)
                l_span = T.alloc_var(T.int32)
                l_p = T.alloc_var(T.int32)
                l_mask = T.alloc_var(T.uint32)
                l_rank = T.alloc_var(T.int32)
                l_base = T.alloc_var(T.int32)
                l_kmin = T.alloc_var(T.uint32)
                l_kmax = T.alloc_var(T.uint32)
                l_word = T.alloc_var(T.uint32)
                l_flags = T.alloc_var(T.uint32)
                l_bits = T.alloc_var(T.int32)
                # The lanes below this one, whose ballot bits give its rank.
                l_lanes_below = (T.Cast(T.uint32, 1) << T.Cast(T.uint32, tx % 32)) - T.Cast(
                    T.uint32, 1
                )

                l_new_topk = topk
                l_start_idx = starts[bx, seq_row]
                l_end_idx = ends[bx, seq_row]

                # A window of at most topk keys selects all of them, in order, and fills the
                # slots past them with FILL, which the consumer of this list treats as
                # selecting nothing. A longer window fills every slot with a selected key.
                if l_end_idx - l_start_idx <= topk:
                    for j in T.serial(T.ceildiv(topk, BLOCK_SIZE)):
                        slot = j * BLOCK_SIZE + tx
                        if slot < topk:
                            index[bx, seq_row, g, slot] = T.if_then_else(
                                slot < l_end_idx - l_start_idx, l_start_idx + slot, FILL
                            )
                else:
                    # Stage 1: count each key's 8-bit bin.

                    for j in T.serial(T.ceildiv(2 * (RADIX + 2), BLOCK_SIZE)):
                        histogram_idx = j * BLOCK_SIZE + tx
                        if histogram_idx < 2 * (RADIX + 2):
                            s_hist[histogram_idx // (RADIX + 2), histogram_idx % (RADIX + 2)] = 0
                    if tx == 0:
                        s_num_input[0] = 0
                        s_threshold_bin_id[0] = -1
                        s_kminmax[0] = T.Cast(T.uint32, 0xFFFFFFFF)
                        s_kminmax[1] = T.Cast(T.uint32, 0)

                    T.sync_threads()

                    # Only the chunks the window touches are read, by one uniform test a chunk
                    # over the whole row. A thread takes four adjacent keys at a time, one vector
                    # load, and keeps their bins as one word. A key outside the window counts in
                    # bin RADIX + 1, which keeps the loop free of branches.
                    chunk = BLOCK_SIZE * ITEMS
                    for s in T.serial(T.ceildiv(seq_len_kv, chunk)):
                        if (s + 1) * chunk > l_start_idx and s * chunk < l_end_idx:
                            for v in T.unroll(ITEMS // 4):
                                base = s * chunk + (v * BLOCK_SIZE + tx) * 4
                                if VEC == 4:
                                    # One 16-byte load: seq_len_kv % 4 == 0 aligns every row.
                                    if base < seq_len_kv:
                                        for e in T.vectorized(4):
                                            l_scores[v * 4 + e] = index_score[
                                                bx, seq_row, base + e, g
                                            ]
                                else:
                                    for e in T.unroll(4):
                                        if base + e < seq_len_kv:
                                            l_scores[v * 4 + e] = index_score[
                                                bx, seq_row, base + e, g
                                            ]
                            for v in T.unroll(ITEMS // 4):
                                base = s * chunk + (v * BLOCK_SIZE + tx) * 4
                                l_word = T.Cast(T.uint32, 0)
                                for e in T.unroll(4):
                                    l_bin_id32 = bin8(l_scores[v * 4 + e])
                                    T.atomic_add(
                                        s_hist[
                                            0,
                                            T.Select(
                                                T.Cast(T.uint32, base + e - l_start_idx)
                                                < T.Cast(T.uint32, l_end_idx - l_start_idx),
                                                l_bin_id32,
                                                RADIX + 1,
                                            ),
                                        ],
                                        1,
                                    )
                                    l_word = l_word | (T.Cast(T.uint32, l_bin_id32) << (8 * e))
                                if base < BIN_CACHE:
                                    s_bin[base // 4] = l_word
                    T.sync_threads()

                    # Suffix sums: each warp scans its 32 bins down by shuffles, then adds the
                    # totals of the warps above it.
                    if tx < RADIX:
                        l_val = s_hist[0, tx]
                        for i in T.unroll(5):
                            l_shfl = T.shfl_down(l_val, 1 << i)
                            if tx % 32 + (1 << i) < 32:
                                l_val = l_val + l_shfl
                        if tx % 32 == 0:
                            s_warp_sum[tx // 32] = l_val
                        T.sync_threads(3, RADIX)
                        for j in T.unroll(RADIX // 32 - 1):
                            if tx // 32 + j + 1 < RADIX // 32:
                                l_val = l_val + s_warp_sum[tx // 32 + j + 1]
                        s_hist[0, tx] = l_val
                        T.sync_threads(3, RADIX)
                        if s_hist[0, tx] > l_new_topk and s_hist[0, tx + 1] <= l_new_topk:
                            s_threshold_bin_id[0] = tx
                    T.sync_threads()
                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_hist[0, l_threshold_bin_id + 1]
                    T.sync_threads()

                    # Collect every key above the threshold bin and stage the threshold bin's,
                    # a word of four bins at a time: a word with none at or above the threshold
                    # bin, most of a long row's, is passed over at once.
                    # Avoid in-loop block barriers on dynamic serial loops: this can deadlock
                    # on newer TileLang codegen.
                    for s in T.serial(T.ceildiv(seq_len_kv, 4 * BLOCK_SIZE)):
                        word_idx = s * BLOCK_SIZE + tx
                        if word_idx * 4 < l_end_idx and word_idx * 4 + 4 > l_start_idx:
                            if word_idx * 4 < BIN_CACHE:
                                l_word = s_bin[word_idx]
                            else:
                                l_word = T.Cast(T.uint32, 0)
                                for e in T.unroll(4):
                                    if word_idx * 4 + e < seq_len_kv:
                                        l_word = l_word | (
                                            T.Cast(
                                                T.uint32,
                                                bin8(index_score[bx, seq_row, word_idx * 4 + e, g]),
                                            )
                                            << (8 * e)
                                        )
                            l_flags = any_at_least(l_word, l_threshold_bin_id)
                            if l_flags != T.Cast(T.uint32, 0):
                                for e in T.unroll(4):
                                    input_idx = word_idx * 4 + e
                                    l_bin_id32 = T.Cast(T.int32, (l_word >> (8 * e)) & 0xFF)
                                    if (
                                        (l_flags >> (8 * e + 7)) & T.Cast(T.uint32, 1)
                                        != T.Cast(T.uint32, 0)
                                        and input_idx >= l_start_idx
                                        and input_idx < l_end_idx
                                    ):
                                        if l_bin_id32 > l_threshold_bin_id:
                                            l_pos = shared_add(s_hist[0, l_bin_id32 + 1], 1)
                                            if l_pos < topk:
                                                index[bx, seq_row, g, l_pos] = input_idx
                                        elif l_new_topk > 0:
                                            l_pos = shared_add(s_num_input[0], 1)
                                            if l_pos < SMEM_INPUT_SIZE:
                                                s_input_idx[0, l_pos] = input_idx
                                                # Copied asynchronously, so the pass does not
                                                # wait on each score.
                                                T.ptx_cp_async(
                                                    T.access_ptr(s_input_key[0, l_pos], "w"),
                                                    T.access_ptr(
                                                        index_score[bx, seq_row, input_idx, g], "r"
                                                    ),
                                                    1,
                                                )

                    if tx == 0:
                        s_refine[0] = T.Cast(T.uint32, l_threshold_bin_id)
                        s_refine[1] = T.Cast(T.uint32, 0)

                    # The staged candidates' keys. The rounds refine only the key bits the
                    # candidates differ in, from their smallest key: a round over bits the stage-1
                    # bin already fixes sorts nothing. More candidates than the buffer holds refine
                    # every key bit, counted from the window.
                    if l_new_topk > 0:
                        T.ptx_wait_group(0)
                        T.sync_threads()
                        rescan = s_num_input[0] > SMEM_INPUT_SIZE
                        l_num_input = T.min(s_num_input[0], SMEM_INPUT_SIZE)
                        if rescan:
                            for s in T.serial(T.ceildiv(seq_len_kv, BLOCK_SIZE)):
                                slot = s * BLOCK_SIZE + tx
                                if slot < l_end_idx and slot >= l_start_idx and slot < seq_len_kv:
                                    l_score = index_score[bx, seq_row, slot, g]
                                    if bin8(l_score) == l_threshold_bin_id:
                                        T.atomic_add(
                                            s_hist[1, T.Cast(T.int32, float_key(l_score) >> 24)],
                                            1,
                                        )
                            if tx == 0:
                                s_kmin[0] = T.Cast(T.uint32, 0)
                                s_bits[0] = 32
                        else:
                            l_kmin = T.Cast(T.uint32, 0xFFFFFFFF)
                            l_kmax = T.Cast(T.uint32, 0)
                            for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                                slot = s * BLOCK_SIZE + tx
                                if slot < l_num_input:
                                    l_key = float_key(
                                        T.reinterpret(s_input_key[0, slot], T.float32)
                                    )
                                    s_input_key[0, slot] = l_key
                                    l_kmin = T.min(l_kmin, l_key)
                                    l_kmax = T.max(l_kmax, l_key)
                            for i in T.unroll(5):
                                l_kmin = T.min(l_kmin, T.shfl_xor(l_kmin, 1 << i))
                                l_kmax = T.max(l_kmax, T.shfl_xor(l_kmax, 1 << i))
                            if tx % 32 == 0:
                                shared_min(s_kminmax[0], l_kmin)
                                shared_max(s_kminmax[1], l_kmax)
                            T.sync_threads()
                            if tx == 0:
                                s_kmin[0] = s_kminmax[0]
                                s_bits[0] = 32 - T.Cast(T.int32, T.clz(s_kminmax[1] - s_kminmax[0]))
                            T.sync_threads()
                            l_kmin = s_kmin[0]
                            l_bits = s_bits[0]
                            for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                                slot = s * BLOCK_SIZE + tx
                                if slot < l_num_input:
                                    T.atomic_add(
                                        s_hist[
                                            1,
                                            T.Cast(
                                                T.int32,
                                                (
                                                    (s_input_key[0, slot] - l_kmin)
                                                    >> T.max(l_bits - 8, 0)
                                                )
                                                & 0xFF,
                                            ),
                                        ],
                                        1,
                                    )

                    # stage 2: tail pass. Round r's histogram, in s_hist[(r + 1) % 2], was counted
                    # by the pass before it; its own pass counts round r + 1's in s_hist[r % 2].
                    # Round r takes the 8 bits from bit max(bits - 8 (r + 1), 0) of a key less the
                    # smallest; the round reaching bit 0 is the last, and settles the ties.
                    for round in T.serial(4):
                        if l_new_topk <= 0:
                            T.loop_break()

                        r_idx = round % 2
                        cur = (round + 1) % 2

                        T.sync_threads()
                        l_kmin = s_kmin[0]
                        l_bits = s_bits[0]
                        shift = T.max(l_bits - 8 * (round + 1), 0)
                        last = l_bits - 8 * (round + 1) <= 0
                        for j in T.serial(T.ceildiv(RADIX + 1, BLOCK_SIZE)):
                            histogram_idx = j * BLOCK_SIZE + tx
                            if histogram_idx < RADIX + 1:
                                s_hist[r_idx, histogram_idx] = 0
                        if tx == 0:
                            if round > 0:
                                s_refine[1] = s_refine[1] | (
                                    T.Cast(T.uint32, s_threshold_bin_id[0]) << (shift + 8)
                                )
                            s_num_input[r_idx ^ 1] = 0
                            s_threshold_bin_id[0] = -1
                            s_cnt[0] = 0
                            s_cnt[1] = 0

                        # Suffix sums, as stage 1 takes them.
                        if tx < RADIX:
                            l_val = s_hist[cur, tx]
                            for i in T.unroll(5):
                                l_shfl = T.shfl_down(l_val, 1 << i)
                                if tx % 32 + (1 << i) < 32:
                                    l_val = l_val + l_shfl
                            if tx % 32 == 0:
                                s_warp_sum[tx // 32] = l_val
                            T.sync_threads(3, RADIX)
                            for j in T.unroll(RADIX // 32 - 1):
                                if tx // 32 + j + 1 < RADIX // 32:
                                    l_val = l_val + s_warp_sum[tx // 32 + j + 1]
                            s_hist[cur, tx] = l_val
                            T.sync_threads(3, RADIX)
                            if s_hist[cur, tx] > l_new_topk and s_hist[cur, tx + 1] <= l_new_topk:
                                s_threshold_bin_id[0] = tx
                        T.sync_threads()

                        # The round's keys above the threshold bin take the slots from
                        # l_start_pos; the last round's ties take the slots after them, to topk.
                        # Each warp takes its slots with one atomic: the round's keys crowd a few
                        # bins, and one atomic a key serialized every lane on the same counter.
                        l_threshold_bin_id = s_threshold_bin_id[0]
                        l_start_pos = topk - l_new_topk
                        l_out_pos = l_start_pos + s_hist[cur, l_threshold_bin_id + 1]
                        l_new_topk = l_new_topk - s_hist[cur, l_threshold_bin_id + 1]

                        # More candidates than the staging buffer holds: the round reads the
                        # window again instead, taking the keys whose bytes match every
                        # threshold settled so far, and stages the next round's in case they
                        # fit. A staged round's count never exceeds the buffer.
                        rescan = s_num_input[r_idx] > SMEM_INPUT_SIZE
                        mask = T.if_then_else(
                            round == 0,
                            T.Cast(T.uint32, 0),
                            T.Cast(T.uint32, 0xFFFFFFFF) << (shift + 8),
                        )
                        l_num_input = T.min(s_num_input[r_idx], SMEM_INPUT_SIZE)
                        l_span = T.if_then_else(rescan, seq_len_kv, l_num_input)
                        for s in T.serial(T.ceildiv(l_span, BLOCK_SIZE)):
                            slot = s * BLOCK_SIZE + tx
                            l_keep = 0
                            # A rescan's slot is a window position; otherwise a staged entry.
                            if T.if_then_else(
                                rescan,
                                slot < l_end_idx and slot >= l_start_idx and slot < seq_len_kv,
                                slot < l_num_input,
                            ):
                                # A rescan reads every key of the window and keeps those
                                # inside the stage-1 bucket and the settled bytes; a staged
                                # entry is inside them and carries its key.
                                if rescan:
                                    l_cand = slot
                                    l_score = index_score[bx, seq_row, l_cand, g]
                                    l_key = float_key(l_score)
                                    l_keep = T.Cast(
                                        T.int32,
                                        T.Cast(T.uint32, bin8(l_score)) == s_refine[0]
                                        and (l_key & mask) == s_refine[1],
                                    )
                                else:
                                    l_cand = s_input_idx[r_idx, T.min(slot, SMEM_INPUT_SIZE - 1)]
                                    l_key = s_input_key[r_idx, T.min(slot, SMEM_INPUT_SIZE - 1)]
                                    l_keep = 1
                            l_bin_id32 = T.if_then_else(
                                l_keep != 0, T.Cast(T.int32, ((l_key - l_kmin) >> shift) & 0xFF), -2
                            )
                            l_p = T.Cast(T.int32, l_bin_id32 > l_threshold_bin_id)
                            l_mask = T.Cast(T.uint32, T.ballot_sync(l_p))
                            l_rank = T.Cast(T.int32, T.popcount(l_mask & l_lanes_below))
                            l_base = 0
                            if l_p != 0 and l_rank == 0:
                                l_base = shared_add(s_cnt[0], T.Cast(T.int32, T.popcount(l_mask)))
                            l_base = T.shfl_sync(
                                l_base,
                                T.Cast(
                                    T.int32,
                                    T.popcount(
                                        (l_mask & (~l_mask + T.Cast(T.uint32, 1)))
                                        - T.Cast(T.uint32, 1)
                                    ),
                                ),
                            )
                            if l_p != 0:
                                index[bx, seq_row, g, l_start_pos + l_base + l_rank] = l_cand
                            if last:
                                l_p = T.Cast(
                                    T.int32, l_bin_id32 == l_threshold_bin_id and l_new_topk > 0
                                )
                                l_mask = T.Cast(T.uint32, T.ballot_sync(l_p))
                                l_rank = T.Cast(T.int32, T.popcount(l_mask & l_lanes_below))
                                l_base = 0
                                if l_p != 0 and l_rank == 0:
                                    l_base = shared_add(
                                        s_cnt[1],
                                        T.Cast(T.int32, T.popcount(l_mask)),
                                    )
                                l_base = T.shfl_sync(
                                    l_base,
                                    T.Cast(
                                        T.int32,
                                        T.popcount(
                                            (l_mask & (~l_mask + T.Cast(T.uint32, 1)))
                                            - T.Cast(T.uint32, 1)
                                        ),
                                    ),
                                )
                                if l_p != 0 and l_out_pos + l_base + l_rank < topk:
                                    index[bx, seq_row, g, l_out_pos + l_base + l_rank] = l_cand
                            else:
                                l_p = T.Cast(
                                    T.int32, l_bin_id32 == l_threshold_bin_id and l_new_topk > 0
                                )
                                l_mask = T.Cast(T.uint32, T.ballot_sync(l_p))
                                l_rank = T.Cast(T.int32, T.popcount(l_mask & l_lanes_below))
                                l_base = 0
                                if l_p != 0 and l_rank == 0:
                                    l_base = shared_add(
                                        s_num_input[r_idx ^ 1],
                                        T.Cast(T.int32, T.popcount(l_mask)),
                                    )
                                l_base = T.shfl_sync(
                                    l_base,
                                    T.Cast(
                                        T.int32,
                                        T.popcount(
                                            (l_mask & (~l_mask + T.Cast(T.uint32, 1)))
                                            - T.Cast(T.uint32, 1)
                                        ),
                                    ),
                                )
                                if l_p != 0:
                                    T.atomic_add(
                                        s_hist[
                                            r_idx,
                                            T.Cast(
                                                T.int32,
                                                (
                                                    (l_key - l_kmin)
                                                    >> T.max(l_bits - 8 * (round + 2), 0)
                                                )
                                                & 0xFF,
                                            ),
                                        ],
                                        1,
                                    )
                                    if l_base + l_rank < SMEM_INPUT_SIZE:
                                        s_input_idx[r_idx ^ 1, l_base + l_rank] = l_cand
                                        s_input_key[r_idx ^ 1, l_base + l_rank] = l_key
                        if last:
                            l_new_topk = 0

        return _topk_select_kernel_main

    return topk_select_fwd_func


# Keys a thread of the register kernel holds at most.
REGS_ITEMS = 16


@target_cache(maxsize=32)
def _topk_select_regs_kernel(
    batch, seq_len, seq_len_kv, kv_group, topk, fill_value, in_dtype, out_dtype
):
    """The top-k of a row whose keys fit the block's registers, read from global memory once.

    One 4096-bin histogram of each key's fp16 high 12 bits finds the threshold bin. The keys
    above it, and the threshold bin's ranked by their 32-bit keys when not all of them fit,
    take their slots in shared memory, which the block copies out coalesced.
    """

    @tilelang.jit(
        out_idx=[1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
    )
    def topk_select_regs_func(BLOCK_SIZE=1024):
        def bin12(x):
            # The fp16 value's top 12 bits as an order-preserving bin, 0 .. 4095.
            return (half_key(x) >> 4) + 2048

        batch = T.dynamic("batch")
        FILL = seq_len_kv if fill_value is None else fill_value
        # A thread's keys load as VEC-wide vectors: positions (v * BLOCK_SIZE + tx) * VEC + e.
        VEC = 4 if kv_group == 1 and seq_len_kv % 4 == 0 else 1
        # Vectors a thread loads; an even count of keys packs their bins two to a register.
        VECS = -(-seq_len_kv // (BLOCK_SIZE * VEC))
        VECS += VECS % 2 if VEC == 1 else 0
        ITEMS = VECS * VEC
        # Keys the histogram counts, the window's and bin 0's.
        TOTAL = ITEMS * BLOCK_SIZE
        HIST = 4096
        BINS = HIST // BLOCK_SIZE
        WARPS = BLOCK_SIZE // 32
        RADIX = 256
        # Threshold-bin keys ranked against each other; more refine by radix rounds.
        TIES = 256
        # Each thread's above-threshold count in the low half, threshold-bin count in the high.
        EQ = 1 << 16

        def pos(i, tx):
            # The position of thread tx's key i.
            return ((i // VEC) * BLOCK_SIZE + tx) * VEC + i % VEC

        def bin_at(pk, i):
            return T.Cast(T.int32, (pk[i // 2] >> (16 * (i % 2))) & T.Cast(T.uint32, 0xFFFF))

        # Slots a thread copies out at once.
        OVEC = 4 if topk % 4 == 0 else 1

        @T.prim_func
        def _topk_select_regs_main(
            index_score: T.Tensor[(batch, seq_len, seq_len_kv, kv_group), in_dtype],
            index: T.Tensor[(batch, seq_len, kv_group, topk), out_dtype],
            starts: T.Tensor[(batch, seq_len), out_dtype],
            ends: T.Tensor[(batch, seq_len), out_dtype],
        ):
            with T.Kernel(batch, seq_len, kv_group, threads=BLOCK_SIZE) as (bx, by, g):
                tx = T.get_thread_binding()
                lane = tx % 32
                warp = tx // 32

                s_hist = T.alloc_shared([HIST], T.int32)
                s_warp = T.alloc_shared([WARPS], T.int32)
                # Threshold bin, keys above it, keys in it.
                s_match = T.alloc_shared([3], T.int32)
                s_cnt = T.alloc_shared([2], T.int32)
                s_tie_key = T.alloc_shared([TIES], T.uint32)
                s_tie_idx = T.alloc_shared([TIES], T.int32)
                s_out = T.alloc_shared([min(topk, seq_len_kv)], T.int32)
                l_x = T.alloc_local([ITEMS], T.float32)
                l_pk = T.alloc_local([ITEMS // 2], T.uint32)
                l_h = T.alloc_local([BINS], T.int32)
                l_start = T.alloc_var(T.int32)
                l_end = T.alloc_var(T.int32)
                l_n = T.alloc_var(T.int32)
                l_sum = T.alloc_var(T.int32)
                l_inc = T.alloc_var(T.int32)
                l_shfl = T.alloc_var(T.int32)
                l_pre = T.alloc_var(T.int32)
                l_thr = T.alloc_var(T.int32)
                l_above = T.alloc_var(T.int32)
                l_eq = T.alloc_var(T.int32)
                l_cnt = T.alloc_var(T.int32)
                l_base = T.alloc_var(T.int32)
                l_pg = T.alloc_var(T.int32)
                l_pe = T.alloc_var(T.int32)
                l_bin = T.alloc_var(T.int32)
                l_remain = T.alloc_var(T.int32)
                l_ki = T.alloc_var(T.uint32)
                l_kj = T.alloc_var(T.uint32)
                l_rank = T.alloc_var(T.int32)
                l_key = T.alloc_var(T.uint32)
                l_prefix = T.alloc_var(T.uint32)
                l_mask = T.alloc_var(T.uint32)
                l_val = T.alloc_var(T.int32)
                l_out = T.alloc_var(T.int32)
                l_byte = T.alloc_var(T.int32)
                l_pos = T.alloc_var(T.int32)

                # The row loads before its window is read, so the two reads' latencies overlap.
                for v in T.unroll(VECS):
                    if (v * BLOCK_SIZE + tx) * VEC < seq_len_kv:
                        for e in T.vectorized(VEC):
                            l_x[v * VEC + e] = index_score[
                                bx, by, (v * BLOCK_SIZE + tx) * VEC + e, g
                            ]
                l_start = starts[bx, by]
                l_end = ends[bx, by]
                l_n = l_end - l_start

                if l_n <= topk:
                    for j in T.serial(T.ceildiv(topk, BLOCK_SIZE)):
                        slot = j * BLOCK_SIZE + tx
                        if slot < topk:
                            index[bx, by, g, slot] = T.if_then_else(
                                slot < l_n, l_start + slot, FILL
                            )
                else:
                    for i in T.unroll(BINS):
                        s_hist[i * BLOCK_SIZE + tx] = 0
                    T.sync_threads()
                    # A key's bin, kept two to a register. Bin 0 holds the keys outside the window
                    # and no other: only negative NaNs bin there, and they bin as 1. Counting them
                    # there keeps the loop free of branches.
                    for j in T.unroll(ITEMS // 2):
                        l_pk[j] = T.Cast(T.uint32, 0)
                        for h in T.unroll(2):
                            l_bin = T.Select(
                                T.Cast(T.uint32, pos(2 * j + h, tx) - l_start)
                                < T.Cast(T.uint32, l_n),
                                T.max(bin12(l_x[2 * j + h]), 1),
                                0,
                            )
                            T.atomic_add(s_hist[l_bin], 1)
                            l_pk[j] = l_pk[j] | (T.Cast(T.uint32, l_bin) << (16 * h))
                    T.sync_threads()

                    # The threshold bin: each thread scans its BINS consecutive bins.
                    l_sum = 0
                    for i in T.unroll(BINS):
                        l_h[i] = s_hist[tx * BINS + i]
                        l_sum = l_sum + l_h[i]
                    l_inc = l_sum
                    for i in T.unroll(5):
                        l_shfl = T.shfl_up(l_inc, 1 << i)
                        if lane >= (1 << i):
                            l_inc = l_inc + l_shfl
                    if lane == 31:
                        s_warp[warp] = l_inc
                    T.sync_threads()
                    l_pre = warp_sum(T.if_then_else(lane < warp, s_warp[T.min(lane, WARPS - 1)], 0))
                    l_pre = l_pre + l_inc - l_sum
                    for i in T.unroll(BINS):
                        l_pre = l_pre + l_h[i]
                        if TOTAL - l_pre < topk and TOTAL - l_pre + l_h[i] >= topk:
                            s_match[0] = tx * BINS + i
                            s_match[1] = TOTAL - l_pre
                            s_match[2] = l_h[i]
                    T.sync_threads()
                    l_thr = s_match[0]
                    l_above = s_match[1]
                    l_eq = s_match[2]

                    # Each thread's slots: a scan of its counts of keys above the threshold bin
                    # and in it, packed in one word, across the block. The keys above take slots
                    # 0 .. l_above - 1, the threshold bin's the slots after, or the tie buffer
                    # when not all fit. The slots stage in shared memory: the index stores
                    # leave the block coalesced.
                    l_cnt = 0
                    for i in T.unroll(ITEMS):
                        l_cnt = (
                            l_cnt
                            + T.Select(bin_at(l_pk, i) > l_thr, 1, 0)
                            + T.Select(bin_at(l_pk, i) == l_thr, EQ, 0)
                        )
                    l_inc = l_cnt
                    for i in T.unroll(5):
                        l_shfl = T.shfl_up(l_inc, 1 << i)
                        if lane >= (1 << i):
                            l_inc = l_inc + l_shfl
                    if lane == 31:
                        s_warp[warp] = l_inc
                    T.sync_threads()
                    l_base = warp_sum(
                        T.if_then_else(lane < warp, s_warp[T.min(lane, WARPS - 1)], 0)
                    )
                    l_base = l_base + l_inc - l_cnt
                    l_pg = l_base & (EQ - 1)
                    l_pe = l_base >> 16
                    for i in T.unroll(ITEMS):
                        if bin_at(l_pk, i) > l_thr:
                            s_out[l_pg] = pos(i, tx)
                            l_pg = l_pg + 1
                    if l_cnt >= EQ:
                        for i in T.unroll(ITEMS):
                            if bin_at(l_pk, i) == l_thr:
                                if l_above + l_eq <= topk:
                                    s_out[l_above + l_pe] = pos(i, tx)
                                elif l_pe < TIES:
                                    s_tie_key[l_pe] = float_key(l_x[i])
                                    s_tie_idx[l_pe] = pos(i, tx)
                                l_pe = l_pe + 1
                    T.sync_threads()

                    l_remain = topk - l_above
                    if l_above + l_eq > topk:
                        if l_eq <= TIES:
                            # A threshold-bin key's slot is its rank among them: the keys
                            # above it, and the equal keys staged before it.
                            for t in T.serial(T.ceildiv(l_eq, WARPS)):
                                if t * WARPS + warp < l_eq:
                                    l_ki = s_tie_key[t * WARPS + warp]
                                    l_rank = 0
                                    for c in T.serial(T.ceildiv(l_eq, 32)):
                                        if c * 32 + lane < l_eq:
                                            l_kj = s_tie_key[c * 32 + lane]
                                            l_rank = l_rank + T.Cast(
                                                T.int32,
                                                l_kj > l_ki
                                                or (
                                                    l_kj == l_ki
                                                    and c * 32 + lane < t * WARPS + warp
                                                ),
                                            )
                                    l_rank = warp_sum(l_rank)
                                    if lane == 0 and l_rank < l_remain:
                                        s_out[l_above + l_rank] = s_tie_idx[t * WARPS + warp]
                        else:
                            # Radix rounds over the threshold bin's 32-bit keys, a byte each,
                            # from the registers.
                            l_prefix = T.Cast(T.uint32, 0)
                            l_mask = T.Cast(T.uint32, 0)
                            l_out = l_above
                            for round in T.serial(4):
                                shift = 24 - round * 8
                                T.sync_threads()
                                # Bins 0 .. RADIX, the suffix sums' sentinel included: a
                                # 256-thread block has no thread RADIX.
                                if tx < RADIX:
                                    s_hist[tx] = 0
                                if tx == 0:
                                    s_hist[RADIX] = 0
                                    s_cnt[0] = 0
                                    s_cnt[1] = 0
                                T.sync_threads()
                                for i in T.unroll(ITEMS):
                                    l_key = float_key(l_x[i])
                                    if bin_at(l_pk, i) == l_thr and (l_key & l_mask) == l_prefix:
                                        T.atomic_add(
                                            s_hist[T.Cast(T.int32, (l_key >> shift) & 0xFF)], 1
                                        )
                                T.sync_threads()
                                # Suffix sums: s_hist[b] counts the candidates of byte >= b.
                                if tx < RADIX:
                                    l_val = s_hist[tx]
                                    for i in T.unroll(5):
                                        l_shfl = T.shfl_down(l_val, 1 << i)
                                        if lane + (1 << i) < 32:
                                            l_val = l_val + l_shfl
                                    if lane == 0:
                                        s_warp[warp] = l_val
                                    T.sync_threads(3, RADIX)
                                    for j in T.unroll(RADIX // 32 - 1):
                                        if warp + j + 1 < RADIX // 32:
                                            l_val = l_val + s_warp[warp + j + 1]
                                    s_hist[tx] = l_val
                                    T.sync_threads(3, RADIX)
                                    if s_hist[tx] >= l_remain and s_hist[tx + 1] < l_remain:
                                        s_match[0] = tx
                                        s_match[1] = s_hist[tx + 1]
                                T.sync_threads()
                                l_byte = s_match[0]
                                for i in T.unroll(ITEMS):
                                    l_key = float_key(l_x[i])
                                    if bin_at(l_pk, i) == l_thr and (l_key & l_mask) == l_prefix:
                                        if T.Cast(T.int32, (l_key >> shift) & 0xFF) > l_byte:
                                            l_pos = shared_add(s_cnt[0], 1)
                                            s_out[l_out + l_pos] = pos(i, tx)
                                        elif (
                                            round == 3
                                            and T.Cast(T.int32, (l_key >> shift) & 0xFF) == l_byte
                                        ):
                                            l_pos = shared_add(s_cnt[1], 1)
                                            if l_pos < l_remain - s_match[1]:
                                                s_out[l_out + s_match[1] + l_pos] = pos(i, tx)
                                l_out = l_out + s_match[1]
                                l_remain = l_remain - s_match[1]
                                l_prefix = l_prefix | (T.Cast(T.uint32, l_byte) << shift)
                                l_mask = l_mask | (T.Cast(T.uint32, 0xFF) << shift)
                        T.sync_threads()
                    for j in T.serial(T.ceildiv(topk, BLOCK_SIZE * OVEC)):
                        if (j * BLOCK_SIZE + tx) * OVEC < topk:
                            for e in T.vectorized(OVEC):
                                index[bx, by, g, (j * BLOCK_SIZE + tx) * OVEC + e] = s_out[
                                    (j * BLOCK_SIZE + tx) * OVEC + e
                                ]

        return _topk_select_regs_main

    return topk_select_regs_func


class TopKSelectKernel(Kernel, TopKSelectFwdInterface):
    """Per-row top-k index selection over an $[B \\times S \\times S\\_kv \\times G]$ score tensor.

    Args:
        batch: Batch size.
        seq_len: Query sequence length.
        seq_len_kv: Key/value sequence length.
        kv_group: Number of key/value groups.
        topk: Number of indices selected per row.
        fill_value: Index written to the slots a short window leaves unselected;
            ``None`` writes ``seq_len_kv``.
        dtype: Torch dtype of the input scores.
        out_dtype: Torch dtype of the emitted indices.
        config: Optional dict with "BLOCK_SIZE".
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: TopKSelectCall) -> Entry:
        return call, lambda: cls(
            call.batch,
            call.seq_len,
            call.seq_len_kv,
            call.kv_group,
            call.topk,
            call.fill_value,
            call.dtype,
            call.out_dtype,
        )

    def __init__(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        kv_group: int,
        topk: int,
        fill_value: int | None,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: dict | None = None,
    ):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.topk = topk
        self.fill_value = fill_value
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.out_dtype_str = self.dtype_to_str(self.out_dtype)

        # A call of too few rows to fill every SM twice takes one row's time, so a row that
        # fits a 1024-thread block's registers is selected from them after one read.
        self.few_rows = batch * seq_len * kv_group < 2 * self.build_target.sm_count
        self.regs = self.few_rows and topk < seq_len_kv <= REGS_ITEMS * 1024
        if self.regs:
            self.kernel = _topk_select_regs_kernel(
                self.batch,
                self.seq_len,
                self.seq_len_kv,
                self.kv_group,
                self.topk,
                self.fill_value,
                self.dtype_str,
                self.out_dtype_str,
            )
        else:
            # The bins of 65536 keys bring a block to about 97 KB of shared memory. An SM80 SM
            # holds two 1024-thread blocks in 164 KB only with half of them; SM86/89 hold one
            # such block whatever its cache, and SM90 two at 97 KB each. A row of fewer keys
            # needs only their bins and stages at most all of them, so a smaller block fits
            # more of them an SM.
            width = max(1024, 1 << max(0, seq_len_kv - 1).bit_length())
            self.kernel = _topk_select_kernel(
                self.batch,
                self.seq_len,
                self.seq_len_kv,
                self.kv_group,
                self.topk,
                self.fill_value,
                self.dtype_str,
                self.out_dtype_str,
                min(32768 if self.build_target.arch == 80 else 65536, width),
                min(2048, width),
            )
        self._supply_prog = self._make_supply_prog()
        self.init_config(config)

    def _make_supply_prog(self):
        import torch as _torch
        from tilelang.utils.device import get_current_device as _get_current_device

        batch = self.batch
        seq_len_kv = self.seq_len_kv

        dim_map = {"batch": batch, "seq_len_kv": seq_len_kv}

        def symbol_name(value):
            name = getattr(value, "name", None)
            if name is None:
                name = getattr(value, "name_hint", None)
            return name() if callable(name) else name

        def resolve_shape(shape):
            result = []
            for s in shape:
                try:
                    result.append(int(s))
                except (TypeError, ValueError):
                    result.append(dim_map.get(symbol_name(s), 1))
            return result

        def supply_prog(params):
            inputs = []
            device = _get_current_device()
            int_tensors = []  # track indices of int tensor params
            for i, param in enumerate(params):
                if param.is_scalar():
                    name = param.name if hasattr(param, "name") else ""
                    inputs.append(dim_map[name])
                else:
                    shape = resolve_shape(param.shape)
                    dtype = param.torch_dtype()
                    if dtype in (_torch.int32, _torch.int64, _torch.int16, _torch.int8):
                        inputs.append(_torch.zeros(shape, dtype=dtype, device=device))
                        int_tensors.append(i)
                    else:
                        # Normal scores, as the workloads draw: uniform [0, 1) puts most keys
                        # in a few float16 buckets and tunes for a rescan the real rows skip.
                        inputs.append(_torch.randn(shape, dtype=dtype, device=device))
            # last int tensor is 'ends' — fill with seq_len_kv so kernel processes all elements
            if int_tensors:
                inputs[int_tensors[-1]].fill_(seq_len_kv)
            return inputs

        return supply_prog

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    @property
    def default_config(self) -> dict:
        # A short row keeps a small block busy, and the small blocks an SM holds hide each
        # other's latency. A long row wants a 1024-thread block's loads in flight, and so does
        # a call with too few rows to fill every SM twice, whose time is one row's.
        if self.regs:
            return {"BLOCK_SIZE": 1024}
        if self.seq_len_kv <= 2048:
            return {"BLOCK_SIZE": 256}
        if self.few_rows:
            return {"BLOCK_SIZE": 1024}
        return {"BLOCK_SIZE": 512 if self.seq_len_kv <= 16384 else 1024}

    @property
    def autotune_configs(self) -> list[dict]:
        blocks = (256, 512, 1024)
        if self.regs:
            blocks = tuple(b for b in blocks if self.seq_len_kv <= REGS_ITEMS * b)
        return [{"BLOCK_SIZE": block} for block in blocks]

    def forward(
        self, index_score: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        return self.program(self._compile)(index_score, starts, ends)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(cfg["BLOCK_SIZE"])
