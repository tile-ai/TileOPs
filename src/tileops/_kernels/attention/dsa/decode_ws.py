"""SM90 warp-specialized DSA decode: a gathering producer and two seesaw consumers."""

import re
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops._csrc import csrc_include
from tileops._kernels.attention.call_spec import DSADecodeCall
from tileops._kernels.attention.dsa.decode import DSADecodeKernelBase
from tileops._kernels.attention.online_softmax import make_sink_scale
from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, chain_cuda_postproc, target_cache
from tileops._kernels.constants import LOG2E, WARPGROUP_THREADS

__all__ = ["DSADecodeWSKernel"]

# mark_role_branches issues this call as a bare statement; _uniform_role_branches finds the
# kernel by that statement and turns it into the role index, so the two must agree.
ROLE_INDEX_CALL = "tileops::canonical_warp_group_idx"


def _uniform_role_branches(code: str) -> str:
    """Make a warp-specialized kernel's role branches test the warpgroup index from lane 0.

    TileLang emits each role branch as a test of threadIdx.x, which ptxas cannot prove
    uniform across a warp, so inside it the WGMMA descriptors stay in ordinary registers
    and every issue pays an R2UR. TileLang reads each role's thread range from that form,
    so the generated CUDA is rewritten instead. Code without the marker statement is
    returned unchanged; so is code whose branches TileLang emitted differently, which
    stays correct without uniform branches.
    """
    marker = re.compile(rf"^([ \t]*){re.escape(ROLE_INDEX_CALL)}\(\);$", re.MULTILINE)
    role_tests = (
        ("if (256 <= ((int)threadIdx.x)) {", 2),
        ("if (((int)threadIdx.x) < 128) {", 0),
        ("if ((128 <= ((int)threadIdx.x)) && (((int)threadIdx.x) < 256)) {", 1),
    )
    if len(marker.findall(code)) != 1 or any(code.count(test) != 1 for test, _ in role_tests):
        return code
    code = marker.sub(rf"\1const int warpgroup_role = {ROLE_INDEX_CALL}();", code)
    for test, role in role_tests:
        code = code.replace(test, f"if (warpgroup_role == {role}) {{")
    return code


chain_cuda_postproc(_uniform_role_branches)


@T.macro
def mark_role_branches():
    """Mark a warp-specialized kernel for _uniform_role_branches, which defines the role index
    here; issue it once, as a bare statement, before the role branches."""
    T.evaluate(T.call_extern("int32", ROLE_INDEX_CALL))


@target_cache(maxsize=32)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
    compile_flags=[
        *CUDA_COMPILE_FLAGS,
        *csrc_include("warp_uniform.h", "row_gather.h", "cluster_partials.h", "pair_merge.h"),
    ],
)
def _dsa_decode_ws_kernel(
    batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride, q_start, kv_group,
    sm_scale, dtype, is_causal, has_sink, ctas, skip, split,
):  # fmt: skip
    """One work item per (token, 64-head block); the selected keys are taken in block pairs.

    Consumer 0 scores the even block of a pair and consumer 1 the odd one. Each turns its
    scores into weights, keeps them in registers for its own half of the value dims and
    hands them to the other through smem. The row max is one chain through both blocks:
    consumer 1 continues from consumer 0's max, and consumer 0 rescales its weights to the
    pair's final max before handing them over.

    A slot is live when its index is in ``[0, seq_len_kv)`` and, if *is_causal*, within
    the query's causal limit; slots past ``topk`` are dead. With *skip*, the producer finds
    the row's last live slot and every role skips the blocks past it. With *has_sink* the
    program takes per-head ``Sinks`` logits that join each row's denominator.

    *ctas* CTAs run: one item each, or, with fewer CTAs than items, every *ctas*-th item in
    turn. A CTA that runs several items loads the next item's Q once both consumers have
    scored the current item's last pair, and stores an item's output while the next one
    starts: consumer 0 stages its half in four 64-column tiles, and consumer 1, while its
    first odd block is scored, stores that half with TMA and then its own through the tiles.

    With *split* 2 a cluster of two CTAs runs each item, each taking half the row's blocks.
    The pair trades each row's max and sum, and each CTA's consumer of one half of the value
    dims adds the peer's half to its own and stores it; ``Sinks`` join the merged denominator.
    """
    # The tiles below are fixed; DSADecodeWSKernel.refusal states the shapes they cover.
    block_h = 64  # query heads per CTA: one m64 WGMMA row block
    block_k = 64  # selected keys per block; blocks are taken in pairs
    half = 256  # value dims each consumer accumulates
    atom = 64  # columns one 128-byte swizzle atom holds
    tail_atoms = 1  # the key tail, when present, is one atom
    vec = 8  # elements one 16-byte cp.async moves
    lanes = atom // vec  # producer lanes per tile row
    warpgroup = WARPGROUP_THREADS
    consumer_threads = 2 * warpgroup  # tx < 256; the producer warpgroup follows
    producer_threads = warpgroup
    rows_per_pass = producer_threads // lanes
    # 2 x 128 x 232 + 128 x 40 = 64512 of the 65536 registers a CTA may hold.
    consumer_regs = 232
    producer_regs = 40

    # A pair's shared-memory parts, each with its own ready and free barrier: K0L / K0R hold
    # value dims [0, 256) / [256, 512) (and the key tail) of the even block, K1L / K1R of the
    # odd block. Each consumer scores both parts of its own block; consumer 0 sums over the
    # L parts, consumer 1 over the R parts.
    part_0l, part_0r, part_1l, part_1r = 0, 1, 2, 3
    # Named barriers the consumer warpgroups sync on; id 0 is __syncthreads.
    even_max_ready = 1  # consumer 0 published the even block's max in sM
    pair_max_ready = 2  # consumer 1 continued it to the pair's max
    s0_ready = 3  # consumer 0 wrote the even block's weights to S0
    s1_ready = 4  # consumer 1 wrote the odd block's weights to S1
    sums_ready = 5  # both consumers' row sums are in sL
    staged_ready = 6  # + w: consumer w staged its output
    tiles_staged = 8  # consumer 0 staged its output in the tiles of a persistent CTA

    dq = dim + tail_dim
    group_heads = heads // kv_group
    head_blocks = group_heads // block_h
    items = batch * kv_group * seq_len * head_blocks
    paired = split == 2
    persistent = not paired and ctas < items
    if persistent and tail_dim:
        # The output leaves through four tiles beside the pipeline: a key tail's tiles take
        # their room.
        raise ValueError("a CTA runs several items only without a key tail")
    blocks = -(-topk // block_k)  # a row's 64-slot blocks, the last dead past topk
    span = -(-blocks // split)  # the blocks a CTA of the item takes at most
    ragged = topk % (2 * block_k) != 0 or paired
    # The producer reads a row's indices in 16-byte vectors when rows keep that alignment,
    # each thread a vector of every `producer_threads * width` slots.
    width = 4 if topk % 4 == 0 else 1
    scan_steps = -(-min(topk, span * block_k) // (producer_threads * width))
    scale = (dq**-0.5 if sm_scale is None else sm_scale) * LOG2E
    accum = "float"
    has_tail = tail_dim > 0
    sink_scale = make_sink_scale(1.0, accum, block_h, rows_per_head=1)

    def coords(item):
        """Item `item`'s batch, token, kv group and first query head; head blocks vary fastest."""
        rest = item // head_blocks
        g = rest // seq_len % kv_group
        h0 = g * group_heads + item % head_blocks * block_h
        return rest // seq_len // kv_group, rest % seq_len, g, h0

    def item_count(cta):
        """The items CTA `cta` runs."""
        return T.ceildiv(items - cta, ctas) if persistent else 1

    def item_of(cta, k):
        """The item CTA `cta` runs `k`-th."""
        return cta // split + k * ctas

    # With an even number of blocks a row takes whole pairs: rounding its blocks up adds at
    # most one block past its last live slot, which weighs nothing, and every pair then has
    # an odd block.
    whole_pairs = not paired and blocks % 2 == 0

    def taken_blocks(sLive, slot, own):
        """The blocks a row takes: those up to its last live slot, at least one."""
        last = T.max(T.max(sLive[slot, 0], sLive[slot, 1]), T.max(sLive[slot, 2], sLive[slot, 3]))
        live = T.max(T.ceildiv(last, block_k), 1)
        return T.min(live + live % 2, own) if whole_pairs else live

    def has_odd(pair, taken):
        """Whether pair `pair` of a row taking `taken` blocks has an odd block."""
        return True if whole_pairs else 2 * pair + 1 < taken

    def slot_index(Indices, b, s, g, slot):
        """Slot `slot`'s key index, -1 past ``topk``.

        The read is clamped into the row when slots run past ``topk`` or the pair count is
        known only at run time.
        """
        idx = Indices[b, s, g, T.min(slot, topk - 1) if skip or ragged else slot]
        return T.if_then_else(slot < topk, idx, -1) if ragged else idx

    def slot_live(idx, limit):
        """Whether key index `idx` is a key the query reads."""
        live = T.And(idx >= 0, idx < seq_len_kv)
        return T.And(live, idx * kv_stride <= limit) if is_causal else live

    @T.macro
    def copy_cols(dst, row, src, i0, i1, i2, col0, ntiles, lane, live):
        """Copy columns [col0, col0 + atom * ntiles) of one source row into `dst` row `row`.

        Each lane moves one vector of each atom. A dead row is zero-filled without a read:
        every dead slot names row 0, and reading it would send them all to the same L2 lines.
        """
        for t in T.unroll(ntiles):
            T.ptx_cp_async(
                T.access_ptr(dst[row, t * atom + lane * vec], "w", vec),
                T.access_ptr(src[i0, i1, i2, col0 + t * atom + lane * vec], "r", vec),
                vec,
                live,
            )

    @T.macro
    def load_part(dst, dst_tail, with_tail, col0, src_rows, live_rows, blk, KV, b, g, group, lane, k_ready, part):  # fmt: skip
        """Gather block `blk`'s value dims from `col0`, and its key tail, then mark `part`."""
        for r in T.unroll(block_k // rows_per_pass):
            row = rows_per_pass * r + group
            copy_cols(
                dst, row, KV, b, src_rows[blk, r], g, col0, half // atom, lane, live_rows[blk, r]
            )
            if with_tail and has_tail:
                copy_cols(
                    dst_tail, row, KV, b, src_rows[blk, r], g, dim, tail_atoms, lane,
                    live_rows[blk, r],
                )  # fmt: skip
        T.cp_async_barrier_noinc(k_ready[part])

    @T.macro
    def gather_pair(pair, odd, phase, odd_phase, Indices, KV, K0L, K0R, K0T, K1L, K1R, K1T, Bias, k_ready, k_free, bias_ready, src_rows, live_rows, b, s, g, limit, off, group, lane, p):  # fmt: skip
        """Gather pair `pair`'s parts, its odd block's only when `odd`, and its mask."""
        # A dead row names row 0 and is zero-filled; its bias makes it weigh nothing.
        for blk in T.unroll(2):
            for r in T.unroll(block_k // rows_per_pass):
                slot = off + (2 * pair + blk) * block_k + rows_per_pass * r + group
                idx = slot_index(Indices, b, s, g, slot)
                live_rows[blk, r] = slot_live(idx, limit)
                src_rows[blk, r] = idx * T.Cast("int32", live_rows[blk, r])
        # Refill each part in the order its last reader releases it.
        T.mbarrier_wait_parity(k_free[part_0l], phase ^ 1)
        load_part(
            K0L, K0T, False, 0, src_rows, live_rows, 0, KV, b, g, group, lane, k_ready, part_0l
        )
        if odd:
            T.mbarrier_wait_parity(k_free[part_1r], odd_phase ^ 1)
            load_part(
                K1R, K1T, True, half, src_rows, live_rows, 1, KV, b, g, group, lane, k_ready,
                part_1r,
            )  # fmt: skip
        T.mbarrier_wait_parity(k_free[part_0r], phase ^ 1)
        load_part(
            K0R, K0T, True, half, src_rows, live_rows, 0, KV, b, g, group, lane, k_ready, part_0r
        )
        T.mbarrier_wait_parity(k_free[part_1l], phase ^ 1)
        if odd:
            load_part(
                K1L, K1T, False, 0, src_rows, live_rows, 1, KV, b, g, group, lane, k_ready,
                part_1l,
            )  # fmt: skip
        # Bias is rewritten after the K1L free wait, which follows every read of the
        # previous pair's mask. Thread p masks slot p; the producer has 2 * block_k threads.
        idx = slot_index(Indices, b, s, g, off + 2 * pair * block_k + p)
        live = slot_live(idx, limit)
        Bias[p // block_k, p % block_k] = T.if_then_else(live, 0.0, -T.infinity(accum))
        T.mbarrier_arrive(bias_ready[0])

    @T.macro
    def gather_pairs(taken, done, done_odd, Indices, KV, K0L, K0R, K0T, K1L, K1R, K1T, Bias, k_ready, k_free, bias_ready, src_rows, live_rows, b, s, g, limit, off, group, lane, p):  # fmt: skip
        """Gather an item's pairs, `taken` blocks of them."""
        fixed = (
            Indices, KV, K0L, K0R, K0T, K1L, K1R, K1T, Bias, k_ready, k_free, bias_ready,
            src_rows, live_rows, b, s, g, limit, off, group, lane, p,
        )  # fmt: skip
        for pair in T.serial(T.ceildiv(taken, 2)):
            gather_pair(
                pair, has_odd(pair, taken), (done + pair) % 2, (done_odd + pair) % 2, *fixed
            )

    @T.macro
    def scan_issue(scanned, Indices, b, s, g, off, own, p):
        """Read the row's indices, every read issued before any is tested."""
        for j in T.unroll(scan_steps):
            first = (j * producer_threads + p) * width
            if first >= T.min(own * block_k, topk - off):
                for v in T.unroll(width):
                    scanned[j, v] = -1
            else:
                for v in T.vectorized(width):
                    scanned[j, v] = Indices[b, s, g, off + first + v]

    @T.macro
    def scan_publish(scanned, sLive, live_ready, slot, limit, p):
        """Publish one past the last live slot the producer read, per warp, in sLive[slot]."""
        last = T.alloc_var("int32", init=0)
        for j in T.unroll(scan_steps):
            for v in T.unroll(width):
                if slot_live(scanned[j, v], limit):
                    last = (j * producer_threads + p) * width + v + 1
        last = T.warp_reduce_max(last)
        if p % 32 == 0:
            sLive[slot, p // 32] = last
            T.mbarrier_arrive(live_ready[0])

    @T.macro
    def producer(Q, KV, Indices, QL, QR, QT, K0L, K0R, K0T, K1L, K1R, K1T, Bias, sLive, q_full, q_free, live_ready, k_ready, k_free, bias_ready, cta, off, own, p):  # fmt: skip
        """Per item: load Q, find the row's last live slot, then gather the pairs up to it.

        The first item's indices are read while its Q loads, a later item's once the CTA's
        previous item is gathered. An item's count is published only after the consumers
        have scored the previous item's last pair, so it never runs two phases ahead of them.

        `lanes` lanes per 128-byte tile row: thread p copies one vector of each atom of rows
        rows_per_pass * r + p // lanes, so a warp request covers whole sectors.
        """
        group = p // lanes
        lane = p % lanes
        src_rows = T.alloc_local([2, block_k // rows_per_pass], "int32")
        live_rows = T.alloc_local([2, block_k // rows_per_pass], "bool")
        if skip:
            scanned = T.alloc_local([scan_steps, width], "int32")
            taken = T.alloc_var("int32")
        # Pairs, and pairs with an odd block, of the CTA's earlier items: they set each
        # barrier's phase.
        done = T.alloc_var("int32", init=0) if persistent else 0
        done_odd = T.alloc_var("int32", init=0) if persistent else 0
        n_items = item_count(cta)
        for k in T.serial(n_items):
            b, s, g, h0 = coords(item_of(cta, k))
            limit = q_start + s - kv_stride + 1
            if persistent:
                # Both consumers have scored the previous item's last pair.
                T.mbarrier_wait_parity(q_free[0], (k % 2) ^ 1)
            if skip and k == 0:
                scan_issue(scanned, Indices, b, s, g, off, own, p)
            if persistent:
                # TMA spares the registers a persistent producer's item loop needs. Q is read
                # once: it leaves L2 first, ahead of the keys other rows gather.
                T.tma_copy(
                    Q[b, s, h0 : h0 + block_h, 0:half], QL, barrier=q_full[0],
                    eviction_policy="evict_first",
                )  # fmt: skip
                T.tma_copy(
                    Q[b, s, h0 : h0 + block_h, half:dim], QR, barrier=q_full[0],
                    eviction_policy="evict_first",
                )  # fmt: skip
                T.mbarrier_arrive(q_full[0])
            else:
                for r in T.unroll(block_h // rows_per_pass):
                    row = rows_per_pass * r + group
                    copy_cols(QL, row, Q, b, s, h0 + row, 0, half // atom, lane, True)
                    copy_cols(QR, row, Q, b, s, h0 + row, half, half // atom, lane, True)
                    if has_tail:
                        copy_cols(QT, row, Q, b, s, h0 + row, dim, tail_atoms, lane, True)
                T.cp_async_barrier_noinc(q_full[0])
            if skip:
                scan_publish(scanned, sLive, live_ready, k % 2, limit, p)
                T.mbarrier_wait_parity(live_ready[0], k % 2)
                taken = taken_blocks(sLive, k % 2, own)
            else:
                taken = own
            gather_pairs(
                taken, done, done_odd, Indices, KV, K0L, K0R, K0T, K1L, K1R, K1T, Bias, k_ready,
                k_free, bias_ready, src_rows, live_rows, b, s, g, limit, off, group, lane, p,
            )  # fmt: skip
            if persistent:
                done = done + T.ceildiv(taken, 2)
                done_odd = done_odd + taken // 2
                if skip and k + 1 < n_items:
                    nb, ns, ng, _ = coords(item_of(cta, k + 1))
                    scan_issue(scanned, Indices, nb, ns, ng, off, own, p)

    @T.macro
    def score_left(QL, KL, acc, clear):
        T.wgmma_gemm(QL, KL, acc, transpose_B=True, clear_accum=clear)

    @T.macro
    def score_right(QR, QT, KR, KT, acc, clear):
        T.wgmma_gemm(QR, KR, acc, transpose_B=True, clear_accum=clear)
        if has_tail:
            T.wgmma_gemm(QT, KT, acc, transpose_B=True)

    @T.macro
    def softmax(rP, rS, rO, rM, rL, prev, cur, keep, Bias, blk, sM):
        """Turn block `blk`'s scores in rP into weights, continuing the max chain from `prev`.

        Masks, rescales rO and rL by the step, leaves the weights in rP (f32) and rS (storage
        dtype), and publishes the new max in sM.
        """
        for i, j in T.Parallel(block_h, block_k):
            rP[i, j] = rP[i, j] + Bias[blk, j]
        T.reduce_max(rP, cur, dim=1, clear=True)
        for i in T.Parallel(block_h):
            keep[i] = rM[i]
            rM[i] = T.max(prev[i], cur[i] * scale)
            # The max the weights subtract: 0 while the row has no valid key, so its
            # weights are 0 instead of NaN.
            cur[i] = T.if_then_else(rM[i] == -T.infinity(accum), 0.0, rM[i])
            keep[i] = T.exp2(keep[i] - cur[i])
        for i, j in T.Parallel(block_h, half):
            rO[i, j] *= keep[i]
        for i, j in T.Parallel(block_h, block_k):
            rP[i, j] = T.exp2(rP[i, j] * scale - cur[i])
            rS[i, j] = rP[i, j]
        T.reduce_sum(rP, cur, dim=1)
        for i in T.Parallel(block_h):
            rL[i] = rL[i] * keep[i] + cur[i]
            sM[i] = rM[i]

    @T.macro
    def rebase(rO, rM, rL, prev, keep):
        """Move rO and rL to the max `prev` of a pair with no odd block."""
        for i in T.Parallel(block_h):
            keep[i] = T.exp2(rM[i] - T.if_then_else(prev[i] == -T.infinity(accum), 0.0, prev[i]))
            rM[i] = prev[i]
            rL[i] *= keep[i]
        for i, j in T.Parallel(block_h, half):
            rO[i, j] *= keep[i]

    @T.macro
    def normalize(w, rM, rL, sL, Sinks, h0):
        """Leave in rL the factor that normalizes consumer `w`'s rows by both row sums."""
        for i in T.Parallel(block_h):
            sL[w, i] = rL[i]
        T.sync_threads(sums_ready, consumer_threads)
        # Only a row with no valid key sums to 0; its rO is 0, and so is its output.
        for i in T.Parallel(block_h):
            rL[i] = sL[0, i] + sL[1, i]
            if not has_sink:
                rL[i] = 1 / T.if_then_else(rL[i] == 0, 1.0, rL[i])
        if has_sink:
            sink_scale(rM, rL, Sinks, h0, rL)

    @T.macro
    def store_staged(w, rO, rL, out, staged, O, b, s, h0):
        """Store consumer `w`'s value dims with TMA from its Q tile, which every score of the
        CTA's one item has finished reading."""
        for i, j in T.Parallel(block_h, half):
            out[i, j] = rO[i, j] * rL[i]
        T.copy(out, staged)
        T.fence_proxy_async()
        T.sync_threads(staged_ready + w, warpgroup)
        T.copy(staged, O[b, s, h0 : h0 + block_h, w * half : (w + 1) * half])

    @T.macro
    def stage_tiles(w, rO, rL, out, X0, X1, S0, S1):
        """Stage consumer `w`'s normalized value dims in the four 64-column tiles."""
        for i, j in T.Parallel(block_h, half):
            out[i, j] = rO[i, j] * rL[i]
        T.copy(out[:, 0:atom], X0)
        T.copy(out[:, atom : 2 * atom], X1)
        T.copy(out[:, 2 * atom : 3 * atom], S0)
        T.copy(out[:, 3 * atom : 4 * atom], S1)
        T.fence_proxy_async()
        T.sync_threads(staged_ready + w, warpgroup)

    @T.macro
    def store_tiles(c0, X0, X1, S0, S1, O, b, s, h0):
        """Store the tiles to value dims [c0, c0 + 256) of one item with TMA; the output
        leaves L2 first, as Q does."""
        hint = "evict_first"
        T.tma_copy(X0, O[b, s, h0 : h0 + block_h, c0 : c0 + atom], eviction_policy=hint)
        T.tma_copy(X1, O[b, s, h0 : h0 + block_h, c0 + atom : c0 + 2 * atom], eviction_policy=hint)
        T.tma_copy(
            S0, O[b, s, h0 : h0 + block_h, c0 + 2 * atom : c0 + 3 * atom], eviction_policy=hint
        )
        T.tma_copy(
            S1, O[b, s, h0 : h0 + block_h, c0 + 3 * atom : c0 + 4 * atom], eviction_policy=hint
        )

    @T.macro
    def store_item(rO, rL, out, X0, X1, S0, S1, O, b, s, h0):
        """Consumer 1 stores an item's output through the tiles: consumer 0's half, which
        consumer 0 staged, then its own once those stores have read the tiles. Its own stays
        in flight until consumer 1 next waits, before S0 and S1 are rewritten."""
        T.sync_threads(tiles_staged, consumer_threads)
        store_tiles(0, X0, X1, S0, S1, O, b, s, h0)
        T.tma_store_wait(0)
        T.sync_threads(staged_ready + 1, warpgroup)
        stage_tiles(1, rO, rL, out, X0, X1, S0, S1)
        store_tiles(half, X0, X1, S0, S1, O, b, s, h0)

    @T.macro
    def merge(w, rank, rO, rM, rL, sL, out, stage, Sinks, O, b, s, h0, sOwn, sPeer, sWeight, QL, QR, s_full, x_free, x_full):  # fmt: skip
        """Consumer `w` of a CTA pair after its last block. The consumer of the value dims
        this CTA stores (`w == rank`) weighs them and the peer's by each CTA's share of the
        merged softmax, adds and stores them; the other sends its own to the peer, with the
        CTA's row maxes and sums. The peer's dims land in QL and QR, which every score here
        has finished reading."""
        x = T.get_thread_binding() % warpgroup
        peer = rank ^ 1
        for i in T.Parallel(block_h):
            sL[w, i] = rL[i]
        T.sync_threads(sums_ready, consumer_threads)
        if w == rank:
            if x == 0:
                T.call_extern(
                    "handle", "tileops::expect_bytes", T.address_of(s_full[0]), block_h * 2 * 4
                )
                T.call_extern(
                    "handle", "tileops::expect_bytes", T.address_of(x_full[0]), block_h * half * 4
                )
                T.call_extern("handle", "tileops::arrive_peer", T.address_of(x_free[0]), peer)
            T.mbarrier_wait_parity(s_full[0], 0)
            for i in T.Parallel(block_h):
                # A row with no live key in either CTA has reference -inf and weighs 0.
                top = T.max(rM[i], sPeer[i, 0])
                mine = T.if_then_else(top == -T.infinity(accum), 0.0, T.exp2(rM[i] - top))
                theirs = T.if_then_else(top == -T.infinity(accum), 0.0, T.exp2(sPeer[i, 0] - top))
                total = (sL[0, i] + sL[1, i]) * mine + sPeer[i, 1] * theirs
                # The sink joins the denominator only: e^sink against the pair's reference.
                sunk = (
                    total
                    + T.if_then_else(
                        top == -T.infinity(accum), 0.0, T.exp2(Sinks[h0 + i] * LOG2E - top)
                    )
                    if has_sink
                    else total
                )
                inverse = T.if_then_else(sunk > 0, 1 / sunk, 0.0)
                sWeight[0, i] = mine * inverse
                sWeight[1, i] = theirs * inverse
            T.sync_threads(staged_ready + w, warpgroup)
            T.mbarrier_wait_parity(x_full[0], 0)
            T.call_extern(
                "handle", "tileops::merge_frag128_halves", rO.data, T.address_of(QL[0, 0]),
                T.address_of(QR[0, 0]), x, T.address_of(sWeight[0, 0]),
                T.address_of(sWeight[1, 0]),
            )  # fmt: skip
            for i, j in T.Parallel(block_h, half):
                out[i, j] = rO[i, j]
            T.copy(out, stage)
            T.fence_proxy_async()
            T.sync_threads(staged_ready + w, warpgroup)
            T.copy(stage, O[b, s, h0 : h0 + block_h, w * half : (w + 1) * half])
        else:
            for i in T.Parallel(block_h):
                sOwn[0, i] = rM[i]
                sOwn[1, i] = sL[0, i] + sL[1, i]
            # Every row of sOwn is written before it is sent.
            T.sync_threads(staged_ready + w, warpgroup)
            if x < block_h:
                T.call_extern(
                    "handle", "tileops::send_partial2", T.address_of(sPeer[x, 0]),
                    T.address_of(s_full[0]), peer, sOwn[0, x], sOwn[1, x],
                )  # fmt: skip
            T.mbarrier_wait_parity(x_free[0], 0)
            T.call_extern(
                "handle", "tileops::send_frag128_halves", rO.data, T.address_of(QL[0, 0]),
                T.address_of(QR[0, 0]), x, T.address_of(x_full[0]), peer,
            )  # fmt: skip

    @T.macro
    def pair0(pair, odd, pairs, done, QL, QR, QT, K0L, K0R, K0T, K1L, S0, S1, Bias, sM, q_free, k_ready, k_free, bias_ready, rP, rS, rO, rM, rL, cur, keep):  # fmt: skip
        """Consumer 0's pair `pair` of `pairs`, with an odd block when `odd`."""
        phase = (done + pair) % 2
        # The first pair's scores are issued inside the loop: one call site keeps the
        # WGMMA descriptors out of local memory.
        if pair == 0:
            T.mbarrier_wait_parity(k_ready[part_0l], phase)
            score_left(QL, K0L, rP, True)
            T.mbarrier_wait_parity(k_ready[part_0r], phase)
            score_right(QR, QT, K0R, K0T, rP, False)
            T.wait_wgmma(0)
            if persistent and pairs == 1:
                T.mbarrier_arrive(q_free[0])
        T.mbarrier_wait_parity(bias_ready[0], phase)
        softmax(rP, rS, rO, rM, rL, rM, cur, keep, Bias, 0, sM)
        T.named_barrier_arrive(even_max_ready, consumer_threads)
        T.wgmma_gemm(rS, K0L, rO)
        T.wait_wgmma(0)
        T.mbarrier_arrive(k_free[part_0l])
        # Consumer 1 continued the chain: bring this block's weights to its max.
        T.sync_threads(pair_max_ready, consumer_threads)
        for i in T.Parallel(block_h):
            keep[i] = T.exp2(rM[i] - T.if_then_else(sM[i] == -T.infinity(accum), 0.0, sM[i]))
            rM[i] = sM[i]
            rL[i] *= keep[i]
        for i, j in T.Parallel(block_h, block_k):
            rS[i, j] = rP[i, j] * keep[i]
        T.copy(rS, S0)
        T.fence_proxy_async()
        T.named_barrier_arrive(s0_ready, consumer_threads)
        if odd:
            T.sync_threads(s1_ready, consumer_threads)
            for i, j in T.Parallel(block_h, half):
                rO[i, j] *= keep[i]
            T.wgmma_gemm(S1, K1L, rO)
        if pair + 1 < pairs:
            T.mbarrier_wait_parity(k_ready[part_0l], phase ^ 1)
            score_left(QL, K0L, rP, True)
            T.wait_wgmma(1)
            T.mbarrier_arrive(k_free[part_1l])
            T.mbarrier_wait_parity(k_ready[part_0r], phase ^ 1)
            score_right(QR, QT, K0R, K0T, rP, False)
            T.wait_wgmma(0)
            if persistent and pair + 2 == pairs:
                T.mbarrier_arrive(q_free[0])
        else:
            T.wait_wgmma(0)
            T.mbarrier_arrive(k_free[part_1l])

    @T.macro
    def consumer0(O, Sinks, QL, QR, QT, K0L, K0R, K0T, K1L, S0, S1, X0, X1, Bias, sM, sL, sLive, q_full, q_free, live_ready, k_ready, k_free, bias_ready, cta, rank, own, pairing):  # fmt: skip
        """Scores the even blocks; accumulates value dims [0, 256)."""
        T.set_max_nreg(consumer_regs, 1)
        rP = T.alloc_fragment([block_h, block_k], accum)
        rS = T.alloc_fragment([block_h, block_k], dtype)
        rO = T.alloc_fragment([block_h, half], accum)
        out = T.alloc_fragment([block_h, half], dtype)
        rM = T.alloc_fragment([block_h], accum)
        rL = T.alloc_fragment([block_h], accum)
        cur = T.alloc_fragment([block_h], accum)
        keep = T.alloc_fragment([block_h], accum)
        if skip:
            taken = T.alloc_var("int32")
        done = T.alloc_var("int32", init=0) if persistent else 0
        for k in T.serial(item_count(cta)):
            b, s, g, h0 = coords(item_of(cta, k))
            T.fill(rO, 0)
            T.fill(rL, 0)
            T.fill(rM, -T.infinity(accum))
            T.mbarrier_wait_parity(q_full[0], k % 2)
            if skip:
                T.mbarrier_wait_parity(live_ready[0], k % 2)
                taken = taken_blocks(sLive, k % 2, own)
            else:
                taken = own
            fixed = (
                T.ceildiv(taken, 2), done, QL, QR, QT, K0L, K0R, K0T, K1L, S0, S1, Bias, sM,
                q_free, k_ready, k_free, bias_ready, rP, rS, rO, rM, rL, cur, keep,
            )  # fmt: skip
            for pair in T.serial(T.ceildiv(taken, 2)):
                pair0(pair, has_odd(pair, taken), *fixed)
            if persistent:
                done = done + T.ceildiv(taken, 2)
            if paired:
                merge(0, rank, rO, rM, rL, sL, out, K0L, Sinks, O, b, s, h0, *pairing)
            else:
                normalize(0, rM, rL, sL, Sinks, h0)
                if persistent:
                    stage_tiles(0, rO, rL, out, X0, X1, S0, S1)
                    T.named_barrier_arrive(tiles_staged, consumer_threads)
                else:
                    store_staged(0, rO, rL, out, QL, O, b, s, h0)

    @T.macro
    def pair1(pair, odd, odd_pairs, done, done_odd, k, pb, ps, ph0, O, QL, QR, QT, K0R, K1L, K1R, K1T, S0, S1, X0, X1, Bias, sM, q_free, k_ready, k_free, bias_ready, rP, rS, rO, out, rM, rL, prev, cur, keep):  # fmt: skip
        """Consumer 1's pair `pair`, with an odd block when `odd`; `odd_pairs` of the item's
        pairs have one."""
        phase = (done + pair) % 2
        odd_phase = (done_odd + pair) % 2
        if odd:
            T.mbarrier_wait_parity(k_ready[part_1r], odd_phase)
            score_right(QR, QT, K1R, K1T, rP, True)
            T.mbarrier_wait_parity(k_ready[part_1l], odd_phase)
            score_left(QL, K1L, rP, False)
        if persistent and pair == 0 and k > 0:
            # The previous item's output leaves while this item's first odd block,
            # if any, is scored.
            store_item(rO, rL, out, X0, X1, S0, S1, O, pb, ps, ph0)
            T.fill(rO, 0)
            T.fill(rL, 0)
            T.fill(rM, -T.infinity(accum))
        # Waited on every pair: ptxas serializes every WGMMA when one issued under a
        # condition is still in flight where the paths meet.
        T.wait_wgmma(0)
        if odd:
            if persistent and pair + 1 == odd_pairs:
                T.mbarrier_arrive(q_free[0])
            T.mbarrier_wait_parity(bias_ready[0], phase)
        # Consumer 0 has scored the even block: continue its max.
        T.sync_threads(even_max_ready, consumer_threads)
        for i in T.Parallel(block_h):
            prev[i] = sM[i]
        if odd:
            softmax(rP, rS, rO, rM, rL, prev, cur, keep, Bias, 1, sM)
        else:
            rebase(rO, rM, rL, prev, keep)
        if persistent and pair == 0:
            # Its stores have read S0 and S1 before either is rewritten.
            T.tma_store_wait(0)
        T.named_barrier_arrive(pair_max_ready, consumer_threads)
        # Each branch waits for every WGMMA it issues, for the same reason.
        if odd:
            T.wgmma_gemm(rS, K1R, rO)
            T.copy(rS, S1)
            T.sync_threads(s0_ready, consumer_threads)
            T.wgmma_gemm(S0, K0R, rO)
            T.fence_proxy_async()
            T.named_barrier_arrive(s1_ready, consumer_threads)
            T.wait_wgmma(1)
            T.mbarrier_arrive(k_free[part_1r])
            T.wait_wgmma(0)
        else:
            T.sync_threads(s0_ready, consumer_threads)
            T.wgmma_gemm(S0, K0R, rO)
            T.wait_wgmma(0)
        T.mbarrier_arrive(k_free[part_0r])

    @T.macro
    def consumer1(O, Sinks, QL, QR, QT, K0R, K1L, K1R, K1T, S0, S1, X0, X1, Bias, sM, sL, sLive, q_full, q_free, live_ready, k_ready, k_free, bias_ready, cta, rank, own, pairing):  # fmt: skip
        """Scores the odd blocks; accumulates value dims [256, 512)."""
        T.set_max_nreg(consumer_regs, 1)
        rP = T.alloc_fragment([block_h, block_k], accum)
        rS = T.alloc_fragment([block_h, block_k], dtype)
        rO = T.alloc_fragment([block_h, half], accum)
        out = T.alloc_fragment([block_h, half], dtype)
        rM = T.alloc_fragment([block_h], accum)
        rL = T.alloc_fragment([block_h], accum)
        prev = T.alloc_fragment([block_h], accum)
        cur = T.alloc_fragment([block_h], accum)
        keep = T.alloc_fragment([block_h], accum)
        if skip:
            taken = T.alloc_var("int32")
            odd_pairs = T.alloc_var("int32")
        done = T.alloc_var("int32", init=0) if persistent else 0
        done_odd = T.alloc_var("int32", init=0) if persistent else 0
        n_items = item_count(cta)
        for k in T.serial(n_items):
            b, s, g, h0 = coords(item_of(cta, k))
            # The previous item, whose output a persistent CTA stores once this one starts.
            pb, ps, _, ph0 = coords(item_of(cta, k - 1))
            T.mbarrier_wait_parity(q_full[0], k % 2)
            if skip:
                T.mbarrier_wait_parity(live_ready[0], k % 2)
                taken = taken_blocks(sLive, k % 2, own)
                odd_pairs = taken // 2
            else:
                taken = own
                odd_pairs = own // 2
            if k == 0:
                T.fill(rO, 0)
                T.fill(rL, 0)
                T.fill(rM, -T.infinity(accum))
            if persistent and odd_pairs == 0:
                T.mbarrier_arrive(q_free[0])
            fixed = (
                O, QL, QR, QT, K0R, K1L, K1R, K1T, S0, S1, X0, X1, Bias, sM, q_free, k_ready,
                k_free, bias_ready, rP, rS, rO, out, rM, rL, prev, cur, keep,
            )  # fmt: skip
            for pair in T.serial(T.ceildiv(taken, 2)):
                pair1(pair, has_odd(pair, taken), odd_pairs, done, done_odd, k, pb, ps, ph0, *fixed)
            if persistent:
                done = done + T.ceildiv(taken, 2)
                done_odd = done_odd + odd_pairs
            if paired:
                merge(1, rank, rO, rM, rL, sL, out, K0R, Sinks, O, b, s, h0, *pairing)
            else:
                normalize(1, rM, rL, sL, Sinks, h0)
                if not persistent:
                    store_staged(1, rO, rL, out, QR, O, b, s, h0)
        if persistent:
            lb, ls, _, lh0 = coords(item_of(cta, n_items - 1))
            store_item(rO, rL, out, X0, X1, S0, S1, O, lb, ls, lh0)
            T.tma_store_wait(0)

    @T.macro
    def run(cta, Q, KV, Indices, O, Sinks):
        # A pair's two CTAs are adjacent in x, as a cluster's are; rank r takes the row's
        # blocks from r * span on.
        rank = cta % 2 if paired else 0
        off = rank * span * block_k
        own = T.min(span, blocks - rank * span) if paired else blocks
        QL = T.alloc_shared([block_h, half], dtype)
        QR = T.alloc_shared([block_h, half], dtype)
        K0L = T.alloc_shared([block_k, half], dtype)
        K0R = T.alloc_shared([block_k, half], dtype)
        K1L = T.alloc_shared([block_k, half], dtype)
        K1R = T.alloc_shared([block_k, half], dtype)
        S1 = T.alloc_shared([block_h, block_k], dtype)
        parts = [QL, QR, K0L, K0R, K1L, K1R, S1]
        X0 = X1 = None
        if has_tail:
            QT = T.alloc_shared([block_h, tail_dim], dtype)
            K0T = T.alloc_shared([block_k, tail_dim], dtype)
            K1T = T.alloc_shared([block_k, tail_dim], dtype)
            parts += [QT, K0T, K1T]
            # The even block's weights take its key tail's place once it is scored.
            S0 = K0T
        else:
            QT, K0T, K1T = QR, K0R, K1R
            S0 = T.alloc_shared([block_h, block_k], dtype)
            parts.append(S0)
        if persistent:
            # With S0 and S1, the tiles an item's output leaves through.
            X0 = T.alloc_shared([block_h, atom], dtype)
            X1 = T.alloc_shared([block_h, atom], dtype)
            parts += [X0, X1]
        T.annotate_layout({buf: make_swizzled_layout(buf) for buf in parts})
        Bias = T.alloc_shared([2, block_k], accum)
        sM = T.alloc_shared([block_h], accum)
        sL = T.alloc_shared([2, block_h], accum)
        q_full = T.alloc_barrier([producer_threads])
        q_free = T.alloc_barrier([consumer_threads]) if persistent else None
        sLive = live_ready = None
        if skip:
            # Each producer warp's last live slot plus one, by item parity.
            sLive = T.alloc_shared([2, producer_threads // 32], "int32")
            live_ready = T.alloc_barrier([producer_threads // 32])
        k_ready = T.alloc_barrier([producer_threads] * 4)
        k_free = T.alloc_barrier([warpgroup] * 4)
        bias_ready = T.alloc_barrier([producer_threads])
        pairing = None
        if paired:
            # The peer's row maxes and sums, this CTA's, each CTA's weight of a row, the
            # peer's sums landed, its dims landed, and this CTA's QL and QR free for them.
            sOwn = T.alloc_shared([2, block_h], accum)
            sPeer = T.alloc_shared([block_h, 2], accum)
            sWeight = T.alloc_shared([2, block_h], accum)
            s_full = T.alloc_barrier([1])
            x_free = T.alloc_barrier([1])
            x_full = T.alloc_barrier([1])
            pairing = (sOwn, sPeer, sWeight, QL, QR, s_full, x_free, x_full)
            # Both CTAs' barriers are initialized before either touches the other's.
            T.cluster_arrive()
            T.cluster_wait()
        tx = T.get_thread_binding()
        mark_role_branches()
        if tx >= consumer_threads:
            T.set_max_nreg(producer_regs, 0)
            producer(
                Q, KV, Indices, QL, QR, QT, K0L, K0R, K0T, K1L, K1R, K1T, Bias, sLive,
                q_full, q_free, live_ready, k_ready, k_free, bias_ready, cta, off, own,
                tx - consumer_threads,
            )  # fmt: skip
        with T.ws(0):
            consumer0(
                O, Sinks, QL, QR, QT, K0L, K0R, K0T, K1L, S0, S1, X0, X1, Bias, sM, sL,
                sLive, q_full, q_free, live_ready, k_ready, k_free, bias_ready, cta, rank,
                own, pairing,
            )  # fmt: skip
        with T.ws(1):
            consumer1(
                O, Sinks, QL, QR, QT, K0R, K1L, K1R, K1T, S0, S1, X0, X1, Bias, sM, sL, sLive,
                q_full, q_free, live_ready, k_ready, k_free, bias_ready, cta, rank, own,
                pairing,
            )  # fmt: skip
        if paired:
            # Neither CTA exits while the peer may still write into it.
            T.cluster_arrive()
            T.cluster_wait()

    @T.macro
    def attend(Q, KV, Indices, O, Sinks):
        if paired:
            with T.ClusterKernel(
                ctas, cluster_dims=(2, 1, 1), threads=consumer_threads + producer_threads
            ) as cta:
                run(cta, Q, KV, Indices, O, Sinks)
        else:
            with T.Kernel(ctas, threads=consumer_threads + producer_threads) as cta:
                run(cta, Q, KV, Indices, O, Sinks)

    if has_sink:

        @T.prim_func
        def with_sinks(
            Q: T.Tensor([batch, seq_len, heads, dq], dtype),
            KV: T.Tensor([batch, seq_len_kv, kv_group, dq], dtype),
            Indices: T.Tensor([batch, seq_len, kv_group, topk], "int32"),
            O: T.Tensor([batch, seq_len, heads, dim], dtype),
            Sinks: T.Tensor([heads], accum),
        ):
            attend(Q, KV, Indices, O, Sinks)

        return with_sinks

    @T.prim_func
    def main(
        Q: T.Tensor([batch, seq_len, heads, dq], dtype),
        KV: T.Tensor([batch, seq_len_kv, kv_group, dq], dtype),
        Indices: T.Tensor([batch, seq_len, kv_group, topk], "int32"),
        O: T.Tensor([batch, seq_len, heads, dim], dtype),
    ):
        attend(Q, KV, Indices, O, None)

    return main


class DSADecodeWSKernel(DSADecodeKernelBase):
    """SM90 sparse MLA decode: value dim 512, a key tail of 0 or 64, 64-head blocks, with
    or without the causal mask and the attention sink."""

    supported_archs: list[int] = [90]
    # Both consumers score and both accumulate, where DSADecodeKernel leaves the scores, the
    # softmax and half the sums to one of them.
    preferred_over: ClassVar[frozenset[str]] = frozenset({"dsa_decode_kernel"})
    serves_attn_sink: ClassVar[bool] = True

    @classmethod
    def refusal(cls, call: DSADecodeCall) -> str | None:
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16"
        # The builder's tiles: 64-head blocks, pairs of 64-key blocks, two 256-wide value
        # halves and a key tail of at most one 64-column swizzle atom.
        if (call.heads // call.kv_group) % 64 != 0:
            return "requires a multiple of 64 heads per KV group"
        if call.dim != 512 or call.tail_dim not in (0, 64):
            return "requires dim 512 and tail_dim 0 or 64"
        return super().refusal(call)

    @classmethod
    def entry_options(cls, call: DSADecodeCall) -> dict:
        """The sink, the CTA count, the dead-block skip and the CTA-pair split for `call`."""
        items = call.batch * call.seq_len * (call.heads // 64)
        split = cls.split_for(items, call.topk, call.tail_dim, call.sm_count)
        ctas = items * split if split > 1 else cls.ctas_for(items, call.tail_dim, call.sm_count)
        return {
            **super().entry_options(call),
            "ctas": ctas,
            "skip": cls.skips_dead_blocks(call),
            "split": split,
        }

    @staticmethod
    def ctas_for(items: int, tail_dim: int, sms: int) -> int:
        """The CTAs that run `items` work items, one per (token, 64-head block).

        A CTA takes an SM's shared memory, so more CTAs than SMs run in waves whose CTAs each
        load Q, gather a first pair and store their output with nothing to overlap. One CTA
        per SM instead runs item after item and overlaps those with its previous item; its
        output leaves through four smem tiles, the room a key tail takes.
        """
        return sms if items > sms and tail_dim == 0 else items

    @staticmethod
    def split_for(items: int, topk: int, tail_dim: int, sms: int) -> int:
        """2 when a CTA pair should share each item's blocks, else 1.

        With every item's pair in one wave, each CTA streams half the row's n 64-slot blocks,
        then the pair trades 64 KiB of float32 output through distributed shared memory,
        merges and stores it, taking time t. The pair beats one CTA when n / 2 + t < n; t is
        about two and a half blocks' time, so the pair starts at six blocks. The trade's smem
        does not fit beside a key tail's tiles.
        """
        return 2 if items * 2 <= sms and -(-topk // 64) >= 6 and tail_dim == 0 else 1

    @staticmethod
    def skips_dead_blocks(call: DSADecodeCall) -> bool:
        """Whether a row can end in dead slots worth finding before its blocks are taken.

        An index-only selection pads a short row with -1. A causal query at position s sees
        the first (q_start_index_s + s + 1) // kv_stride keys, so when even the first query
        sees topk of them, a full row has no dead slot to skip.
        """
        visible = min(call.seq_len_kv, (call.q_start_index_s + 1) // call.kv_stride)
        return not call.is_causal or visible < call.topk

    def __init__(
        self, *args, has_attn_sink: bool = False, ctas: int, skip: bool, split: int, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.has_attn_sink = has_attn_sink
        self.ctas = ctas
        self.skip = skip
        self.split = split
        self.program(self._compile)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches; it reads no configuration."""
        self.kernel = kernel = _dsa_decode_ws_kernel(
            self.batch, self.seq_len, self.seq_len_kv, self.heads, self.dim, self.tail_dim,
            self.topk, self.kv_stride, self.q_start_index_s, self.kv_group, self.sm_scale,
            self.dtype_str, self.is_causal, self.has_attn_sink, self.ctas, self.skip, self.split,
        )  # fmt: skip
        return kernel

    @property
    def default_config(self) -> dict:
        return {}

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        attn_sink: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_devices(q=q, kv=kv, indices=indices, attn_sink=attn_sink)
        out = torch.empty(
            (self.batch, self.seq_len, self.heads, self.dim), dtype=q.dtype, device=q.device
        )
        sinks = () if attn_sink is None else (attn_sink,)
        self.program(self._compile)(q, kv, indices, out, *sinks)
        return out
