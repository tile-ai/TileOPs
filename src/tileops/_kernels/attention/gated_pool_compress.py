import dataclasses
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface

__all__ = ["GatedPoolCompressCall", "GatedPoolCompressFwdInterface", "GatedPoolCompressKernel"]


@dataclasses.dataclass(frozen=True)
class GatedPoolCompressCall(CallSpec):
    """One gated pooling of packed tokens into an entry every ``ratio`` positions."""

    ratio: int = 0
    overlap: bool = False
    dim: int = 0
    dtype: torch.dtype | None = None


class GatedPoolCompressFwdInterface(KernelInterface):
    """The softmax-gated sum of every window of ``ratio`` positions, and the state it leaves."""

    request = GatedPoolCompressCall

    @abstractmethod
    def forward(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        start_pos: torch.Tensor,
        state_slots: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Pool the windows the packed tokens close and write their state rows in place.

        Request ``i`` holds tokens ``cu_seqlens[i]:cu_seqlens[i + 1]`` at positions from
        ``start_pos[i]``. The token at position ``p`` with ``(p + 1) % call.ratio == 0``
        closes a window: per channel, the softmax over the window's positions of
        ``score + ape`` in float32, the weights and their products with ``kv`` rounded to
        ``call.dtype``, summed in float32 and rounded to bfloat16. With ``call.overlap`` the
        window also takes the first ``call.dim`` channels of the previous window's
        positions, its own giving the last ``call.dim``. Positions before ``start_pos[i]``
        come from the ring of slot ``state_slots[i]``, row ``p % (c * ratio)``; the call
        writes the rows of the request's last ``c * ratio`` positions.

        Args:
            kv: ``(T_q, c * call.dim)`` in ``call.dtype``, contiguous.
            score: As ``kv``.
            ape: ``(call.ratio, c * call.dim)`` in ``call.dtype``, contiguous.
            kv_state: ``(NS, c * call.ratio, c * call.dim)`` float32, contiguous; written.
            score_state: As ``kv_state``, holding ``score + ape``; written.
            cu_seqlens: ``(N + 1,)`` int32 packed offsets.
            start_pos: ``(N,)`` int32 first position of each request.
            state_slots: ``(N,)`` int32 state slot of each request.

        Returns:
            ``entries`` ``(T_q, call.dim)`` bfloat16, defined at the closing tokens, and
            ``valid`` ``(T_q,)`` bool, true at them.
        """


@target_cache(maxsize=32)
def _gated_pool_compress_kernel(ratio, overlap, dim, in_dtype):
    C = 2 if overlap else 1
    # The positions a window pools, and the ring rows a request keeps.
    SPAN = C * ratio
    WIDTH = C * dim
    # A short window's blocks keep SM_WARPS warps an SM: fewer starve the loads a prefill
    # keeps in flight.
    SM_WARPS = 16
    WARP_VEC = 4
    # A window of at most REG_SPAN positions pools in a warp from registers, rounding its
    # weights and products to kv's dtype as the reference does: its l_sc and l_kv, SPAN *
    # WARP_VEC float32 each, take at most a thread's share of an SM's registers. A longer one
    # pools across the block, BATCH positions a warp at a time folded in online, in float32:
    # for float32 kv the same sum, for bfloat16 kv within the entry's bfloat16 tolerance.
    REG_SPAN = 65536 // (SM_WARPS * 32) // (2 * WARP_VEC)
    BATCH = 8
    LONG = SPAN > REG_SPAN
    # A long window's warps, one per BATCH positions and at most 16 (at least 3, as SPAN > 16),
    # and the batches each folds.
    LONG_WARPS = min(16, -(-SPAN // BATCH))
    N_BATCH = -(-SPAN // (LONG_WARPS * BATCH))
    # Channels a lane pools at once: WARP_VEC for a warp's window, two for a block's, which
    # spreads a long window's loads over twice the blocks.
    VEC = (2 if dim % 2 == 0 else 1) if LONG else (WARP_VEC if dim % WARP_VEC == 0 else 1)
    LANE_CH = 32 * VEC
    CHUNKS = -(-dim // LANE_CH)
    # Whether a long window's block slots cover its positions and its lanes a chunk's
    # channels exactly, with none to mask.
    LONG_EXACT = LONG_WARPS * N_BATCH * BATCH == SPAN and dim % LANE_CH == 0
    # Short windows a block pools, a warp each: at least the C head windows a block owns.
    WARPS = 4
    # Rounding to kv's dtype; float32 rounds nothing.
    ROUND = in_dtype != "float32"

    # Every index is in range by construction: a position within its window, a ring row
    # modulo SPAN, a token within its request. TileLang's own checks would test the dynamic
    # bounds ahead of every load and hold each load behind the one before it.
    @tilelang.jit(
        out_idx=[8, 9],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        },
    )
    def gated_pool_compress_func():
        t_q = T.dynamic("T_q")
        n_req = T.dynamic("N")
        n_slots = T.dynamic("NS")
        # Slots of the windows that read no ring, those closing at or after the request's
        # first position + SPAN - 1: request i's start at cu_seqlens[i] // ratio. It has at
        # most tokens // ratio of them, as SPAN >= ratio, and floor sums keep that many slots
        # before the next request's.
        n_windows = t_q // ratio

        # A long window takes a block, a short one a warp of WARPS.
        BLOCK_WARPS = LONG_WARPS if LONG else WARPS
        n_units = n_windows if LONG else T.ceildiv(n_windows, BLOCK_WARPS)
        # Ring rows a thread carries from the call's tokens to the states at once, and how
        # many times: the first rows load before the head windows pool.
        WRITE_ROWS = min(2 if LONG else 4, -(-SPAN // BLOCK_WARPS))
        N_WRITE = -(-SPAN // (BLOCK_WARPS * WRITE_ROWS))

        def rounded(x):
            return T.Cast(T.float32, T.Cast(in_dtype, x)) if ROUND else x

        @T.macro
        def load_slot(
            l_sc, l_kv, u, q, h, col, base, first, slot, kv, score, ape, kv_state, score_state, ring
        ):
            # Slot u's score + ape and kv at position q, channels h * dim + col .. + VEC: from
            # the call's tokens at and after first; with ring, from the ring before it, masked
            # below 0. Without ring every position is a token's, and nothing branches; with it
            # the branches test registers alone, alike across the warp.
            if not ring or q >= first:
                for e in T.vectorized(VEC):
                    l_sc[u * VEC + e] = T.Cast(
                        T.float32, score[base + q - first, h * dim + col + e]
                    ) + T.Cast(T.float32, ape[q % ratio, h * dim + col + e])
                for e in T.vectorized(VEC):
                    l_kv[u * VEC + e] = T.Cast(T.float32, kv[base + q - first, h * dim + col + e])
            elif q >= 0:
                for e in T.vectorized(VEC):
                    l_sc[u * VEC + e] = score_state[slot, q % SPAN, h * dim + col + e]
                for e in T.vectorized(VEC):
                    l_kv[u * VEC + e] = kv_state[slot, q % SPAN, h * dim + col + e]
            else:
                for e in T.unroll(VEC):
                    l_sc[u * VEC + e] = -T.infinity(T.float32)
                    l_kv[u * VEC + e] = T.float32(0)

        @T.macro
        def load_lone(
            l_sc, l_kv, u, w, q, h, col, base, first, slot, kv, score, ape, kv_state, score_state
        ):
            # load_slot for a window whose closing position is the request's first token: its
            # last slot is that token, the others the ring's rows (masked below 0 after). Each
            # slot's source is fixed, so the loads branch on nothing loaded and all issue
            # before any is used; the token's slot comes last.
            if w < SPAN - 1:
                for e in T.vectorized(VEC):
                    l_sc[u * VEC + e] = score_state[slot, (q + SPAN) % SPAN, h * dim + col + e]
                for e in T.vectorized(VEC):
                    l_kv[u * VEC + e] = kv_state[slot, (q + SPAN) % SPAN, h * dim + col + e]
            else:
                for e in T.vectorized(VEC):
                    l_sc[u * VEC + e] = T.Cast(T.float32, score[base, h * dim + col + e]) + T.Cast(
                        T.float32, ape[w % ratio, h * dim + col + e]
                    )
                for e in T.vectorized(VEC):
                    l_kv[u * VEC + e] = T.Cast(T.float32, kv[base, h * dim + col + e])

        @T.macro
        def load_slot_live(
            l_sc, l_kv, u, w, p, col, base, first, slot, kv, score, ape, kv_state, score_state, ring
        ):
            # load_slot of window slot w of the window closing at p, masked where the block's
            # slots run past the window or its lanes past the channels; where they never do,
            # nothing branches around the loads.
            if LONG_EXACT or (w < SPAN and col < dim):
                load_slot(
                    l_sc,
                    l_kv,
                    u,
                    p - SPAN + 1 + w,
                    T.if_then_else(w < SPAN - ratio, 0, C - 1),
                    col,
                    base,
                    first,
                    slot,
                    kv,
                    score,
                    ape,
                    kv_state,
                    score_state,
                    ring,
                )
            else:
                for e in T.unroll(VEC):
                    l_sc[u * VEC + e] = -T.infinity(T.float32)
                    l_kv[u * VEC + e] = T.float32(0)

        @T.macro
        def mask_before_zero(l_sc, l_kv, u, q):
            # A position before 0, loaded from a ring row only to keep the loads unbranched.
            for e in T.unroll(VEC):
                l_sc[u * VEC + e] = T.if_then_else(
                    q >= 0, l_sc[u * VEC + e], -T.infinity(T.float32)
                )
                l_kv[u * VEC + e] = T.if_then_else(q >= 0, l_kv[u * VEC + e], T.float32(0))

        @T.macro
        def pool_window(
            kv,
            score,
            ape,
            kv_state,
            score_state,
            entries,
            l_sc,
            l_kv,
            l_out,
            l_m,
            l_s,
            p,
            col,
            base,
            first,
            slot,
            ring,
        ):
            # A warp's entry of the window closing at position p of the request whose tokens
            # start at base and position first, channels col .. col + VEC, from registers:
            # each exponential once, in place of its score, and one reciprocal a channel. A
            # rounding dtype rounds each weight, then each product, as the reference does.
            if ring and p == first:
                for u in T.unroll(SPAN):
                    load_lone(
                        l_sc,
                        l_kv,
                        u,
                        u,
                        p - SPAN + 1 + u,
                        T.if_then_else(u < SPAN - ratio, 0, C - 1),
                        col,
                        base,
                        first,
                        slot,
                        kv,
                        score,
                        ape,
                        kv_state,
                        score_state,
                    )
                for u in T.unroll(SPAN):
                    mask_before_zero(l_sc, l_kv, u, p - SPAN + 1 + u)
            else:
                for u in T.unroll(SPAN):
                    load_slot(
                        l_sc,
                        l_kv,
                        u,
                        p - SPAN + 1 + u,
                        T.if_then_else(u < SPAN - ratio, 0, C - 1),
                        col,
                        base,
                        first,
                        slot,
                        kv,
                        score,
                        ape,
                        kv_state,
                        score_state,
                        ring,
                    )
            for e in T.unroll(VEC):
                l_m[e] = -T.infinity(T.float32)
                for u in T.unroll(SPAN):
                    l_m[e] = T.max(l_m[e], l_sc[u * VEC + e])
                l_s[e] = T.float32(0)
                for u in T.unroll(SPAN):
                    l_sc[u * VEC + e] = T.__exp(l_sc[u * VEC + e] - l_m[e])
                    l_s[e] = l_s[e] + l_sc[u * VEC + e]
                l_out[e] = T.float32(0)
                if ROUND:
                    l_s[e] = T.fast_rcp(l_s[e])
                    for u in T.unroll(SPAN):
                        l_out[e] = l_out[e] + rounded(
                            rounded(l_sc[u * VEC + e] * l_s[e]) * rounded(l_kv[u * VEC + e])
                        )
                else:
                    for u in T.unroll(SPAN):
                        l_out[e] = l_out[e] + l_sc[u * VEC + e] * l_kv[u * VEC + e]
                    l_out[e] = l_out[e] * T.fast_rcp(l_s[e])
            for e in T.vectorized(VEC):
                entries[base + p - first, col + e] = T.Cast(T.bfloat16, l_out[e])

        @T.macro
        def pool_window_long(
            kv,
            score,
            ape,
            kv_state,
            score_state,
            entries,
            l_sc,
            l_kv,
            l_out,
            l_m,
            l_s,
            l_bm,
            l_bs,
            s_part,
            p,
            col,
            base,
            first,
            slot,
            ring,
            warp,
            tx,
            chunk,
        ):
            # The block's entry of a window longer than a warp's registers hold: warp w
            # pools positions w * N_BATCH * BATCH on, BATCH at a time, into its max, its sum
            # of exponentials past it and its weighted sum; threads 0 .. LANE_CH - 1 merge the
            # warps' partials, a channel each. A masked position, at -inf, adds nothing; a
            # warp whose positions are all masked keeps -inf and zero sums.
            for b in T.unroll(N_BATCH):
                if LONG_EXACT and ring and p == first:
                    for u in T.unroll(BATCH):
                        load_lone(
                            l_sc,
                            l_kv,
                            u,
                            (warp * N_BATCH + b) * BATCH + u,
                            p - SPAN + 1 + (warp * N_BATCH + b) * BATCH + u,
                            T.if_then_else(
                                (warp * N_BATCH + b) * BATCH + u < SPAN - ratio, 0, C - 1
                            ),
                            col,
                            base,
                            first,
                            slot,
                            kv,
                            score,
                            ape,
                            kv_state,
                            score_state,
                        )
                    for u in T.unroll(BATCH):
                        mask_before_zero(
                            l_sc, l_kv, u, p - SPAN + 1 + (warp * N_BATCH + b) * BATCH + u
                        )
                else:
                    for u in T.unroll(BATCH):
                        load_slot_live(
                            l_sc,
                            l_kv,
                            u,
                            (warp * N_BATCH + b) * BATCH + u,
                            p,
                            col,
                            base,
                            first,
                            slot,
                            kv,
                            score,
                            ape,
                            kv_state,
                            score_state,
                            ring,
                        )
                for e in T.unroll(VEC):
                    # The batch's max, then the running max; a max still at -inf subtracts
                    # zero instead, so its exponentials are zero rather than NaN.
                    l_bm[e] = -T.infinity(T.float32)
                    for u in T.unroll(BATCH):
                        l_bm[e] = T.max(l_bm[e], l_sc[u * VEC + e])
                    if b > 0:
                        l_bm[e] = T.max(l_bm[e], l_m[e])
                    l_bs[e] = T.if_then_else(l_bm[e] > -T.infinity(T.float32), l_bm[e], 0)
                    if b == 0:
                        l_s[e] = T.float32(0)
                        l_out[e] = T.float32(0)
                    else:
                        l_s[e] = l_s[e] * T.__exp(l_m[e] - l_bs[e])
                        l_out[e] = l_out[e] * T.__exp(l_m[e] - l_bs[e])
                    for u in T.unroll(BATCH):
                        l_sc[u * VEC + e] = T.__exp(l_sc[u * VEC + e] - l_bs[e])
                        l_s[e] = l_s[e] + l_sc[u * VEC + e]
                        l_out[e] = l_out[e] + l_sc[u * VEC + e] * l_kv[u * VEC + e]
                    l_m[e] = l_bm[e]
            for e in T.unroll(VEC):
                s_part[0, warp, (tx % 32) * VEC + e] = l_m[e]
                s_part[1, warp, (tx % 32) * VEC + e] = l_s[e]
                s_part[2, warp, (tx % 32) * VEC + e] = l_out[e]
            T.sync_threads()
            if tx < LANE_CH and chunk * LANE_CH + tx < dim:
                l_bm[0] = -T.infinity(T.float32)
                for w in T.unroll(LONG_WARPS):
                    l_bm[0] = T.max(l_bm[0], s_part[0, w, tx])
                l_s[0] = T.float32(0)
                l_out[0] = T.float32(0)
                for w in T.unroll(LONG_WARPS):
                    l_bs[0] = T.__exp(s_part[0, w, tx] - l_bm[0])
                    l_s[0] = l_s[0] + s_part[1, w, tx] * l_bs[0]
                    l_out[0] = l_out[0] + s_part[2, w, tx] * l_bs[0]
                entries[base + p - first, chunk * LANE_CH + tx] = T.Cast(
                    T.bfloat16, l_out[0] * T.fast_rcp(l_s[0])
                )
            # The partials are read before a next window of the block writes its own.
            T.sync_threads()

        @T.prim_func
        def gated_pool_compress_main(
            kv: T.Tensor[(t_q, WIDTH), in_dtype],
            score: T.Tensor[(t_q, WIDTH), in_dtype],
            ape: T.Tensor[(ratio, WIDTH), in_dtype],
            kv_state: T.Tensor[(n_slots, SPAN, WIDTH), T.float32],
            score_state: T.Tensor[(n_slots, SPAN, WIDTH), T.float32],
            cu_seqlens: T.Tensor[(n_req + 1,), T.int32],
            start_pos: T.Tensor[(n_req,), T.int32],
            state_slots: T.Tensor[(n_req,), T.int32],
            entries: T.Tensor[(t_q, dim), T.bfloat16],
            valid: T.Tensor[(t_q,), T.bool],
        ):
            # Blocks 0 .. N * CHUNKS - 1 each own a request's windows that read its ring, a
            # chunk of their channels, and then write those channels of the ring (a later
            # token of a short request takes a row an earlier window reads) and, in turn, the
            # request's valid flags. They come first, the longest chain of loads in the grid.
            # The blocks after them pool the other windows: a warp each, or the whole block a
            # window longer than a warp's registers hold. A block's chunk is its index modulo
            # CHUNKS.
            with T.Kernel((n_req + n_units) * CHUNKS, threads=BLOCK_WARPS * 32) as bx:
                if not LONG:
                    T.annotate_min_blocks_per_sm(max(1, SM_WARPS // BLOCK_WARPS))
                if LONG and N_BATCH == 1:
                    # Two blocks an SM: 16 warps' threads each fit 64 registers with a single
                    # batch live.
                    T.annotate_min_blocks_per_sm(2)
                tx = T.get_thread_binding()
                warp = tx // 32
                lane = tx % 32
                chunk = bx % CHUNKS
                col = chunk * LANE_CH + lane * VEC
                l_sc = T.alloc_local([(BATCH if LONG else SPAN) * VEC], T.float32)
                l_kv = T.alloc_local([(BATCH if LONG else SPAN) * VEC], T.float32)
                # Ring rows on their way from the call's tokens to the states.
                l_wkv = T.alloc_local([WRITE_ROWS * C * VEC], T.float32)
                l_wsc = T.alloc_local([WRITE_ROWS * C * VEC], T.float32)
                l_out = T.alloc_local([VEC], T.float32)
                l_m = T.alloc_local([VEC], T.float32)
                l_s = T.alloc_local([VEC], T.float32)
                # A batch's max, and the max its exponentials subtract.
                l_bm = T.alloc_local([VEC], T.float32)
                l_bs = T.alloc_local([VEC], T.float32)
                # Each warp's max, exponential sum and weighted sum of a long window.
                s_part = T.alloc_shared([3, LONG_WARPS, LANE_CH], T.float32) if LONG else None
                l_lo = T.alloc_var(T.int32)
                l_hi = T.alloc_var(T.int32)
                l_step = T.alloc_var(T.int32)
                l_req = T.alloc_var(T.int32)
                l_p = T.alloc_var(T.int32)
                # The request's first token, its first position, its token count and its slot.
                l_base = T.alloc_var(T.int32)
                l_first = T.alloc_var(T.int32)
                l_count = T.alloc_var(T.int32)
                l_slot = T.alloc_var(T.int32)
                l_g = T.alloc_var(T.int32)
                # The first of the request's last SPAN positions, whose rows the call writes.
                l_w0 = T.alloc_var(T.int32)

                if bx < n_req * CHUNKS:
                    l_req = bx // CHUNKS
                    l_base = cu_seqlens[l_req]
                    l_first = start_pos[l_req]
                    l_count = cu_seqlens[l_req + 1] - l_base
                    l_slot = state_slots[l_req]
                    l_w0 = T.max(l_first, l_first + l_count - SPAN)
                    # The first WRITE_ROWS rows a thread writes, loaded now so their loads wait
                    # out the global latency with the windows'.
                    for r in T.unroll(WRITE_ROWS):
                        q = l_w0 + r * BLOCK_WARPS + warp
                        if q < l_first + l_count and col < dim:
                            for h in T.unroll(C):
                                for e in T.vectorized(VEC):
                                    l_wkv[(r * C + h) * VEC + e] = T.Cast(
                                        T.float32, kv[l_base + q - l_first, h * dim + col + e]
                                    )
                                for e in T.vectorized(VEC):
                                    l_wsc[(r * C + h) * VEC + e] = T.Cast(
                                        T.float32, score[l_base + q - l_first, h * dim + col + e]
                                    ) + T.Cast(T.float32, ape[q % ratio, h * dim + col + e])
                    # The request's windows that read its ring close before position
                    # first + SPAN - 1: at most C of them.
                    if LONG:
                        for w in T.serial(C):
                            l_p = l_first + (ratio - 1 - l_first % ratio) % ratio + w * ratio
                            if l_p < l_first + l_count and l_p < l_first + SPAN - 1:
                                pool_window_long(
                                    kv,
                                    score,
                                    ape,
                                    kv_state,
                                    score_state,
                                    entries,
                                    l_sc,
                                    l_kv,
                                    l_out,
                                    l_m,
                                    l_s,
                                    l_bm,
                                    l_bs,
                                    s_part,
                                    l_p,
                                    col,
                                    l_base,
                                    l_first,
                                    l_slot,
                                    True,
                                    warp,
                                    tx,
                                    chunk,
                                )
                    elif warp < C:
                        l_p = l_first + (ratio - 1 - l_first % ratio) % ratio + warp * ratio
                        if l_p < l_first + l_count and l_p < l_first + SPAN - 1 and col < dim:
                            pool_window(
                                kv,
                                score,
                                ape,
                                kv_state,
                                score_state,
                                entries,
                                l_sc,
                                l_kv,
                                l_out,
                                l_m,
                                l_s,
                                l_p,
                                col,
                                l_base,
                                l_first,
                                l_slot,
                                True,
                            )
                    # A later token's row may be one a head window read; a request's only token
                    # takes the row of the position SPAN before it, which no window reads.
                    if l_count > 1:
                        T.sync_threads()
                    for k in T.serial(N_WRITE):
                        if k > 0:
                            for r in T.unroll(WRITE_ROWS):
                                q = l_w0 + (k * WRITE_ROWS + r) * BLOCK_WARPS + warp
                                if q < l_first + l_count and col < dim:
                                    for h in T.unroll(C):
                                        for e in T.vectorized(VEC):
                                            l_wkv[(r * C + h) * VEC + e] = T.Cast(
                                                T.float32,
                                                kv[l_base + q - l_first, h * dim + col + e],
                                            )
                                        for e in T.vectorized(VEC):
                                            l_wsc[(r * C + h) * VEC + e] = T.Cast(
                                                T.float32,
                                                score[l_base + q - l_first, h * dim + col + e],
                                            ) + T.Cast(T.float32, ape[q % ratio, h * dim + col + e])
                        for r in T.unroll(WRITE_ROWS):
                            q = l_w0 + (k * WRITE_ROWS + r) * BLOCK_WARPS + warp
                            if q < l_first + l_count and col < dim:
                                for h in T.unroll(C):
                                    for e in T.vectorized(VEC):
                                        kv_state[l_slot, q % SPAN, h * dim + col + e] = l_wkv[
                                            (r * C + h) * VEC + e
                                        ]
                                    for e in T.vectorized(VEC):
                                        score_state[l_slot, q % SPAN, h * dim + col + e] = l_wsc[
                                            (r * C + h) * VEC + e
                                        ]
                    # The valid flags of the request's tokens, its CHUNKS blocks in turn.
                    for j in T.serial(T.ceildiv(l_count, CHUNKS * BLOCK_WARPS * 32)):
                        q = (j * CHUNKS + chunk) * (BLOCK_WARPS * 32) + tx
                        if q < l_count:
                            valid[l_base + q] = (l_first + q + 1) % ratio == 0
                else:
                    l_g = (
                        bx // CHUNKS - n_req
                        if LONG
                        else (bx // CHUNKS - n_req) * BLOCK_WARPS + warp
                    )
                    if l_g < n_windows:
                        # The request whose slots hold g: the last i with
                        # cu_seqlens[i] // ratio <= g, which i = 0 meets; an earlier request
                        # with the same start has no slot. The warp's lanes test 32 evenly
                        # spaced candidates at once, and the range shrinks to the gap after the
                        # last that meets it: 32 times smaller.
                        l_lo = 0
                        l_hi = n_req - 1
                        for _round in T.serial(7):
                            if l_lo < l_hi:
                                l_step = T.ceildiv(l_hi - l_lo + 1, 32)
                                l_lo = (
                                    l_lo
                                    + (
                                        T.Cast(
                                            T.int32,
                                            T.popcount(
                                                T.ballot_sync(
                                                    lane * l_step <= l_hi - l_lo
                                                    and cu_seqlens[
                                                        T.min(l_lo + lane * l_step, l_hi)
                                                    ]
                                                    // ratio
                                                    <= l_g
                                                )
                                            ),
                                        )
                                        - 1
                                    )
                                    * l_step
                                )
                                l_hi = T.min(l_hi, l_lo + l_step - 1)
                        l_req = l_lo
                        l_base = cu_seqlens[l_req]
                        l_first = start_pos[l_req]
                        l_count = cu_seqlens[l_req + 1] - l_base
                        # Slot g - base // ratio: the window closing that many windows after
                        # the first at or after first + SPAN - 1.
                        l_p = (T.ceildiv(l_first + SPAN, ratio) + l_g - l_base // ratio) * ratio - 1
                        if LONG:
                            if l_p < l_first + l_count:
                                pool_window_long(
                                    kv,
                                    score,
                                    ape,
                                    kv_state,
                                    score_state,
                                    entries,
                                    l_sc,
                                    l_kv,
                                    l_out,
                                    l_m,
                                    l_s,
                                    l_bm,
                                    l_bs,
                                    s_part,
                                    l_p,
                                    col,
                                    l_base,
                                    l_first,
                                    l_slot,
                                    False,
                                    warp,
                                    tx,
                                    chunk,
                                )
                        elif l_p < l_first + l_count and col < dim:
                            pool_window(
                                kv,
                                score,
                                ape,
                                kv_state,
                                score_state,
                                entries,
                                l_sc,
                                l_kv,
                                l_out,
                                l_m,
                                l_s,
                                l_p,
                                col,
                                l_base,
                                l_first,
                                l_slot,
                                False,
                            )

        return gated_pool_compress_main

    return gated_pool_compress_func


class GatedPoolCompressKernel(Kernel, GatedPoolCompressFwdInterface):
    """Gated pooling of packed tokens into an entry every ``ratio`` positions, with its ring.

    Args:
        ratio: Positions a window pools.
        overlap: Whether a window also pools the previous window's first ``dim`` channels.
        dim: Channels of an entry.
        dtype: Torch dtype of ``kv``, ``score`` and ``ape``.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: GatedPoolCompressCall) -> Entry:
        return call, lambda: cls(call.ratio, call.overlap, call.dim, call.dtype)

    def __init__(
        self,
        ratio: int,
        overlap: bool,
        dim: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.ratio = ratio
        self.overlap = overlap
        self.dim = dim
        self.dtype = dtype
        self.kernel = _gated_pool_compress_kernel(ratio, overlap, dim, self.dtype_str)
        self.init_config()

    def forward(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        start_pos: torch.Tensor,
        state_slots: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        if start_pos.shape[0] == 0:
            # No request: no token, and no block to launch.
            return (
                torch.empty((0, self.dim), dtype=torch.bfloat16, device=kv.device),
                torch.empty((0,), dtype=torch.bool, device=kv.device),
            )
        return self.program(self._compile)(
            kv, score, ape, kv_state, score_state, cu_seqlens, start_pos, state_slots
        )

    def _compile(self, cfg: dict):
        """The program ``forward`` launches; the kernel takes no configuration."""
        return self.kernel()
