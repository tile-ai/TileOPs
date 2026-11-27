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
            kv: ``(T_q, c * call.dim)`` in ``call.dtype``, its channels contiguous and its
                rows any stride apart.
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
def _gated_pool_compress_kernel(
    ratio, overlap, dim, in_dtype, one_token, long_warps, kv_stride, score_stride
):
    """The pooling program; ``kv`` and ``score`` rows lie ``kv_stride`` and ``score_stride``
    elements apart, ``None`` for contiguous rows."""
    K = GatedPoolCompressKernel
    SM_WARPS, BATCH = K.SM_WARPS, K.BATCH
    C = 2 if overlap else 1
    # The positions a window pools, and the ring rows a request keeps.
    SPAN = C * ratio
    WIDTH = C * dim
    LONG, VEC = K._lane_vec(ratio, overlap, dim)
    # A long window's warps, one per BATCH positions and at most long_warps (at least 3, as
    # SPAN > 16), and the batches each folds.
    WINDOW_WARPS = min(long_warps, -(-SPAN // BATCH))
    N_BATCH = -(-SPAN // (WINDOW_WARPS * BATCH))
    # The 8-warp program keeps each batch's partial apart, the one a 16-warp block's warp
    # computes over the same BATCH positions, and merges them as the 16 warps' partials: a
    # window's entry is the same whichever program pools it. Rows of those partials.
    SPLIT = long_warps < K.LONG_WARPS
    PARTS = WINDOW_WARPS * N_BATCH if SPLIT else WINDOW_WARPS
    LANE_CH = 32 * VEC
    CHUNKS = -(-dim // LANE_CH)
    # Whether a long window's block slots cover its positions and its lanes a chunk's
    # channels exactly, with none to mask.
    LONG_EXACT = WINDOW_WARPS * N_BATCH * BATCH == SPAN and dim % LANE_CH == 0
    # Short windows a block pools, a warp each: at least the C head windows a block owns.
    WARPS = 4
    # A request's only token, alone in its request, leaves the block's other warps idle: a
    # short window splits its chunk's channels over TOKEN_WARPS warps, TOKEN_VEC a lane, so
    # that the blocks two requests place on one SM do not issue from one scheduler. A long
    # window's token is the last slot of warp TOKEN_WARP.
    TOKEN_VEC = max(1, VEC // WARPS)
    TOKEN_WARPS = VEC // TOKEN_VEC
    TOKEN_WARP = (SPAN - 1) // (N_BATCH * BATCH) if LONG else 0
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
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            # Binds stay binds: TileLang 0.1.15's IfStmtBinding would inline them into one if a
            # statement, which MergeIfStmt does not rejoin, putting window loads behind branches.
            tilelang.PassConfigKey.TL_IF_STMT_BINDING_INLINE_REPLAYABLE_BINDS: False,
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
        BLOCK_WARPS = WINDOW_WARPS if LONG else WARPS
        n_units = n_windows if LONG else T.ceildiv(n_windows, BLOCK_WARPS)
        # The warps that write a request's ring rows: a long window's all, a short window's
        # those that pool no head window. The rows a thread carries from the call's tokens to
        # the states at once, and how many times: 8 * VEC floats each of kv and score for a
        # short window, which takes a request's ring rows in one batch for ratio 4; for a long
        # window 4 * VEC, the fewest dependent batches 64 registers a thread allow, but 2 * VEC
        # in the one-token program, where a request of several tokens is rare and its rows'
        # registers stay out of the one-token path's.
        ROW_WARPS = BLOCK_WARPS if LONG else WARPS - C
        ROW_FLOATS = (2 if one_token else 4) if LONG else 8
        WRITE_ROWS = min(ROW_FLOATS // C, -(-SPAN // ROW_WARPS))
        N_WRITE = -(-SPAN // (ROW_WARPS * WRITE_ROWS))

        def row_of(w0, k, r, warp):
            # The position of row r of a thread's batch k of ring rows.
            return w0 + (k * WRITE_ROWS + r) * ROW_WARPS + (warp if LONG else warp - C)

        def rounded(x):
            return T.Cast(T.float32, T.Cast(in_dtype, x)) if ROUND else x

        def head_of(w):
            # The channel half window slot w pools: the previous window's slots take the
            # first, the window's own the last.
            return T.if_then_else(w < SPAN - ratio, 0, C - 1)

        @T.macro
        def load_rows(l_wkv, l_wsc, k, kv, score, ape, base, first, count, w0, warp, col):
            # Pass k of the ring rows a thread carries from the call's tokens to the states.
            for r in T.unroll(WRITE_ROWS):
                q = row_of(w0, k, r, warp)
                if q < first + count and col < dim:
                    for h in T.unroll(C):
                        for e in T.vectorized(VEC):
                            l_wkv[(r * C + h) * VEC + e] = T.Cast(
                                T.float32, kv[base + q - first, h * dim + col + e]
                            )
                        for e in T.vectorized(VEC):
                            l_wsc[(r * C + h) * VEC + e] = T.Cast(
                                T.float32, score[base + q - first, h * dim + col + e]
                            ) + T.Cast(T.float32, ape[q % ratio, h * dim + col + e])

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

        def lone_row(w, l_off):
            # The ring row of slot w < SPAN - 1 of a window closing at a request's first
            # position p: that of position p - SPAN + 1 + w, (w + l_off) % SPAN for
            # l_off = (p + 1) % SPAN, which is 0 or ratio as ratio divides p + 1. Over one
            # window's positions it is w; over two, w + l_off for the previous window's and
            # w - l_off for the window's own, an offset from one row with no modulo.
            if C == 1:
                return w
            return T.if_then_else(w < ratio, w + l_off, w - l_off)

        @T.macro
        def load_lone(
            l_sc,
            l_kv,
            u,
            vec,
            w,
            col,
            base,
            slot,
            l_off,
            tok_regs,
            kv,
            score,
            ape,
            kv_state,
            score_state,
        ):
            # Slot w of the window closing at the request's first token, into slot u: the
            # ring's row, or for w = SPAN - 1 the token, which with tok_regs the caller takes
            # from the registers its ring row is loaded into. Each slot's source is fixed, so
            # the loads branch on nothing loaded and all issue before any is used.
            if w < SPAN - 1:
                for e in T.vectorized(vec):
                    l_sc[u * vec + e] = score_state[
                        slot,
                        lone_row(w, l_off),
                        head_of(w) * dim + col + e,
                    ]
                for e in T.vectorized(vec):
                    l_kv[u * vec + e] = kv_state[
                        slot,
                        lone_row(w, l_off),
                        head_of(w) * dim + col + e,
                    ]
            elif not tok_regs:
                for e in T.vectorized(vec):
                    l_sc[u * vec + e] = T.Cast(
                        T.float32, score[base, (C - 1) * dim + col + e]
                    ) + T.Cast(T.float32, ape[ratio - 1, (C - 1) * dim + col + e])
                for e in T.vectorized(vec):
                    l_kv[u * vec + e] = T.Cast(T.float32, kv[base, (C - 1) * dim + col + e])

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
                    head_of(w),
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
        def load_lone_window(
            l_sc,
            l_kv,
            vec,
            col,
            base,
            first,
            slot,
            l_off,
            tok_regs,
            kv,
            score,
            ape,
            kv_state,
            score_state,
        ):
            # A warp's window closing at the request's first token. A position before 0 is
            # one of the previous window's, which the window at position ratio - 1 alone
            # lacks; one test for the whole block keeps it off the other windows' path.
            if C == 2:
                if first >= SPAN - 1:
                    for u in T.unroll(ratio, explicit=True):
                        load_lone(
                            l_sc,
                            l_kv,
                            u,
                            vec,
                            u,
                            col,
                            base,
                            slot,
                            l_off,
                            tok_regs,
                            kv,
                            score,
                            ape,
                            kv_state,
                            score_state,
                        )
                else:
                    for u in T.unroll(ratio):
                        for e in T.unroll(vec):
                            l_sc[u * vec + e] = -T.infinity(T.float32)
                            l_kv[u * vec + e] = T.float32(0)
            for u in T.unroll(SPAN - ratio, SPAN, explicit=True):
                load_lone(
                    l_sc,
                    l_kv,
                    u,
                    vec,
                    u,
                    col,
                    base,
                    slot,
                    l_off,
                    tok_regs,
                    kv,
                    score,
                    ape,
                    kv_state,
                    score_state,
                )

        @T.macro
        def load_lone_batch(
            l_sc,
            l_kv,
            b,
            warp,
            col,
            base,
            first,
            slot,
            l_off,
            tok_regs,
            kv,
            score,
            ape,
            kv_state,
            score_state,
        ):
            # Batch b of a block's window closing at the request's first token, whose slots
            # cover the window exactly. Over two windows' positions, those before 0 are masked
            # after the loads.
            for u in T.unroll(BATCH, explicit=True):
                load_lone(
                    l_sc,
                    l_kv,
                    u,
                    VEC,
                    (warp * N_BATCH + b) * BATCH + u,
                    col,
                    base,
                    slot,
                    l_off,
                    tok_regs,
                    kv,
                    score,
                    ape,
                    kv_state,
                    score_state,
                )
            if C == 2:
                for u in T.unroll(BATCH):
                    mask_before_zero(
                        l_sc, l_kv, u, first - SPAN + 1 + (warp * N_BATCH + b) * BATCH + u
                    )

        @T.macro
        def pool_registers(entries, l_sc, l_kv, vec, l_out, l_m, l_s, row, col):
            # A warp's entry of a window whose slots are in l_sc and l_kv, channels col .. col
            # + vec, into entries row row: each exponential once, in place of its score, and
            # one reciprocal a channel. A rounding dtype rounds each weight, then each product,
            # as the reference does.
            for e in T.unroll(vec):
                l_m[e] = -T.infinity(T.float32)
                for u in T.unroll(SPAN):
                    l_m[e] = T.max(l_m[e], l_sc[u * vec + e])
                l_s[e] = T.float32(0)
                for u in T.unroll(SPAN):
                    l_sc[u * vec + e] = T.__exp(l_sc[u * vec + e] - l_m[e])
                    l_s[e] = l_s[e] + l_sc[u * vec + e]
                l_out[e] = T.float32(0)
                if ROUND:
                    l_s[e] = T.fast_rcp(l_s[e])
                    for u in T.unroll(SPAN):
                        l_out[e] = l_out[e] + rounded(
                            rounded(l_sc[u * vec + e] * l_s[e]) * rounded(l_kv[u * vec + e])
                        )
                else:
                    for u in T.unroll(SPAN):
                        l_out[e] = l_out[e] + l_sc[u * vec + e] * l_kv[u * vec + e]
                    l_out[e] = l_out[e] * T.fast_rcp(l_s[e])
            for e in T.vectorized(vec):
                entries[row, col + e] = T.Cast(T.bfloat16, l_out[e])

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
            l_off,
            ring,
        ):
            # A warp's entry of the window closing at position p of the request whose tokens
            # start at base and position first, channels col .. col + VEC, from registers.
            if ring and p == first:
                load_lone_window(
                    l_sc,
                    l_kv,
                    VEC,
                    col,
                    base,
                    first,
                    slot,
                    l_off,
                    False,
                    kv,
                    score,
                    ape,
                    kv_state,
                    score_state,
                )
            else:
                for u in T.unroll(SPAN):
                    load_slot(
                        l_sc,
                        l_kv,
                        u,
                        p - SPAN + 1 + u,
                        head_of(u),
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
            pool_registers(entries, l_sc, l_kv, VEC, l_out, l_m, l_s, base + p - first, col)

        @T.macro
        def fold_batch(l_sc, l_kv, l_out, l_m, l_s, l_bm, l_bs, b):
            # Batch b of a warp's slots folded into its max, its sum of exponentials past it
            # and its weighted sum. A masked position, at -inf, adds nothing; a warp whose
            # positions are all masked keeps -inf and zero sums.
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

        @T.macro
        def fold_part(l_sc, l_kv, l_out, l_m, l_s, l_bm, l_bs, s_part, warp, tx, b):
            # Batch b of a warp's slots folded in: into the warp's partial, or with SPLIT into
            # a partial of its own, row warp * N_BATCH + b of s_part.
            fold_batch(l_sc, l_kv, l_out, l_m, l_s, l_bm, l_bs, 0 if SPLIT else b)
            if SPLIT or b == N_BATCH - 1:
                for e in T.unroll(VEC):
                    s_part[0, warp * N_BATCH + b if SPLIT else warp, (tx % 32) * VEC + e] = l_m[e]
                    s_part[1, warp * N_BATCH + b if SPLIT else warp, (tx % 32) * VEC + e] = l_s[e]
                    s_part[2, warp * N_BATCH + b if SPLIT else warp, (tx % 32) * VEC + e] = l_out[e]

        @T.macro
        def merge_partials(entries, l_out, l_s, l_bm, l_bs, s_part, row, tx, chunk):
            # Threads 0 .. LANE_CH - 1 merge the PARTS partials, a channel each, into entries
            # row row.
            T.sync_threads()
            if tx < LANE_CH and chunk * LANE_CH + tx < dim:
                l_bm[0] = -T.infinity(T.float32)
                for w in T.unroll(PARTS):
                    l_bm[0] = T.max(l_bm[0], s_part[0, w, tx])
                l_s[0] = T.float32(0)
                l_out[0] = T.float32(0)
                for w in T.unroll(PARTS):
                    l_bs[0] = T.__exp(s_part[0, w, tx] - l_bm[0])
                    l_s[0] = l_s[0] + s_part[1, w, tx] * l_bs[0]
                    l_out[0] = l_out[0] + s_part[2, w, tx] * l_bs[0]
                entries[row, chunk * LANE_CH + tx] = T.Cast(
                    T.bfloat16, l_out[0] * T.fast_rcp(l_s[0])
                )

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
            l_off,
            ring,
            warp,
            tx,
            chunk,
        ):
            # The block's entry of a window longer than a warp's registers hold: warp w
            # pools positions w * N_BATCH * BATCH on, BATCH at a time, and the warps' partials
            # merge.
            for b in T.unroll(N_BATCH):
                if LONG_EXACT and ring and p == first:
                    load_lone_batch(
                        l_sc,
                        l_kv,
                        b,
                        warp,
                        col,
                        base,
                        first,
                        slot,
                        l_off,
                        False,
                        kv,
                        score,
                        ape,
                        kv_state,
                        score_state,
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
                fold_part(l_sc, l_kv, l_out, l_m, l_s, l_bm, l_bs, s_part, warp, tx, b)
            merge_partials(entries, l_out, l_s, l_bm, l_bs, s_part, base + p - first, tx, chunk)
            # The partials are read before a next window of the block writes its own.
            T.sync_threads()

        @T.macro
        def load_token_row(kv, score, ape, l_wkv, l_wsc, l_wape, vec, base, q, col):
            # Token base, at position q: its kv, score and ape in all C halves, added by
            # add_ape once loads that must not wait on these have issued.
            for h in T.unroll(C):
                for e in T.vectorized(vec):
                    l_wkv[h * vec + e] = T.Cast(T.float32, kv[base, h * dim + col + e])
                for e in T.vectorized(vec):
                    l_wsc[h * vec + e] = T.Cast(T.float32, score[base, h * dim + col + e])
                for e in T.vectorized(vec):
                    l_wape[h * vec + e] = T.Cast(T.float32, ape[q % ratio, h * dim + col + e])

        @T.macro
        def add_ape(l_wsc, l_wape, vec):
            for i in T.unroll(C * vec):
                l_wsc[i] = l_wsc[i] + l_wape[i]

        @T.macro
        def store_token_row(kv_state, score_state, l_wkv, l_wsc, vec, slot, q, col):
            # A token's kv and score + ape, held in l_wkv and l_wsc, into its ring row.
            for h in T.unroll(C):
                for e in T.vectorized(vec):
                    kv_state[slot, q % SPAN, h * dim + col + e] = l_wkv[h * vec + e]
                for e in T.vectorized(vec):
                    score_state[slot, q % SPAN, h * dim + col + e] = l_wsc[h * vec + e]

        kv_param, score_param = (
            T.Tensor[(t_q, WIDTH), in_dtype]
            if stride is None
            else T.StridedTensor[(t_q, WIDTH), (stride, 1), in_dtype]
            for stride in (kv_stride, score_stride)
        )

        @T.prim_func
        def gated_pool_compress_main(
            kv: kv_param,
            score: score_param,
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
            # valid flags of the request's tokens no other block's window closes. They come
            # first, the longest chain of loads in the grid. The blocks after them pool the
            # other windows, and write the flags of their windows' positions: a warp each, or
            # the whole block a window longer than a warp's registers hold. A block's chunk is
            # its index modulo CHUNKS.
            with T.Kernel((n_req + n_units) * CHUNKS, threads=BLOCK_WARPS * 32) as bx:
                if not LONG:
                    T.annotate_min_blocks_per_sm(max(1, SM_WARPS // BLOCK_WARPS))
                if LONG and N_BATCH == 1:
                    T.annotate_min_blocks_per_sm(K.LONG_BLOCKS_PER_SM)
                elif LONG and SPLIT:
                    T.annotate_min_blocks_per_sm(K.SPLIT_BLOCKS_PER_SM)
                tx = T.get_thread_binding()
                warp = tx // 32
                lane = tx % 32
                chunk = bx % CHUNKS
                col = chunk * LANE_CH + lane * VEC
                # A request's only token's channels: a chunk's split over TOKEN_WARPS warps.
                token_col = chunk * LANE_CH + (warp * 32 + lane) * TOKEN_VEC
                l_sc = T.alloc_local([(BATCH if LONG else SPAN) * VEC], T.float32)
                l_kv = T.alloc_local([(BATCH if LONG else SPAN) * VEC], T.float32)
                # Ring rows on their way from the call's tokens to the states.
                l_wkv = T.alloc_local([WRITE_ROWS * C * VEC], T.float32)
                l_wsc = T.alloc_local([WRITE_ROWS * C * VEC], T.float32)
                # A request's only token's ape, added to its score once its window's loads issue.
                l_wape = T.alloc_local([C * VEC], T.float32)
                l_out = T.alloc_local([VEC], T.float32)
                l_m = T.alloc_local([VEC], T.float32)
                l_s = T.alloc_local([VEC], T.float32)
                # A batch's max, and the max its exponentials subtract.
                l_bm = T.alloc_local([VEC], T.float32)
                l_bs = T.alloc_local([VEC], T.float32)
                # The partials of a long window: max, exponential sum and weighted sum.
                s_part = T.alloc_shared([3, PARTS, LANE_CH], T.float32) if LONG else None
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
                # (first + 1) % SPAN, the ring row offset of a window closing at first.
                l_off = T.alloc_var(T.int32)

                if bx < n_req * CHUNKS:
                    l_req = bx // CHUNKS
                    l_base = cu_seqlens[l_req]
                    l_first = start_pos[l_req]
                    l_count = cu_seqlens[l_req + 1] - l_base
                    l_slot = state_slots[l_req]
                    l_off = (l_first + 1) % SPAN
                    if one_token and SPAN > 1 and l_count == 1 and (not LONG or LONG_EXACT):
                        # A request's only token: no window of the call reads its ring row,
                        # the position SPAN before it. The threads whose window slot the token
                        # is load it once, for the row and the slot; its score + ape, which
                        # waits on the loads, comes once the window's ring loads have issued.
                        if chunk == 0 and tx == 0:
                            valid[l_base] = (l_first + 1) % ratio == 0
                        if LONG:
                            if warp == TOKEN_WARP:
                                load_token_row(
                                    kv, score, ape, l_wkv, l_wsc, l_wape, VEC, l_base, l_first, col
                                )
                            if (l_first + 1) % ratio == 0:
                                for b in T.unroll(N_BATCH, explicit=True):
                                    load_lone_batch(
                                        l_sc,
                                        l_kv,
                                        b,
                                        warp,
                                        col,
                                        l_base,
                                        l_first,
                                        l_slot,
                                        l_off,
                                        True,
                                        kv,
                                        score,
                                        ape,
                                        kv_state,
                                        score_state,
                                    )
                                    if warp == TOKEN_WARP:
                                        if b == 0:
                                            add_ape(l_wsc, l_wape, VEC)
                                            store_token_row(
                                                kv_state,
                                                score_state,
                                                l_wkv,
                                                l_wsc,
                                                VEC,
                                                l_slot,
                                                l_first,
                                                col,
                                            )
                                        if b == N_BATCH - 1:
                                            for e in T.unroll(VEC):
                                                l_sc[(BATCH - 1) * VEC + e] = l_wsc[
                                                    (C - 1) * VEC + e
                                                ]
                                                l_kv[(BATCH - 1) * VEC + e] = l_wkv[
                                                    (C - 1) * VEC + e
                                                ]
                                    fold_part(
                                        l_sc, l_kv, l_out, l_m, l_s, l_bm, l_bs, s_part, warp, tx, b
                                    )
                                merge_partials(
                                    entries, l_out, l_s, l_bm, l_bs, s_part, l_base, tx, chunk
                                )
                            elif warp == TOKEN_WARP:
                                add_ape(l_wsc, l_wape, VEC)
                                store_token_row(
                                    kv_state, score_state, l_wkv, l_wsc, VEC, l_slot, l_first, col
                                )
                        elif warp < TOKEN_WARPS and token_col < dim:
                            if (l_first + 1) % ratio == 0:
                                load_lone_window(
                                    l_sc,
                                    l_kv,
                                    TOKEN_VEC,
                                    token_col,
                                    l_base,
                                    l_first,
                                    l_slot,
                                    l_off,
                                    True,
                                    kv,
                                    score,
                                    ape,
                                    kv_state,
                                    score_state,
                                )
                            load_token_row(
                                kv,
                                score,
                                ape,
                                l_wkv,
                                l_wsc,
                                l_wape,
                                TOKEN_VEC,
                                l_base,
                                l_first,
                                token_col,
                            )
                            add_ape(l_wsc, l_wape, TOKEN_VEC)
                            store_token_row(
                                kv_state,
                                score_state,
                                l_wkv,
                                l_wsc,
                                TOKEN_VEC,
                                l_slot,
                                l_first,
                                token_col,
                            )
                            if (l_first + 1) % ratio == 0:
                                for e in T.unroll(TOKEN_VEC):
                                    l_sc[(SPAN - 1) * TOKEN_VEC + e] = l_wsc[
                                        (C - 1) * TOKEN_VEC + e
                                    ]
                                    l_kv[(SPAN - 1) * TOKEN_VEC + e] = l_wkv[
                                        (C - 1) * TOKEN_VEC + e
                                    ]
                                pool_registers(
                                    entries,
                                    l_sc,
                                    l_kv,
                                    TOKEN_VEC,
                                    l_out,
                                    l_m,
                                    l_s,
                                    l_base,
                                    token_col,
                                )
                    else:
                        l_w0 = T.max(l_first, l_first + l_count - SPAN)
                        # The request's windows that read its ring close before position
                        # first + SPAN - 1: at most C of them. A long window's block pools
                        # them, and its threads first load the first WRITE_ROWS rows they
                        # write, so that those loads wait out the global latency with the
                        # windows'. A short window's warp pools each, and the block's other
                        # ROW_WARPS warps load all the rows: neither waits on the other's loads.
                        if LONG:
                            load_rows(
                                l_wkv, l_wsc, 0, kv, score, ape, l_base, l_first, l_count, l_w0,
                                warp, col,
                            )  # fmt: skip
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
                                        l_off,
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
                                    l_off,
                                    True,
                                )
                        else:
                            load_rows(
                                l_wkv, l_wsc, 0, kv, score, ape, l_base, l_first, l_count, l_w0,
                                warp, col,
                            )  # fmt: skip
                        # The valid flags of the request's tokens no window block owns, its CHUNKS
                        # blocks in turn: those before the first window block's own positions,
                        # and those after the last closing position. A window block owns the
                        # ratio positions its window closes.
                        l_p = (T.ceildiv(l_first + SPAN, ratio) - 1) * ratio
                        l_g = (l_first + l_count) // ratio * ratio
                        if l_p >= l_g:
                            l_p = l_first + l_count
                            l_g = l_first + l_count
                        for j in T.serial(T.ceildiv(l_p - l_first, CHUNKS * BLOCK_WARPS * 32)):
                            q = l_first + (j * CHUNKS + chunk) * (BLOCK_WARPS * 32) + tx
                            if q < l_p:
                                valid[l_base + q - l_first] = (q + 1) % ratio == 0
                        for j in T.serial(
                            T.ceildiv(l_first + l_count - l_g, CHUNKS * BLOCK_WARPS * 32)
                        ):
                            q = l_g + (j * CHUNKS + chunk) * (BLOCK_WARPS * 32) + tx
                            if q < l_first + l_count:
                                valid[l_base + q - l_first] = (q + 1) % ratio == 0
                        # A later token's row may be one a head window read.
                        if l_count > 1:
                            T.sync_threads()
                        for k in T.serial(N_WRITE):
                            if k > 0 and (LONG or warp >= C):
                                load_rows(
                                    l_wkv, l_wsc, k, kv, score, ape, l_base, l_first, l_count, l_w0,
                                    warp, col,
                                )  # fmt: skip
                            if LONG or warp >= C:
                                for r in T.unroll(WRITE_ROWS):
                                    q = row_of(l_w0, k, r, warp)
                                    if q < l_first + l_count and col < dim:
                                        for h in T.unroll(C):
                                            for e in T.vectorized(VEC):
                                                kv_state[l_slot, q % SPAN, h * dim + col + e] = (
                                                    l_wkv[(r * C + h) * VEC + e]
                                                )
                                            for e in T.vectorized(VEC):
                                                score_state[l_slot, q % SPAN, h * dim + col + e] = (
                                                    l_wsc[(r * C + h) * VEC + e]
                                                )
                else:
                    l_g = (
                        bx // CHUNKS - n_req
                        if LONG
                        else (bx // CHUNKS - n_req) * BLOCK_WARPS + warp
                    )
                    if l_g < n_windows:
                        # Let-bound: one if guards the statements below, not one each reading l_g.
                        g = l_g
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
                                                    <= g
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
                        l_p = (T.ceildiv(l_first + SPAN, ratio) + g - l_base // ratio) * ratio - 1
                        if l_p < l_first + l_count and chunk == 0:
                            # The valid flags of the window's own ratio positions, the last true.
                            if LONG:
                                for j in T.serial(T.ceildiv(ratio, BLOCK_WARPS * 32)):
                                    q = j * BLOCK_WARPS * 32 + tx
                                    if q < ratio:
                                        valid[l_base + l_p - l_first - ratio + 1 + q] = (
                                            q == ratio - 1
                                        )
                            elif lane < ratio:
                                valid[l_base + l_p - l_first - ratio + 1 + lane] = lane == ratio - 1
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
                                    l_off,
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
                                l_off,
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

    # A short window's blocks keep SM_WARPS warps an SM: fewer starve the loads a prefill
    # keeps in flight. A window of at most REG_SPAN positions pools in a warp from registers,
    # rounding its weights and products to kv's dtype as the reference does: its l_sc and
    # l_kv, SPAN * WARP_VEC float32 each, take at most a thread's share of an SM's registers.
    # A longer one pools across a block, BATCH positions a warp at a time folded in online, in
    # float32: for float32 kv the same sum, for bfloat16 kv within the entry's bfloat16
    # tolerance. The program builder and _program_facts agree on these.
    SM_WARPS = 16
    WARP_VEC = 4
    REG_SPAN = 65536 // (SM_WARPS * 32) // (2 * WARP_VEC)
    BATCH = 8
    # A long window's warps: the full program's, and the split program's, which keeps each
    # batch's partial apart (SPLIT).
    LONG_WARPS = 16
    SPLIT_WARPS = 8
    # The registers a thread of a long window's program takes, and the blocks an SM keeps at
    # that of its full program with a single batch live and of its split program: the kernel
    # asks for them, and _program_facts counts a call's waves by them.
    LONG_REGS = 64
    LONG_BLOCKS_PER_SM = 65536 // (LONG_WARPS * 32 * LONG_REGS)
    SPLIT_BLOCKS_PER_SM = 65536 // (SPLIT_WARPS * 32 * LONG_REGS)

    @classmethod
    def _lane_vec(cls, ratio: int, overlap: bool, dim: int) -> "tuple[bool, int]":
        """Whether a window pools across a block, and the channels a lane pools at once:
        WARP_VEC for a warp's window, two for a block's, which spreads a long window's loads
        over twice the blocks."""
        if (2 if overlap else 1) * ratio > cls.REG_SPAN:
            return True, 2 if dim % 2 == 0 else 1
        return False, cls.WARP_VEC if dim % cls.WARP_VEC == 0 else 1

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
        self.kernel = _gated_pool_compress_kernel(
            ratio, overlap, dim, self.dtype_str, False, self.LONG_WARPS, None, None
        )
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
        facts = self._program_facts(kv.shape[0], start_pos.shape[0])
        strides = (self._row_stride(kv), self._row_stride(score))
        return self.program(self._compile, *facts, *strides)(
            kv, score, ape, kv_state, score_state, cu_seqlens, start_pos, state_slots
        )

    @staticmethod
    def _row_stride(t: torch.Tensor) -> "int | None":
        """The row stride the program declares for *t*, ``None`` for contiguous rows."""
        return None if t.is_contiguous() else t.stride(0)

    def _program_facts(self, t_q: int, n: int) -> "tuple[bool, int]":
        """Whether the program pools a request's only token apart, and a long window's warps.

        A call of at most one token a request (T_q <= N) takes the one-token path. A long window
        takes 16 warps, or 8 where 16 would pool it in one batch and a one-token call's blocks fit
        one wave of the SMs.
        """
        one_token = t_q <= n
        long, vec = self._lane_vec(self.ratio, self.overlap, self.dim)
        span = (2 if self.overlap else 1) * self.ratio
        blocks = (n + t_q // self.ratio) * -(-self.dim // (32 * vec))
        sms = self.build_target.sm_count
        one_split_wave = self.LONG_BLOCKS_PER_SM * sms < blocks <= self.SPLIT_BLOCKS_PER_SM * sms
        eight = long and one_token and -(-span // self.BATCH) == self.LONG_WARPS and one_split_wave
        return one_token, self.SPLIT_WARPS if eight else self.LONG_WARPS

    def _compile(
        self,
        cfg: dict,
        one_token: bool,
        long_warps: int,
        kv_stride: "int | None",
        score_stride: "int | None",
    ):
        """The program for ``_program_facts`` over ``kv`` and ``score`` rows of the strides
        ``_row_stride`` declares; the kernel takes no configuration."""
        return _gated_pool_compress_kernel(
            self.ratio,
            self.overlap,
            self.dim,
            self.dtype_str,
            one_token,
            long_warps,
            kv_stride,
            score_stride,
        )()
