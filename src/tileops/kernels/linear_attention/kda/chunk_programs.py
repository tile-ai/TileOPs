"""The TileLang programs the chunked Kimi Delta Attention (KDA) forward runs.

``chunk_prepare`` is chunk-local: nothing in it carries state from one chunk to
the next, so one CTA takes one (chunk, value head) and the launch covers the
machine. It produces the WY vectors ``w`` and ``u``, the gated ``q`` and ``k``
the scan multiplies the state by, the intra-chunk attention matrix, and the
chunk's decay. ``chunk_scan`` is what is left: one CTA per (sequence, value
head) walking that sequence's chunks in order. ``chunk_state`` and ``chunk_output``
split that scan: the first walks the chunks carrying only the state, the second
forms every chunk's output in parallel from the states the first stored.

The WY inverse uses stable 16-row diagonal solves and block matrix products.
Channel gates use reference-centered tiles; head gates scale the dot product.
"""

import functools

import tilelang
import tilelang.language as T

from tileops.kernels.constants import LOG2E
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.linear_attention.gdn.prefill_common import step_size
from tileops.kernels.linear_attention.kda.inverse import inverse_program

__all__ = [
    "chunk_output_program",
    "chunk_prepare_program",
    "chunk_scan_program",
    "chunk_state_program",
]

# Center each channel-gate exponent within 16 rows: its log-gate difference
# spans at most 8 increments rather than a whole chunk. This limits, but does
# not universally bound, low-precision scaling; the fused program shares it.
REFERENCE_BLOCK = 16
# Added under the square root of each row norm before taking its reciprocal. The
# fused program normalizes the same rows, so it imports this value.
L2_EPS = 1e-6


@functools.lru_cache(maxsize=64)
def chunk_prepare_program(
    heads: int,
    value_heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    scale: float,
    l2norm: bool,
    total_tokens: int,
    num_seqs: int,
    threads: int = 256,
    lean: bool = False,
    *,
    gate_mode: str = "channel",
    beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
):
    """Build the chunk-local program for one static packed shape.

    *lean* writes the gated query and key, the intra-chunk attention and the decay
    as soon as the Gram matrices exist and holds the causal mask in registers. The
    inverse's two tiles then reuse the shared memory those writes freed.
    """
    BT = chunk_size
    tiling = GroupTiling(num_seqs, chunk_size)
    num_chunks = tiling.tile_upper_bound(total_tokens)
    BC = REFERENCE_BLOCK
    NC = BT // BC
    REF = BC // 2
    H, HV, K, V = heads, value_heads, dim_k, dim_v
    group = HV // H
    accum = "float32"
    gate_shape = [1, total_tokens, HV, K] if gate_mode == "channel" else [1, total_tokens, HV]
    gate_dtype = dtype if gate_mode == "channel" else accum

    @tilelang.jit(
        out_idx=[-6, -5, -4, -3, -2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        invert = inverse_program(chunk_size=BT, block_size=BC, threads=threads, dtype=dtype)

        def gated_query(qa_s, gc_s, i, j):
            reference = (i // BC) * BC + REF if gate_mode == "channel" else i
            return T.cast(T.cast(qa_s[i, j], accum) * T.exp2(gc_s[reference, j]), dtype)

        def gated_key(key, gc_s, rows, i, j):
            reference = (i // BC) * BC + REF if gate_mode == "channel" else i
            return T.cast(
                T.cast(key[i, j], accum) * T.exp2(gc_s[rows - 1, j] - gc_s[reference, j]),
                dtype,
            )

        @T.macro
        def fill_causal(mask):
            for i, j in T.Parallel(BT, BT):
                mask[i, j] = T.cast(T.if_then_else(i >= j, 1.0, 0.0), dtype)

        @T.macro
        def store_attention_and_decay(aqk, dec, aq, gc_s, rows, bos, ic, ihv):
            for i, j in T.Parallel(BT, BT):
                if i < rows:
                    aqk[0, bos + i, ihv, j] = T.cast(T.if_then_else(i >= j, aq[i, j], 0.0), dtype)
            last = T.max(rows - 1, 0)
            for j in T.Parallel(K):
                dec[ic, ihv, j] = T.exp2(gc_s[last, j])

        @T.macro
        def prepare(q, k, v, g, beta, cu_seqlens, w, u, qg, kg, aqk, dec, inverse):
            with T.Kernel(num_chunks, HV, threads=threads) as (ic, ihv):
                ih = ihv // group
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")

                qa_s = T.alloc_shared([BT, K], dtype)
                ka_s = T.alloc_shared([BT, K], dtype)
                if gate_mode == "channel":
                    kb_s = T.alloc_shared([BT, K], dtype)
                b_s = T.alloc_shared([BT, K], dtype)
                m_s = T.alloc_shared([BT, BT], dtype)
                p_s = T.alloc_shared([BT, BT], dtype)
                gc_s = T.alloc_shared([BT, K], accum)
                bt_s = T.alloc_shared([BT], accum)
                if lean:
                    mask_s = T.alloc_fragment([BT, BT], dtype)

                akk = T.alloc_fragment([BT, BT], accum)
                aq = T.alloc_fragment([BT, BT], accum)
                tkk = T.alloc_fragment([BT, BT], accum)
                tqk = T.alloc_fragment([BT, BT], accum)
                acc = T.alloc_fragment([BT, K], accum)
                rowsum = T.alloc_fragment([BT], accum)

                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                # The chunk axis runs to a bound. A CTA past the last real chunk
                # decodes the last one and then holds no row, so every load reads
                # zero and every store is predicated off: it falls out of the data
                # flow rather than branching around the body. Guarding the body with
                # `if ic < live` instead costs 24% to 47% here, because the syncs it
                # carries hoist out of the branch -- these CTAs pay every barrier
                # anyway, and the full ones lose their pipeline to the branch.
                live = tile_cum[num_seqs]
                tiling.decode(T.min(ic, live - 1), tile_cum, lo, hi, seq, first)
                bos = T.cast(cu_seqlens[seq[0]], "int32") + first[0]
                rows = T.if_then_else(
                    ic < live, T.min(BT, T.cast(cu_seqlens[seq[0] + 1], "int32") - bos), 0
                )

                for i, j in T.Parallel(BT, K):
                    qa_s[i, j] = T.if_then_else(i < rows, q[0, bos + i, ih, j], T.cast(0, dtype))
                    ka_s[i, j] = T.if_then_else(i < rows, k[0, bos + i, ih, j], T.cast(0, dtype))
                    if gate_mode == "channel":
                        b_s[i, j] = T.if_then_else(
                            i < rows, g[0, bos + i, ihv, j], T.cast(0, dtype)
                        )
                fill_causal(mask_s if lean else m_s)
                for i in T.Parallel(BT):
                    bt_s[i] = T.if_then_else(
                        i < rows,
                        T.cast(
                            step_size(beta[0, bos + i, ihv], beta_sigmoid, allow_neg_eigval), accum
                        ),
                        T.cast(0, accum),
                    )
                T.sync_threads()

                # The chunk-local inclusive prefix sum of the log gate, in log2
                # space: bf16 gates summed into an f32 accumulator.
                if gate_mode == "channel":
                    T.clear(acc)
                    T.gemm(mask_s if lean else m_s, b_s, acc)
                    for i, j in T.Parallel(BT, K):
                        gc_s[i, j] = acc[i, j] * LOG2E
                elif gate_mode == "head":
                    for i, j in T.Parallel(BT, K):
                        gc_s[i, j] = T.if_then_else(
                            rows > 0,
                            g[0, bos + T.min(i, T.max(rows - 1, 0)), ihv] * LOG2E,
                            0.0,
                        )
                else:
                    T.clear(gc_s)

                for i, j in T.Parallel(BT, K):
                    acc[i, j] = T.cast(qa_s[i, j], accum) * T.cast(qa_s[i, j], accum)
                T.reduce_sum(acc, rowsum, dim=1)
                for i, j in T.Parallel(BT, K):
                    qa_s[i, j] = T.cast(
                        T.cast(qa_s[i, j], accum)
                        * (T.rsqrt(rowsum[i] + L2_EPS) if l2norm else 1.0)
                        * scale,
                        dtype,
                    )
                for i, j in T.Parallel(BT, K):
                    acc[i, j] = T.cast(ka_s[i, j], accum) * T.cast(ka_s[i, j], accum)
                T.reduce_sum(acc, rowsum, dim=1)
                for i, j in T.Parallel(BT, K):
                    ka_s[i, j] = T.cast(
                        T.cast(ka_s[i, j], accum)
                        * (T.rsqrt(rowsum[i] + L2_EPS) if l2norm else 1.0),
                        dtype,
                    )
                T.sync_threads()

                if gate_mode == "channel":
                    # Three exponentials serve both Gram matrices and all four
                    # reference blocks: every later factor is a rescale of these.
                    for i, j in T.Parallel(BT, K):
                        kb_s[i, j] = T.cast(
                            T.cast(ka_s[i, j], accum)
                            * T.exp2(gc_s[(i // BC) * BC + REF, j] - gc_s[i, j]),
                            dtype,
                        )
                    T.sync_threads()
                    for i, j in T.Parallel(BT, K):
                        ka_s[i, j] = T.cast(
                            T.cast(ka_s[i, j], accum)
                            * T.exp2(gc_s[i, j] - gc_s[(i // BC) * BC + REF, j]),
                            dtype,
                        )
                        qa_s[i, j] = T.cast(
                            T.cast(qa_s[i, j], accum)
                            * T.exp2(gc_s[i, j] - gc_s[(i // BC) * BC + REF, j]),
                            dtype,
                        )
                    T.sync_threads()

                    T.clear(akk)
                    T.clear(aq)
                    for sub in T.serial(NC):
                        for i, j in T.Parallel(BT, K):
                            b_s[i, j] = T.cast(
                                T.if_then_else(
                                    i // BC <= sub,
                                    T.cast(kb_s[i, j], accum)
                                    * T.exp2(
                                        gc_s[sub * BC + REF, j] - gc_s[(i // BC) * BC + REF, j]
                                    ),
                                    0.0,
                                ),
                                dtype,
                            )
                        T.sync_threads()
                        T.clear(tkk)
                        T.clear(tqk)
                        T.gemm(ka_s, b_s, tkk, transpose_B=True)
                        T.gemm(qa_s, b_s, tqk, transpose_B=True)
                        for i, j in T.Parallel(BT, BT):
                            akk[i, j] = T.if_then_else(i // BC == sub, tkk[i, j], akk[i, j])
                            aq[i, j] = T.if_then_else(i // BC == sub, tqk[i, j], aq[i, j])
                        T.sync_threads()

                else:
                    # A head gate factors outside the dot product. Apply its
                    # causal difference after GEMM, without FP16 inverse scaling.
                    T.clear(aq)
                    T.gemm(qa_s, ka_s, aq, transpose_B=True)
                    # Mask before exponentiation so future keys cannot produce
                    # growing gate ratios; the shared store also masks the
                    # channel-gate path, whose GEMMs leave future entries live.
                    for i, j in T.Parallel(BT, BT):
                        aq[i, j] = T.if_then_else(
                            i >= j, aq[i, j] * T.exp2(gc_s[i, 0] - gc_s[j, 0]), 0.0
                        )
                    T.sync_threads()

                if lean:
                    for i, j in T.Parallel(BT, K):
                        if i < rows:
                            qg[0, bos + i, ihv, j] = gated_query(qa_s, gc_s, i, j)
                            kg[0, bos + i, ihv, j] = gated_key(
                                kb_s if gate_mode == "channel" else ka_s, gc_s, rows, i, j
                            )
                    store_attention_and_decay(aqk, dec, aq, gc_s, rows, bos, ic, ihv)

                if gate_mode == "channel":
                    for i, j in T.Parallel(BT, BT):
                        m_s[i, j] = T.cast(T.if_then_else(i > j, -akk[i, j] * bt_s[i], 0.0), dtype)
                    T.sync_threads()
                    invert(m_s, p_s, tkk)
                else:
                    for i, j in T.Parallel(BT, BT):
                        p_s[i, j] = T.if_then_else(
                            i < rows, inverse[0, bos + i, ihv, j], T.cast(0, dtype)
                        )
                    T.sync_threads()

                for i, j in T.Parallel(BT, K):
                    b_s[i, j] = T.cast(
                        T.cast(ka_s[i, j], accum)
                        * T.exp2(gc_s[(i // BC) * BC + REF if gate_mode == "channel" else i, j])
                        * bt_s[i],
                        dtype,
                    )
                T.sync_threads()
                T.clear(acc)
                T.gemm(p_s, b_s, acc)
                for i, j in T.Parallel(BT, K):
                    if i < rows:
                        w[0, bos + i, ihv, j] = T.cast(acc[i, j], dtype)
                        if not lean:
                            qg[0, bos + i, ihv, j] = gated_query(qa_s, gc_s, i, j)
                            kg[0, bos + i, ihv, j] = gated_key(
                                kb_s if gate_mode == "channel" else ka_s, gc_s, rows, i, j
                            )
                if not lean:
                    store_attention_and_decay(aqk, dec, aq, gc_s, rows, bos, ic, ihv)
                T.sync_threads()
                for i, j in T.Parallel(BT, V):
                    b_s[i, j] = T.cast(
                        T.if_then_else(i < rows, T.cast(v[0, bos + i, ihv, j], accum), 0.0)
                        * bt_s[i],
                        dtype,
                    )
                T.sync_threads()
                T.clear(acc)
                T.gemm(p_s, b_s, acc)
                for i, j in T.Parallel(BT, V):
                    if i < rows:
                        u[0, bos + i, ihv, j] = T.cast(acc[i, j], dtype)

        if gate_mode == "channel":

            @T.prim_func
            def main(
                q: T.Tensor([1, total_tokens, H, K], dtype),
                k: T.Tensor([1, total_tokens, H, K], dtype),
                v: T.Tensor([1, total_tokens, HV, V], dtype),
                g: T.Tensor(gate_shape, gate_dtype),
                beta: T.Tensor([1, total_tokens, HV], dtype),
                cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
                w: T.Tensor([1, total_tokens, HV, K], dtype),
                u: T.Tensor([1, total_tokens, HV, V], dtype),
                qg: T.Tensor([1, total_tokens, HV, K], dtype),
                kg: T.Tensor([1, total_tokens, HV, K], dtype),
                aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
                dec: T.Tensor([num_chunks, HV, K], accum),
            ):
                prepare(q, k, v, g, beta, cu_seqlens, w, u, qg, kg, aqk, dec, None)
        else:

            @T.prim_func
            def main(
                q: T.Tensor([1, total_tokens, H, K], dtype),
                k: T.Tensor([1, total_tokens, H, K], dtype),
                v: T.Tensor([1, total_tokens, HV, V], dtype),
                g: T.Tensor(gate_shape, gate_dtype),
                beta: T.Tensor([1, total_tokens, HV], dtype),
                cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
                inverse: T.Tensor([1, total_tokens, HV, BT], dtype),
                w: T.Tensor([1, total_tokens, HV, K], dtype),
                u: T.Tensor([1, total_tokens, HV, V], dtype),
                qg: T.Tensor([1, total_tokens, HV, K], dtype),
                kg: T.Tensor([1, total_tokens, HV, K], dtype),
                aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
                dec: T.Tensor([num_chunks, HV, K], accum),
            ):
                prepare(q, k, v, g, beta, cu_seqlens, w, u, qg, kg, aqk, dec, inverse)

        return main

    return build()


@functools.lru_cache(maxsize=64)
def chunk_scan_program(
    value_heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    total_tokens: int,
    num_seqs: int,
    threads: int = 256,
    value_tile: int | None = None,
    *,
    state_v_first: bool = False,
):
    """Build the sequential chunk scan for one static packed shape.

    The state's value columns update independently, so a CTA may take a *value_tile*
    wide slice of them (all of them by default).
    """
    BT = chunk_size
    tiling = GroupTiling(num_seqs, chunk_size)
    num_chunks = tiling.tile_upper_bound(total_tokens)
    HV, K, V = value_heads, dim_k, dim_v
    BV = V if value_tile is None else value_tile
    value_tiles = V // BV
    accum = "float32"
    state_shape = [num_seqs, HV, V, K] if state_v_first else [num_seqs, HV, K, V]

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.prim_func
        def main(
            w: T.Tensor([1, total_tokens, HV, K], dtype),
            u: T.Tensor([1, total_tokens, HV, V], dtype),
            qg: T.Tensor([1, total_tokens, HV, K], dtype),
            kg: T.Tensor([1, total_tokens, HV, K], dtype),
            aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
            dec: T.Tensor([num_chunks, HV, K], accum),
            h0: T.Tensor(state_shape, accum),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            o: T.Tensor([1, total_tokens, HV, V], dtype),
            ht: T.Tensor(state_shape, accum),
        ):
            with T.Kernel(num_seqs, HV, value_tiles, threads=threads) as (iseq, ihv, iv):
                v0 = iv * BV if value_tiles > 1 else 0
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")

                # The scan walks one sequence, so it needs where that sequence's
                # chunks start on the axis the chunk-local half wrote dec along.
                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                bos = T.cast(cu_seqlens[iseq], "int32")
                length = T.cast(cu_seqlens[iseq + 1], "int32") - bos
                chunk0 = tile_cum[iseq]

                state_s = T.alloc_shared([K, BV], dtype)
                w_s = T.alloc_shared([BT, K], dtype)
                qg_s = T.alloc_shared([BT, K], dtype)
                kg_s = T.alloc_shared([BT, K], dtype)
                a_s = T.alloc_shared([BT, BT], dtype)
                vn_s = T.alloc_shared([BT, BV], dtype)
                state_f = T.alloc_fragment([K, BV], accum)
                vn_f = T.alloc_fragment([BT, BV], accum)
                ws_f = T.alloc_fragment([BT, BV], accum)
                o_f = T.alloc_fragment([BT, BV], accum)
                upd = T.alloc_fragment([K, BV], accum)

                for i, j in T.Parallel(K, BV):
                    if state_v_first:
                        state_f[i, j] = h0[iseq, ihv, v0 + j, i]
                    else:
                        state_f[i, j] = h0[iseq, ihv, i, v0 + j]

                for c in T.serial(T.ceildiv(length, BT)):
                    base = bos + c * BT
                    rows = T.min(BT, length - c * BT)
                    for i, j in T.Parallel(K, BV):
                        state_s[i, j] = T.cast(state_f[i, j], dtype)
                    for i, j in T.Parallel(BT, K):
                        w_s[i, j] = T.if_then_else(
                            i < rows, w[0, base + i, ihv, j], T.cast(0, dtype)
                        )
                        qg_s[i, j] = T.if_then_else(
                            i < rows, qg[0, base + i, ihv, j], T.cast(0, dtype)
                        )
                        kg_s[i, j] = T.if_then_else(
                            i < rows, kg[0, base + i, ihv, j], T.cast(0, dtype)
                        )
                    for i, j in T.Parallel(BT, BT):
                        a_s[i, j] = T.if_then_else(
                            i < rows, aqk[0, base + i, ihv, j], T.cast(0, dtype)
                        )
                    for i, j in T.Parallel(BT, BV):
                        vn_f[i, j] = T.if_then_else(
                            i < rows, T.cast(u[0, base + i, ihv, v0 + j], accum), T.cast(0, accum)
                        )
                    T.sync_threads()

                    T.clear(ws_f)
                    T.gemm(w_s, state_s, ws_f)
                    for i, j in T.Parallel(BT, BV):
                        vn_s[i, j] = T.cast(vn_f[i, j] - ws_f[i, j], dtype)
                    T.clear(o_f)
                    T.sync_threads()
                    T.gemm(qg_s, state_s, o_f)
                    T.gemm(a_s, vn_s, o_f)
                    T.clear(upd)
                    T.gemm(kg_s, vn_s, upd, transpose_A=True)
                    for i, j in T.Parallel(K, BV):
                        state_f[i, j] = state_f[i, j] * dec[chunk0 + c, ihv, i] + upd[i, j]
                    for i, j in T.Parallel(BT, BV):
                        if i < rows:
                            o[0, base + i, ihv, v0 + j] = T.cast(o_f[i, j], dtype)
                    T.sync_threads()

                for i, j in T.Parallel(K, BV):
                    if state_v_first:
                        ht[iseq, ihv, v0 + j, i] = state_f[i, j]
                    else:
                        ht[iseq, ihv, i, v0 + j] = state_f[i, j]

        return main

    return build()


@functools.lru_cache(maxsize=64)
def chunk_state_program(
    value_heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    total_tokens: int,
    num_seqs: int,
    value_tile: int,
    threads: int = 128,
):
    """Build the sequential scan that carries only the state, for one static packed shape.

    Per chunk a CTA forms the new values ``vn = u - w S``, stores them and the state the
    chunk starts from, and updates ``S <- dec * S + kg^T vn``; the output reads both in
    ``chunk_output_program``, one CTA a chunk. A CTA takes a *value_tile* wide slice of
    the state's value columns.
    """
    BT = chunk_size
    num_chunks = GroupTiling(num_seqs, chunk_size).tile_upper_bound(total_tokens)
    HV, K, V = value_heads, dim_k, dim_v
    BV = value_tile
    value_tiles = V // BV
    accum = "float32"

    @tilelang.jit(
        out_idx=[-3, -2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        # Built here so the jit builder closes over scalars only.
        tiling = GroupTiling(num_seqs, chunk_size)

        @T.macro
        def walk(w, u, kg, dec, h, vn, tiles, frags, bos, chunk0, ihv, v0, c, rows, partial: bool):
            """One chunk: store its starting state and new values, then update the state.

            A partial chunk's rows past the sequence's end belong to the next one: its
            keys and new values are zeroed so those rows reach no output or state.
            """
            w_s, kg_s, u_s, state_s, vn_s = tiles
            state_f, ws_f, upd = frags
            base = bos + c * BT
            if partial:
                for i, j in T.Parallel(BT, K):
                    w_s[i, j] = T.if_then_else(i < rows, w[0, base + i, ihv, j], T.cast(0, dtype))
                    kg_s[i, j] = T.if_then_else(i < rows, kg[0, base + i, ihv, j], T.cast(0, dtype))
            else:
                T.copy(w[0, base : base + BT, ihv, :], w_s)
                T.copy(kg[0, base : base + BT, ihv, :], kg_s)
            T.copy(u[0, base : base + BT, ihv, v0 : v0 + BV], u_s)
            for i, j in T.Parallel(K, BV):
                state_s[i, j] = T.cast(state_f[i, j], dtype)
            for i, j in T.Parallel(K, BV):
                h[chunk0 + c, ihv, i, v0 + j] = state_s[i, j]
            T.clear(ws_f)
            T.gemm(w_s, state_s, ws_f)
            for i, j in T.Parallel(BT, BV):
                vn_s[i, j] = T.if_then_else(
                    i < rows,
                    T.cast(T.cast(u_s[i, j], accum) - ws_f[i, j], dtype),
                    T.cast(0, dtype),
                )
            for i, j in T.Parallel(BT, BV):
                if i < rows:
                    vn[0, base + i, ihv, v0 + j] = vn_s[i, j]
            T.clear(upd)
            T.gemm(kg_s, vn_s, upd, transpose_A=True)
            for i, j in T.Parallel(K, BV):
                state_f[i, j] = state_f[i, j] * dec[chunk0 + c, ihv, i] + upd[i, j]

        @T.prim_func
        def main(
            w: T.Tensor([1, total_tokens, HV, K], dtype),
            u: T.Tensor([1, total_tokens, HV, V], dtype),
            kg: T.Tensor([1, total_tokens, HV, K], dtype),
            dec: T.Tensor([num_chunks, HV, K], accum),
            h0: T.Tensor([num_seqs, HV, K, V], accum),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            h: T.Tensor([num_chunks, HV, K, V], dtype),
            vn: T.Tensor([1, total_tokens, HV, V], dtype),
            ht: T.Tensor([num_seqs, HV, K, V], accum),
        ):
            with T.Kernel(num_seqs, HV, value_tiles, threads=threads) as (iseq, ihv, iv):
                v0 = iv * BV
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")

                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                bos = T.cast(cu_seqlens[iseq], "int32")
                length = T.cast(cu_seqlens[iseq + 1], "int32") - bos
                chunk0 = tile_cum[iseq]

                state_s = T.alloc_shared([K, BV], dtype)
                w_s = T.alloc_shared([BT, K], dtype)
                kg_s = T.alloc_shared([BT, K], dtype)
                vn_s = T.alloc_shared([BT, BV], dtype)
                state_f = T.alloc_fragment([K, BV], accum)
                u_s = T.alloc_shared([BT, BV], dtype)
                ws_f = T.alloc_fragment([BT, BV], accum)
                upd = T.alloc_fragment([K, BV], accum)

                for i, j in T.Parallel(K, BV):
                    state_f[i, j] = h0[iseq, ihv, i, v0 + j]

                tiles = (w_s, kg_s, u_s, state_s, vn_s)
                frags = (state_f, ws_f, upd)
                whole = length // BT
                # A one-stage pipeline issues the chunk's loads asynchronously: 1357 us for
                # 65536 tokens of four heads against 1915 as a serial loop (H200).
                for c in T.Pipelined(whole, num_stages=1):
                    walk(w, u, kg, dec, h, vn, tiles, frags, bos, chunk0, ihv, v0, c, BT, False)
                if length % BT != 0:
                    walk(
                        w,
                        u,
                        kg,
                        dec,
                        h,
                        vn,
                        tiles,
                        frags,
                        bos,
                        chunk0,
                        ihv,
                        v0,
                        whole,
                        length % BT,
                        True,
                    )

                for i, j in T.Parallel(K, BV):
                    ht[iseq, ihv, i, v0 + j] = state_f[i, j]

        return main

    return build()


@functools.lru_cache(maxsize=64)
def chunk_output_program(
    value_heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: str,
    total_tokens: int,
    num_seqs: int,
    threads: int = 128,
):
    """Build the chunk-parallel output ``o = qg h + A vn`` for one static packed shape.

    One CTA takes one (chunk, value head): the gated query against the state the chunk
    starts from, plus the intra-chunk attention against the chunk's new values, both
    written by ``chunk_state_program``.
    """
    BT = chunk_size
    num_chunks = GroupTiling(num_seqs, chunk_size).tile_upper_bound(total_tokens)
    HV, K, V = value_heads, dim_k, dim_v
    accum = "float32"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        # Built here so the jit builder closes over scalars only.
        tiling = GroupTiling(num_seqs, chunk_size)

        @T.prim_func
        def main(
            qg: T.Tensor([1, total_tokens, HV, K], dtype),
            aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
            h: T.Tensor([num_chunks, HV, K, V], dtype),
            vn: T.Tensor([1, total_tokens, HV, V], dtype),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            o: T.Tensor([1, total_tokens, HV, V], dtype),
        ):
            with T.Kernel(num_chunks, HV, threads=threads) as (ic, ihv):
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                qg_s = T.alloc_shared([BT, K], dtype)
                h_s = T.alloc_shared([K, V], dtype)
                a_s = T.alloc_shared([BT, BT], dtype)
                vn_s = T.alloc_shared([BT, V], dtype)
                o_f = T.alloc_fragment([BT, V], accum)

                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                # A CTA past the last real chunk decodes the last one and holds no row.
                live = tile_cum[num_seqs]
                tiling.decode(T.min(ic, live - 1), tile_cum, lo, hi, seq, first)
                bos = T.cast(cu_seqlens[seq[0]], "int32") + first[0]
                rows = T.if_then_else(
                    ic < live, T.min(BT, T.cast(cu_seqlens[seq[0] + 1], "int32") - bos), 0
                )
                chunk = T.min(ic, live - 1)

                for i, j in T.Parallel(BT, K):
                    qg_s[i, j] = T.if_then_else(i < rows, qg[0, bos + i, ihv, j], T.cast(0, dtype))
                for i, j in T.Parallel(BT, BT):
                    a_s[i, j] = T.if_then_else(i < rows, aqk[0, bos + i, ihv, j], T.cast(0, dtype))
                for i, j in T.Parallel(BT, V):
                    vn_s[i, j] = T.if_then_else(i < rows, vn[0, bos + i, ihv, j], T.cast(0, dtype))
                T.copy(h[chunk, ihv, :, :], h_s)
                T.clear(o_f)
                T.gemm(qg_s, h_s, o_f)
                T.gemm(a_s, vn_s, o_f)
                for i, j in T.Parallel(BT, V):
                    if i < rows:
                        o[0, bos + i, ihv, j] = T.cast(o_f[i, j], dtype)

        return main

    return build()
