"""The two TileLang programs the chunked Kimi Delta Attention forward runs.

``chunk_prepare`` is chunk-local: nothing in it carries state from one chunk to
the next, so one CTA takes one (chunk, value head) and the launch covers the
machine. It produces the WY vectors ``w`` and ``u``, the gated ``q`` and ``k``
the scan multiplies the state by, the intra-chunk attention matrix, and the
chunk's decay. ``chunk_scan`` is what is left: one CTA per (sequence, value
head) walking that sequence's chunks in order.

Inside ``chunk_prepare`` the unit-triangular WY inverse is six nilpotent
Neumann factors, so every step of it is a tensor-core matmul rather than a
row-sequential forward substitution. Each of the four reference blocks rescales
one pre-exponentiated tile instead of taking its own exponential, which is what
keeps the shared-memory footprint at two CTAs per SM.
"""

import functools

import tilelang
import tilelang.language as T

from tileops.kernels.constants import LOG2E
from tileops.kernels.grouped_tiling import GroupTiling

__all__ = ["chunk_prepare_program", "chunk_scan_program"]

# One gate reference per this many rows keeps every exponent this kernel takes
# inside float32, whatever the gate magnitude the caller supplies.
REFERENCE_BLOCK = 16
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
):
    """Build the chunk-local program for one static packed shape."""
    BT = chunk_size
    tiling = GroupTiling(num_seqs, chunk_size)
    num_chunks = tiling.tile_upper_bound(total_tokens)
    BC = REFERENCE_BLOCK
    NC = BT // BC
    REF = BC // 2
    H, HV, K, V = heads, value_heads, dim_k, dim_v
    group = HV // H
    accum = "float32"

    @tilelang.jit(
        out_idx=[-6, -5, -4, -3, -2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.prim_func
        def main(
            q: T.Tensor([1, total_tokens, H, K], dtype),
            k: T.Tensor([1, total_tokens, H, K], dtype),
            v: T.Tensor([1, total_tokens, HV, V], dtype),
            g: T.Tensor([1, total_tokens, HV, K], dtype),
            beta: T.Tensor([1, total_tokens, HV], dtype),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            w: T.Tensor([1, total_tokens, HV, K], dtype),
            u: T.Tensor([1, total_tokens, HV, V], dtype),
            qg: T.Tensor([1, total_tokens, HV, K], dtype),
            kg: T.Tensor([1, total_tokens, HV, K], dtype),
            aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
            dec: T.Tensor([num_chunks, HV, K], accum),
        ):
            with T.Kernel(num_chunks, HV, threads=threads) as (ic, ihv):
                ih = ihv // group
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")

                qa_s = T.alloc_shared([BT, K], dtype)
                ka_s = T.alloc_shared([BT, K], dtype)
                kb_s = T.alloc_shared([BT, K], dtype)
                b_s = T.alloc_shared([BT, K], dtype)
                m_s = T.alloc_shared([BT, BT], dtype)
                p_s = T.alloc_shared([BT, BT], dtype)
                gc_s = T.alloc_shared([BT, K], accum)
                bt_s = T.alloc_shared([BT], accum)

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
                    b_s[i, j] = T.if_then_else(i < rows, g[0, bos + i, ihv, j], T.cast(0, dtype))
                for i, j in T.Parallel(BT, BT):
                    m_s[i, j] = T.cast(T.if_then_else(i >= j, 1.0, 0.0), dtype)
                for i in T.Parallel(BT):
                    bt_s[i] = T.if_then_else(
                        i < rows, T.cast(beta[0, bos + i, ihv], accum), T.cast(0, accum)
                    )
                T.sync_threads()

                # The chunk-local inclusive prefix sum of the log gate, in log2
                # space: bf16 gates summed into an f32 accumulator.
                T.clear(acc)
                T.gemm(m_s, b_s, acc)
                for i, j in T.Parallel(BT, K):
                    gc_s[i, j] = acc[i, j] * LOG2E

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
                                * T.exp2(gc_s[sub * BC + REF, j] - gc_s[(i // BC) * BC + REF, j]),
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

                # (I - N)^-1 = (I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^16)(I+N^32) for the
                # nilpotent strictly-lower N: six tensor-core factors, no row
                # the chunk solves in sequence.
                for i, j in T.Parallel(BT, BT):
                    m_s[i, j] = T.cast(T.if_then_else(i > j, -akk[i, j] * bt_s[i], 0.0), dtype)
                    p_s[i, j] = T.cast(
                        T.if_then_else(i > j, -akk[i, j] * bt_s[i], 0.0)
                        + T.if_then_else(i == j, 1.0, 0.0),
                        dtype,
                    )
                T.sync_threads()
                for _level in T.serial(5):
                    T.clear(tkk)
                    T.gemm(m_s, m_s, tkk)
                    T.sync_threads()
                    for i, j in T.Parallel(BT, BT):
                        m_s[i, j] = T.cast(tkk[i, j] + T.if_then_else(i == j, 1.0, 0.0), dtype)
                    T.sync_threads()
                    T.clear(tqk)
                    T.gemm(p_s, m_s, tqk)
                    T.sync_threads()
                    for i, j in T.Parallel(BT, BT):
                        m_s[i, j] = T.cast(T.if_then_else(i > j, tkk[i, j], 0.0), dtype)
                        p_s[i, j] = T.cast(tqk[i, j], dtype)
                    T.sync_threads()

                for i, j in T.Parallel(BT, K):
                    b_s[i, j] = T.cast(
                        T.cast(ka_s[i, j], accum) * T.exp2(gc_s[(i // BC) * BC + REF, j]) * bt_s[i],
                        dtype,
                    )
                T.sync_threads()
                T.clear(acc)
                T.gemm(p_s, b_s, acc)
                for i, j in T.Parallel(BT, K):
                    if i < rows:
                        w[0, bos + i, ihv, j] = T.cast(acc[i, j], dtype)
                        qg[0, bos + i, ihv, j] = T.cast(
                            T.cast(qa_s[i, j], accum) * T.exp2(gc_s[(i // BC) * BC + REF, j]),
                            dtype,
                        )
                        kg[0, bos + i, ihv, j] = T.cast(
                            T.cast(kb_s[i, j], accum)
                            * T.exp2(gc_s[rows - 1, j] - gc_s[(i // BC) * BC + REF, j]),
                            dtype,
                        )
                for i, j in T.Parallel(BT, BT):
                    if i < rows:
                        aqk[0, bos + i, ihv, j] = T.cast(
                            T.if_then_else(i >= j, aq[i, j], 0.0), dtype
                        )
                last = T.max(rows - 1, 0)
                for j in T.Parallel(K):
                    dec[ic, ihv, j] = T.exp2(gc_s[last, j])
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
):
    """Build the sequential chunk scan for one static packed shape."""
    BT = chunk_size
    tiling = GroupTiling(num_seqs, chunk_size)
    num_chunks = tiling.tile_upper_bound(total_tokens)
    HV, K, V = value_heads, dim_k, dim_v
    accum = "float32"

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
            h0: T.Tensor([num_seqs, HV, K, V], accum),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            o: T.Tensor([1, total_tokens, HV, V], dtype),
            ht: T.Tensor([num_seqs, HV, K, V], accum),
        ):
            with T.Kernel(num_seqs, HV, threads=threads) as (iseq, ihv):
                tile_cum = T.alloc_shared([num_seqs + 1], "int32")

                # The scan walks one sequence, so it needs where that sequence's
                # chunks start on the axis the chunk-local half wrote dec along.
                tiling.cumsum_offsets(cu_seqlens, tile_cum)
                bos = T.cast(cu_seqlens[iseq], "int32")
                length = T.cast(cu_seqlens[iseq + 1], "int32") - bos
                chunk0 = tile_cum[iseq]

                state_s = T.alloc_shared([K, V], dtype)
                w_s = T.alloc_shared([BT, K], dtype)
                qg_s = T.alloc_shared([BT, K], dtype)
                kg_s = T.alloc_shared([BT, K], dtype)
                a_s = T.alloc_shared([BT, BT], dtype)
                vn_s = T.alloc_shared([BT, V], dtype)
                state_f = T.alloc_fragment([K, V], accum)
                vn_f = T.alloc_fragment([BT, V], accum)
                ws_f = T.alloc_fragment([BT, V], accum)
                o_f = T.alloc_fragment([BT, V], accum)
                upd = T.alloc_fragment([K, V], accum)

                for i, j in T.Parallel(K, V):
                    state_f[i, j] = h0[iseq, ihv, i, j]

                for c in T.serial(T.ceildiv(length, BT)):
                    base = bos + c * BT
                    rows = T.min(BT, length - c * BT)
                    for i, j in T.Parallel(K, V):
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
                    for i, j in T.Parallel(BT, V):
                        vn_f[i, j] = T.if_then_else(
                            i < rows, T.cast(u[0, base + i, ihv, j], accum), T.cast(0, accum)
                        )
                    T.sync_threads()

                    T.clear(ws_f)
                    T.gemm(w_s, state_s, ws_f)
                    for i, j in T.Parallel(BT, V):
                        vn_s[i, j] = T.cast(vn_f[i, j] - ws_f[i, j], dtype)
                    T.clear(o_f)
                    T.sync_threads()
                    T.gemm(qg_s, state_s, o_f)
                    T.gemm(a_s, vn_s, o_f)
                    T.clear(upd)
                    T.gemm(kg_s, vn_s, upd, transpose_A=True)
                    for i, j in T.Parallel(K, V):
                        state_f[i, j] = state_f[i, j] * dec[chunk0 + c, ihv, i] + upd[i, j]
                    for i, j in T.Parallel(BT, V):
                        if i < rows:
                            o[0, base + i, ihv, j] = T.cast(o_f[i, j], dtype)
                    T.sync_threads()

                for i, j in T.Parallel(K, V):
                    ht[iseq, ihv, i, j] = state_f[i, j]

        return main

    return build()
