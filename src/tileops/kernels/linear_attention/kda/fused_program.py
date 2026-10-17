"""The fused Kimi Delta Attention (KDA) chunk program: one CTA walks one sequence.

Everything a chunk decides on its own and the recurrence that joins the chunks
live in the same CTA, so the WY vectors, the gated query and key and the
intra-chunk attention never leave the chip. What it costs is parallelism: the
launch is one block per (sequence, value head), so it is the right shape only
when there are enough of those to cover the machine.

The WY inverse uses stable 16-row diagonal solves and block matrix products.
Gate references are computed once per 16 rows and rescaled per block.
"""

import functools

import tilelang
import tilelang.language as T

from tileops.kernels.constants import LOG2E
from tileops.kernels.linear_attention.kda.chunk_programs import L2_EPS, REFERENCE_BLOCK
from tileops.kernels.linear_attention.kda.inverse import inverse_program

__all__ = ["fused_chunk_program"]


@functools.lru_cache(maxsize=64)
def fused_chunk_program(
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
    compact: bool = False,
):
    """Build the fused chunk program for one static packed shape."""
    if compact and (dim_k != 128 or dim_v != 128 or chunk_size != 64):
        raise ValueError("compact storage requires a 64-token chunk and 128-wide state")
    BT = chunk_size
    BC = REFERENCE_BLOCK
    NC = BT // BC
    REF = BC // 2
    H, HV, K, V = heads, value_heads, dim_k, dim_v
    group = HV // H
    accum = "float32"

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        invert = inverse_program(
            chunk_size=BT,
            block_size=BC,
            threads=threads,
            dtype=dtype,
            output_offset=BT if compact else 0,
        )

        def reference_gate(gc, metadata, block, channel):
            return metadata[block * K + channel] if compact else gc[block * BC + REF, channel]

        def final_gate(gc, metadata, rows, channel):
            return metadata[NC * K + channel] if compact else gc[rows - 1, channel]

        def step(bt, i):
            return bt[(NC + 1) * K + i] if compact else bt[i]

        def matrix(buffer, second=False):
            if compact:
                return buffer[:, BT:] if second else buffer[:, :BT]
            return buffer

        @T.macro
        def load_state(state_s, state_f):
            for i, j in T.Parallel(K, V):
                state_s[i, j] = T.cast(state_f[i, j], dtype)

        @T.macro
        def prepare_key(b_s, ka_s, gc_s, metadata, bt_s):
            for i, j in T.Parallel(BT, K):
                b_s[i, j] = T.cast(
                    T.cast(ka_s[i, j], accum)
                    * T.exp2(reference_gate(gc_s, metadata, i // BC, j))
                    * step(bt_s, i),
                    dtype,
                )
            T.sync_threads()

        @T.macro
        def load_value(vb_s, v, bt_s, rows, bos, ihv):
            for i, j in T.Parallel(BT, V):
                vb_s[i, j] = T.cast(
                    T.if_then_else(i < rows, T.cast(v[0, bos + i, ihv, j], accum), 0.0)
                    * step(bt_s, i),
                    dtype,
                )
            T.sync_threads()

        @T.macro
        def prepare_query(b_s, qa_s, gc_s, metadata):
            for i, j in T.Parallel(BT, K):
                b_s[i, j] = T.cast(
                    T.cast(qa_s[i, j], accum) * T.exp2(reference_gate(gc_s, metadata, i // BC, j)),
                    dtype,
                )

        @T.prim_func
        def main(
            q: T.Tensor([1, total_tokens, H, K], dtype),
            k: T.Tensor([1, total_tokens, H, K], dtype),
            v: T.Tensor([1, total_tokens, HV, V], dtype),
            g: T.Tensor([1, total_tokens, HV, K], dtype),
            beta: T.Tensor([1, total_tokens, HV], dtype),
            h0: T.Tensor([num_seqs, HV, K, V], accum),
            cu_seqlens: T.Tensor([num_seqs + 1], "int64"),
            o: T.Tensor([1, total_tokens, HV, V], dtype),
            ht: T.Tensor([num_seqs, HV, K, V], accum),
        ):
            with T.Kernel(num_seqs, HV, threads=threads) as (iseq, ihv):
                ih = ihv // group
                origin = T.cast(cu_seqlens[iseq], "int32")
                length = T.cast(cu_seqlens[iseq + 1], "int32") - origin

                if compact:
                    qa_s = T.alloc_shared([BT, K], dtype)
                    ka_s = T.alloc_shared([BT, K], dtype)
                    kb_s = T.alloc_shared([BT, K], dtype)
                    b_s = T.alloc_shared([BT, K], dtype)
                    metadata = T.alloc_shared([(NC + 1) * K + BT], accum)
                    state_s = T.alloc_shared([K, V], dtype)
                    gc_s = state_s
                    vb_s = T.view(b_s, [BT, V], dtype)
                    vn_s = T.view(ka_s, [BT, V], dtype)
                    m_s = ka_s
                    p_s = ka_s
                    bt_s = metadata
                else:
                    qa_s = T.alloc_shared([BT, K], dtype)
                    ka_s = T.alloc_shared([BT, K], dtype)
                    kb_s = T.alloc_shared([BT, K], dtype)
                    b_s = T.alloc_shared([BT, K], dtype)
                    vb_s = T.alloc_shared([BT, V], dtype)
                    vn_s = T.alloc_shared([BT, V], dtype)
                    m_s = T.alloc_shared([BT, BT], dtype)
                    p_s = T.alloc_shared([BT, BT], dtype)
                    state_s = T.alloc_shared([K, V], dtype)
                    gc_s = T.alloc_shared([BT, K], accum)
                    bt_s = T.alloc_shared([BT], accum)
                    metadata = bt_s

                akk = T.alloc_fragment([BT, BT], accum)
                aq = T.alloc_fragment([BT, BT], accum)
                tkk = T.alloc_fragment([BT, BT], accum)
                tqk = T.alloc_fragment([BT, BT], accum)
                acc = T.alloc_fragment([BT, K], accum)
                if compact:
                    norm_f = T.alloc_fragment([BT, K], accum)
                    T.annotate_layout(
                        {
                            norm_f: T.Fragment(
                                [BT, K],
                                forward_thread_fn=lambda i, j: (i % (threads // 32)) * 32 + j % 32,
                                forward_index_fn=lambda i, j: (i // (threads // 32)) * (K // 32)
                                + j // 32,
                            )
                        }
                    )
                else:
                    norm_f = acc
                rowsum = T.alloc_fragment([BT], accum)
                state_f = T.alloc_fragment([K, V], accum)
                vn_f = T.alloc_fragment([BT, V], accum)
                o_f = T.alloc_fragment([BT, V], accum)
                upd = T.alloc_fragment([K, V], accum)

                prefix_s = kb_s if compact else m_s
                causal_s = qa_s if compact else m_s

                for i, j in T.Parallel(K, V):
                    state_f[i, j] = h0[iseq, ihv, i, j]

                for ic in T.serial(T.ceildiv(length, BT)):
                    bos = origin + ic * BT
                    rows = T.min(BT, length - ic * BT)
                    for i, j in T.Parallel(BT, K):
                        qa_s[i, j] = T.if_then_else(
                            i < rows, q[0, bos + i, ih, j], T.cast(0, dtype)
                        )
                        ka_s[i, j] = T.if_then_else(
                            i < rows, k[0, bos + i, ih, j], T.cast(0, dtype)
                        )
                        b_s[i, j] = T.if_then_else(
                            i < rows, g[0, bos + i, ihv, j], T.cast(0, dtype)
                        )
                    for i, j in T.Parallel(BT, BT):
                        prefix_s[i, j] = T.cast(T.if_then_else(i >= j, 1.0, 0.0), dtype)
                    for i in T.Parallel(BT):
                        bt_s[(NC + 1) * K + i if compact else i] = T.if_then_else(
                            i < rows, T.cast(beta[0, bos + i, ihv], accum), T.cast(0, accum)
                        )
                    T.sync_threads()

                    T.clear(acc)
                    T.gemm(matrix(prefix_s), b_s, acc)
                    if compact:
                        # Keep FP32 prefixes in acc; only reference rows cross threads.
                        for i, j in T.Parallel(BT, K):
                            if i % BC == REF:
                                metadata[(i // BC) * K + j] = acc[i, j] * LOG2E
                            if i == rows - 1:
                                metadata[NC * K + j] = acc[i, j] * LOG2E
                        T.sync_threads()
                    else:
                        for i, j in T.Parallel(BT, K):
                            gc_s[i, j] = acc[i, j] * LOG2E
                    for i, j in T.Parallel(BT, K):
                        norm_f[i, j] = T.cast(qa_s[i, j], accum) * T.cast(qa_s[i, j], accum)
                    T.reduce_sum(norm_f, rowsum, dim=1)
                    for i, j in T.Parallel(BT, K):
                        qa_s[i, j] = T.cast(
                            T.cast(qa_s[i, j], accum)
                            * (T.rsqrt(rowsum[i] + L2_EPS) if l2norm else 1.0)
                            * scale,
                            dtype,
                        )
                    for i, j in T.Parallel(BT, K):
                        norm_f[i, j] = T.cast(ka_s[i, j], accum) * T.cast(ka_s[i, j], accum)
                    T.reduce_sum(norm_f, rowsum, dim=1)
                    for i, j in T.Parallel(BT, K):
                        ka_s[i, j] = T.cast(
                            T.cast(ka_s[i, j], accum)
                            * (T.rsqrt(rowsum[i] + L2_EPS) if l2norm else 1.0),
                            dtype,
                        )
                    T.sync_threads()

                    for i, j in T.Parallel(BT, K):
                        kb_s[i, j] = T.cast(
                            T.cast(ka_s[i, j], accum)
                            * T.exp2(
                                reference_gate(gc_s, metadata, i // BC, j)
                                - (acc[i, j] * LOG2E if compact else gc_s[i, j])
                            ),
                            dtype,
                        )
                    T.sync_threads()
                    for i, j in T.Parallel(BT, K):
                        ka_s[i, j] = T.cast(
                            T.cast(ka_s[i, j], accum)
                            * T.exp2(
                                (acc[i, j] * LOG2E if compact else gc_s[i, j])
                                - reference_gate(gc_s, metadata, i // BC, j)
                            ),
                            dtype,
                        )
                        qa_s[i, j] = T.cast(
                            T.cast(qa_s[i, j], accum)
                            * T.exp2(
                                (acc[i, j] * LOG2E if compact else gc_s[i, j])
                                - reference_gate(gc_s, metadata, i // BC, j)
                            ),
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
                                        reference_gate(gc_s, metadata, sub, j)
                                        - reference_gate(gc_s, metadata, i // BC, j)
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

                    if compact:
                        prepare_key(b_s, ka_s, gc_s, metadata, bt_s)
                    for i, j in T.Parallel(BT, BT):
                        m_s[i, j] = T.cast(
                            T.if_then_else(i > j, -akk[i, j] * step(bt_s, i), 0.0), dtype
                        )
                    T.sync_threads()
                    invert(m_s, p_s, tkk)

                    # The chunk's step over the state this CTA carries.
                    if not compact:
                        load_state(state_s, state_f)
                    if not compact:
                        prepare_key(b_s, ka_s, gc_s, metadata, bt_s)
                    T.clear(acc)
                    T.gemm(matrix(p_s, True), b_s, acc)
                    for i, j in T.Parallel(BT, K):
                        b_s[i, j] = T.cast(acc[i, j], dtype)
                    if compact:
                        load_state(state_s, state_f)
                        T.sync_threads()
                    if not compact:
                        load_value(vb_s, v, bt_s, rows, bos, ihv)
                    T.clear(vn_f)
                    T.gemm(b_s, state_s, vn_f)
                    for i, j in T.Parallel(BT, V):
                        o_f[i, j] = -vn_f[i, j]
                    if compact:
                        T.sync_threads()
                        load_value(vb_s, v, bt_s, rows, bos, ihv)
                    T.clear(vn_f)
                    T.gemm(matrix(p_s, True), vb_s, vn_f)
                    if compact:
                        T.sync_threads()
                        prepare_query(b_s, qa_s, gc_s, metadata)
                        T.sync_threads()
                    for i, j in T.Parallel(BT, V):
                        vn_s[i, j] = T.cast(vn_f[i, j] + o_f[i, j], dtype)
                    if not compact:
                        prepare_query(b_s, qa_s, gc_s, metadata)
                    for i, j in T.Parallel(BT, BT):
                        causal_s[i, j] = T.cast(T.if_then_else(i >= j, aq[i, j], 0.0), dtype)
                    T.sync_threads()
                    T.clear(o_f)
                    T.gemm(b_s, state_s, o_f)
                    T.gemm(matrix(causal_s), vn_s, o_f)
                    for i, j in T.Parallel(BT, K):
                        b_s[i, j] = T.cast(
                            T.cast(kb_s[i, j], accum)
                            * T.exp2(
                                final_gate(gc_s, metadata, rows, j)
                                - reference_gate(gc_s, metadata, i // BC, j)
                            ),
                            dtype,
                        )
                    T.sync_threads()
                    T.clear(upd)
                    T.gemm(b_s, vn_s, upd, transpose_A=True)
                    for i, j in T.Parallel(K, V):
                        state_f[i, j] = (
                            state_f[i, j] * T.exp2(final_gate(gc_s, metadata, rows, i)) + upd[i, j]
                        )
                    for i, j in T.Parallel(BT, V):
                        if i < rows:
                            o[0, bos + i, ihv, j] = T.cast(o_f[i, j], dtype)
                    T.sync_threads()

                for i, j in T.Parallel(K, V):
                    ht[iseq, ihv, i, j] = state_f[i, j]

        return main

    return build()
