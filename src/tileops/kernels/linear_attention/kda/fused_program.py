"""The fused Kimi Delta Attention chunk program: one CTA walks one sequence.

Everything a chunk decides on its own and the recurrence that joins the chunks
live in the same CTA, so the WY vectors, the gated query and key and the
intra-chunk attention never leave the chip. What it costs is parallelism: the
launch is one block per (sequence, value head), so it is the right shape only
when there are enough of those to cover the machine.

Inside the chunk, the unit-triangular WY inverse is six nilpotent Neumann
factors -- every step of it a tensor-core matmul rather than a row-sequential
forward substitution -- and one gate reference per 16 rows is produced once and
rescaled per block, so one reference block's operands are gone before the next
is built.
"""

import functools

import tilelang
import tilelang.language as T

from tileops.kernels.constants import LOG2E
from tileops.kernels.linear_attention.kda.chunk_programs import L2_EPS, REFERENCE_BLOCK

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
):
    """Build the fused chunk program for one static packed shape."""
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

                akk = T.alloc_fragment([BT, BT], accum)
                aq = T.alloc_fragment([BT, BT], accum)
                tkk = T.alloc_fragment([BT, BT], accum)
                tqk = T.alloc_fragment([BT, BT], accum)
                acc = T.alloc_fragment([BT, K], accum)
                rowsum = T.alloc_fragment([BT], accum)
                state_f = T.alloc_fragment([K, V], accum)
                vn_f = T.alloc_fragment([BT, V], accum)
                o_f = T.alloc_fragment([BT, V], accum)
                upd = T.alloc_fragment([K, V], accum)

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
                        m_s[i, j] = T.cast(T.if_then_else(i >= j, 1.0, 0.0), dtype)
                    for i in T.Parallel(BT):
                        bt_s[i] = T.if_then_else(
                            i < rows, T.cast(beta[0, bos + i, ihv], accum), T.cast(0, accum)
                        )
                    T.sync_threads()

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

                    # The chunk's step over the state this CTA carries.
                    for i, j in T.Parallel(K, V):
                        state_s[i, j] = T.cast(state_f[i, j], dtype)
                    for i, j in T.Parallel(BT, K):
                        b_s[i, j] = T.cast(
                            T.cast(ka_s[i, j], accum)
                            * T.exp2(gc_s[(i // BC) * BC + REF, j])
                            * bt_s[i],
                            dtype,
                        )
                    T.sync_threads()
                    T.clear(acc)
                    T.gemm(p_s, b_s, acc)
                    for i, j in T.Parallel(BT, K):
                        b_s[i, j] = T.cast(acc[i, j], dtype)
                    for i, j in T.Parallel(BT, V):
                        vb_s[i, j] = T.cast(
                            T.if_then_else(i < rows, T.cast(v[0, bos + i, ihv, j], accum), 0.0)
                            * bt_s[i],
                            dtype,
                        )
                    T.sync_threads()
                    T.clear(vn_f)
                    T.gemm(b_s, state_s, vn_f)
                    for i, j in T.Parallel(BT, V):
                        o_f[i, j] = -vn_f[i, j]
                    T.clear(vn_f)
                    T.gemm(p_s, vb_s, vn_f)
                    for i, j in T.Parallel(BT, V):
                        vn_s[i, j] = T.cast(vn_f[i, j] + o_f[i, j], dtype)
                    for i, j in T.Parallel(BT, K):
                        b_s[i, j] = T.cast(
                            T.cast(qa_s[i, j], accum) * T.exp2(gc_s[(i // BC) * BC + REF, j]),
                            dtype,
                        )
                    for i, j in T.Parallel(BT, BT):
                        m_s[i, j] = T.cast(T.if_then_else(i >= j, aq[i, j], 0.0), dtype)
                    T.sync_threads()
                    T.clear(o_f)
                    T.gemm(b_s, state_s, o_f)
                    T.gemm(m_s, vn_s, o_f)
                    for i, j in T.Parallel(BT, K):
                        b_s[i, j] = T.cast(
                            T.cast(kb_s[i, j], accum)
                            * T.exp2(gc_s[rows - 1, j] - gc_s[(i // BC) * BC + REF, j]),
                            dtype,
                        )
                    T.sync_threads()
                    T.clear(upd)
                    T.gemm(b_s, vn_s, upd, transpose_A=True)
                    for i, j in T.Parallel(K, V):
                        state_f[i, j] = state_f[i, j] * T.exp2(gc_s[rows - 1, i]) + upd[i, j]
                    for i, j in T.Parallel(BT, V):
                        if i < rows:
                            o[0, bos + i, ihv, j] = T.cast(o_f[i, j], dtype)
                    T.sync_threads()

                for i, j in T.Parallel(K, V):
                    ht[iseq, ihv, i, j] = state_f[i, j]

        return main

    return build()
