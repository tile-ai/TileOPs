"""The fused Kimi Delta Attention (KDA) chunk program: one CTA walks one sequence.

Everything a chunk decides on its own and the recurrence that joins the chunks
live in the same CTA, so the WY vectors, the gated query and key and the
intra-chunk attention never leave the chip. What it costs is parallelism: the
launch is one block per (sequence, value head), so it is the right shape only
when there are enough of those to cover the machine.

The WY inverse uses stable 16-row diagonal solves and block matrix products.
Precomputed gates use reference-centred tiles where ``centred_bounds`` holds; other
chunks form each Gram entry from its pair's decay.
"""

import functools

import tilelang
import tilelang.language as T

from tileops._csrc import csrc_include
from tileops._kernels.constants import LOG2E
from tileops._kernels.linear_attention.kda.chunk_programs import (
    L2_EPS,
    REFERENCE_BLOCK,
    centred_bounds,
    centred_row_bound,
)
from tileops._kernels.linear_attention.kda.gate import raw_gate_gram
from tileops._kernels.linear_attention.kda.inverse import inverse_program

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
    *,
    stage: str | None = None,
):
    """Build the fused chunk program for one static packed shape.

    For precomputed gates, *stage* ``None`` checks each chunk and takes per-pair entries for
    that chunk alone. ``"check"`` forms every chunk from centred tiles and flags each
    (sequence, value head) with a chunk past the bound; ``"rerun"`` walks the flagged ones
    again from their initial state on per-pair entries, overwriting their outputs.
    """
    if compact and (dim_k != 128 or dim_v != 128 or chunk_size != 64):
        raise ValueError("compact storage requires a 64-token chunk and 128-wide state")
    if compact and stage is not None:
        raise ValueError("compact storage serves only the one-launch precomputed-gate program")
    BT = chunk_size
    BC = REFERENCE_BLOCK
    NC = BT // BC
    REF = BC // 2
    H, HV, K, V = heads, value_heads, dim_k, dim_v
    group = HV // H
    accum = "float32"
    guarded = stage is None
    checked = stage != "rerun"
    bounds = centred_bounds(dtype, scale)
    checkers = threads - BT
    ends = 2 * NC * K

    @tilelang.jit(
        out_idx=[] if stage == "rerun" else ([-3, -2, -1] if stage == "check" else [-2, -1]),
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            "--use_fast_math",
            *csrc_include("streaming_load.h"),
        ],
    )
    def build():
        def reference_gate(gc, metadata, block, channel):
            return metadata[block * K + channel] if compact else gc[block * BC + REF, channel]

        def final_gate(gc, metadata, rows, channel):
            return metadata[NC * K + channel] if compact else gc[rows - 1, channel]

        def scratch_load(scratch, i, j):
            """An FP32 view of the compact state tile would be allocated apart from it, so
            compact CTAs split each FP32 value over two of its 16-bit elements."""
            if not compact:
                return scratch[i, j]
            high = T.cast(T.reinterpret(scratch[2 * i, j], "uint16"), "uint32")
            low = T.cast(T.reinterpret(scratch[2 * i + 1, j], "uint16"), "uint32")
            return T.reinterpret(T.bitwise_or(T.shift_left(high, T.uint32(16)), low), "float32")

        @T.macro
        def scratch_store(scratch, i, j, value):
            if compact:
                bits = T.reinterpret(value, "uint32")
                scratch[2 * i, j] = T.reinterpret(
                    T.cast(T.shift_right(bits, T.uint32(16)), "uint16"), dtype
                )
                scratch[2 * i + 1, j] = T.reinterpret(
                    T.cast(T.bitwise_and(bits, T.uint32(65535)), "uint16"), dtype
                )
            else:
                scratch[i, j] = value

        gram = raw_gate_gram(BT, K, scratch_load)

        # Past the block ends (compact scratch) or the beta row (metadata): each row's
        # exponent bound for unnormalized rows, then the one-launch chunk flag.
        def row_bound(metadata, scratch, i):
            return scratch_load(scratch, 2 * NC, i) if compact else metadata[BT + 1 + i]

        @T.macro
        def store_row_bound(metadata, scratch, i, value):
            if compact:
                scratch_store(scratch, 2 * NC, i, value)
            else:
                metadata[BT + 1 + i] = value

        def chunk_flag(metadata, scratch):
            return scratch_load(scratch, 2 * NC + 1, 0) if compact else metadata[BT]

        @T.macro
        def store_chunk_flag(metadata, scratch, value):
            if compact:
                scratch_store(scratch, 2 * NC + 1, 0, value)
            else:
                metadata[BT] = value

        def block_bound(metadata, scratch, block):
            if l2norm:
                return bounds.normalized_limit
            return functools.reduce(
                T.min, [row_bound(metadata, scratch, block * BC + r) for r in range(BC)]
            )

        def past_bound(gc_s, metadata, scratch, block, end, channel):
            """Whether block *block*'s first (*end* 0) or last row lies past the bound; a
            block's prefix is monotone for one-signed gates, so its end rows lie farthest."""
            if compact:
                # The metadata keeps log2 references, the scratch unscaled end rows.
                distance = (
                    T.abs(
                        scratch_load(scratch, 2 * block + end, channel)
                        - metadata[block * K + channel] / LOG2E
                    )
                    * LOG2E
                )
            else:
                distance = T.abs(
                    gc_s[block * BC + end * (BC - 1), channel] - gc_s[block * BC + REF, channel]
                )
            return distance > block_bound(metadata, scratch, block)

        @T.macro
        def flag_chunk(gc_s, metadata, scratch):
            for e, j in T.Parallel(2 * NC, K):
                if past_bound(gc_s, metadata, scratch, e // 2, e % 2, j):
                    store_chunk_flag(metadata, scratch, T.float32(1.0))

        @T.macro
        def flag_sequence(t, gc_s, metadata, scratch, flags, iseq, ihv):
            """Run by the threads the diagonal solves leave idle, so it adds no step."""
            for e in T.serial(T.ceildiv(ends, checkers)):
                index = e * checkers + t
                if index < ends and past_bound(
                    gc_s, metadata, scratch, index // (2 * K), (index // K) % 2, index % K
                ):
                    flags[iseq, ihv] = 1

        invert = inverse_program(
            chunk_size=BT,
            block_size=BC,
            threads=threads,
            dtype=dtype,
            output_offset=BT if compact else 0,
            idle=flag_sequence if stage == "check" else None,
        )

        def centred(per_pair):
            return per_pair[0] == 0 if guarded else stage == "check"

        def step(bt, i):
            return bt[(NC + 1) * K + i] if compact else bt[i]

        def matrix(buffer, second=False):
            if compact:
                return buffer[:, BT:] if second else buffer[:, :BT]
            return buffer

        @T.macro
        def prefetch_row(x, row, head, column):
            T.call_extern(
                "handle", "tileops::prefetch_l2", T.access_ptr(x[0, row, head, column], "r")
            )

        @T.macro
        def load_state(state_s, state_f):
            for i, j in T.Parallel(K, V):
                state_s[i, j] = T.cast(state_f[i, j], dtype)

        @T.macro
        def prepare_key(b_s, ka_s, gc_s, metadata, bt_s, centred):
            if centred:
                for i, j in T.Parallel(BT, K):
                    b_s[i, j] = T.cast(
                        T.cast(ka_s[i, j], accum)
                        * T.exp2(reference_gate(gc_s, metadata, i // BC, j))
                        * step(bt_s, i),
                        dtype,
                    )
            else:
                for i, j in T.Parallel(BT, K):
                    b_s[i, j] = T.cast(T.cast(ka_s[i, j], accum) * step(bt_s, i), dtype)
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
        def prepare_query(b_s, qa_s, gc_s, metadata, centred):
            if centred:
                for i, j in T.Parallel(BT, K):
                    b_s[i, j] = T.cast(
                        T.cast(qa_s[i, j], accum)
                        * T.exp2(reference_gate(gc_s, metadata, i // BC, j)),
                        dtype,
                    )
            else:
                for i, j in T.Parallel(BT, K):
                    b_s[i, j] = T.cast(T.cast(qa_s[i, j], accum), dtype)

        @T.macro
        def update_keys(b_s, kb_s, gc_s, metadata, rows, centred):
            if centred:
                for i, j in T.Parallel(BT, K):
                    b_s[i, j] = T.cast(
                        T.cast(kb_s[i, j], accum)
                        * T.exp2(
                            final_gate(gc_s, metadata, rows, j)
                            - reference_gate(gc_s, metadata, i // BC, j)
                        ),
                        dtype,
                    )
            else:
                for i, j in T.Parallel(BT, K):
                    b_s[i, j] = T.cast(T.cast(kb_s[i, j], accum), dtype)

        @T.macro
        def per_pair_gram(qa_s, ka_s, kb_s, gc_s, acc, akk, aq, rows):
            if compact:
                # The compact prefix lives in registers; the per-pair entries read it shared.
                for i, j in T.Parallel(BT, K):
                    scratch_store(gc_s, i, j, acc[i, j] * LOG2E)
                T.sync_threads()
            gram(qa_s, ka_s, gc_s, akk, aq)
            T.sync_threads()
            for i, j in T.Parallel(BT, K):
                kb_s[i, j] = T.cast(
                    T.cast(ka_s[i, j], accum)
                    * T.exp2(scratch_load(gc_s, rows - 1, j) - scratch_load(gc_s, i, j)),
                    dtype,
                )
                qa_s[i, j] = T.cast(
                    T.cast(qa_s[i, j], accum) * T.exp2(scratch_load(gc_s, i, j)), dtype
                )
                ka_s[i, j] = T.cast(
                    T.cast(ka_s[i, j], accum) * T.exp2(scratch_load(gc_s, i, j)), dtype
                )
            T.sync_threads()

        @T.macro
        def centred_gram(qa_s, ka_s, kb_s, b_s, gc_s, metadata, acc, akk, aq, tkk, tqk):
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

        @T.macro
        def walk(q, k, v, g, beta, h0, cu_seqlens, o, ht, flags, iseq, ihv):
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
                # The state tile is idle until the chunk's state contractions; the check's
                # block ends, row bounds and flag, or a per-pair chunk's prefix, borrow it.
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
                bt_s = T.alloc_shared(
                    [BT + (int(guarded) + (0 if l2norm else 1 + BT) if checked else 0)], accum
                )
                metadata = bt_s

            akk = T.alloc_fragment([BT, BT], accum)
            aq = T.alloc_fragment([BT, BT], accum)
            tkk = T.alloc_fragment([BT, BT], accum)
            tqk = T.alloc_fragment([BT, BT], accum) if checked else None
            acc = T.alloc_fragment([BT, K], accum)
            if compact:
                norm_f = T.alloc_fragment([BT, K], accum)
                T.annotate_layout(
                    {
                        norm_f: T.Fragment(
                            [BT, K],
                            forward_thread_fn=lambda i, j: (i % (threads // 32)) * 32 + j % 32,
                            forward_index_fn=lambda i, j: (
                                (i // (threads // 32)) * (K // 32) + j // 32
                            ),
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
            # One flag per chunk keeps the branch on it uniform across the CTA.
            per_pair = T.alloc_local([1], "int32") if guarded else None

            prefix_s = kb_s if compact else m_s
            causal_s = qa_s if compact else m_s

            for i, j in T.Parallel(K, V):
                state_f[i, j] = h0[iseq, ihv, i, j]
            if stage == "check":
                flags[iseq, ihv] = 0

            for ic in T.serial(T.ceildiv(length, BT)):
                bos = origin + ic * BT
                rows = T.min(BT, length - ic * BT)
                for i, j in T.Parallel(BT, K):
                    qa_s[i, j] = T.if_then_else(i < rows, q[0, bos + i, ih, j], T.cast(0, dtype))
                    ka_s[i, j] = T.if_then_else(i < rows, k[0, bos + i, ih, j], T.cast(0, dtype))
                    b_s[i, j] = T.if_then_else(i < rows, g[0, bos + i, ihv, j], T.cast(0, dtype))
                for i, j in T.Parallel(BT, BT):
                    prefix_s[i, j] = T.cast(T.if_then_else(i >= j, 1.0, 0.0), dtype)
                for i in T.Parallel(BT):
                    bt_s[(NC + 1) * K + i if compact else i] = T.if_then_else(
                        i < rows, T.cast(beta[0, bos + i, ihv], accum), T.cast(0, accum)
                    )
                if guarded:
                    store_chunk_flag(metadata, state_s, T.float32(0.0))
                if not compact:
                    # The walk is serial, so each chunk would first wait on its own rows' loads;
                    # bringing the next chunk's rows into L2 now overlaps that wait with this chunk.
                    lane = T.get_thread_binding()
                    ahead = bos + BT + lane % BT
                    if lane < 2 * BT and ahead < origin + length:
                        prefetch_row(q, ahead, ih, lane // BT * (K // 2))
                        prefetch_row(k, ahead, ih, lane // BT * (K // 2))
                        prefetch_row(g, ahead, ihv, lane // BT * (K // 2))
                        prefetch_row(v, ahead, ihv, lane // BT * (V // 2))
                T.sync_threads()

                T.clear(acc)
                T.gemm(matrix(prefix_s), b_s, acc)
                if compact:
                    for i, j in T.Parallel(BT, K):
                        if i % BC == REF:
                            metadata[(i // BC) * K + j] = acc[i, j] * LOG2E
                        if i == rows - 1:
                            metadata[NC * K + j] = acc[i, j] * LOG2E
                        if checked and (i % BC == 0 or i % BC == BC - 1):
                            # Unscaled: a product shared with the centred factors would
                            # change the exponents they contract it into.
                            scratch_store(
                                state_s, 2 * (i // BC) + (i % BC) // (BC - 1), j, acc[i, j]
                            )
                    T.sync_threads()
                else:
                    for i, j in T.Parallel(BT, K):
                        gc_s[i, j] = acc[i, j] * LOG2E
                for i, j in T.Parallel(BT, K):
                    norm_f[i, j] = T.cast(qa_s[i, j], accum) * T.cast(qa_s[i, j], accum)
                T.reduce_sum(norm_f, rowsum, dim=1)
                if checked and not l2norm:
                    for i in T.Parallel(BT):
                        store_row_bound(
                            metadata,
                            state_s,
                            i,
                            centred_row_bound(
                                bounds, bounds.flush_limit, rowsum[i], bounds.query_scale_log2
                            ),
                        )
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
                if checked and not l2norm:
                    for i in T.Parallel(BT):
                        store_row_bound(
                            metadata,
                            state_s,
                            i,
                            centred_row_bound(
                                bounds, row_bound(metadata, state_s, i), rowsum[i], 0.0
                            ),
                        )
                for i, j in T.Parallel(BT, K):
                    ka_s[i, j] = T.cast(
                        T.cast(ka_s[i, j], accum)
                        * (T.rsqrt(rowsum[i] + L2_EPS) if l2norm else 1.0),
                        dtype,
                    )
                if guarded and l2norm:
                    flag_chunk(gc_s, metadata, state_s)
                T.sync_threads()

                if guarded:
                    if not l2norm:
                        # The block bounds read rows other threads stored.
                        flag_chunk(gc_s, metadata, state_s)
                        T.sync_threads()
                    per_pair[0] = T.if_then_else(chunk_flag(metadata, state_s) != 0.0, 1, 0)
                    if per_pair[0] != 0:
                        if compact:
                            # Every thread has read the flag before the prefix overwrites it.
                            T.sync_threads()
                        per_pair_gram(qa_s, ka_s, kb_s, gc_s, acc, akk, aq, rows)
                    else:
                        centred_gram(qa_s, ka_s, kb_s, b_s, gc_s, metadata, acc, akk, aq, tkk, tqk)
                elif stage == "check":
                    centred_gram(qa_s, ka_s, kb_s, b_s, gc_s, metadata, acc, akk, aq, tkk, tqk)
                else:
                    per_pair_gram(qa_s, ka_s, kb_s, gc_s, acc, akk, aq, rows)
                if compact:
                    prepare_key(b_s, ka_s, gc_s, metadata, bt_s, centred(per_pair))
                for i, j in T.Parallel(BT, BT):
                    m_s[i, j] = T.cast(
                        T.if_then_else(i > j, -akk[i, j] * step(bt_s, i), 0.0), dtype
                    )
                T.sync_threads()
                if stage == "check":
                    invert(m_s, p_s, tkk, (gc_s, metadata, state_s, flags, iseq, ihv))
                else:
                    invert(m_s, p_s, tkk)

                # The chunk's step over the state this CTA carries.
                if not compact:
                    load_state(state_s, state_f)
                if not compact:
                    prepare_key(b_s, ka_s, gc_s, metadata, bt_s, centred(per_pair))
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
                    prepare_query(b_s, qa_s, gc_s, metadata, centred(per_pair))
                    T.sync_threads()
                for i, j in T.Parallel(BT, V):
                    vn_s[i, j] = T.cast(vn_f[i, j] + o_f[i, j], dtype)
                if not compact:
                    prepare_query(b_s, qa_s, gc_s, metadata, centred(per_pair))
                for i, j in T.Parallel(BT, BT):
                    causal_s[i, j] = T.cast(T.if_then_else(i >= j, aq[i, j], 0.0), dtype)
                T.sync_threads()
                T.clear(o_f)
                T.gemm(b_s, state_s, o_f)
                T.gemm(matrix(causal_s), vn_s, o_f)
                update_keys(b_s, kb_s, gc_s, metadata, rows, centred(per_pair))
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

        @T.macro
        def compute(q, k, v, g, beta, h0, cu_seqlens, o, ht, flags):
            with T.Kernel(num_seqs, HV, threads=threads) as (iseq, ihv):
                # A rerun walks only the sequences the first launch flagged.
                if stage != "rerun" or flags[iseq, ihv] != 0:
                    walk(q, k, v, g, beta, h0, cu_seqlens, o, ht, flags, iseq, ihv)

        if stage is not None:

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
                flags: T.Tensor([num_seqs, HV], "int32"),
            ):
                compute(q, k, v, g, beta, h0, cu_seqlens, o, ht, flags)
        else:

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
                compute(q, k, v, g, beta, h0, cu_seqlens, o, ht, None)

        return main

    return build()
