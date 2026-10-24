"""Split KDA chunk preparation: ordered FP32 gate prefixes and per-pair Gram products.

Off-diagonal products are exactly rounded FP32 fused multiply-add chains of the
boundary-referenced factors, which preserve cancellation residuals. The stable inverse
uses the arithmetic of kda/inverse.py and is multiplied into the WY vectors in one launch.
"""

import functools

import tilelang
import tilelang.language as T

from tileops.kernels.constants import LOG2E
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.linear_attention.chunk_plan import CHUNK_SIZE
from tileops.kernels.linear_attention.gdn.prefill_common import step_size
from tileops.kernels.linear_attention.kda.chunk_programs import L2_EPS, REFERENCE_BLOCK
from tileops.kernels.linear_attention.kda.gate import log_decay

# Rows per CTA of ``_staged_norm_program``, one warp each; ``split_prepare_program`` sends
# it only row counts this divides. A larger CTA would share no data between rows.
_STAGED_NORM_ROWS = 4


@functools.lru_cache(maxsize=64)
def _gate_program(tokens, seqs, HV, K, dtype, has_bias, bound, grid_chunks=None, raw=True):
    """The FP32 chunk-local inclusive prefix of the log decay, times LOG2E.

    A raw gate becomes its log decay first; a given gate (*raw* false) is the log decay.
    Each channel adds its tokens left to right in FP32.
    """
    BT = CHUNK_SIZE
    tiling = GroupTiling(seqs, BT)
    chunks = grid_chunks or tiling.tile_upper_bound(tokens)
    # Two channels per thread halve the per-element address, bias and store work but double
    # each thread's serial chain; the saved work wins only on grids of many chunk heads.
    vec = 2 if chunks * HV >= 1056 else 1
    # Tokens per register block: more loads per batch against more registers per thread;
    # only the bounded gate at one channel per thread gains from holding the whole chunk.
    SUB = BT if vec == 1 and bound is not None else 8

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.macro
        def walk(g, gc, head, bos, rows, raw_tile, out, running, log_rate, offset, full):
            # Each channel adds its tokens left to right in FP32, one register block at a time.
            for s in T.serial(BT // SUB):
                base = s * SUB
                for t, c in T.Parallel(SUB, K):
                    if full:
                        raw_tile[t, c] = g[0, bos + base + t, head, c]
                    else:
                        raw_tile[t, c] = T.if_then_else(
                            base + t < rows, g[0, bos + base + t, head, c], T.cast(0, dtype)
                        )
                for t in T.unroll(SUB):
                    for c in T.Parallel(K):
                        if not raw:
                            # Padded rows loaded as zero add nothing.
                            running[c] += T.cast(raw_tile[t, c], "float32")
                        elif full:
                            running[c] += log_decay(raw_tile[t, c], log_rate[0], offset[c], bound)
                        else:
                            running[c] += T.if_then_else(
                                base + t < rows,
                                log_decay(raw_tile[t, c], log_rate[0], offset[c], bound),
                                0.0,
                            )
                        out[t, c] = running[c] * LOG2E
                for t, c in T.Parallel(SUB, K):
                    if full:
                        gc[0, bos + base + t, head, c] = out[t, c]
                    else:
                        if base + t < rows:
                            gc[0, bos + base + t, head, c] = out[t, c]

        @T.macro
        def body(g, cu, a, bias, gc):
            with T.Kernel(chunks, HV, threads=K // vec) as (ic, head):
                cum = T.alloc_shared([seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                log_rate = T.alloc_local([1], "float32")
                raw_tile = T.alloc_fragment([SUB, K], dtype)
                out = T.alloc_fragment([SUB, K], "float32")
                running = T.alloc_fragment([K], "float32")
                offset = T.alloc_fragment([K], "float32")
                tile = T.Fragment(
                    [SUB, K],
                    forward_thread_fn=lambda t, c: c // vec,
                    forward_index_fn=lambda t, c: t * vec + c % vec,
                )
                channel = T.Fragment(
                    [K], forward_thread_fn=lambda c: c // vec, forward_index_fn=lambda c: c % vec
                )
                T.annotate_layout({raw_tile: tile, out: tile, running: channel, offset: channel})
                tiling.cumsum_offsets(cu, cum)
                tiling.decode(T.min(ic, cum[seqs] - 1), cum, lo, hi, seq, first)
                bos = T.cast(cu[seq[0]], "int32") + first[0]
                rows = T.if_then_else(
                    ic < cum[seqs], T.min(BT, T.cast(cu[seq[0] + 1], "int32") - bos), 0
                )
                if raw:
                    # Loaded once per program: in-loop parameter loads serialize each token's chain.
                    log_rate[0] = a[head]
                    for c in T.Parallel(K):
                        offset[c] = bias[head * K + c] if has_bias else T.float32(0)
                T.clear(running)
                # Full chunks take the predicate-free walk.
                if rows == BT:
                    walk(g, gc, head, bos, rows, raw_tile, out, running, log_rate, offset, True)
                else:
                    walk(g, gc, head, bos, rows, raw_tile, out, running, log_rate, offset, False)

        if raw:

            @T.prim_func
            def main(
                g: T.Tensor([1, tokens, HV, K], dtype),
                cu: T.Tensor([seqs + 1], "int64"),
                a: T.Tensor([HV], "float32"),
                bias: T.Tensor([HV * K if has_bias else HV], "float32"),
                gc: T.Tensor([1, tokens, HV, K], "float32"),
            ):
                body(g, cu, a, bias, gc)

        else:

            @T.prim_func
            def main(
                g: T.Tensor([1, tokens, HV, K], dtype),
                cu: T.Tensor([seqs + 1], "int64"),
                gc: T.Tensor([1, tokens, HV, K], "float32"),
            ):
                body(g, cu, None, None, gc)

        return main

    return build()


@functools.lru_cache(maxsize=64)
def _norm_program(tokens, H, K, dtype, scale):
    """L2-normalized q (times *scale*) and k of *H* heads.

    One warp reduces each of a CTA's *R* rows.
    """
    R = 8

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.prim_func
        def main(
            q: T.Tensor([1, tokens, H, K], dtype),
            k: T.Tensor([1, tokens, H, K], dtype),
            qn: T.Tensor([1, tokens, H, K], dtype),
            kn: T.Tensor([1, tokens, H, K], dtype),
        ):
            with T.Kernel(T.ceildiv(tokens * H, R), threads=R * 32) as (block,):
                qf = T.alloc_fragment([R, K], "float32")
                kf = T.alloc_fragment([R, K], "float32")
                qs = T.alloc_fragment([R, K], "float32")
                ks = T.alloc_fragment([R, K], "float32")
                qr = T.alloc_fragment([R], "float32")
                kr = T.alloc_fragment([R], "float32")
                T.annotate_layout(
                    {
                        qf: T.Fragment(
                            [R, K],
                            forward_thread_fn=lambda r, c: r * 32 + c % 32,
                            forward_index_fn=lambda r, c: c // 32,
                        )
                    }
                )
                for r, c in T.Parallel(R, K):
                    row = block * R + r
                    qf[r, c] = T.if_then_else(
                        row < tokens * H,
                        T.cast(q[0, row // H, row % H, c], "float32"),
                        0.0,
                    )
                    kf[r, c] = T.if_then_else(
                        row < tokens * H,
                        T.cast(k[0, row // H, row % H, c], "float32"),
                        0.0,
                    )
                    qs[r, c] = qf[r, c] * qf[r, c]
                    ks[r, c] = kf[r, c] * kf[r, c]
                T.reduce_sum(qs, qr, dim=1)
                T.reduce_sum(ks, kr, dim=1)
                for r, c in T.Parallel(R, K):
                    row = block * R + r
                    if row < tokens * H:
                        qn[0, row // H, row % H, c] = T.cast(
                            qf[r, c] * T.rsqrt(qr[r] + L2_EPS) * scale, dtype
                        )
                        kn[0, row // H, row % H, c] = T.cast(
                            kf[r, c] * T.rsqrt(kr[r] + L2_EPS), dtype
                        )

        return main

    return build()


@functools.lru_cache(maxsize=64)
def _staged_norm_program(rows, K, dtype, scale):
    """``_norm_program`` over the flattened ``[rows, K]`` rows, a multiple of R of them.

    Each CTA moves its rows through shared memory with vector copies, then reduces them
    in the same thread layout as ``_norm_program``, so the outputs are the same.
    """
    R = _STAGED_NORM_ROWS

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.prim_func
        def main(
            q: T.Tensor([rows, K], dtype),
            k: T.Tensor([rows, K], dtype),
            qn: T.Tensor([rows, K], dtype),
            kn: T.Tensor([rows, K], dtype),
        ):
            with T.Kernel(rows // R, threads=R * 32) as (block,):
                q_s = T.alloc_shared([R, K], dtype)
                k_s = T.alloc_shared([R, K], dtype)
                qf = T.alloc_fragment([R, K], "float32")
                kf = T.alloc_fragment([R, K], "float32")
                qs = T.alloc_fragment([R, K], "float32")
                ks = T.alloc_fragment([R, K], "float32")
                qr = T.alloc_fragment([R], "float32")
                kr = T.alloc_fragment([R], "float32")
                T.annotate_layout(
                    {
                        qf: T.Fragment(
                            [R, K],
                            forward_thread_fn=lambda r, c: r * 32 + c % 32,
                            forward_index_fn=lambda r, c: c // 32,
                        )
                    }
                )
                T.copy(q[block * R, 0], q_s)
                T.copy(k[block * R, 0], k_s)
                for r, c in T.Parallel(R, K):
                    qf[r, c] = T.cast(q_s[r, c], "float32")
                    kf[r, c] = T.cast(k_s[r, c], "float32")
                    qs[r, c] = qf[r, c] * qf[r, c]
                    ks[r, c] = kf[r, c] * kf[r, c]
                T.reduce_sum(qs, qr, dim=1)
                T.reduce_sum(ks, kr, dim=1)
                for r, c in T.Parallel(R, K):
                    q_s[r, c] = T.cast(qf[r, c] * T.rsqrt(qr[r] + L2_EPS) * scale, dtype)
                    k_s[r, c] = T.cast(kf[r, c] * T.rsqrt(kr[r] + L2_EPS), dtype)
                T.copy(q_s, qn[block * R, 0])
                T.copy(k_s, kn[block * R, 0])

        return main

    return build()


def _key_head(head, group):
    """The q/k head a value head reads: q and k may hold one head per *group* value heads."""
    return head if group == 1 else head // group


def _scaled(query, qscale):
    """An FP32 query value times *qscale*, or unchanged when the staged query carries its scale."""
    return query if qscale is None else query * qscale


def _fma(a, b, c):
    """``a * b + c`` rounded once to FP32, keeping subnormal values under fast math."""
    # T.ieee_fmaf emits __fmaf_rn, which -ftz flushes; __fmaf_ieee_rn ignores the flag.
    return T.call_extern(  # lint: extern-target allow: no T.* intrinsic ignores -ftz
        "float32", "__fmaf_ieee_rn", a, b, c
    )


@T.macro
def _group_sum8(values, part):
    """Sums of eight per-lane partials over the eight lanes of a lane group; lane part ends with entry part."""
    for m in T.unroll(4):
        values[m] = _fma(
            1.0,
            T.if_then_else((part >> 2) & 1 == 1, values[m + 4], values[m]),
            T.shfl_xor(T.if_then_else((part >> 2) & 1 == 1, values[m], values[m + 4]), 4),
        )
    for m in T.unroll(2):
        values[m] = _fma(
            1.0,
            T.if_then_else((part >> 1) & 1 == 1, values[m + 2], values[m]),
            T.shfl_xor(T.if_then_else((part >> 1) & 1 == 1, values[m], values[m + 2]), 2),
        )
    values[0] = _fma(
        1.0,
        T.if_then_else(part & 1 == 1, values[1], values[0]),
        T.shfl_xor(T.if_then_else(part & 1 == 1, values[0], values[1]), 1),
    )


@T.macro
def _warp_sum8(values, lane):
    """Warp sums of eight per-lane partials along the xor butterfly (16, 8, 4, 2, 1).

    Each step keeps the half of the partials its lane bit selects and adds the partner
    lane's partial of the same entry, so every entry is the exact pairwise sum tree of a
    full butterfly (T.reduce_sum over 32 lanes) at 9 shuffles instead of 40. Afterwards
    values[0] holds entry lane >> 2.
    """
    for m in T.unroll(4):
        values[m] = T.if_then_else((lane >> 4) & 1 == 1, values[m + 4], values[m]) + T.shfl_xor(
            T.if_then_else((lane >> 4) & 1 == 1, values[m], values[m + 4]), 16
        )
    for m in T.unroll(2):
        values[m] = T.if_then_else((lane >> 3) & 1 == 1, values[m + 2], values[m]) + T.shfl_xor(
            T.if_then_else((lane >> 3) & 1 == 1, values[m], values[m + 2]), 8
        )
    values[0] = T.if_then_else((lane >> 2) & 1 == 1, values[1], values[0]) + T.shfl_xor(
        T.if_then_else((lane >> 2) & 1 == 1, values[0], values[1]), 4
    )
    values[0] = values[0] + T.shfl_xor(values[0], 2)
    values[0] = values[0] + T.shfl_xor(values[0], 1)


@functools.lru_cache(maxsize=64)
def _diagonal_reduce_program(
    total_tokens, num_seqs, heads, dim_k, dtype, grid_chunks=None, key_heads=None
):
    """Four 16x16 diagonal Gram blocks per chunk as FP32 32-lane reductions (small grids).

    One CTA per 16-row block stages its q, k and gate rows in shared memory by one
    asynchronous copy; warp w owns rows w and 15-w, whose column batches together cost
    the same for every warp. Each entry is the per-lane FMA chain over channels
    {lane + 32 i} followed by the 32-lane butterfly, which rounds as T.reduce_sum over
    the 32 lanes does. A row's batches past the diagonal hold only zeros and skip the
    products.
    """
    BT, BC, K, HV = CHUNK_SIZE, REFERENCE_BLOCK, dim_k, heads
    G = HV // (key_heads or HV)
    # Two rows per warp: eight warps cover the block with balanced column batches.
    warps = BC // 2
    threads = warps * 32
    E = K // 32
    # Four columns per batch: the eight partials ``_warp_sum8`` reduces together.
    BW = 4
    assert 2 * BW == 8, "the partials _warp_sum8 reduces"
    tiling = GroupTiling(num_seqs, BT)
    chunks = grid_chunks or tiling.tile_upper_bound(total_tokens)

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.macro
        def stage(q, k, gc, kr, qr, gr, bos, head, base):
            # Rows past the tensor arrive as zeros; rows past the chunk feed only entries
            # that are masked or never stored.
            T.async_copy(k[0, bos + base : bos + base + BC, _key_head(head, G), :], kr)
            T.async_copy(q[0, bos + base : bos + base + BC, _key_head(head, G), :], qr)
            T.async_copy(gc[0, bos + base : bos + base + BC, head, :], gr)

        @T.macro
        def block(kk, qk, kr, qr, gr, bufs, bos, rows, head, base, lane, warp):
            qv, kv, gv, key, gk, vals = bufs
            for r in T.unroll(2):
                row = T.if_then_else(r == 0, warp, BC - 1 - warp)
                for i in T.unroll(E):
                    qv[r * E + i] = T.cast(qr[row, lane + 32 * i], "float32")
                    kv[r * E + i] = T.cast(kr[row, lane + 32 * i], "float32")
                    gv[r * E + i] = gr[row, lane + 32 * i]
            for r in T.unroll(2):
                row = T.if_then_else(r == 0, warp, BC - 1 - warp)
                for b in T.serial(row // BW + 1):
                    for jj in T.unroll(2 * BW):
                        vals[jj] = 0.0
                    for jj in T.unroll(BW):
                        for i in T.unroll(E):
                            key[jj * E + i] = T.cast(kr[b * BW + jj, lane + 32 * i], "float32")
                            gk[jj * E + i] = gr[b * BW + jj, lane + 32 * i]
                    # The four columns' chains are independent and interleave.
                    for jj in T.unroll(BW):
                        for i in T.unroll(E):
                            vals[jj] = vals[jj] + kv[r * E + i] * (
                                key[jj * E + i] * T.exp2(gv[r * E + i] - gk[jj * E + i])
                            )
                            vals[BW + jj] = vals[BW + jj] + qv[r * E + i] * (
                                key[jj * E + i] * T.exp2(gv[r * E + i] - gk[jj * E + i])
                            )
                        vals[jj] = T.if_then_else(b * BW + jj <= row, vals[jj], 0.0)
                        vals[BW + jj] = T.if_then_else(b * BW + jj <= row, vals[BW + jj], 0.0)
                    _warp_sum8(vals, lane)
                    # Lane 4 m holds kk column m (m < 4) or qk column m - 4.
                    if lane % 4 == 0 and base + row < rows:
                        if lane < 16:
                            kk[0, bos + base + row, head, base + b * BW + lane // 4] = vals[0]
                        else:
                            qk[0, bos + base + row, head, base + b * BW + lane // 4 - BW] = vals[0]
                # Columns of the skipped batches: lanes below tail store kk, the next qk.
                tail = BC - BW - (row // BW) * BW
                if lane < 2 * tail and base + row < rows:
                    if lane < tail:
                        kk[0, bos + base + row, head, base + BC - tail + lane] = 0.0
                    else:
                        qk[0, bos + base + row, head, base + BC - 2 * tail + lane] = 0.0

        @T.prim_func
        def main(
            q: T.Tensor([1, total_tokens, HV // G, K], dtype),
            k: T.Tensor([1, total_tokens, HV // G, K], dtype),
            gc: T.Tensor([1, total_tokens, HV, K], "float32"),
            cu: T.Tensor([num_seqs + 1], "int64"),
            kk: T.Tensor([1, total_tokens, HV, BT], "float32"),
            qk: T.Tensor([1, total_tokens, HV, BT], "float32"),
        ):
            with T.Kernel(chunks, HV, BT // BC, threads=threads) as (ic, head, blk):
                # More resident CTAs hide more of the per-lane chains' latency but leave
                # each thread fewer registers; four per SM balances the two.
                T.annotate_min_blocks_per_sm(4)
                tx = T.get_thread_binding()
                lane = tx % 32
                warp = tx // 32
                cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                kr = T.alloc_shared([BC, K], dtype)
                qr = T.alloc_shared([BC, K], dtype)
                gr = T.alloc_shared([BC, K], "float32")
                bufs = (
                    T.alloc_local([2 * E], "float32"),
                    T.alloc_local([2 * E], "float32"),
                    T.alloc_local([2 * E], "float32"),
                    T.alloc_local([BW * E], "float32"),
                    T.alloc_local([BW * E], "float32"),
                    T.alloc_local([2 * BW], "float32"),
                )
                tiling.cumsum_offsets(cu, cum)
                tiling.decode(T.min(ic, cum[num_seqs] - 1), cum, lo, hi, seq, first)
                bos = T.cast(cu[seq[0]], "int32") + first[0]
                rows = T.if_then_else(
                    ic < cum[num_seqs], T.min(BT, T.cast(cu[seq[0] + 1], "int32") - bos), 0
                )
                if rows > blk * BC:
                    stage(q, k, gc, kr, qr, gr, bos, head, blk * BC)
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    block(kk, qk, kr, qr, gr, bufs, bos, rows, head, blk * BC, lane, warp)

        return main

    return build()


@functools.lru_cache(maxsize=64)
def _diagonal_program(
    total_tokens, num_seqs, heads, dim_k, dtype, grid_chunks=None, qscale=None, key_heads=None
):
    """Four 16x16 diagonal Gram blocks per chunk as FP32 direct products.

    One small CTA per 16-row block stages its q, k and gate rows with one asynchronous
    copy. Lane group i owns rows i and 15-i, whose column batches together cost the same
    for every group; a row's batches past the diagonal hold only zeros and skip the
    products. Every pair keeps its own exp2(g_r - g_s): kv * (key * exp2(gv - gk)) with
    both products rounded once and no flushed subnormals. A lane sums four channels per
    segment, the segments pairwise and the eight lanes along a butterfly, the reduction
    depth of the 32-lane form in ``_diagonal_reduce_program``.
    """
    BT, BC, K, HV = CHUNK_SIZE, REFERENCE_BLOCK, dim_k, heads
    G = HV // (key_heads or HV)
    # Eight lanes share a row; each lane holds K / 32 4-channel vectors {32 m + 4 part}, so
    # a lane group's 16-byte reads of a staged row fill distinct banks.
    GROUP = 8
    VEC = 4
    SEGS = K // (GROUP * VEC)
    CH = SEGS * VEC
    BW = 4
    # ``_group_sum8`` and the pairwise segment sums below are written for these widths.
    assert GROUP == 8 and 2 * BW == 8 and SEGS in (2, 4), "the widths these sums are written for"
    # Eight lane groups, one CTA per block: many small CTAs hide the per-pair chains.
    threads = BC // 2 * GROUP
    tiling = GroupTiling(num_seqs, BT)
    chunks = grid_chunks or tiling.tile_upper_bound(total_tokens)
    # The 32-lane form spreads each block over four times these threads: shorter per-lane
    # chains against a wider butterfly, which pays off only on small grids. The two forms
    # round tiny products differently, so the choice reads the host bound: a dense call's
    # exact grid must not change which form a shape gets.
    # Unnormalized rows keep this form: its chains add neighbouring channels, which keeps
    # finite the cancelling sums of products near FP32's range that chains of channels 32
    # apart overflow.
    if qscale is None and tiling.tile_upper_bound(total_tokens) * HV < 48:
        return _diagonal_reduce_program(
            total_tokens, num_seqs, heads, dim_k, dtype, grid_chunks, key_heads
        )

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.prim_func
        def main(
            q: T.Tensor([1, total_tokens, HV // G, K], dtype),
            k: T.Tensor([1, total_tokens, HV // G, K], dtype),
            gc: T.Tensor([1, total_tokens, HV, K], "float32"),
            cu: T.Tensor([num_seqs + 1], "int64"),
            kk: T.Tensor([1, total_tokens, HV, BT], "float32"),
            qk: T.Tensor([1, total_tokens, HV, BT], "float32"),
        ):
            with T.Kernel(chunks, HV, BT // BC, threads=threads) as (ic, head, blk):
                tx = T.get_thread_binding()
                part = tx % GROUP
                gi = tx // GROUP
                cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                kr = T.alloc_shared([BC, K], dtype)
                qr = T.alloc_shared([BC, K], dtype)
                gr = T.alloc_shared([BC, K], "float32")
                tq = T.alloc_local([VEC], dtype)
                tk = T.alloc_local([VEC], dtype)
                qv = T.alloc_local([CH], "float32")
                kv = T.alloc_local([CH], "float32")
                gv = T.alloc_local([CH], "float32")
                key = T.alloc_local([CH], "float32")
                gk = T.alloc_local([CH], "float32")
                acc = T.alloc_local([2 * BW], "float32")
                sub = T.alloc_local([2 * SEGS], "float32")
                tiling.cumsum_offsets(cu, cum)
                tiling.decode(T.min(ic, cum[num_seqs] - 1), cum, lo, hi, seq, first)
                bos = T.cast(cu[seq[0]], "int32") + first[0]
                rows = T.if_then_else(
                    ic < cum[num_seqs], T.min(BT, T.cast(cu[seq[0] + 1], "int32") - bos), 0
                )
                base = blk * BC
                if rows > base:
                    T.async_copy(k[0, bos + base : bos + base + BC, _key_head(head, G), :], kr)
                    T.async_copy(q[0, bos + base : bos + base + BC, _key_head(head, G), :], qr)
                    T.async_copy(gc[0, bos + base : bos + base + BC, head, :], gr)
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    for r in T.unroll(2):
                        row = T.if_then_else(r == 0, gi, BC - 1 - gi)
                        for m in T.unroll(SEGS):
                            c0 = m * GROUP * VEC + part * VEC
                            for v in T.vectorized(VEC):
                                tq[v] = qr[row, c0 + v]
                            for v in T.vectorized(VEC):
                                tk[v] = kr[row, c0 + v]
                            for v in T.vectorized(VEC):
                                gv[m * VEC + v] = gr[row, c0 + v]
                            for v in T.unroll(VEC):
                                qv[m * VEC + v] = _scaled(T.cast(tq[v], "float32"), qscale)
                                kv[m * VEC + v] = T.cast(tk[v], "float32")
                        for b in T.serial(row // BW + 1):
                            for e in T.unroll(2 * BW):
                                acc[e] = 0.0
                            for j in T.unroll(BW):
                                for m in T.unroll(SEGS):
                                    c0 = m * GROUP * VEC + part * VEC
                                    for v in T.vectorized(VEC):
                                        tk[v] = kr[b * BW + j, c0 + v]
                                    for v in T.vectorized(VEC):
                                        gk[m * VEC + v] = gr[b * BW + j, c0 + v]
                                    for v in T.unroll(VEC):
                                        key[m * VEC + v] = T.cast(tk[v], "float32")
                                for m in T.unroll(SEGS):
                                    sub[m] = 0.0
                                    sub[SEGS + m] = 0.0
                                    for v in T.unroll(VEC):
                                        # key * exp2(g_r - g_s) rounded once, then the two products;
                                        # one chain of four channels per segment.
                                        i = m * VEC + v
                                        gk[i] = _fma(key[i], T.exp2(gv[i] - gk[i]), -0.0)
                                        sub[m] = _fma(kv[i], gk[i], sub[m])
                                        sub[SEGS + m] = _fma(qv[i], gk[i], sub[SEGS + m])
                                # Pairwise over the segments, as the warp butterfly pairs lanes.
                                if SEGS == 4:
                                    acc[2 * j] = _fma(
                                        1.0, _fma(1.0, sub[0], sub[1]), _fma(1.0, sub[2], sub[3])
                                    )
                                    acc[2 * j + 1] = _fma(
                                        1.0,
                                        _fma(1.0, sub[SEGS], sub[SEGS + 1]),
                                        _fma(1.0, sub[SEGS + 2], sub[SEGS + 3]),
                                    )
                                else:
                                    acc[2 * j] = _fma(1.0, sub[0], sub[1])
                                    acc[2 * j + 1] = _fma(1.0, sub[SEGS], sub[SEGS + 1])
                                acc[2 * j] = T.if_then_else(b * BW + j <= row, acc[2 * j], 0.0)
                                acc[2 * j + 1] = T.if_then_else(
                                    b * BW + j <= row, acc[2 * j + 1], 0.0
                                )
                            _group_sum8(acc, part)
                            # Lane part holds column part // 2, kk (even) or qk (odd).
                            if base + row < rows:
                                if part % 2 == 0:
                                    kk[0, bos + base + row, head, base + b * BW + part // 2] = acc[
                                        0
                                    ]
                                else:
                                    qk[0, bos + base + row, head, base + b * BW + part // 2] = acc[
                                        0
                                    ]
                        tail = BC - BW - (row // BW) * BW
                        for z in T.serial(T.ceildiv(2 * tail, GROUP)):
                            zl = z * GROUP + part
                            if zl < 2 * tail and base + row < rows:
                                if zl < tail:
                                    kk[0, bos + base + row, head, base + BC - tail + zl] = 0.0
                                else:
                                    qk[0, bos + base + row, head, base + BC - 2 * tail + zl] = 0.0

        return main

    return build()


@T.macro
def _group_sum16(values, part):
    """Sums of sixteen per-lane partials over the eight lanes of a lane group.

    Each xor step (4, 2, 1) keeps the half of the partials its lane bit selects and adds
    the partner lane's partial of the same entry, so lane ``part`` of the group ends
    with entries 2 part and 2 part + 1. The additions round like the products' chains.
    """
    for m in T.unroll(8):
        values[m] = _fma(
            1.0,
            T.if_then_else((part >> 2) & 1 == 1, values[m + 8], values[m]),
            T.shfl_xor(T.if_then_else((part >> 2) & 1 == 1, values[m], values[m + 8]), 4),
        )
    for m in T.unroll(4):
        values[m] = _fma(
            1.0,
            T.if_then_else((part >> 1) & 1 == 1, values[m + 4], values[m]),
            T.shfl_xor(T.if_then_else((part >> 1) & 1 == 1, values[m], values[m + 4]), 2),
        )
    for m in T.unroll(2):
        values[m] = _fma(
            1.0,
            T.if_then_else(part & 1 == 1, values[m + 2], values[m]),
            T.shfl_xor(T.if_then_else(part & 1 == 1, values[m], values[m + 2]), 1),
        )


@functools.lru_cache(maxsize=64)
def _offdiagonal_program(
    total_tokens, num_seqs, heads, dim_k, dtype, grid_chunks=None, qscale=None, key_heads=None
):
    """Six off-diagonal 16x16 tiles of both Gram matrices as FP32 dot products.

    Each row block keeps its first row as the reference: the left factor
    x_r * exp2(g_r - g_ref) and the right factor k_s * exp2(g_ref - g_s) are formed in
    FP32. A raw gate's log decays share one sign, so its prefix is monotone and neither
    factor exceeds the decay of the pair it belongs to; with a nonincreasing prefix
    neither exceeds its input. Every entry is then an exactly rounded FP32 fused
    multiply-add chain over a lane's K / 8 channels and an eight-lane butterfly, without
    flushing subnormal values. These keep
    the lower-order cross terms that tensor-core products of BF16 splits of the factors
    drop.
    """
    BT, BC, K, HV = CHUNK_SIZE, REFERENCE_BLOCK, dim_k, heads
    G = HV // (key_heads or HV)
    # Eight lanes share a row pair; each lane holds K / 32 4-channel vectors {32 m + 4 part},
    # so a lane group's 16-byte reads of a right-factor row fill distinct banks.
    GROUP = 8
    VEC = 4
    SEGS = K // (GROUP * VEC)
    CH = SEGS * VEC
    # Two rows per lane: every shared right-factor vector feeds four FMAs. A warp's four
    # lane groups and a second warp cover a row block's sixteen rows.
    PAIR = 2
    # Columns per butterfly: two rows of two Gram entries fill the sixteen partials.
    COLS = 4
    assert GROUP == 8 and 2 * PAIR * COLS == 16, "the widths _group_sum16 reduces"
    accum = "float32"
    tiling = GroupTiling(num_seqs, BT)
    chunks = grid_chunks or tiling.tile_upper_bound(total_tokens)
    # Spreading a chunk over more CTAs runs shorter chains side by side, but each extra CTA
    # repeats the setup and barriers, which pays off only on small grids:
    # - "split", below 256: each row block gets its own CTA, whose warp pairs split the
    #   block's columns; four pairs up to 96 chunk CTAs, two above;
    # - "walk", from 2048: one 64-thread CTA walks the chunk's row blocks;
    # - "concurrent", between: one 128-thread CTA forms all three blocks' right factors in
    #   one load phase and barrier, then one warp pair takes block 3 and the other blocks
    #   2 and 1.
    grid = chunks * HV
    mode = "split" if grid < 256 else ("concurrent" if grid < 2048 else "walk")
    threads = 64 if mode == "walk" else (256 if grid <= 96 else 128)
    # Warp pairs splitting a row block's columns.
    splits = threads // 64

    @tilelang.jit(
        out_idx=[],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.macro
        def right_factors(k, gc, rs, bos, ih, rb, roff):
            # Threads form a row block's right factors over its K channels, from row roff.
            for s, c in T.Parallel(rb, K):
                rs[roff + s, c] = T.cast(k[0, bos + s, _key_head(ih, G), c], accum) * T.exp2(
                    gc[0, bos + rb, ih, c] - gc[0, bos + s, ih, c]
                )

        @T.macro
        def left_factors(q, k, gc, bufs, bos, rows, ih, tx, rb):
            tq, tk, tg, tr, lq, lk, rv, acc = bufs
            lane = tx % 32
            warp = tx // 32
            part = lane % GROUP
            slot = lane // GROUP
            row0 = rb + (warp % 2) * (BC // 2) + slot * PAIR
            for p in T.unroll(PAIR):
                token = T.min(bos + row0 + p, total_tokens - 1)
                for m in T.unroll(SEGS):
                    c0 = m * GROUP * VEC + part * VEC
                    for v in T.vectorized(VEC):
                        tq[v] = q[0, token, _key_head(ih, G), c0 + v]
                    for v in T.vectorized(VEC):
                        tk[v] = k[0, token, _key_head(ih, G), c0 + v]
                    for v in T.vectorized(VEC):
                        tg[v] = gc[0, token, ih, c0 + v]
                    for v in T.vectorized(VEC):
                        tr[v] = gc[0, bos + rb, ih, c0 + v]
                    for v in T.unroll(VEC):
                        lq[p * CH + m * VEC + v] = T.if_then_else(
                            row0 + p < rows,
                            _scaled(T.cast(tq[v], accum), qscale) * T.exp2(tg[v] - tr[v]),
                            0.0,
                        )
                        lk[p * CH + m * VEC + v] = T.if_then_else(
                            row0 + p < rows, T.cast(tk[v], accum) * T.exp2(tg[v] - tr[v]), 0.0
                        )

        @T.macro
        def columns(kk, qk, rs, bufs, bos, rows, ih, tx, rb, roff, col0, batches):
            # A warp pair's sixteen rows against right-factor rows roff + col0 onward.
            tq, tk, tg, tr, lq, lk, rv, acc = bufs
            lane = tx % 32
            warp = tx // 32
            part = lane % GROUP
            slot = lane // GROUP
            row0 = rb + (warp % 2) * (BC // 2) + slot * PAIR
            for cg in T.serial(batches):
                for e in T.unroll(2 * PAIR * COLS):
                    acc[e] = 0.0
                for m in T.unroll(SEGS):
                    # All columns' vectors first: sixteen independent chains per channel.
                    for j in T.unroll(COLS):
                        for v in T.vectorized(VEC):
                            rv[j * VEC + v] = rs[
                                roff + col0 + cg * COLS + j, m * GROUP * VEC + part * VEC + v
                            ]
                    for v in T.unroll(VEC):
                        for j in T.unroll(COLS):
                            for p in T.unroll(PAIR):
                                acc[p * 2 * COLS + j * 2] = _fma(
                                    lk[p * CH + m * VEC + v],
                                    rv[j * VEC + v],
                                    acc[p * 2 * COLS + j * 2],
                                )
                                acc[p * 2 * COLS + j * 2 + 1] = _fma(
                                    lq[p * CH + m * VEC + v],
                                    rv[j * VEC + v],
                                    acc[p * 2 * COLS + j * 2 + 1],
                                )
                _group_sum16(acc, part)
                # Lane part now holds row part // 4, column part % 4 of the batch.
                if row0 + part // COLS < rows:
                    kk[0, bos + row0 + part // COLS, ih, col0 + cg * COLS + part % COLS] = acc[0]
                    qk[0, bos + row0 + part // COLS, ih, col0 + cg * COLS + part % COLS] = acc[1]

        @T.macro
        def row_block(q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, rb):
            right_factors(k, gc, rs, bos, ih, rb, 0)
            left_factors(q, k, gc, bufs, bos, rows, ih, tx, rb)
            T.sync_threads()
            col0 = (tx // 64) * (rb // splits)
            columns(kk, qk, rs, bufs, bos, rows, ih, tx, rb, 0, col0, rb // splits // COLS)
            T.sync_threads()

        @T.macro
        def pair_block(q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, rb, roff):
            left_factors(q, k, gc, bufs, bos, rows, ih, tx, rb)
            columns(kk, qk, rs, bufs, bos, rows, ih, tx, rb, roff, 0, rb // COLS)

        @T.prim_func
        def main(
            q: T.Tensor([1, total_tokens, HV // G, K], dtype),
            k: T.Tensor([1, total_tokens, HV // G, K], dtype),
            gc: T.Tensor([1, total_tokens, HV, K], accum),
            cu: T.Tensor([num_seqs + 1], "int64"),
            kk: T.Tensor([1, total_tokens, HV, BT], accum),
            qk: T.Tensor([1, total_tokens, HV, BT], accum),
        ):
            with T.Kernel(3 if mode == "split" else 1, chunks, HV, threads=threads) as (ib, ic, ih):
                tx = T.get_thread_binding()
                cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                # Right factors of the current row block, one row per earlier token, or of
                # all three (rows 0-15 block 1, 16-47 block 2, 48-95 block 3) at once.
                rs = T.alloc_shared([BT + 2 * BC if mode == "concurrent" else BT - BC, K], accum)
                tq = T.alloc_local([VEC], dtype)
                tk = T.alloc_local([VEC], dtype)
                tg = T.alloc_local([VEC], accum)
                tr = T.alloc_local([VEC], accum)
                lq = T.alloc_local([PAIR * CH], accum)
                lk = T.alloc_local([PAIR * CH], accum)
                rv = T.alloc_local([COLS * VEC], accum)
                acc = T.alloc_local([2 * PAIR * COLS], accum)
                bufs = (tq, tk, tg, tr, lq, lk, rv, acc)
                tiling.cumsum_offsets(cu, cum)
                tiling.decode(T.min(ic, cum[num_seqs] - 1), cum, lo, hi, seq, first)
                bos = T.cast(cu[seq[0]], "int32") + first[0]
                rows = T.if_then_else(
                    ic < cum[num_seqs], T.min(BT, T.cast(cu[seq[0] + 1], "int32") - bos), 0
                )
                if mode == "concurrent":
                    if rows > BC:
                        right_factors(k, gc, rs, bos, ih, BC, 0)
                    if rows > 2 * BC:
                        right_factors(k, gc, rs, bos, ih, 2 * BC, BC)
                    if rows > 3 * BC:
                        right_factors(k, gc, rs, bos, ih, 3 * BC, 3 * BC)
                    T.sync_threads()
                    # Twelve column batches a warp pair: block 3, or blocks 2 and 1.
                    if tx < 64:
                        if rows > 3 * BC:
                            pair_block(
                                q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, 3 * BC, 3 * BC
                            )
                    else:
                        if rows > 2 * BC:
                            pair_block(q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, 2 * BC, BC)
                        if rows > BC:
                            pair_block(q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, BC, 0)
                elif mode == "split":
                    # The heaviest row block's CTAs start first.
                    if ib == 0:
                        if rows > 3 * BC:
                            row_block(q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, 3 * BC)
                    elif ib == 1:
                        if rows > 2 * BC:
                            row_block(q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, 2 * BC)
                    elif rows > BC:
                        row_block(q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, BC)
                else:
                    if rows > BC:
                        row_block(q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, BC)
                    if rows > 2 * BC:
                        row_block(q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, 2 * BC)
                    if rows > 3 * BC:
                        row_block(q, k, gc, kk, qk, rs, bufs, bos, rows, ih, tx, 3 * BC)

        return main

    return build()


@functools.lru_cache(maxsize=64)
def _inverse_wy_program(
    total_tokens,
    num_seqs,
    heads,
    dim_k,
    dim_v,
    dtype,
    beta_sigmoid,
    allow_neg_eigval,
    grid_chunks=None,
    qscale=None,
    key_heads=None,
):
    """The stable inverse, the stored intra-chunk attention and the WY vectors of every chunk.

    It reads both FP32 Gram matrices and forms the inverse in shared memory with the
    arithmetic of the blockwise program in kda/inverse.py, bitwise: forward substitution
    of the four diagonal blocks, then two doubling levels of BF16 products, each
    elementwise pass folded into the store of the product before it. The inverse never
    leaves the CTA: it multiplies, in the activation dtype, into the step-scaled gated keys
    and values, and the gated query and key and the chunk decay come from the same reads
    of the gate prefix.
    """
    BT, BC, K, V, HV = CHUNK_SIZE, REFERENCE_BLOCK, dim_k, dim_v, heads
    G = HV // (key_heads or HV)
    # Two warpgroups held to three CTAs (85 registers) per SM: the inverse is a chain of
    # small dependent steps and each chunk streams about 170 KB, so resident CTAs hide
    # both, and the registers left over keep a pass's key-side loads in flight.
    threads = 256
    blocks_per_sm = 3
    accum = "float32"
    # Eight lanes a thread (16 B loads and stores) over a row of the Gram matrices and of
    # the key side, written out because T.Parallel keeps the causally masked Gram loads
    # scalar and vectorizes the key side only to the FP32 gate's four lanes.
    lanes = 8
    gram_row_threads = BT // lanes
    gram_rows_per_pass = threads // gram_row_threads
    row_threads = K // lanes
    rows_per_pass = threads // row_threads
    tiling = GroupTiling(num_seqs, BT)
    chunks = grid_chunks or tiling.tile_upper_bound(total_tokens)

    @tilelang.jit(
        out_idx=[-6, -5, -4, -3, -2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def build():
        @T.prim_func
        def main(
            q: T.Tensor([1, total_tokens, HV // G, K], dtype),
            k: T.Tensor([1, total_tokens, HV // G, K], dtype),
            v: T.Tensor([1, total_tokens, HV, V], dtype),
            g: T.Tensor([1, total_tokens, HV, K], accum),
            beta: T.Tensor([1, total_tokens, HV], dtype),
            cu: T.Tensor([num_seqs + 1], "int64"),
            kk: T.Tensor([1, total_tokens, HV, BT], accum),
            qk: T.Tensor([1, total_tokens, HV, BT], accum),
            w: T.Tensor([1, total_tokens, HV, K], dtype),
            u: T.Tensor([1, total_tokens, HV, V], dtype),
            qg: T.Tensor([1, total_tokens, HV, K], dtype),
            kg: T.Tensor([1, total_tokens, HV, K], dtype),
            aqk: T.Tensor([1, total_tokens, HV, BT], dtype),
            # Shaped by the bound the state walk declares, whatever the grid.
            dec: T.Tensor([tiling.tile_upper_bound(total_tokens), HV, K], accum),
        ):
            with T.Kernel(chunks, HV, threads=threads) as (ic, ihv):
                T.annotate_min_blocks_per_sm(blocks_per_sm)
                tx = T.get_thread_binding()
                cum = T.alloc_shared([num_seqs + 1], "int32")
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                seq = T.alloc_local([1], "int32")
                first = T.alloc_local([1], "int32")
                bt_s = T.alloc_shared([BT], accum)
                last_s = T.alloc_shared([K], accum)
                m_s = T.alloc_shared([BT, BT], dtype)
                p_s = T.alloc_shared([BT, BT], dtype)
                # The diagonal blocks of the coefficient matrix, held apart so the
                # product operand is masked to its strictly lower blocks on load.
                d_s = T.alloc_shared([BT, BC], accum)
                bk_s = T.alloc_shared([BT, K], dtype)
                bv_s = T.alloc_shared([BT, V], dtype)
                x = T.alloc_local([BC], accum)
                scratch = T.alloc_fragment([BT, BT], accum)
                acc = T.alloc_fragment([BT, K], accum)
                accv = T.alloc_fragment([BT, V], accum)
                gram = T.alloc_local([lanes], accum)
                score = T.alloc_local([lanes], accum)
                coeff = T.alloc_local([lanes], dtype)
                attention = T.alloc_local([lanes], dtype)
                gate = T.alloc_local([lanes], accum)
                key = T.alloc_local([lanes], dtype)
                query = T.alloc_local([lanes], dtype)
                scaled = T.alloc_local([lanes], dtype)
                gated_query = T.alloc_local([lanes], dtype)
                gated_key = T.alloc_local([lanes], dtype)

                tiling.cumsum_offsets(cu, cum)
                live = cum[num_seqs]
                tiling.decode(T.min(ic, live - 1), cum, lo, hi, seq, first)
                bos = T.cast(cu[seq[0]], "int32") + first[0]
                rows = T.if_then_else(
                    ic < live, T.min(BT, T.cast(cu[seq[0] + 1], "int32") - bos), 0
                )
                last = T.max(rows - 1, 0)
                for i in T.Parallel(BT):
                    bt_s[i] = T.if_then_else(
                        i < rows,
                        T.cast(
                            step_size(beta[0, bos + i, ihv], beta_sigmoid, allow_neg_eigval), accum
                        ),
                        T.cast(0, accum),
                    )
                for j in T.Parallel(K):
                    last_s[j] = T.if_then_else(rows > 0, g[0, bos + last, ihv, j], 0.0)
                T.sync_threads()
                # Blocks above the diagonal are never read.
                c0 = (tx % gram_row_threads) * lanes
                for step in T.unroll(BT // gram_rows_per_pass):
                    i = step * gram_rows_per_pass + tx // gram_row_threads
                    if (i < rows) & (c0 // BC <= i // BC):
                        for e in T.vectorized(lanes):
                            gram[e] = kk[0, bos + i, ihv, c0 + e]
                        for e in T.vectorized(lanes):
                            score[e] = qk[0, bos + i, ihv, c0 + e]
                    else:
                        for e in T.vectorized(lanes):
                            gram[e] = 0.0
                            score[e] = 0.0
                    for e in T.vectorized(lanes):
                        coeff[e] = T.if_then_else(
                            i // BC > (c0 + e) // BC,
                            T.cast(-gram[e] * bt_s[i], dtype),
                            T.cast(0, dtype),
                        )
                        attention[e] = T.cast(T.if_then_else(i >= c0 + e, score[e], 0.0), dtype)
                    for e in T.vectorized(lanes):
                        m_s[i, c0 + e] = coeff[e]
                    for e in T.vectorized(lanes):
                        p_s[i, c0 + e] = T.cast(0, dtype)
                    if c0 // BC == i // BC:
                        for e in T.serial(lanes):
                            d_s[i, (c0 + e) % BC] = T.cast(
                                T.cast(T.if_then_else(i > c0 + e, -gram[e] * bt_s[i], 0.0), dtype),
                                accum,
                            )
                    if i < rows:
                        for e in T.vectorized(lanes):
                            aqk[0, bos + i, ihv, c0 + e] = attention[e]
                T.sync_threads()
                # Forward substitution, one thread per column of a diagonal block.
                if tx < BT:
                    for row in T.unroll(BC):
                        value = T.alloc_var(accum)
                        value = T.if_then_else(row == tx % BC, T.float32(1.0), T.float32(0.0))
                        for prior in T.unroll(row):
                            value += d_s[(tx // BC) * BC + row, prior] * x[prior]
                        x[row] = value
                        p_s[(tx // BC) * BC + row, tx] = T.cast(value, dtype)
                T.sync_threads()
                T.clear(scratch)
                T.gemm(m_s, p_s, scratch)
                T.sync_threads()
                # Level 0: the coefficient gains its unit diagonal as it is stored.
                for i, j in T.Parallel(BT, BT):
                    m_s[i, j] = T.cast(scratch[i, j], dtype) + T.cast(
                        T.if_then_else(i == j, 1.0, 0.0), dtype
                    )
                T.sync_threads()
                T.clear(scratch)
                T.gemm(p_s, m_s, scratch)
                T.sync_threads()
                # Two loops: one loop with both stores puts a barrier in every iteration.
                for i, j in T.Parallel(BT, BT):
                    p_s[i, j] = T.cast(scratch[i, j], dtype)
                for i, j in T.Parallel(BT, BT):
                    m_s[i, j] = T.if_then_else(i // BC > j // BC, m_s[i, j], T.cast(0, dtype))
                T.sync_threads()
                T.clear(scratch)
                T.gemm(m_s, m_s, scratch)
                T.sync_threads()
                for i, j in T.Parallel(BT, BT):
                    m_s[i, j] = T.cast(scratch[i, j], dtype) + T.cast(
                        T.if_then_else(i == j, 1.0, 0.0), dtype
                    )
                T.sync_threads()
                T.clear(scratch)
                T.gemm(p_s, m_s, scratch)
                T.sync_threads()
                for i, j in T.Parallel(BT, BT):
                    p_s[i, j] = T.cast(scratch[i, j], dtype)

                for j in T.Parallel(K):
                    dec[ic, ihv, j] = T.exp2(last_s[j])
                j0 = (tx % row_threads) * lanes
                for step in T.unroll(BT // rows_per_pass):
                    i = step * rows_per_pass + tx // row_threads
                    # Loads sit at a clamped row outside any branch and are masked after:
                    # in a guarded branch they wait for the stores of the pass before.
                    src = T.min(bos + i, total_tokens - 1)
                    for e in T.vectorized(lanes):
                        gate[e] = g[0, src, ihv, j0 + e]
                    for e in T.vectorized(lanes):
                        key[e] = k[0, src, _key_head(ihv, G), j0 + e]
                    for e in T.vectorized(lanes):
                        query[e] = q[0, src, _key_head(ihv, G), j0 + e]
                    for e in T.vectorized(lanes):
                        gate[e] = T.if_then_else(i < rows, gate[e], 0.0)
                        key[e] = T.if_then_else(i < rows, key[e], T.cast(0, dtype))
                        query[e] = T.if_then_else(i < rows, query[e], T.cast(0, dtype))
                    for e in T.vectorized(lanes):
                        scaled[e] = T.cast(T.cast(key[e], accum) * T.exp2(gate[e]) * bt_s[i], dtype)
                        gated_query[e] = T.cast(
                            _scaled(T.cast(query[e], accum), qscale) * T.exp2(gate[e]), dtype
                        )
                        gated_key[e] = T.cast(
                            T.cast(key[e], accum) * T.exp2(last_s[j0 + e] - gate[e]), dtype
                        )
                    for e in T.vectorized(lanes):
                        bk_s[i, j0 + e] = scaled[e]
                    if i < rows:
                        for e in T.vectorized(lanes):
                            qg[0, bos + i, ihv, j0 + e] = gated_query[e]
                        for e in T.vectorized(lanes):
                            kg[0, bos + i, ihv, j0 + e] = gated_key[e]
                for i, j in T.Parallel(BT, V):
                    bv_s[i, j] = T.cast(
                        T.if_then_else(i < rows, T.cast(v[0, bos + i, ihv, j], accum), 0.0)
                        * bt_s[i],
                        dtype,
                    )
                T.sync_threads()
                # Each product goes back through its operand tile so the rows leave in
                # 16 B stores rather than scattered from the accumulator fragment.
                T.clear(acc)
                T.gemm(p_s, bk_s, acc)
                T.sync_threads()
                T.copy(acc, bk_s)
                T.sync_threads()
                for i, j in T.Parallel(BT, K):
                    if i < rows:
                        w[0, bos + i, ihv, j] = bk_s[i, j]
                T.clear(accv)
                T.gemm(p_s, bv_s, accv)
                T.sync_threads()
                T.copy(accv, bv_s)
                T.sync_threads()
                for i, j in T.Parallel(BT, V):
                    if i < rows:
                        u[0, bos + i, ihv, j] = bv_s[i, j]

        return main

    return build()


@functools.lru_cache(maxsize=64)
def split_prepare_program(*parameters, **options):
    """Build the split preparation with the ordinary preparation call interface.

    ``grid_chunks`` is the exact chunk count of a dense call. Without it every launch
    takes the bound of one partial chunk per sequence, and the chunk CTAs past the live
    chunks find no rows.
    """
    H, heads, dim_k, dim_v, chunk_size, dtype, scale, l2, tokens, sequences = parameters[:10]
    # The lane mappings below cover 64 or 128 channels in 64-token chunks.
    assert dim_k in (64, 128) and chunk_size == CHUNK_SIZE
    beta_sigmoid = options.get("beta_sigmoid", False)
    allow_neg_eigval = options.get("allow_neg_eigval", False)
    grid_chunks = options.get("grid_chunks")
    raw = options.get("gate_in_kernel", True)
    if raw:
        gate = _gate_program(
            tokens,
            sequences,
            heads,
            dim_k,
            dtype,
            options["has_dt_bias"],
            options["lower_bound"],
            grid_chunks,
        )
    else:
        gate = _gate_program(
            tokens, sequences, heads, dim_k, dtype, False, None, grid_chunks, raw=False
        )
    # q and k keep their H heads; each value head reads the one its group shares.
    # Unnormalized rows are read as given, the query scaled in FP32 where it is read.
    qscale = None if l2 else scale
    if not l2:

        def norm(q, k):
            return q, k

    elif tokens * H % _STAGED_NORM_ROWS == 0:
        staged = _staged_norm_program(tokens * H, dim_k, dtype, scale)

        def norm(q, k):
            qn, kn = staged(q.reshape(tokens * H, dim_k), k.reshape(tokens * H, dim_k))
            return qn.view(1, tokens, H, dim_k), kn.view(1, tokens, H, dim_k)

    else:
        norm = _norm_program(tokens, H, dim_k, dtype, scale)
    diagonal = _diagonal_program(tokens, sequences, heads, dim_k, dtype, grid_chunks, qscale, H)
    offdiagonal = _offdiagonal_program(
        tokens, sequences, heads, dim_k, dtype, grid_chunks, qscale, H
    )
    inverse_wy = _inverse_wy_program(
        tokens,
        sequences,
        heads,
        dim_k,
        dim_v,
        dtype,
        beta_sigmoid,
        allow_neg_eigval,
        grid_chunks,
        qscale,
        H,
    )

    def run(q, k, v, g, beta, cu, a_log=None, dt_bias=None):
        qn, kn = norm(q, k)
        gc = gate(g, cu, a_log, dt_bias) if raw else gate(g, cu)
        kk, qk = diagonal(qn, kn, gc, cu)
        offdiagonal(qn, kn, gc, cu, kk, qk)
        return inverse_wy(qn, kn, v, gc, beta, cu, kk, qk)

    return run
