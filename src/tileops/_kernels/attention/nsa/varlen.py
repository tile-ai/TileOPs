import math
from collections.abc import Callable

import tilelang
import torch
from tilelang import language as T
from tilelang.layout import make_swizzled_layout

from tileops._kernels.attention.call_spec import NSACall, NSAFwdInterface
from tileops._kernels.attention.online_softmax import make_online_softmax, make_rescale
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import LOG2E, MAX_GRID_YZ_BLOCKS
from tileops._kernels.kernel_base import Entry, Kernel


@target_cache(maxsize=32)
def _nsa_fwd_varlen_kernel(
    batch: int,
    heads: int,
    c_seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    if scale is None:
        scale = (1.0 / dim) ** 0.5
    scale *= LOG2E

    @tilelang.jit(
        out_idx=[3],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _nsa_fwd_varlen_func(threads: int):
        head_kv = heads // groups
        q_shape = [c_seq_len, heads, dim]
        kv_shape = [c_seq_len, head_kv, dim]
        o_slc_shape = [c_seq_len, heads, dim]
        block_indices_shape = [c_seq_len, head_kv, selected_blocks]
        block_counts_shape = [c_seq_len, head_kv]
        offsets_shape = [batch + 1]
        token_indices_shape = [c_seq_len, 2]
        block_indices_dtype = T.int32
        block_counts_dtype = T.int32
        offsets_dtype = T.int32
        token_indices_dtype = T.int32

        block_s = block_size
        block_t = min(128, tilelang.math.next_power_of_2(dim))

        nk = tilelang.cdiv(dim, block_t)
        nv = tilelang.cdiv(dim, block_t)
        if nk != 1:
            raise ValueError("The key dimension can not be larger than 128")

        g = groups
        bs = block_s
        bk = bv = block_t
        num_stages = 4

        online_softmax = make_online_softmax(scale, accum_dtype, g, bs)
        rescale = make_rescale(g, bv)

        @T.prim_func
        def _nsa_fwd_varlen_main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(kv_shape, dtype),
            v: T.Tensor(kv_shape, dtype),
            o_slc: T.Tensor(o_slc_shape, dtype),
            block_indices: T.Tensor(block_indices_shape, block_indices_dtype),
            block_counts: T.Tensor(block_counts_shape, block_counts_dtype),
            offsets: T.Tensor(offsets_shape, offsets_dtype),
            token_indices: T.Tensor(token_indices_shape, token_indices_dtype),
        ):
            with T.Kernel(c_seq_len, nv, head_kv, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared([g, bk], dtype)
                k_shared = T.alloc_shared([bs, bk], dtype)
                v_shared = T.alloc_shared([bs, bv], dtype)
                o_shared = T.alloc_shared([g, bv], dtype)

                acc_s = T.alloc_fragment([g, bs], accum_dtype)
                acc_s_cast = T.alloc_fragment([g, bs], dtype)
                acc_o = T.alloc_fragment([g, bv], accum_dtype)
                scores_max = T.alloc_fragment([g], accum_dtype)
                scores_max_prev = T.alloc_fragment([g], accum_dtype)
                scores_scale = T.alloc_fragment([g], accum_dtype)
                scores_sum = T.alloc_fragment([g], accum_dtype)
                logsum = T.alloc_fragment([g], accum_dtype)

                i_c, i_v, i_bh = bx, by, bz
                _, i_h = i_bh // head_kv, i_bh % head_kv

                i_n, i_t = token_indices[i_c, 0], token_indices[i_c, 1]

                bos = offsets[i_n]
                # Causal: keys up to the token. Otherwise: up to the sequence end.
                limit = i_t + 1 if is_causal else offsets[i_n + 1] - bos

                ns = block_counts[bos + i_t, i_h]
                T.copy(q[bos + i_t, i_h * g : (i_h + 1) * g, :bk], q_shared)

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for i in T.Pipelined(ns, num_stages=num_stages):
                    i_s = block_indices[bos + i_t, i_h, i] * bs
                    if i_s < limit and i_s >= 0:
                        # Rows past the bound load as zeros: a non-finite value survives the mask.
                        if i_s + bs <= limit:
                            T.copy(k[bos + i_s : bos + i_s + bs, i_h, :bk], k_shared)
                            T.copy(v[bos + i_s : bos + i_s + bs, i_h, :bv], v_shared)
                        else:
                            for j, d in T.Parallel(bs, bk):
                                if (i_s + j < limit) & (d < dim):
                                    k_shared[j, d] = k[bos + i_s + j, i_h, d]
                                    v_shared[j, d] = v[bos + i_s + j, i_h, d]
                                else:
                                    k_shared[j, d] = T.cast(0, dtype)
                                    v_shared[j, d] = T.cast(0, dtype)

                        for i, j in T.Parallel(g, bs):
                            acc_s[i, j] = T.if_then_else(
                                i_s + j < limit, 0, -T.infinity(acc_s.dtype)
                            )

                        T.gemm(
                            q_shared,
                            k_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                        online_softmax(
                            acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                        )
                        T.copy(acc_s, acc_s_cast)

                        rescale(acc_o, scores_scale)

                        # V * softmax(Q * K)
                        T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                # A token no selected block gives a key outputs zeros.
                for i, j in T.Parallel(g, bv):
                    acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                T.copy(acc_o, o_shared)
                T.copy(
                    o_shared, o_slc[bos + i_t, i_h * g : (i_h + 1) * g, i_v * bv : (i_v + 1) * bv]
                )

        return _nsa_fwd_varlen_main

    return _nsa_fwd_varlen_func


@target_cache(maxsize=32)
def _nsa_fwd_varlen_tma_kernel(
    batch: int,
    heads: int,
    c_seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    """One warp per (token, KV head) on SM90: tiles move by TMA, Q stays in registers.

    With one selected block the ring is a single stage. TMA zero-fills rows past the packed
    tensor; keys past the token's limit are masked.
    """
    if scale is None:
        scale = (1.0 / dim) ** 0.5
    scale *= LOG2E
    head_kv = heads // groups
    g = groups
    bs = block_size
    stages = min(2, selected_blocks)

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _nsa_fwd_varlen_tma_func(threads: int):
        @T.prim_func
        def _nsa_fwd_varlen_tma_main(
            q: T.Tensor([c_seq_len, heads, dim], dtype),
            k: T.Tensor([c_seq_len, head_kv, dim], dtype),
            v: T.Tensor([c_seq_len, head_kv, dim], dtype),
            block_indices: T.Tensor([c_seq_len, head_kv, selected_blocks], T.int32),
            block_counts: T.Tensor([c_seq_len, head_kv], T.int32),
            offsets: T.Tensor([batch + 1], T.int32),
            token_indices: T.Tensor([c_seq_len, 2], T.int32),
            o_slc: T.Tensor([c_seq_len, heads, dim], dtype),
        ):
            with T.Kernel(c_seq_len, head_kv, threads=threads) as (i_c, i_h):
                q_shared = T.alloc_shared([g, dim], dtype)
                k_shared = T.alloc_shared([stages, bs, dim], dtype)
                v_shared = T.alloc_shared([stages, bs, dim], dtype)
                T.annotate_layout(
                    {
                        q_shared: make_swizzled_layout(q_shared),
                        k_shared: make_swizzled_layout(k_shared),
                        v_shared: make_swizzled_layout(v_shared),
                    }
                )
                q_ready = T.alloc_barrier([threads])
                kv_ready = T.alloc_barrier([threads] * stages)

                q_local = T.alloc_fragment([g, dim], dtype)
                acc_s = T.alloc_fragment([g, bs], accum_dtype)
                acc_s_cast = T.alloc_fragment([g, bs], dtype)
                acc_o = T.alloc_fragment([g, dim], accum_dtype)
                scores_max = T.alloc_fragment([g], accum_dtype)
                scores_max_prev = T.alloc_fragment([g], accum_dtype)
                scores_scale = T.alloc_fragment([g], accum_dtype)
                scores_sum = T.alloc_fragment([g], accum_dtype)
                logsum = T.alloc_fragment([g], accum_dtype)
                inv_logsum = T.alloc_fragment([g], accum_dtype)
                # Live slots not yet loaded, live slots not yet attended, and their count.
                to_load = T.alloc_local([1], T.int32)
                to_attend = T.alloc_local([1], T.int32)
                n_live = T.alloc_local([1], T.int32)

                i_t = token_indices[i_c, 1]
                # A packed token sits at its request's start plus its position.
                bos = i_c - i_t
                # Causal: keys up to the token. Otherwise: up to the request end.
                limit = i_t + 1 if is_causal else offsets[token_indices[i_c, 0] + 1] - bos
                ns = block_counts[i_c, i_h]

                T.tma_copy(q[i_c, i_h * g : (i_h + 1) * g, :], q_shared, barrier=q_ready[0])
                T.mbarrier_arrive(q_ready[0])

                to_load[0] = 0
                n_live[0] = 0
                for i in T.unroll(selected_blocks):
                    i_s = block_indices[i_c, i_h, i] * bs
                    if i < ns and i_s >= 0 and i_s < limit:
                        to_load[0] = to_load[0] | (1 << i)
                        n_live[0] = n_live[0] + 1
                to_attend[0] = to_load[0]

                for p in T.unroll(stages - 1):
                    if to_load[0] != 0:
                        row = bos + block_indices[i_c, i_h, T.__ffs(to_load[0]) - 1] * bs
                        to_load[0] = to_load[0] & (to_load[0] - 1)
                        T.tma_copy(
                            k[row : row + bs, i_h, :], k_shared[p, :, :], barrier=kv_ready[p]
                        )
                        T.tma_copy(
                            v[row : row + bs, i_h, :], v_shared[p, :, :], barrier=kv_ready[p]
                        )
                        T.mbarrier_arrive(kv_ready[p])

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.mbarrier_wait_parity(q_ready[0], 0)
                T.copy(q_shared, q_local)

                for j in T.serial(n_live[0]):
                    if to_load[0] != 0:
                        st_next = (j + stages - 1) % stages
                        row = bos + block_indices[i_c, i_h, T.__ffs(to_load[0]) - 1] * bs
                        to_load[0] = to_load[0] & (to_load[0] - 1)
                        # The stage was last read by this warp's own MMAs; order those
                        # generic-proxy reads before the TMA write lands.
                        T.fence_proxy_async()
                        T.tma_copy(
                            k[row : row + bs, i_h, :],
                            k_shared[st_next, :, :],
                            barrier=kv_ready[st_next],
                        )
                        T.tma_copy(
                            v[row : row + bs, i_h, :],
                            v_shared[st_next, :, :],
                            barrier=kv_ready[st_next],
                        )
                        T.mbarrier_arrive(kv_ready[st_next])
                    st = j % stages
                    i_s = block_indices[i_c, i_h, T.__ffs(to_attend[0]) - 1] * bs
                    to_attend[0] = to_attend[0] & (to_attend[0] - 1)
                    T.mbarrier_wait_parity(kv_ready[st], (j // stages) % 2)

                    T.clear(acc_s)
                    T.gemm(
                        q_local,
                        k_shared[st, :, :],
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, s in T.Parallel(g, bs):
                        acc_s[i, s] = T.if_then_else(
                            i_s + s < limit, acc_s[i, s] * scale, -T.infinity(accum_dtype)
                        )
                    # A live block holds the key at its start, so every row's max is finite.
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(g):
                        scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                    for i, s in T.Parallel(g, bs):
                        acc_s[i, s] = T.exp2(acc_s[i, s] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(g):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, d in T.Parallel(g, dim):
                        acc_o[i, d] *= scores_scale[i]
                    T.gemm(acc_s_cast, v_shared[st, :, :], acc_o, policy=T.GemmWarpPolicy.FullRow)

                # A token no selected block gives a key outputs zeros.
                for i in T.Parallel(g):
                    inv_logsum[i] = T.if_then_else(logsum[i] > 0, 1.0 / logsum[i], 0)
                for i, d in T.Parallel(g, dim):
                    acc_o[i, d] = acc_o[i, d] * inv_logsum[i]
                T.copy(acc_o, q_shared)
                T.copy(q_shared, o_slc[i_c, i_h * g : (i_h + 1) * g, :])

        return _nsa_fwd_varlen_tma_main

    return _nsa_fwd_varlen_tma_func


@target_cache(maxsize=32)
def _nsa_fwd_varlen_gather_kernel(
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    separate_keys: bool,
    block_size: int,
    selected_blocks: int,
    splits: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    """One CTA per (token, KV head, key split): selected rows gathered into a K/V tile.

    With one split the CTA writes the output; with several, each split writes its normalized
    output, its row max and its softmax denominator, and a second kernel merges them. Rows the
    selection does not reach load as zeros: a non-finite value would survive a zero weight.
    """
    groups = heads // heads_kv
    rows = -(-groups // 16) * 16

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _nsa_fwd_varlen_gather_func(block_n: int, num_stages: int, threads: int):
        c_seq_len = T.dynamic("c_seq_len")
        n_keys = T.dynamic("n_keys")
        batch = T.dynamic("batch")
        # A step covers whole blocks; a block longer than the tile takes a step to itself.
        blocks_per_step = max(1, block_n // block_size)
        step_n = blocks_per_step * block_size
        chunks = dim // 8  # 16-byte copies per key row
        split_slots = -(-selected_blocks // (blocks_per_step * splits)) * blocks_per_step
        rescale = make_rescale(rows, dim)

        def scaled_gap(score, peak, root):
            """``(score - peak) * scale`` in log2 units, for score <= peak. The halves are
            subtracted, so two finite scores of opposite sign cannot overflow, and equal
            scores still cancel exactly. *root* is the program's runtime argument
            ``sqrt(2 * scale * log2(e))``, applied twice: a finite float32 for every
            admitted scale from below float32's smallest normal to its largest value,
            where the factor itself would flush to zero or overflow."""
            return (score * 0.5 - peak * 0.5) * root * root

        def key_of(block_indices, i_c, i_h, slot, r, limit):
            """The in-request key that row *r* of a step starting at *slot* holds. A block id
            is clamped to the first block past *limit* before it is scaled, so a padding id
            stays past the limit instead of passing the int32 range."""
            block = block_indices[i_c, i_h, T.min(slot + r // block_size, selected_blocks - 1)]
            return T.min(block, T.ceildiv(limit, block_size)) * block_size + r % block_size

        def visible(block_indices, i_c, i_h, slot, r, end, limit):
            """Whether that key is selected and visible: its slot is below the token's count,
            and the key lies in [0, limit)."""
            key = key_of(block_indices, i_c, i_h, slot, r, limit)
            return (slot + r // block_size < end) & (key >= r % block_size) & (key < limit)

        @T.macro
        def issue(
            k,
            v,
            block_indices,
            k_shared,
            v_shared,
            i_c,
            i_h,
            tx,
            first,
            end,
            limit,
            bos_k,
            step,
            stage,
        ):
            """Start copying step *step*'s key and value rows into *stage*; a row the
            selection does not reach lands as zeros."""
            # Each 16-byte copy moves 8 elements of one key row; a warp copies whole rows.
            for it in T.unroll(step_n * chunks // threads):
                c = it * threads + tx
                r = c // chunks
                col = c % chunks * 8
                slot = first + step * blocks_per_step
                live = visible(block_indices, i_c, i_h, slot, r, end, limit)
                row = T.if_then_else(
                    live, bos_k + key_of(block_indices, i_c, i_h, slot, r, limit), 0
                )
                T.ptx_cp_async(
                    T.access_ptr(k_shared[stage, r, col], "w", 8),
                    T.access_ptr(k[row, i_h, col], "r", 8),
                    8,
                    live,
                )
                T.ptx_cp_async(
                    T.access_ptr(v_shared[stage, r, col], "w", 8),
                    T.access_ptr(v[row, i_h, col], "r", 8),
                    8,
                    live,
                )

        @T.macro
        def attend(
            q,
            k,
            v,
            block_indices,
            block_counts,
            offsets,
            token_indices,
            offsets_k,
            scale_root,
            i_c,
            i_h,
            i_s,
            acc_o,
            logsum,
            scores_max,
        ):
            q_shared = T.alloc_shared([rows, dim], dtype)
            k_shared = T.alloc_shared([num_stages, step_n, dim], dtype)
            v_shared = T.alloc_shared([num_stages, step_n, dim], dtype)
            p_shared = T.alloc_shared([rows, step_n], dtype)
            T.annotate_layout(
                {
                    k_shared: make_swizzled_layout(k_shared),
                    v_shared: make_swizzled_layout(v_shared),
                }
            )
            q_local = T.alloc_fragment([rows, dim], dtype)
            acc_s = T.alloc_fragment([rows, step_n], accum_dtype)
            seen = T.alloc_fragment([rows, step_n], accum_dtype)
            scores_max_prev = T.alloc_fragment([rows], accum_dtype)
            scores_scale = T.alloc_fragment([rows], accum_dtype)
            scores_sum = T.alloc_fragment([rows], accum_dtype)

            i_n = token_indices[i_c, 0]
            bos = offsets[i_n]
            if separate_keys:
                bos_k = offsets_k[i_n]
                n_k = offsets_k[i_n + 1] - bos_k
            else:
                bos_k = bos
                n_k = offsets[i_n + 1] - bos
            # A request's queries are the last of its keys. Causal: keys up to the token;
            # otherwise up to the request end.
            pos = token_indices[i_c, 1] + n_k - (offsets[i_n + 1] - bos)
            limit = pos + 1 if is_causal else n_k
            first = i_s * split_slots
            live_slots = T.min(block_counts[i_c, i_h], selected_blocks) - first
            n_steps = T.max(
                0, T.min(T.ceildiv(live_slots, blocks_per_step), split_slots // blocks_per_step)
            )

            for i, d in T.Parallel(rows, dim):
                q_shared[i, d] = T.if_then_else(
                    i < groups, q[i_c, i_h * groups + T.min(i, groups - 1), d], T.cast(0, dtype)
                )
            # Q stays in registers: every step's scores read it.
            T.copy(q_shared, q_local)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, T.min_value(accum_dtype))

            tx = T.get_thread_binding()
            end = first + live_slots

            for p in T.unroll(num_stages - 1):
                if p < n_steps:
                    issue(
                        k,
                        v,
                        block_indices,
                        k_shared,
                        v_shared,
                        i_c,
                        i_h,
                        tx,
                        first,
                        end,
                        limit,
                        bos_k,
                        p,
                        p,
                    )
                T.ptx_commit_group()
            for j in T.serial(n_steps):
                if j + num_stages - 1 < n_steps:
                    issue(
                        k,
                        v,
                        block_indices,
                        k_shared,
                        v_shared,
                        i_c,
                        i_h,
                        tx,
                        first,
                        end,
                        limit,
                        bos_k,
                        j + num_stages - 1,
                        (j + num_stages - 1) % num_stages,
                    )
                T.ptx_commit_group()
                T.ptx_wait_group(num_stages - 1)
                T.sync_threads()
                st = j % num_stages
                slot0 = first + j * blocks_per_step
                T.clear(acc_s)
                T.gemm(
                    q_local,
                    k_shared[st, :, :],
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                # The online softmax over unscaled scores: the max is taken first and only
                # a score's distance below it is scaled (scaled_gap), so equal scores cancel
                # exactly and no finite score overflows. A key the selection does not reach scores the
                # lowest finite value, which no reached key's max can fall below, and weighs
                # exactly zero.
                for i, r in T.Parallel(rows, step_n):
                    seen[i, r] = T.if_then_else(
                        visible(block_indices, i_c, i_h, slot0, r, end, limit), 1, 0
                    )
                    acc_s[i, r] = T.if_then_else(
                        seen[i, r] > 0, acc_s[i, r], T.min_value(accum_dtype)
                    )
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(rows):
                    scores_scale[i] = T.exp2(
                        scaled_gap(scores_max_prev[i], scores_max[i], scale_root)
                    )
                for i, r in T.Parallel(rows, step_n):
                    acc_s[i, r] = T.if_then_else(
                        seen[i, r] > 0,
                        T.exp2(scaled_gap(acc_s[i, r], scores_max[i], scale_root)),
                        0,
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(rows):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, p_shared)
                rescale(acc_o, scores_scale)
                T.gemm(p_shared, v_shared[st, :, :], acc_o, policy=T.GemmWarpPolicy.FullCol)
                # The next issue overwrites this stage.
                T.sync_threads()

        if splits == 1:

            @T.prim_func
            def _nsa_fwd_varlen_gather_main(
                q: T.Tensor([c_seq_len, heads, dim], dtype),
                k: T.Tensor([n_keys, heads_kv, dim], dtype),
                v: T.Tensor([n_keys, heads_kv, dim], dtype),
                block_indices: T.Tensor([c_seq_len, heads_kv, selected_blocks], T.int32),
                block_counts: T.Tensor([c_seq_len, heads_kv], T.int32),
                offsets: T.Tensor([batch + 1], T.int32),
                token_indices: T.Tensor([c_seq_len, 2], T.int32),
                offsets_k: T.Tensor([batch + 1], T.int32),
                scale_root: T.float32,
                o: T.Tensor([c_seq_len, heads, dim], dtype),
            ):
                # Tokens on grid x: y and z hold at most MAX_GRID_YZ_BLOCKS blocks.
                with T.Kernel(c_seq_len, heads_kv, threads=threads) as (i_c, i_h):
                    acc_o = T.alloc_fragment([rows, dim], accum_dtype)
                    logsum = T.alloc_fragment([rows], accum_dtype)
                    scores_max = T.alloc_fragment([rows], accum_dtype)
                    attend(
                        q,
                        k,
                        v,
                        block_indices,
                        block_counts,
                        offsets,
                        token_indices,
                        offsets_k,
                        scale_root,
                        i_c,
                        i_h,
                        0,
                        acc_o,
                        logsum,
                        scores_max,
                    )
                    # A token no selected key reaches outputs zeros.
                    for i, d in T.Parallel(rows, dim):
                        if i < groups:
                            o[i_c, i_h * groups + i, d] = T.if_then_else(
                                logsum[i] > 0, acc_o[i, d] / logsum[i], 0
                            )

            return _nsa_fwd_varlen_gather_main

        @T.prim_func
        def _nsa_fwd_varlen_gather_split(
            q: T.Tensor([c_seq_len, heads, dim], dtype),
            k: T.Tensor([n_keys, heads_kv, dim], dtype),
            v: T.Tensor([n_keys, heads_kv, dim], dtype),
            block_indices: T.Tensor([c_seq_len, heads_kv, selected_blocks], T.int32),
            block_counts: T.Tensor([c_seq_len, heads_kv], T.int32),
            offsets: T.Tensor([batch + 1], T.int32),
            token_indices: T.Tensor([c_seq_len, 2], T.int32),
            offsets_k: T.Tensor([batch + 1], T.int32),
            scale_root: T.float32,
            max_part: T.Tensor([splits, c_seq_len, heads], accum_dtype),
            sum_part: T.Tensor([splits, c_seq_len, heads], accum_dtype),
            o_part: T.Tensor([splits, c_seq_len, heads, dim], accum_dtype),
            o: T.Tensor([c_seq_len, heads, dim], dtype),
        ):
            with T.Kernel(c_seq_len, heads_kv, splits, threads=threads) as (i_c, i_h, i_s):
                acc_o = T.alloc_fragment([rows, dim], accum_dtype)
                logsum = T.alloc_fragment([rows], accum_dtype)
                scores_max = T.alloc_fragment([rows], accum_dtype)
                attend(
                    q,
                    k,
                    v,
                    block_indices,
                    block_counts,
                    offsets,
                    token_indices,
                    offsets_k,
                    scale_root,
                    i_c,
                    i_h,
                    i_s,
                    acc_o,
                    logsum,
                    scores_max,
                )
                for i, d in T.Parallel(rows, dim):
                    if i < groups:
                        o_part[i_s, i_c, i_h * groups + i, d] = T.if_then_else(
                            logsum[i] > 0, acc_o[i, d] / logsum[i], 0
                        )
                # Each split's row max and softmax denominator, kept apart: at a large max
                # the logarithm of the denominator would round away in their sum. A split
                # no key reached has a zero denominator and weighs nothing in the merge.
                for i in T.Parallel(rows):
                    if i < groups:
                        max_part[i_s, i_c, i_h * groups + i] = scores_max[i]
                        sum_part[i_s, i_c, i_h * groups + i] = logsum[i]

            # One thread per two value dims of a 256-wide head; narrower heads mask the rest.
            combine_threads = 128
            # A one-dimensional fragment needs every thread to own the same number of its
            # elements, so the accumulator spans the head dim rounded up to the thread count.
            combine_width = -(-dim // combine_threads) * combine_threads

            with T.Kernel(c_seq_len, heads, threads=combine_threads) as (i_c, i_hq):
                row_max = T.alloc_local([1], accum_dtype)
                total = T.alloc_local([1], accum_dtype)
                weight = T.alloc_local([splits], accum_dtype)
                o_acc = T.alloc_fragment([combine_width], accum_dtype)
                row_max[0] = T.min_value(accum_dtype)
                # Unrolled, so the split statistics load together rather than one by one.
                for s in T.unroll(splits):
                    weight[s] = sum_part[s, i_c, i_hq]
                    if weight[s] > 0:
                        row_max[0] = T.max(row_max[0], max_part[s, i_c, i_hq])
                # A split's weight is its denominator in the frame of the largest max.
                total[0] = 0
                for s in T.unroll(splits):
                    weight[s] = T.if_then_else(
                        weight[s] > 0,
                        weight[s]
                        * T.exp2(scaled_gap(max_part[s, i_c, i_hq], row_max[0], scale_root)),
                        0,
                    )
                    total[0] += weight[s]
                T.clear(o_acc)
                for s in T.serial(splits):
                    for d in T.Parallel(combine_width):
                        if d < dim:
                            o_acc[d] += o_part[s, i_c, i_hq, d] * weight[s]
                # A token no split saw a key for outputs zeros.
                for d in T.Parallel(combine_width):
                    if d < dim:
                        o[i_c, i_hq, d] = T.if_then_else(total[0] > 0, o_acc[d] / total[0], 0)

        return _nsa_fwd_varlen_gather_split

    return _nsa_fwd_varlen_gather_func


class NSAFwdVarlenKernel(Kernel, NSAFwdInterface):
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: NSACall) -> str | None:
        if call.separate_keys:
            return "attends to the queries' own tokens only"
        if call.dim > 128:
            return f"tiles the head dimension whole, at most 128, got {call.dim}"
        if (call.heads // call.heads_kv) % 16 != 0:
            return (
                f"feeds each query group to a 16-row MMA, got {call.heads // call.heads_kv} heads"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: NSACall) -> Entry:
        return call, lambda: cls(
            batch=call.batch,
            heads=call.heads,
            c_seq_len=call.c_seq_len,
            dim=call.dim,
            is_causal=call.is_causal,
            scale=call.scale,
            block_size=call.block_size,
            groups=call.heads // call.heads_kv,
            selected_blocks=call.selected_blocks,
            dtype=call.dtype,
            accum_dtype=torch.float32,
        )

    def __init__(
        self,
        batch: int,
        heads: int,
        c_seq_len: int,
        dim: int,
        is_causal: bool,
        scale: float,
        block_size: int,
        groups: int,
        selected_blocks: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.c_seq_len = c_seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.scale = scale
        self.block_size = block_size
        self.groups = groups
        self.selected_blocks = selected_blocks
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.accum_dtype_str = self.dtype_to_str(self.accum_dtype)

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"threads": 32}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": 32}]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        offsets: torch.Tensor,
        token_indices: torch.Tensor,
        offsets_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.program(self._compile)(
            q, k, v, block_indices, block_counts, offsets, token_indices
        )

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return _nsa_fwd_varlen_kernel(
            self.batch,
            self.heads,
            self.c_seq_len,
            self.dim,
            self.is_causal,
            self.scale,
            self.block_size,
            self.groups,
            self.selected_blocks,
            self.dtype_str,
            self.accum_dtype_str,
        )(cfg["threads"])


class NSAFwdVarlenTMAKernel(NSAFwdVarlenKernel):
    """SM90 NSA forward: TMA tiles, Q held in registers, a K/V ring over live slots."""

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"nsa_fwd_varlen_kernel"})

    @classmethod
    def refusal(cls, call: NSACall) -> str | None:
        """Head dims the MMA K step tiles, and a slot mask that fits one int32."""
        if call.dim % 16 != 0:
            return f"head dim {call.dim} is not a multiple of the MMA K step 16"
        if call.selected_blocks >= 32:
            return f"{call.selected_blocks} selected blocks overflow the int32 slot mask"
        return super().refusal(call)

    def _compile(self, cfg: dict):
        """The SM90 TMA program ``forward`` launches, compiled from *cfg*."""
        return _nsa_fwd_varlen_tma_kernel(
            self.batch,
            self.heads,
            self.c_seq_len,
            self.dim,
            self.is_causal,
            self.scale,
            self.block_size,
            self.groups,
            self.selected_blocks,
            self.dtype_str,
            self.accum_dtype_str,
        )(cfg["threads"])


class NSAFwdVarlenGatherKernel(Kernel, NSAFwdInterface):
    """NSA forward for any query group, block size and key layout the MMA tiles.

    Each step gathers the next selected keys row by row, so blocks of a few keys fill a
    tile; a group under 16 is padded to the MMA's 16 rows. With too few (token, KV head)
    pairs to fill the device, each pair's selection splits across CTAs and a second kernel
    merges the splits.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general: bool = True

    @classmethod
    def refusal(cls, call: NSACall) -> str | None:
        """Head dims the MMA K step tiles, and tiles that fit the device's shared memory."""
        if call.dim % 16 != 0:
            return f"head dim {call.dim} is not a multiple of the MMA K step 16"
        if call.heads > MAX_GRID_YZ_BLOCKS:
            return (
                f"{call.heads} query heads pass the {MAX_GRID_YZ_BLOCKS} blocks of a grid's y axis"
            )
        # The scale reaches the program as a float32 root of 2 * scale * log2(e), which
        # must be a normal float32: fast math flushes subnormals.
        root = math.sqrt(2 * call.scale * LOG2E)
        if not 2.0**-126 <= root <= (2 - 2.0**-23) * 2.0**127:
            return f"scale {call.scale} has no normal float32 root of 2 * scale * log2(e)"
        if cls._placement(call) is None:
            return (
                f"no step of whole {call.block_size}-key blocks is a multiple of the 16-key "
                f"MMA tile and fits head dim {call.dim} in {call.smem_budget} bytes of shared "
                "memory"
            )
        return super().refusal(call)

    @staticmethod
    def _placement(call: NSACall) -> tuple[dict, int] | None:
        """The config whose steps keep the most gathered key rows in flight per SM, and how
        many of its CTAs an SM holds; ``None`` where no step fits.

        A gather is bound by the latency of row-sized loads from L2, so what counts is the
        rows in flight across the CTAs an SM holds: two warps per CTA and two stages, the
        step width trading rows per CTA against CTAs per SM; a tie keeps the wider step.
        """
        element = torch.finfo(call.dtype).bits // 8
        rows = -(-(call.heads // call.heads_kv) // 16) * 16
        best, best_rows = None, 0
        for block_n in (64, 32, 16):
            # Two warps split a step's keys for the scores and the value dims for P V; two
            # stages overlap one step's copies with the last one's MMAs. A deeper ring or
            # more warps cost CTAs per SM, which a gather needs more.
            config = {"block_n": block_n, "num_stages": 2, "threads": 64}
            step_n = max(1, block_n // call.block_size) * call.block_size
            # The staged Q rows, the K and V tiles of every stage, and the softmax weights.
            tiles = 2 * config["num_stages"] * step_n * call.dim
            smem = (rows * call.dim + tiles + rows * step_n) * element
            # The MMA tiles 16 keys, and the step's 16-byte copies divide over the threads.
            copies = step_n * call.dim // 8
            if step_n % 16 or copies % config["threads"] or smem > call.smem_budget:
                continue
            ctas = min(call.smem_per_sm // smem, call.threads_per_sm // config["threads"])
            if ctas * step_n > best_rows:
                best, best_rows = (config, ctas), ctas * step_n
        return best

    @classmethod
    def entry_for(cls, call: NSACall) -> Entry:
        config, ctas = cls._placement(call)
        # Split each selection until the (token, KV head) pairs fill every resident slot
        # of the device, while each split keeps a step of keys. The count rounds up to a
        # power of two: the split count keys the program, and a decode batch of any size
        # reuses one of a few.
        steps = -(-call.selected_blocks // max(1, config["block_n"] // call.block_size))
        fill = -(-call.sm_count * ctas // (call.c_seq_len * call.heads_kv))
        splits = max(1, min(1 << (fill - 1).bit_length(), steps))
        args = dict(
            heads=call.heads,
            heads_kv=call.heads_kv,
            dim=call.dim,
            is_causal=call.is_causal,
            separate_keys=call.separate_keys,
            scale=call.scale,
            block_size=call.block_size,
            selected_blocks=call.selected_blocks,
            splits=splits,
            dtype=call.dtype,
            accum_dtype=torch.float32,
        )
        # The request count and the per-request lengths are dynamic in the program; the
        # token count reaches the entry only through the split count.
        key = (*args.values(), *config.values())
        return key, lambda: cls(**args, config=config)

    def __init__(
        self,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        separate_keys: bool,
        scale: float,
        block_size: int,
        selected_blocks: int,
        splits: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        config: dict,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.separate_keys = separate_keys
        self.scale = scale
        self._scale_root = math.sqrt(2 * scale * LOG2E)
        self.block_size = block_size
        self.selected_blocks = selected_blocks
        self.splits = splits
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.accum_dtype_str = self.dtype_to_str(self.accum_dtype)
        self._config = config
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return dict(self._config)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        offsets: torch.Tensor,
        token_indices: torch.Tensor,
        offsets_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        offsets_k = offsets if offsets_k is None else offsets_k
        program = self.program(self._compile)
        args = (
            q,
            k,
            v,
            block_indices,
            block_counts,
            offsets,
            token_indices,
            offsets_k,
            self._scale_root,
        )
        if self.splits == 1:
            return program(*args)
        c_seq_len = q.shape[0]
        max_part = torch.empty(
            (self.splits, c_seq_len, self.heads), dtype=torch.float32, device=q.device
        )
        sum_part = torch.empty_like(max_part)
        o_part = torch.empty(
            (self.splits, c_seq_len, self.heads, self.dim), dtype=torch.float32, device=q.device
        )
        return program(*args, max_part, sum_part, o_part)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return _nsa_fwd_varlen_gather_kernel(
            self.heads,
            self.heads_kv,
            self.dim,
            self.is_causal,
            self.separate_keys,
            self.block_size,
            self.selected_blocks,
            self.splits,
            self.dtype_str,
            self.accum_dtype_str,
        )(cfg["block_n"], cfg["num_stages"], cfg["threads"])
