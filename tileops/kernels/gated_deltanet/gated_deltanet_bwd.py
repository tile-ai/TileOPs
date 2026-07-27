"""
Gated DeltaNet backward: given dL/do, compute dL/d(q, k, v, g, beta).

Backward (split for SM utilisation):
  1. fused_prepare_compute_w_u: recompute w, u from forward
  2. bwd_parallel:    per-chunk gradients (grid: num_chunks x B x H)
  3. dh_recurrence_bwd: sequential dh propagation + corrections (grid: B x H)
  4. complete prepare backward: dw, du, A -> dk_prepare, dv, dbeta, dg_prepare
  5. merge: combine dk paths and apply the chunk-local reverse cumsum to dg
"""
import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from .gated_deltanet_fwd import _LOG2E

__all__ = [
    "GatedDeltaNetBwdKernel",
]


# Split kernel: bwd_parallel (fully parallel over chunks)

@functools.lru_cache(maxsize=32)
def _bwd_parallel_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Parallel per-chunk backward gradients.

    Grid: (num_chunks, batch, head) — fully parallel across chunks.
    Computes everything that does NOT depend on dh_buf from other chunks.

    Outputs: dq, dk_partial, dg_partial, dw, du_partial, v_new, dh_local
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(
        out_idx=[-7, -6, -5, -4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=256):
        @T.prim_func
        def bwd_parallel_kernel(
            do: T.Tensor([batch, head, seq_len, dim_v], dtype),
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            # Outputs
            dq: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dk_partial: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dg_partial: T.Tensor([batch, head, seq_len], dtype),
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_partial: T.Tensor([batch, head, seq_len, dim_v], dtype),
            v_new_out: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
        ):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                # Shared buffers
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, dim_v], dtype)
                do_c = T.alloc_shared([block_C, dim_v], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                o_part = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                # Gradients
                d_q_c = T.alloc_shared([block_C, dim_k], dtype)
                d_k_c = T.alloc_shared([block_C, dim_k], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                d_w_c = T.alloc_shared([block_C, dim_k], dtype)
                d_v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                d_attn = T.alloc_shared([block_C, block_C], dtype)
                # Working
                exp_g = T.alloc_shared([block_C], dtype)
                P = T.alloc_shared([block_C, dim_k], dtype)
                dP = T.alloc_shared([block_C, dim_k], dtype)
                # Fragments
                ws_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_v_new_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                d_attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_q_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                d_k_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dh_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)

                # Load chunk data
                T.copy(q[bid, hid, tid * block_C : (tid + 1) * block_C, :], q_c, disable_tma=True)
                T.copy(k[bid, hid, tid * block_C : (tid + 1) * block_C, :], k_c, disable_tma=True)
                T.copy(g[bid, hid, tid * block_C : (tid + 1) * block_C], g_c, disable_tma=True)
                T.copy(w[bid, hid, tid * block_C : (tid + 1) * block_C, :], w_c, disable_tma=True)
                T.copy(u[bid, hid, tid * block_C : (tid + 1) * block_C, :], u_c, disable_tma=True)
                T.copy(do[bid, hid, tid * block_C : (tid + 1) * block_C, :], do_c, disable_tma=True)
                T.copy(S[bid, hid, tid, :, :], h_c, disable_tma=True)

                # Recompute forward: v_new_c, o_part, attn
                T.clear(ws_frag)
                T.gemm(w_c, h_c, ws_frag)
                for i in T.Parallel(block_C):
                    exp_g[i] = T.exp2(g_c[i] * _LOG2E)
                for i, j in T.Parallel(block_C, dim_v):
                    v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                        (g_c[i] + g_c[block_C - 1]) * _LOG2E)

                # Store v_new for recurrence kernel
                T.copy(v_new_c, v_new_out[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)

                T.clear(ws_frag)
                T.gemm(q_c, h_c, ws_frag)
                for i, j in T.Parallel(block_C, dim_v):
                    o_part[i, j] = ws_frag[i, j] * exp_g[i]

                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(
                        i >= j,
                        attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        T.float32(0.0))

                T.clear(dh_frag)

                # Step 2: d_v_new_c = attn^T @ do_c (partial du)
                T.clear(d_v_new_frag)
                T.gemm(attn, do_c, d_v_new_frag, transpose_A=True)
                T.copy(d_v_new_frag, d_v_new_c)

                # d_attn = do_c @ v_new_c^T (causal masked)
                T.clear(d_attn_frag)
                T.gemm(do_c, v_new_c, d_attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(i >= j, d_attn_frag[i, j], T.float32(0.0))

                # Step 3: dg from o_part, dq from h, dh from q
                T.clear(d_q_c_frag)
                for i, j in T.Parallel(block_C, dim_v):
                    o_part[i, j] = do_c[i, j] * o_part[i, j]
                T.reduce_sum(o_part, dg_c, dim=1)
                for i, j in T.Parallel(block_C, dim_v):
                    o_part[i, j] = do_c[i, j] * exp_g[i]
                T.gemm(o_part, h_c, d_q_c_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, dim_k):
                    P[i, j] = q_c[i, j] * exp_g[i]
                T.gemm(P, do_c, dh_frag, transpose_A=True)

                # Step 4: dg from Γ, dq/dk from d_attn*Gamma
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = d_attn[i, j] * attn[i, j]
                dg_step4_row = T.alloc_shared([block_C], dtype)
                T.reduce_sum(attn, dg_step4_row, dim=1)
                dg_step4_col = T.alloc_shared([block_C], dtype)
                T.reduce_sum(attn, dg_step4_col, dim=0)
                for i in T.Parallel(block_C):
                    dg_c[i] = dg_c[i] + dg_step4_row[i] - dg_step4_col[i]

                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(
                        i >= j,
                        d_attn[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        T.float32(0.0))

                T.gemm(d_attn, k_c, d_q_c_frag)
                T.copy(d_q_c_frag, d_q_c)

                T.clear(d_k_c_frag)
                T.gemm(d_attn, q_c, d_k_c_frag, transpose_A=True)
                T.copy(d_k_c_frag, d_k_c)

                # Step 5: dh from w/v_new, dw, dg from P
                for i, j in T.Parallel(block_C, dim_k):
                    P[i, j] = w_c[i, j] * T.exp2(
                        (g_c[i] + g_c[block_C - 1]) * _LOG2E)
                T.clear(dP_frag)
                T.gemm(d_v_new_c, h_c, dP_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, dim_k):
                    dP[i, j] = -dP_frag[i, j]
                dh_sub_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                T.clear(dh_sub_frag)
                T.gemm(P, d_v_new_c, dh_sub_frag, transpose_A=True)
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_frag[i, j] -= dh_sub_frag[i, j]
                # dw
                for i, j in T.Parallel(block_C, dim_k):
                    d_w_c[i, j] = dP[i, j] * T.exp2(
                        (g_c[i] + g_c[block_C - 1]) * _LOG2E)
                # dg from P*dP
                for i, j in T.Parallel(block_C, dim_k):
                    P[i, j] = P[i, j] * dP[i, j]
                dg_step5_tmp = T.alloc_shared([block_C], dtype)
                T.reduce_sum(P, dg_step5_tmp, dim=1)
                dg_step5_total = T.alloc_shared([1], accum_dtype)
                T.reduce_sum(dg_step5_tmp, dg_step5_total, dim=0)
                for i in T.Parallel(block_C):
                    dg_c[i] += dg_step5_tmp[i]
                dg_c[block_C - 1] = dg_c[block_C - 1] + dg_step5_total[0]

                # Write outputs
                T.copy(d_q_c, dq[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)
                T.copy(d_k_c, dk_partial[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)
                for i in T.Parallel(block_C):
                    dg_partial[bid, hid, tid * block_C + i] = dg_c[i]
                T.copy(d_w_c, dw[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)
                T.copy(d_v_new_c, du_partial[bid, hid, tid * block_C : (tid + 1) * block_C, :], disable_tma=True)
                # Store dh_local for recurrence kernel
                T.copy(dh_frag, dh_local[bid, hid, tid, :, :], disable_tma=True)

        return bwd_parallel_kernel

    return _func


# Split kernel: dh_recurrence_bwd (sequential backward over chunks)

@functools.lru_cache(maxsize=32)
def _dh_recurrence_bwd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
):
    """Sequential backward dh recurrence with corrections.

    Grid: (num_v_tiles, batch, head) — sequential over chunks (backward),
    parallel over V tiles.
    Reads dh_local from bwd_parallel, propagates dh backward, and computes
    corrections for dk, du, dg that depend on dh_buf from other chunks.

    Outputs: dk_correction_partial, du_correction, dg_correction_partial.
    The partial dk/dg outputs must be reduced over the V-tile dimension by
    the caller. This keeps the per-CTA state buffer small enough for d128.
    """
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = dim_v if block_v <= 0 else block_v
    if dim_v % BV != 0:
        raise ValueError(f"dim_v ({dim_v}) must be divisible by block_v ({BV})")
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dh_recurrence_bwd_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            # Outputs
            dk_corr_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dg_corr_partial: T.Tensor([batch, head, num_v_tiles, seq_len], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                # Shared buffers
                g_c = T.alloc_shared([block_C], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                k_scaled = T.alloc_shared([block_C, dim_k], dtype)
                dP = T.alloc_shared([block_C, dim_k], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                # dh_buf carries gradient from the next chunk (backward)
                dh_buf = T.alloc_shared([dim_k, BV], dtype)
                dh_h_tmp = T.alloc_shared([dim_k, BV], dtype)
                # Fragments
                dh_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                du_corr_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                v_offset = vid * BV

                # Zero dh_buf (last chunk has no successor)
                for i, j in T.Parallel(dim_k, BV):
                    dh_buf[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    # Load data
                    T.copy(g[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C], g_c, disable_tma=True)
                    T.copy(k[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], k_c, disable_tma=True)
                    T.copy(
                        v_new[
                            bid,
                            hid,
                            t_bwd * block_C : (t_bwd + 1) * block_C,
                            v_offset : v_offset + BV,
                        ],
                        v_new_c,
                        disable_tma=True,
                    )
                    T.copy(S[bid, hid, t_bwd, :, v_offset : v_offset + BV], h_c, disable_tma=True)
                    T.copy(dh_local[bid, hid, t_bwd, :, v_offset : v_offset + BV], dh_loc, disable_tma=True)

                    # k_scaled = k * exp(g_last - g)
                    for pn, sk in T.Parallel(block_C, dim_k):
                        k_scaled[pn, sk] = k_c[pn, sk] * T.exp2((g_c[block_C - 1] - g_c[pn]) * _LOG2E)

                    # dh = dh_local + dh_buf * exp(g_last)
                    for i, j in T.Parallel(dim_k, BV):
                        dh_frag[i, j] = dh_loc[i, j] + dh_buf[i, j] * T.exp2(g_c[block_C - 1] * _LOG2E)

                    # du_correction = k_scaled @ dh_buf
                    T.clear(du_corr_frag)
                    T.gemm(k_scaled, dh_buf, du_corr_frag)
                    T.copy(
                        du_corr_frag,
                        du_corr[
                            bid,
                            hid,
                            t_bwd * block_C : (t_bwd + 1) * block_C,
                            v_offset : v_offset + BV,
                        ],
                        disable_tma=True,
                    )

                    # dk_correction = (v_new @ dh_buf^T) * exp(g_last - g)
                    T.clear(dP_frag)
                    T.gemm(v_new_c, dh_buf, dP_frag, transpose_B=True)
                    T.copy(dP_frag, dP)
                    for n, kk in T.Parallel(block_C, dim_k):
                        dk_corr_partial[bid, hid, vid, t_bwd * block_C + n, kk] = dP[n, kk] * T.exp2((g_c[block_C - 1] - g_c[n]) * _LOG2E)

                    # dg_correction: per-position and g_last terms
                    # Per-position: -sum_k(dP * k_scaled) per row n
                    for n, kk in T.Parallel(block_C, dim_k):
                        dP[n, kk] = dP[n, kk] * k_scaled[n, kk]
                    d_g_pos = T.alloc_shared([block_C], dtype)
                    T.reduce_sum(dP, d_g_pos, dim=1)
                    for n in T.Parallel(block_C):
                        dg_c[n] = -d_g_pos[n]

                    # g_last term 1: sum(dh_buf * h_c) * exp_g_last
                    for i, j in T.Parallel(dim_k, BV):
                        dh_h_tmp[i, j] = dh_buf[i, j] * h_c[i, j]
                    d_g_last_partial = T.alloc_shared([dim_k], dtype)
                    T.reduce_sum(dh_h_tmp, d_g_last_partial, dim=1)
                    d_g_last_scalar1 = T.alloc_shared([1], accum_dtype)
                    T.reduce_sum(d_g_last_partial, d_g_last_scalar1, dim=0)

                    # g_last term 2: sum_n(d_g_pos)
                    d_g_last_scalar2 = T.alloc_shared([1], accum_dtype)
                    T.reduce_sum(d_g_pos, d_g_last_scalar2, dim=0)
                    dg_c[block_C - 1] = dg_c[block_C - 1] + d_g_last_scalar1[0] * T.exp2(
                        g_c[block_C - 1] * _LOG2E) + d_g_last_scalar2[0]

                    # Write dg_correction
                    for i in T.Parallel(block_C):
                        dg_corr_partial[bid, hid, vid, t_bwd * block_C + i] = dg_c[i]

                    # Carry dh to next iteration
                    T.copy(dh_frag, dh_buf)

        return dh_recurrence_bwd_kernel

    return _func


@functools.lru_cache(maxsize=32)
def _reduce_dh_recurrence_partials_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
):
    """Reduce V-tiled dh-recurrence correction partials.

    ``dk`` and ``dg`` corrections are additive across V tiles. Keeping this
    reduction inside TileLang avoids generic torch reductions in the full
    backward hot path.
    """
    block_C = chunk_size
    BV = dim_v if block_v <= 0 else block_v
    if dim_v % BV != 0:
        raise ValueError(f"dim_v ({dim_v}) must be divisible by block_v ({BV})")
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=256):
        @T.prim_func
        def reduce_dh_recurrence_partials(
            dk_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
            dg_partial: T.Tensor([batch, head, num_v_tiles, seq_len], dtype),
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dg: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, cid):
                for n, kk in T.Parallel(block_C, dim_k):
                    acc = T.alloc_var(T.float32, init=0.0)
                    for vid in T.Serial(num_v_tiles):
                        acc += T.cast(dk_partial[bid, hid, vid, cid * block_C + n, kk], "float32")
                    dk[bid, hid, cid * block_C + n, kk] = T.cast(acc, dtype)

                for n in T.Parallel(block_C):
                    acc = T.alloc_var(T.float32, init=0.0)
                    for vid in T.Serial(num_v_tiles):
                        acc += T.cast(dg_partial[bid, hid, vid, cid * block_C + n], "float32")
                    dg[bid, hid, cid * block_C + n] = T.cast(acc, dtype)

        return reduce_dh_recurrence_partials

    return _func


@functools.lru_cache(maxsize=32)
def _merge_bwd_outputs_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dtype: str = "float32",
):
    """Merge dk contributions and convert chunk-local dg_cum to dg."""
    block_C = chunk_size

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def merge_bwd_outputs(
            dk_partial: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dk_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dk_prepare: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dg_partial: T.Tensor([batch, head, seq_len], dtype),
            dg_corr: T.Tensor([batch, head, seq_len], dtype),
            dg_prepare: T.Tensor([batch, head, seq_len], dtype),
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dg: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, cid):
                offset = cid * block_C
                for n, kk in T.Parallel(block_C, dim_k):
                    dk[bid, hid, offset + n, kk] = (
                        dk_partial[bid, hid, offset + n, kk]
                        + dk_corr[bid, hid, offset + n, kk]
                        + dk_prepare[bid, hid, offset + n, kk]
                    )

                for n in T.Parallel(block_C):
                    acc = T.alloc_var(T.float32, init=0.0)
                    for j in T.Serial(block_C):
                        if j >= n:
                            acc += T.cast(
                                dg_partial[bid, hid, offset + j]
                                + dg_corr[bid, hid, offset + j]
                                + dg_prepare[bid, hid, offset + j],
                                "float32",
                            )
                    dg[bid, hid, offset + n] = T.cast(acc, dtype)

        return merge_bwd_outputs

    return _func


@functools.lru_cache(maxsize=32)
def _compute_dw_correction_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Backpropagate future-state du into w for each independent chunk."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def compute_dw_correction(
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            dw_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
        ):
            with T.Kernel(batch, head, num_chunks, threads=threads) as (bid, hid, cid):
                offset = cid * block_C
                du_s = T.alloc_shared([block_C, dim_v], dtype)
                state_s = T.alloc_shared([dim_k, dim_v], dtype)
                g_s = T.alloc_shared([block_C], dtype)
                dw_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                T.copy(
                    du_corr[bid, hid, offset : offset + block_C, :],
                    du_s,
                    disable_tma=True,
                )
                T.copy(S[bid, hid, cid, :, :], state_s, disable_tma=True)
                T.copy(
                    g[bid, hid, offset : offset + block_C],
                    g_s,
                    disable_tma=True,
                )
                T.clear(dw_frag)
                T.gemm(du_s, state_s, dw_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, dim_k):
                    dw_corr[bid, hid, offset + i, j] = (
                        -dw_frag[i, j]
                        * T.exp2((g_s[i] + g_s[block_C - 1]) * _LOG2E)
                    )

        return compute_dw_correction

    return _func


@functools.lru_cache(maxsize=32)
def _dh_carry_after_scan_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
):
    """Produce per-chunk successor dh carries for a split correction path."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = dim_v if block_v <= 0 else block_v
    if dim_v % BV != 0:
        raise ValueError(f"dim_v ({dim_v}) must be divisible by block_v ({BV})")
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dh_carry_after_scan_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            dh_carry_after: T.Tensor(
                [batch, head, num_v_tiles, num_chunks, dim_k, BV], dtype
            ),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                g_c = T.alloc_shared([block_C], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                dh_buf = T.alloc_shared([dim_k, BV], dtype)
                dh_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                v_offset = vid * BV

                for i, j in T.Parallel(dim_k, BV):
                    dh_buf[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    for i, j in T.Parallel(dim_k, BV):
                        dh_carry_after[bid, hid, vid, t_bwd, i, j] = dh_buf[i, j]

                    T.copy(g[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C], g_c, disable_tma=True)
                    T.copy(
                        dh_local[bid, hid, t_bwd, :, v_offset : v_offset + BV],
                        dh_loc,
                        disable_tma=True,
                    )

                    for i, j in T.Parallel(dim_k, BV):
                        dh_frag[i, j] = dh_loc[i, j] + dh_buf[i, j] * T.exp2(g_c[block_C - 1] * _LOG2E)
                    T.copy(dh_frag, dh_buf)

        return dh_carry_after_scan_kernel

    return _func


@functools.lru_cache(maxsize=32)
def _dh_correction_from_carry_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
):
    """Compute dh corrections from precomputed carries, parallel over chunks."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = dim_v if block_v <= 0 else block_v
    if dim_v % BV != 0:
        raise ValueError(f"dim_v ({dim_v}) must be divisible by block_v ({BV})")
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=256):
        @T.prim_func
        def dh_correction_from_carry_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            dh_carry_after: T.Tensor(
                [batch, head, num_v_tiles, num_chunks, dim_k, BV], dtype
            ),
            dk_corr_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dg_corr_partial: T.Tensor([batch, head, num_v_tiles, seq_len], dtype),
        ):
            with T.Kernel(num_v_tiles, num_chunks, batch * head, threads=threads) as (
                vid,
                cid,
                bhid,
            ):
                bid = bhid // head
                hid = bhid - bid * head
                g_c = T.alloc_shared([block_C], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                k_scaled = T.alloc_shared([block_C, dim_k], dtype)
                dP = T.alloc_shared([block_C, dim_k], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                dh_buf = T.alloc_shared([dim_k, BV], dtype)
                dh_h_tmp = T.alloc_shared([dim_k, BV], dtype)
                du_corr_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                d_g_pos = T.alloc_shared([block_C], dtype)
                d_g_last_partial = T.alloc_shared([dim_k], dtype)
                d_g_last_scalar1 = T.alloc_shared([1], accum_dtype)
                d_g_last_scalar2 = T.alloc_shared([1], accum_dtype)
                v_offset = vid * BV

                T.copy(g[bid, hid, cid * block_C : (cid + 1) * block_C], g_c, disable_tma=True)
                T.copy(k[bid, hid, cid * block_C : (cid + 1) * block_C, :], k_c, disable_tma=True)
                T.copy(
                    v_new[
                        bid,
                        hid,
                        cid * block_C : (cid + 1) * block_C,
                        v_offset : v_offset + BV,
                    ],
                    v_new_c,
                    disable_tma=True,
                )
                T.copy(S[bid, hid, cid, :, v_offset : v_offset + BV], h_c, disable_tma=True)
                T.copy(dh_carry_after[bid, hid, vid, cid, :, :], dh_buf, disable_tma=True)

                g_last = g_c[block_C - 1]
                exp_g_last = T.exp2(g_last * _LOG2E)

                for pn, sk in T.Parallel(block_C, dim_k):
                    k_scaled[pn, sk] = k_c[pn, sk] * T.exp2((g_last - g_c[pn]) * _LOG2E)

                T.clear(du_corr_frag)
                T.gemm(k_scaled, dh_buf, du_corr_frag)
                T.copy(
                    du_corr_frag,
                    du_corr[
                        bid,
                        hid,
                        cid * block_C : (cid + 1) * block_C,
                        v_offset : v_offset + BV,
                    ],
                    disable_tma=True,
                )

                T.clear(dP_frag)
                T.gemm(v_new_c, dh_buf, dP_frag, transpose_B=True)
                T.copy(dP_frag, dP)
                for n, kk in T.Parallel(block_C, dim_k):
                    dk_corr_partial[bid, hid, vid, cid * block_C + n, kk] = (
                        dP[n, kk] * T.exp2((g_last - g_c[n]) * _LOG2E)
                    )

                for n, kk in T.Parallel(block_C, dim_k):
                    dP[n, kk] = dP[n, kk] * k_scaled[n, kk]
                T.reduce_sum(dP, d_g_pos, dim=1)
                for n in T.Parallel(block_C):
                    dg_c[n] = -d_g_pos[n]

                for i, j in T.Parallel(dim_k, BV):
                    dh_h_tmp[i, j] = dh_buf[i, j] * h_c[i, j]
                T.reduce_sum(dh_h_tmp, d_g_last_partial, dim=1)
                T.reduce_sum(d_g_last_partial, d_g_last_scalar1, dim=0)
                T.reduce_sum(d_g_pos, d_g_last_scalar2, dim=0)
                dg_c[block_C - 1] = (
                    dg_c[block_C - 1]
                    + d_g_last_scalar1[0] * exp_g_last
                    + d_g_last_scalar2[0]
                )

                for i in T.Parallel(block_C):
                    dg_corr_partial[bid, hid, vid, cid * block_C + i] = dg_c[i]

        return dh_correction_from_carry_kernel

    return _func


@functools.lru_cache(maxsize=32)
def _dh_segment_summary_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
    segment_chunks: int = 8,
):
    """Summarize a short reverse chunk segment as X[left] = B + A * X[right]."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    if num_chunks % segment_chunks != 0:
        raise ValueError("num_chunks must be divisible by segment_chunks")
    num_segments = num_chunks // segment_chunks
    BV = dim_v if block_v <= 0 else block_v
    if dim_v % BV != 0:
        raise ValueError(f"dim_v ({dim_v}) must be divisible by block_v ({BV})")
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dh_segment_summary_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            segment_alpha: T.Tensor([batch, head, num_v_tiles, num_segments], dtype),
            segment_local: T.Tensor(
                [batch, head, num_v_tiles, num_segments, dim_k, BV], dtype
            ),
        ):
            with T.Kernel(num_v_tiles, num_segments, batch * head, threads=threads) as (
                vid,
                sid,
                bhid,
            ):
                bid = bhid // head
                hid = bhid - bid * head
                g_c = T.alloc_shared([block_C], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                summary = T.alloc_shared([dim_k, BV], dtype)
                summary_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                v_offset = vid * BV
                alpha_acc = T.alloc_var(T.float32, init=1.0)

                for i, j in T.Parallel(dim_k, BV):
                    summary[i, j] = T.float32(0.0)

                for step in T.Serial(segment_chunks):
                    cid = sid * segment_chunks + (segment_chunks - 1 - step)
                    T.copy(
                        g[bid, hid, cid * block_C : (cid + 1) * block_C],
                        g_c,
                        disable_tma=True,
                    )
                    T.copy(
                        dh_local[bid, hid, cid, :, v_offset : v_offset + BV],
                        dh_loc,
                        disable_tma=True,
                    )
                    alpha = T.exp2(g_c[block_C - 1] * _LOG2E)
                    for i, j in T.Parallel(dim_k, BV):
                        summary_frag[i, j] = dh_loc[i, j] + summary[i, j] * alpha
                    T.copy(summary_frag, summary)
                    alpha_acc = alpha * alpha_acc

                segment_alpha[bid, hid, vid, sid] = T.cast(alpha_acc, dtype)
                T.copy(summary, segment_local[bid, hid, vid, sid, :, :], disable_tma=True)

        return dh_segment_summary_kernel

    return _func


@functools.lru_cache(maxsize=32)
def _dh_segment_boundary_scan_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
    segment_chunks: int = 8,
):
    """Reverse scan segment summaries to produce each segment's successor carry."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    if num_chunks % segment_chunks != 0:
        raise ValueError("num_chunks must be divisible by segment_chunks")
    num_segments = num_chunks // segment_chunks
    BV = dim_v if block_v <= 0 else block_v
    if dim_v % BV != 0:
        raise ValueError(f"dim_v ({dim_v}) must be divisible by block_v ({BV})")
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dh_segment_boundary_scan_kernel(
            segment_alpha: T.Tensor([batch, head, num_v_tiles, num_segments], dtype),
            segment_local: T.Tensor(
                [batch, head, num_v_tiles, num_segments, dim_k, BV], dtype
            ),
            segment_carry_after: T.Tensor(
                [batch, head, num_v_tiles, num_segments, dim_k, BV], dtype
            ),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                local = T.alloc_shared([dim_k, BV], dtype)
                carry = T.alloc_shared([dim_k, BV], dtype)
                carry_frag = T.alloc_fragment([dim_k, BV], accum_dtype)

                for i, j in T.Parallel(dim_k, BV):
                    carry[i, j] = T.float32(0.0)

                for step in T.Pipelined(num_segments, num_stages=num_stages):
                    sid = num_segments - 1 - step
                    T.copy(
                        carry,
                        segment_carry_after[bid, hid, vid, sid, :, :],
                        disable_tma=True,
                    )
                    T.copy(
                        segment_local[bid, hid, vid, sid, :, :],
                        local,
                        disable_tma=True,
                    )
                    alpha = segment_alpha[bid, hid, vid, sid]
                    for i, j in T.Parallel(dim_k, BV):
                        carry_frag[i, j] = local[i, j] + carry[i, j] * alpha
                    T.copy(carry_frag, carry)

        return dh_segment_boundary_scan_kernel

    return _func


@functools.lru_cache(maxsize=32)
def _dh_segment_local_carry_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
    segment_chunks: int = 8,
):
    """Expand segment boundary carries into per-chunk successor carries."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    if num_chunks % segment_chunks != 0:
        raise ValueError("num_chunks must be divisible by segment_chunks")
    num_segments = num_chunks // segment_chunks
    BV = dim_v if block_v <= 0 else block_v
    if dim_v % BV != 0:
        raise ValueError(f"dim_v ({dim_v}) must be divisible by block_v ({BV})")
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dh_segment_local_carry_kernel(
            g: T.Tensor([batch, head, seq_len], dtype),
            dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype),
            segment_carry_after: T.Tensor(
                [batch, head, num_v_tiles, num_segments, dim_k, BV], dtype
            ),
            dh_carry_after: T.Tensor(
                [batch, head, num_v_tiles, num_chunks, dim_k, BV], dtype
            ),
        ):
            with T.Kernel(num_v_tiles, num_segments, batch * head, threads=threads) as (
                vid,
                sid,
                bhid,
            ):
                bid = bhid // head
                hid = bhid - bid * head
                g_c = T.alloc_shared([block_C], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                carry = T.alloc_shared([dim_k, BV], dtype)
                carry_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                v_offset = vid * BV

                T.copy(
                    segment_carry_after[bid, hid, vid, sid, :, :],
                    carry,
                    disable_tma=True,
                )

                for step in T.Serial(segment_chunks):
                    local_idx = segment_chunks - 1 - step
                    cid = sid * segment_chunks + local_idx
                    T.copy(
                        carry,
                        dh_carry_after[bid, hid, vid, cid, :, :],
                        disable_tma=True,
                    )
                    T.copy(
                        g[bid, hid, cid * block_C : (cid + 1) * block_C],
                        g_c,
                        disable_tma=True,
                    )
                    T.copy(
                        dh_local[bid, hid, cid, :, v_offset : v_offset + BV],
                        dh_loc,
                        disable_tma=True,
                    )
                    alpha = T.exp2(g_c[block_C - 1] * _LOG2E)
                    for i, j in T.Parallel(dim_k, BV):
                        carry_frag[i, j] = dh_loc[i, j] + carry[i, j] * alpha
                    T.copy(carry_frag, carry)

        return dh_segment_local_carry_kernel

    return _func


@torch.library.custom_op("tileops::gated_deltanet_bwd_kernel", mutates_args=())
def _gated_deltanet_bwd_wrapped_kernel(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    num_stages: int, threads: int,
    parallel_threads: int, recurrence_threads: int, recurrence_block_v: int,
    recurrence_split_carry: int,
    do: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    S: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from .compute_w_u_bwd import compute_w_u_bwd_full_tl
    from .fused_prepare_compute_w_u import fused_prepare_compute_w_u_tl
    from .gated_deltanet_fwd import _chunk_local_cumsum

    g_cum = _chunk_local_cumsum(g.float(), chunk_size).to(g.dtype)

    fused_fn = fused_prepare_compute_w_u_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
        write_duplicate_A=False,
        fast_math=True,
    )(num_stages, threads)
    bwd_parallel_fn = _bwd_parallel_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(parallel_threads)
    reduce_dh_partials_fn = _reduce_dh_recurrence_partials_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
        block_v=recurrence_block_v,
    )(recurrence_threads)
    merge_bwd_outputs_fn = _merge_bwd_outputs_tl(
        batch, head, seq_len, chunk_size, dim_k, dtype,
    )(threads)
    wu_bwd_fn = compute_w_u_bwd_full_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(num_stages, threads)
    dw_correction_fn = _compute_dw_correction_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
    )(threads)

    Aw, _Au, w, u = fused_fn(k, v, g_cum, beta)
    dq, dk_partial, dg_partial, dw, du_partial, v_new, dh_local = \
        bwd_parallel_fn(do, q, k, g_cum, w, u, S)
    if recurrence_split_carry == 0:
        dh_recurrence_bwd_fn = _dh_recurrence_bwd_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
            block_v=recurrence_block_v,
        )(num_stages, recurrence_threads)
        dk_corr_partial, du_corr, dg_corr_partial = \
            dh_recurrence_bwd_fn(g_cum, k, v_new, S, dh_local)
    elif recurrence_split_carry == 1:
        dh_carry_after_scan_fn = _dh_carry_after_scan_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
            block_v=recurrence_block_v,
        )(num_stages, recurrence_threads)
        dh_correction_from_carry_fn = _dh_correction_from_carry_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
            block_v=recurrence_block_v,
        )(recurrence_threads)
        dh_carry_after = dh_carry_after_scan_fn(g_cum, dh_local)
        dk_corr_partial, du_corr, dg_corr_partial = \
            dh_correction_from_carry_fn(g_cum, k, v_new, S, dh_carry_after)
    else:
        num_chunks = seq_len // chunk_size
        segment_chunks = 8 if num_chunks % 8 == 0 else (
            4 if num_chunks % 4 == 0 else (2 if num_chunks % 2 == 0 else 1)
        )
        dh_segment_summary_fn = _dh_segment_summary_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
            block_v=recurrence_block_v, segment_chunks=segment_chunks,
        )(num_stages, recurrence_threads)
        dh_segment_boundary_scan_fn = _dh_segment_boundary_scan_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
            block_v=recurrence_block_v, segment_chunks=segment_chunks,
        )(num_stages, recurrence_threads)
        dh_segment_local_carry_fn = _dh_segment_local_carry_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
            block_v=recurrence_block_v, segment_chunks=segment_chunks,
        )(num_stages, recurrence_threads)
        dh_correction_from_carry_fn = _dh_correction_from_carry_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, dtype,
            block_v=recurrence_block_v,
        )(recurrence_threads)
        segment_alpha, segment_local = dh_segment_summary_fn(g_cum, dh_local)
        segment_carry_after = dh_segment_boundary_scan_fn(segment_alpha, segment_local)
        dh_carry_after = dh_segment_local_carry_fn(g_cum, dh_local, segment_carry_after)
        dk_corr_partial, du_corr, dg_corr_partial = \
            dh_correction_from_carry_fn(g_cum, k, v_new, S, dh_carry_after)
    dk_corr, dg_corr = reduce_dh_partials_fn(dk_corr_partial, dg_corr_partial)

    dw_corr = dw_correction_fn(du_corr, S, g_cum)
    dk_prepare, dv, dbeta, dg_prepare = wu_bwd_fn(
        dw, dw_corr, du_partial, du_corr, Aw, k, v, g_cum, beta,
    )

    dk, dg = merge_bwd_outputs_fn(
        dk_partial, dk_corr, dk_prepare, dg_partial, dg_corr, dg_prepare,
    )
    return dq, dk, dv, dg, dbeta


@_gated_deltanet_bwd_wrapped_kernel.register_fake
def _gated_deltanet_bwd_wrapped_kernel_fake(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    num_stages: int, threads: int,
    parallel_threads: int, recurrence_threads: int, recurrence_block_v: int,
    recurrence_split_carry: int,
    do: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    S: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty(batch, head, seq_len, dim_k, dtype=q.dtype, device=q.device)
    dk = torch.empty_like(dq)
    dv = torch.empty(batch, head, seq_len, dim_v, dtype=v.dtype, device=v.device)
    dg = torch.empty(batch, head, seq_len, dtype=g.dtype, device=g.device)
    dbeta = torch.empty(batch, head, seq_len, dtype=beta.dtype, device=beta.device)
    return dq, dk, dv, dg, dbeta


class GatedDeltaNetBwdKernel(Kernel):
    """Gated DeltaNet backward kernel.

    Full backward: do -> (dq, dk, dv, dg, dbeta).

    Split pipeline (Phase 2 optimisation):
      1. fused_prepare_compute_w_u: recompute w, u
      2. bwd_parallel: per-chunk gradients (grid: num_chunks x B x H)
      3. dh_recurrence_bwd: sequential dh propagation + corrections (grid: B x H)
      4. complete prepare backward: dw, du, A -> dk_prepare, dv, dbeta, dg_prepare
      5. merge: combine dk paths and apply the chunk-local reverse cumsum to dg
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        head: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: str = "float32",
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.head = head
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        use_split_carry = self.chunk_size >= 64 and self.dim_v > 64 and self.dim_v % 64 == 0
        threads = 128 if use_split_carry else (256 if self.chunk_size >= 64 else 128)
        recurrence_threads = 128 if use_split_carry else threads
        recurrence_block_v = (
            64 if use_split_carry
            else (32 if self.dim_v > 64 and self.dim_v % 32 == 0 else 0)
        )
        parallel_threads = 256 if use_split_carry else threads
        return {
            "num_stages": 2,
            "threads": threads,
            "parallel_threads": parallel_threads,
            "recurrence_threads": recurrence_threads,
            "recurrence_block_v": recurrence_block_v,
            "recurrence_split_carry": 2 if use_split_carry else 0,
        }

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        default_config = self.default_config
        return _gated_deltanet_bwd_wrapped_kernel(
            self.batch, self.head, self.seq_len, self.chunk_size,
            self.dim_k, self.dim_v, self.dtype_str,
            self.config.get("num_stages", default_config["num_stages"]),
            self.config.get("threads", default_config["threads"]),
            self.config.get("parallel_threads", default_config["parallel_threads"]),
            self.config.get("recurrence_threads", default_config["recurrence_threads"]),
            self.config.get("recurrence_block_v", default_config["recurrence_block_v"]),
            self.config.get("recurrence_split_carry", default_config["recurrence_split_carry"]),
            do, q, k, v, g, beta, S,
        )
