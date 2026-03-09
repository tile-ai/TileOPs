"""
Gated DeltaNet backward: given dL/do, compute dL/d(q, k, v, g, beta).

Forward dependency (same as gated_deltanet_fwd):
  1. w, u = compute_w_u(Aw, Au, k, v, beta)
  2. S_0 = 0; kernel2(q, k, g, w, u, S_0) -> o

Backward:
  1. kernel2_bwd: do -> dq, dk_k2, dg, dw, du  (steps 1-8, reverse chunk order)
  2. compute_w_u_bwd: dw, du -> dAw, dAu, dk_wu, dv, dbeta  (step 9)
  3. dk = dk_k2 + dk_wu
"""
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

from .gated_deltanet_fwd import _LOG2E

__all__ = [
    "GatedDeltaNetBwdKernel",
]


def _gated_deltanet_kernel2_bwd_chunk_tl(
    batch: int,
    head: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """TileLang single-chunk backward: (do_c, q_c, k_c, g_c, w_c, u_c, h_c, dh_in) -> (dq_c, dk_c, dg_c, dw_c, du_c, dh_out).
    Caller must run chunks in reverse order and pass dh_in (zeros for last chunk), dh_out -> next chunk's dh_in."""
    accum_dtype = "float32"
    block_C = chunk_size

    @tilelang.jit(
        out_idx=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _chunk_bwd_func(num_stages, threads=128):
        @T.macro
        def _chunk_bwd_body(
            do_c: T.Tensor([batch, head, block_C, dim_v], dtype),
            q_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            k_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            g_c: T.Tensor([batch, head, block_C], dtype),
            w_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            u_c: T.Tensor([batch, head, block_C, dim_v], dtype),
            h_c: T.Tensor([batch, head, dim_k, dim_v], dtype),
            dh_in: T.Tensor([batch, head, dim_k, dim_v], dtype),
            dq_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            dk_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            dg_c: T.Tensor([batch, head, block_C], dtype),
            dw_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            du_c: T.Tensor([batch, head, block_C, dim_v], dtype),
            dh_out: T.Tensor([batch, head, dim_k, dim_v], dtype),
            d_attn_out: T.Tensor([batch, head, block_C, block_C], dtype),
            d_k_scaled_out: T.Tensor([batch, head, block_C, dim_k], dtype),
            dg_after_step3_out: T.Tensor([batch, head, block_C], dtype),
            dg_after_step5_out: T.Tensor([batch, head, block_C], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                v_new_c = T.alloc_shared([block_C, dim_v], accum_dtype)
                o_part = T.alloc_shared([block_C, dim_v], accum_dtype)
                attn = T.alloc_shared([block_C, block_C], accum_dtype)
                d_v_new_c = T.alloc_shared([block_C, dim_v], accum_dtype)
                d_attn = T.alloc_shared([block_C, block_C], accum_dtype)
                d_q_c_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                d_k_c_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                dg_c_s = T.alloc_shared([block_C], accum_dtype)
                d_w_c_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                exp_g = T.alloc_shared([block_C], accum_dtype)
                P = T.alloc_shared([block_C, dim_k], accum_dtype)
                dP = T.alloc_shared([block_C, dim_k], accum_dtype)
                dh_s = T.alloc_shared([dim_k, dim_v], accum_dtype)
                w_s_product = T.alloc_shared([block_C, dim_v], accum_dtype)
                k_scaled = T.alloc_shared([block_C, dim_k], accum_dtype)
                d_k_scaled = T.alloc_shared([block_C, dim_k], accum_dtype)
                d_g_last_buf = T.alloc_shared([1], accum_dtype)
                h_c_s = T.alloc_shared([dim_k, dim_v], accum_dtype)
                q_c_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                k_c_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                w_c_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                u_c_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                do_c_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                g_c_s = T.alloc_shared([block_C], accum_dtype)

                # Load chunk from global to shared (per bid, hid)
                for i, j in T.Parallel(dim_k, dim_v):
                    h_c_s[i, j] = h_c[bid, hid, i, j]
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_s[i, j] = dh_in[bid, hid, i, j]
                for i, j in T.Parallel(block_C, dim_k):
                    q_c_s[i, j] = q_c[bid, hid, i, j]
                    k_c_s[i, j] = k_c[bid, hid, i, j]
                    w_c_s[i, j] = w_c[bid, hid, i, j]
                for i, j in T.Parallel(block_C, dim_v):
                    u_c_s[i, j] = u_c[bid, hid, i, j]
                    do_c_s[i, j] = do_c[bid, hid, i, j]
                for i in T.Parallel(block_C):
                    g_c_s[i] = g_c[bid, hid, i]

                # Recompute forward: v_new_c, o_part, attn, k_scaled
                for i, j in T.Parallel(block_C, dim_v):
                    w_s_product[i, j] = T.float32(0.0)
                for i, j in T.Parallel(block_C, dim_v):
                    for kk in T.Serial(dim_k):
                        w_s_product[i, j] += w_c_s[i, kk] * h_c_s[kk, j]
                for i in T.Parallel(block_C):
                    exp_g[i] = T.exp2(g_c_s[i] * _LOG2E)
                for i, j in T.Parallel(block_C, dim_v):
                    v_new_c[i, j] = u_c_s[i, j] - w_s_product[i, j] * exp_g[i]
                for i, j in T.Parallel(block_C, dim_v):
                    o_part[i, j] = T.float32(0.0)
                for i, j in T.Parallel(block_C, dim_v):
                    for kk in T.Serial(dim_k):
                        o_part[i, j] += q_c_s[i, kk] * h_c_s[kk, j]
                for i, j in T.Parallel(block_C, dim_v):
                    o_part[i, j] = o_part[i, j] * exp_g[i]
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    for i, j in T.Parallel(block_C, block_C):
                        attn[i, j] += q_c_s[i, kk] * k_c_s[j, kk]
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(i >= j, attn[i, j], T.float32(0.0))
                g_last = g_c_s[block_C - 1]
                for n, kk in T.Parallel(block_C, dim_k):
                    k_scaled[n, kk] = k_c_s[n, kk] * T.exp2((g_last - g_c_s[n]) * _LOG2E)

                # Step 2
                for i, j in T.Parallel(block_C, dim_v):
                    d_v_new_c[i, j] = T.float32(0.0)
                for m,v in T.Parallel(block_C, dim_v):
                    for n in T.Serial(block_C):
                        d_v_new_c[m, v] += attn[n, m] * do_c_s[n, v]
                # d_attn = do_c @ v_new_c^T (causal), ref: einsum("bhnv,bhmv->bhnm", do_c, v_new_c) => d_attn[n,m]=sum_v do_c[n,v]*v_new_c[m,v]
                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.float32(0.0)
                for v in T.Serial(dim_v):
                    for i, j in T.Parallel(block_C, block_C):
                        d_attn[i, j] += do_c_s[i, v] * v_new_c[j, v]
                for i, j in T.Parallel(block_C, block_C):
                    d_attn[i, j] = T.if_then_else(i >= j, d_attn[i, j], T.float32(0.0))

                # Step 3
                for i, j in T.Parallel(block_C, dim_k):
                    d_q_c_s[i, j] = T.float32(0.0)
                # dg_c = sum_v(do_c * o_part) via T.reduce_sum (o_part last use, overwrite)
                for i, j in T.Parallel(block_C, dim_v):
                    o_part[i, j] = do_c_s[i, j] * o_part[i, j]
                T.reduce_sum(o_part, dg_c_s, dim=1)
                for v in T.Serial(dim_v):
                    for i, kk in T.Parallel(block_C, dim_k):
                        d_q_c_s[i, kk] += do_c_s[i, v] * exp_g[i] * h_c_s[kk, v]
                # dh += q_c^T @ d_x: per (kk,v) sum over n in serial to avoid race
                for kk, v in T.Parallel(dim_k, dim_v):
                    for n in T.Serial(block_C):
                        dh_s[kk, v] += q_c_s[n, kk] * do_c_s[n, v] * exp_g[n]
                for i in T.Parallel(block_C):
                    dg_after_step3_out[bid, hid, i] = dg_c_s[i]

                # Step 4
                for i, kk in T.Parallel(block_C, dim_k):
                    for m in T.Serial(block_C):
                        d_q_c_s[i, kk] += d_attn[i, m] * k_c_s[m, kk]
                for i, j in T.Parallel(block_C, dim_k):
                    d_k_c_s[i, j] = T.float32(0.0)
                # d_k_c = d_attn^T @ q_c: per (m,kk) sum over n in serial to avoid race
                for m, kk in T.Parallel(block_C, dim_k):
                    for n in T.Serial(block_C):
                        d_k_c_s[m, kk] += d_attn[n, m] * q_c_s[n, kk]

                # Step 5: P = w_c*exp_g, dP = -d_v_new_c @ h^T
                for i, j in T.Parallel(block_C, dim_k):
                    P[i, j] = w_c_s[i, j] * exp_g[i]
                for i, j in T.Parallel(block_C, dim_k):
                    dP[i, j] = T.float32(0.0)
                for v in T.Serial(dim_v):
                    for i,kk in T.Parallel(block_C, dim_k):
                        dP[i, kk] -= d_v_new_c[i, v] * h_c_s[kk, v]
                # dh -= P^T @ d_v_new_c (torch ref: dh = dh - torch.einsum("bhnk,bhnv->bhkv", P, d_v_new_c))
                for n in T.Serial(block_C):
                    for kk, v in T.Parallel(dim_k, dim_v):
                        dh_s[kk, v] -= P[n, kk] * d_v_new_c[n, v]
                for i, j in T.Parallel(block_C, dim_k):
                    d_w_c_s[i, j] = dP[i, j] * exp_g[i]
                # dg_c += sum_k(P * dP) via T.reduce_sum (P, dP last use)
                for i, j in T.Parallel(block_C, dim_k):
                    P[i, j] = P[i, j] * dP[i, j]
                dg_step5_tmp = T.alloc_shared([block_C], accum_dtype)
                T.reduce_sum(P, dg_step5_tmp, dim=1)
                for i in T.Parallel(block_C):
                    dg_c_s[i] += dg_step5_tmp[i]
                for i in T.Parallel(block_C):
                    dg_after_step5_out[bid, hid, i] = dg_c_s[i]

                # Step 6: dh_s += dh_in*exp_g_last, d_v_new_c += k_scaled @ dh_in, d_k_scaled = v_new_c @ dh_in^T
                exp_g_last = T.exp2(g_last * _LOG2E)
                dh_in_shared = T.alloc_shared([dim_k, dim_v], accum_dtype)
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_in_shared[i, j] = dh_in[bid, hid, i, j]
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_s[i, j] = dh_s[i, j] + dh_in_shared[i, j] * exp_g_last
                for n, v in T.Parallel(block_C, dim_v):
                    for kk in T.Serial(dim_k):
                        d_v_new_c[n, v] += k_scaled[n, kk] * dh_in_shared[kk, v]
                for n, kk in T.Parallel(block_C, dim_k):
                    d_k_scaled[n, kk] = T.float32(0.0)
                for v in T.Serial(dim_v):
                    for n, kk in T.Parallel(block_C, dim_k):
                        d_k_scaled[n, kk] += v_new_c[n, v] * dh_in_shared[kk, v]
                for n, kk in T.Parallel(block_C, dim_k):
                    d_k_c_s[n, kk] = d_k_c_s[n, kk] + d_k_scaled[n, kk] * T.exp2((g_last - g_c_s[n]) * _LOG2E)
                # Compute row sums of d_k_scaled * k_scaled once via T.reduce_sum
                for n, kk in T.Parallel(block_C, dim_k):
                    d_k_scaled[n, kk] = d_k_scaled[n, kk] * k_scaled[n, kk]
                d_g_last_partial2 = T.alloc_shared([block_C], accum_dtype)
                T.reduce_sum(d_k_scaled, d_g_last_partial2, dim=1)
                for n in T.Parallel(block_C):
                    dg_c_s[n] = dg_c_s[n] - d_g_last_partial2[n]

                # d_g_last part 1: sum(dh_in * h_c * exp_g_last)
                d_g_last_partial = T.alloc_shared([dim_k], accum_dtype)
                for kk in T.Parallel(dim_k):
                    d_g_last_partial[kk] = T.float32(0.0)
                for v in T.Serial(dim_v):
                    for kk in T.Parallel(dim_k):
                        d_g_last_partial[kk] += dh_in_shared[kk, v] * h_c_s[kk, v]
                d_g_last_buf[0] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    d_g_last_buf[0] += d_g_last_partial[kk] * exp_g_last

                # d_g_last part 2: sum_n(d_g_last_partial2)
                for n in T.Serial(block_C):
                    d_g_last_buf[0] = d_g_last_buf[0] + d_g_last_partial2[n]
                dg_c_s[block_C - 1] = dg_c_s[block_C - 1] + d_g_last_buf[0]

                # du_c = d_v_new_c
                for i, j in T.Parallel(block_C, dim_v):
                    du_c[bid, hid, i, j] = d_v_new_c[i, j]
                for i, j in T.Parallel(block_C, dim_k):
                    dq_c[bid, hid, i, j] = d_q_c_s[i, j]
                    dk_c[bid, hid, i, j] = d_k_c_s[i, j]
                    dw_c[bid, hid, i, j] = d_w_c_s[i, j]
                for i in T.Parallel(block_C):
                    dg_c[bid, hid, i] = dg_c_s[i]
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_out[bid, hid, i, j] = dh_s[i, j]
                for i, j in T.Parallel(block_C, block_C):
                    d_attn_out[bid, hid, i, j] = d_attn[i, j]
                for i, j in T.Parallel(block_C, dim_k):
                    d_k_scaled_out[bid, hid, i, j] = d_k_scaled[i, j]

        @T.prim_func
        def kernel2_bwd_chunk(
            do_c: T.Tensor([batch, head, block_C, dim_v], dtype),
            q_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            k_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            g_c: T.Tensor([batch, head, block_C], dtype),
            w_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            u_c: T.Tensor([batch, head, block_C, dim_v], dtype),
            h_c: T.Tensor([batch, head, dim_k, dim_v], dtype),
            dh_in: T.Tensor([batch, head, dim_k, dim_v], dtype),
            dq_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            dk_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            dg_c: T.Tensor([batch, head, block_C], dtype),
            dw_c: T.Tensor([batch, head, block_C, dim_k], dtype),
            du_c: T.Tensor([batch, head, block_C, dim_v], dtype),
            dh_out: T.Tensor([batch, head, dim_k, dim_v], dtype),
            d_attn_out: T.Tensor([batch, head, block_C, block_C], dtype),
            d_k_scaled_out: T.Tensor([batch, head, block_C, dim_k], dtype),
            dg_after_step3_out: T.Tensor([batch, head, block_C], dtype),
            dg_after_step5_out: T.Tensor([batch, head, block_C], dtype),
        ):
            _chunk_bwd_body(do_c, q_c, k_c, g_c, w_c, u_c, h_c, dh_in, dq_c, dk_c, dg_c, dw_c, du_c, dh_out, d_attn_out, d_k_scaled_out, dg_after_step3_out, dg_after_step5_out)

        return kernel2_bwd_chunk

    return _chunk_bwd_func


def _gated_deltanet_kernel2_bwd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """TileLang kernel2 backward: do, q, k, g, w, u, S -> dq, dk, dg, dw, du.
    S is the per-chunk state buffer from the forward pass."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(
        out_idx=[-5, -4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _kernel2_bwd_func(num_stages, threads=128):
        @T.macro
        def _kernel2_bwd_body(
            do: T.Tensor([batch, head, seq_len, dim_v], dtype),
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            dq: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dg: T.Tensor([batch, head, seq_len], dtype),
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                # Shared buffers for one chunk
                q_c = T.alloc_shared([block_C, dim_k], accum_dtype)
                k_c = T.alloc_shared([block_C, dim_k], accum_dtype)
                g_c = T.alloc_shared([block_C], accum_dtype)
                w_c = T.alloc_shared([block_C, dim_k], accum_dtype)
                u_c = T.alloc_shared([block_C, dim_v], accum_dtype)
                do_c = T.alloc_shared([block_C, dim_v], accum_dtype)
                h_c = T.alloc_shared([dim_k, dim_v], accum_dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], accum_dtype)
                o_part = T.alloc_shared([block_C, dim_v], accum_dtype)
                attn = T.alloc_shared([block_C, block_C], accum_dtype)
                # Gradients (chunk)
                d_q_c = T.alloc_shared([block_C, dim_k], accum_dtype)
                d_k_c = T.alloc_shared([block_C, dim_k], accum_dtype)
                dg_c = T.alloc_shared([block_C], accum_dtype)
                d_w_c = T.alloc_shared([block_C, dim_k], accum_dtype)
                d_v_new_c = T.alloc_shared([block_C, dim_v], accum_dtype)
                d_attn = T.alloc_shared([block_C, block_C], accum_dtype)
                # Working
                exp_g = T.alloc_shared([block_C], accum_dtype)
                P = T.alloc_shared([block_C, dim_k], accum_dtype)
                dP = T.alloc_shared([block_C, dim_k], accum_dtype)
                # dh for current chunk; carry to next iteration as dh_from_next
                dh_buf = T.alloc_shared([dim_k, dim_v], accum_dtype)
                dh = T.alloc_shared([dim_k, dim_v], accum_dtype)
                k_scaled = T.alloc_shared([block_C, dim_k], accum_dtype)
                d_k_scaled = T.alloc_shared([block_C, dim_k], accum_dtype)
                d_g_last_buf = T.alloc_shared([1], accum_dtype)
                # Fragments for T.gemm
                ws_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_v_new_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                d_attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_q_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                d_k_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                d_k_scaled_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)

                # Zero dh_buf (gradient from "next" chunk; zero for last chunk)
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_buf[i, j] = T.float32(0.0)

                for t in T.Serial(num_chunks):
                    t_bwd = num_chunks - 1 - t
                    # Load chunk inputs
                    for i, j in T.Parallel(block_C, dim_k):
                        q_c[i, j] = q[bid, hid, t_bwd * block_C + i, j]
                    for i, j in T.Parallel(block_C, dim_k):
                        k_c[i, j] = k[bid, hid, t_bwd * block_C + i, j]
                    for i in T.Parallel(block_C):
                        g_c[i] = g[bid, hid, t_bwd * block_C + i]
                    for i, j in T.Parallel(block_C, dim_k):
                        w_c[i, j] = w[bid, hid, t_bwd * block_C + i, j]
                    for i, j in T.Parallel(block_C, dim_v):
                        u_c[i, j] = u[bid, hid, t_bwd * block_C + i, j]
                    for i, j in T.Parallel(block_C, dim_v):
                        do_c[i, j] = do[bid, hid, t_bwd * block_C + i, j]
                    for i, j in T.Parallel(dim_k, dim_v):
                        h_c[i, j] = S[bid, hid, t_bwd, i, j]

                    # Recompute forward: v_new_c, o_part, attn, g_last, k_scaled
                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i in T.Parallel(block_C):
                        exp_g[i] = T.exp2(g_c[i] * _LOG2E)
                    for i, j in T.Parallel(block_C, dim_v):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * exp_g[i]

                    T.clear(ws_frag)
                    T.gemm(q_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, dim_v):
                        o_part[i, j] = ws_frag[i, j] * exp_g[i]

                    T.clear(attn_frag)
                    T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        attn[i, j] = T.if_then_else(i >= j, attn_frag[i, j], T.float32(0.0))
                    g_last = g_c[block_C - 1]
                    for pn, sk in T.Parallel(block_C, dim_k):
                        k_scaled[pn, sk] = k_c[pn, sk] * T.exp2((g_last - g_c[pn]) * _LOG2E)

                    # dh = dh_from_next (from dh_buf)
                    for i, j in T.Parallel(dim_k, dim_v):
                        dh[i, j] = dh_buf[i, j]

                    # Step 2: d_v_new_c = attn^T @ do_c, d_attn = do_c @ v_new_c^T (causal)
                    T.clear(d_v_new_frag)
                    T.gemm(attn, do_c, d_v_new_frag, transpose_A=True)
                    T.copy(d_v_new_frag, d_v_new_c)

                    T.clear(d_attn_frag)
                    T.gemm(do_c, v_new_c, d_attn_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        d_attn[i, j] = T.if_then_else(i >= j, d_attn_frag[i, j], T.float32(0.0))

                    # Step 3: dg_c, d_q_c, dh
                    T.clear(d_q_c_frag)
                    # dg_c = sum_v(do_c * o_part) via T.reduce_sum (o_part last use, overwrite)
                    for i, j in T.Parallel(block_C, dim_v):
                        o_part[i, j] = do_c[i, j] * o_part[i, j]
                    T.reduce_sum(o_part, dg_c, dim=1)
                    for sv in T.Serial(dim_v):
                        for i, kk in T.Parallel(block_C, dim_k):
                            d_q_c_frag[i, kk] += do_c[i, sv] * exp_g[i] * h_c[kk, sv]
                    # dh += q_c^T @ d_x
                    for kk, sv in T.Parallel(dim_k, dim_v):
                        for n in T.Serial(block_C):
                            dh[kk, sv] += q_c[n, kk] * do_c[n, sv] * exp_g[n]

                    # Step 4: d_q_c += d_attn @ k_c, d_k_c = d_attn^T @ q_c
                    T.gemm(d_attn, k_c, d_q_c_frag)  # accumulate into d_q_c_frag
                    T.copy(d_q_c_frag, d_q_c)

                    T.clear(d_k_c_frag)
                    T.gemm(d_attn, q_c, d_k_c_frag, transpose_A=True)
                    T.copy(d_k_c_frag, d_k_c)

                    # Step 5: P = w_c*exp_g, dP = -d_v_new_c @ h_c^T, dh -= P^T @ d_v_new_c
                    for i, j in T.Parallel(block_C, dim_k):
                        P[i, j] = w_c[i, j] * exp_g[i]
                    T.clear(dP_frag)
                    T.gemm(d_v_new_c, h_c, dP_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, dim_k):
                        dP[i, j] = -dP_frag[i, j]
                    for n in T.Serial(block_C):
                        for kk, sv in T.Parallel(dim_k, dim_v):
                            dh[kk, sv] -= P[n, kk] * d_v_new_c[n, sv]
                    for i, j in T.Parallel(block_C, dim_k):
                        d_w_c[i, j] = dP[i, j] * exp_g[i]
                    # dg_c += sum_k(P * dP) via T.reduce_sum (P, dP last use)
                    for i, j in T.Parallel(block_C, dim_k):
                        P[i, j] = P[i, j] * dP[i, j]
                    dg_step5_tmp = T.alloc_shared([block_C], accum_dtype)
                    T.reduce_sum(P, dg_step5_tmp, dim=1)
                    for i in T.Parallel(block_C):
                        dg_c[i] += dg_step5_tmp[i]

                    # Step 6: h_next = h*exp(g_last)+k_scaled^T@v_new_c
                    exp_g_last = T.exp2(g_last * _LOG2E)
                    for i, j in T.Parallel(dim_k, dim_v):
                        dh[i, j] = dh[i, j] + dh_buf[i, j] * exp_g_last
                    T.copy(d_v_new_c, d_v_new_frag)
                    T.gemm(k_scaled, dh_buf, d_v_new_frag)
                    T.copy(d_v_new_frag, d_v_new_c)

                    T.clear(d_k_scaled_frag)
                    T.gemm(v_new_c, dh_buf, d_k_scaled_frag, transpose_B=True)
                    T.copy(d_k_scaled_frag, d_k_scaled)
                    for n, kk in T.Parallel(block_C, dim_k):
                        d_k_c[n, kk] = d_k_c[n, kk] + d_k_scaled[n, kk] * T.exp2((g_last - g_c[n]) * _LOG2E)
                    # Compute row sums of d_k_scaled * k_scaled once via T.reduce_sum
                    # Used for both dg_c subtraction and d_g_last addition
                    for n, kk in T.Parallel(block_C, dim_k):
                        d_k_scaled[n, kk] = d_k_scaled[n, kk] * k_scaled[n, kk]
                    d_g_last_partial2 = T.alloc_shared([block_C], accum_dtype)
                    T.reduce_sum(d_k_scaled, d_g_last_partial2, dim=1)
                    for n in T.Parallel(block_C):
                        dg_c[n] = dg_c[n] - d_g_last_partial2[n]

                    # d_g_last part 1: sum(dh_buf * h_c * exp_g_last)
                    d_g_last_partial = T.alloc_shared([dim_k], accum_dtype)
                    for kk in T.Parallel(dim_k):
                        d_g_last_partial[kk] = T.float32(0.0)
                    for s6j in T.Serial(dim_v):
                        for kk in T.Parallel(dim_k):
                            d_g_last_partial[kk] += dh_buf[kk, s6j] * h_c[kk, s6j]
                    d_g_last_buf[0] = T.float32(0.0)
                    for kk in T.Serial(dim_k):
                        d_g_last_buf[0] += d_g_last_partial[kk] * exp_g_last

                    # d_g_last part 2: sum_n(d_g_last_partial2)
                    for s6n in T.Serial(block_C):
                        d_g_last_buf[0] = d_g_last_buf[0] + d_g_last_partial2[s6n]
                    dg_c[block_C - 1] = dg_c[block_C - 1] + d_g_last_buf[0]

                    # d_u_c = d_v_new_c (after step 6)
                    for i, j in T.Parallel(block_C, dim_v):
                        du[bid, hid, t_bwd * block_C + i, j] = d_v_new_c[i, j]

                    # Write dq, dk, dg, dw
                    for i, j in T.Parallel(block_C, dim_k):
                        dq[bid, hid, t_bwd * block_C + i, j] = d_q_c[i, j]
                    for i, j in T.Parallel(block_C, dim_k):
                        dk[bid, hid, t_bwd * block_C + i, j] = d_k_c[i, j]
                    for i in T.Parallel(block_C):
                        dg[bid, hid, t_bwd * block_C + i] = dg_c[i]
                    for i, j in T.Parallel(block_C, dim_k):
                        dw[bid, hid, t_bwd * block_C + i, j] = d_w_c[i, j]

                    # Carry dh to next iteration (as dh_buf for next t_bwd)
                    for i, j in T.Parallel(dim_k, dim_v):
                        dh_buf[i, j] = dh[i, j]

        @T.prim_func
        def gated_deltanet_kernel2_bwd(
            do: T.Tensor([batch, head, seq_len, dim_v], dtype),
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            dq: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dg: T.Tensor([batch, head, seq_len], dtype),
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            _kernel2_bwd_body(do, q, k, g, w, u, S, dq, dk, dg, dw, du)

        return gated_deltanet_kernel2_bwd

    return _kernel2_bwd_func


def _gated_deltanet_bwd_from_aw_au(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aw: torch.Tensor,
    Au: torch.Tensor,
    chunk_size: int,
    num_stages: int = 2,
    threads: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full backward using TileLang kernels: compute_w_u_tl + kernel2_fwd (for S) + kernel2_bwd_chunk_tl + compute_w_u_bwd."""
    from .compute_w_u import compute_w_u_tl
    from .compute_w_u_bwd import compute_w_u_bwd_tl
    from .gated_deltanet_fwd import _gated_deltanet_kernel2_tl

    B, H, S, DK = k.shape
    _, _, _, DV = v.shape
    BC = chunk_size
    device = q.device

    # Step 1: Recompute w, u via TileLang
    k1 = compute_w_u_tl(B, H, S, BC, DK, DV, "float32")(num_stages, threads)
    w, u = k1(Aw, Au, k.float(), v.float(), beta.float())

    # Step 2: Run forward kernel2 to get S states
    S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=device)
    k2_fwd = _gated_deltanet_kernel2_tl(B, H, S, BC, DK, DV, "float32")(num_stages, threads)
    S_buf, _o = k2_fwd(q.float(), k.float(), g.float(), w, u, S_0)

    # Step 3: Run fused backward kernel (all chunks processed in single GPU launch)
    k2_bwd = _gated_deltanet_kernel2_bwd_tl(B, H, S, BC, DK, DV, "float32")(num_stages, threads)
    dq, dk, dg, dw, du = k2_bwd(do.float(), q.float(), k.float(), g.float(), w, u, S_buf)

    # Step 4: compute_w_u backward (step 9) via TileLang
    wu_bwd = compute_w_u_bwd_tl(B, H, S, BC, DK, DV, "float32")(num_stages, threads)
    _dAw, _dAu, dk_wu, dv, dbeta = wu_bwd(
        dw, du, Aw.float(), Au.float(), k.float(), v.float(), beta.float(),
    )

    dk = dk + dk_wu
    return dq, dk, dv, dg, dbeta


# ---- Custom op and Kernel class ----

@torch.library.custom_op("tileops::gated_deltanet_bwd_kernel", mutates_args=())
def _gated_deltanet_bwd_wrapped_kernel(
    batch: int, head: int, seq_len: int, chunk_size: int,
    dim_k: int, dim_v: int,
    do: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor,
    Aw: torch.Tensor, Au: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _gated_deltanet_bwd_from_aw_au(
        do, q, k, v, g, beta, Aw, Au, chunk_size,
    )


@_gated_deltanet_bwd_wrapped_kernel.register_fake
def _gated_deltanet_bwd_wrapped_kernel_fake(
    batch: int, head: int, seq_len: int, chunk_size: int,
    dim_k: int, dim_v: int,
    do: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor,
    Aw: torch.Tensor, Au: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty(batch, head, seq_len, dim_k, dtype=q.dtype, device=q.device)
    dk = torch.empty(batch, head, seq_len, dim_k, dtype=k.dtype, device=k.device)
    dv = torch.empty(batch, head, seq_len, dim_v, dtype=v.dtype, device=v.device)
    dg = torch.empty(batch, head, seq_len, dtype=g.dtype, device=g.device)
    dbeta = torch.empty(batch, head, seq_len, dtype=beta.dtype, device=beta.device)
    return dq, dk, dv, dg, dbeta


class GatedDeltaNetBwdKernel(Kernel):
    """Gated DeltaNet backward kernel.

    Full backward: do -> (dq, dk, dv, dg, dbeta).
    Chains kernel2_bwd (steps 1-8) + compute_w_u_bwd (step 9).
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
        return {}

    def forward(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        Aw: torch.Tensor,
        Au: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _gated_deltanet_bwd_wrapped_kernel(
            self.batch, self.head, self.seq_len, self.chunk_size,
            self.dim_k, self.dim_v,
            do, q, k, v, g, beta, Aw, Au,
        )
