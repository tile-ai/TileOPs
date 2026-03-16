"""GLA (Gated Linear Attention) backward kernel — TileLang implementation.

Two-pass architecture:
  Pass 1 (sequential reverse, B*H blocks): Accumulate dh per chunk, store dh_out.
  Pass 2 (parallel, B*H*NC blocks): Given h[i_c] and dh[i_c], compute dq,dk,dv,dg.
    A (intra-chunk attention) is recomputed internally — no external input needed.

h is read from the forward pass's h_out (no recomputation).

Reference:
    https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gla/chunk.py
"""

from typing import Callable, Optional, Tuple

import tilelang
import torch
from tilelang import language as T
from tilelang.profiler import do_bench

from tileops.kernels.kernel import Kernel
from tileops.kernels.linear_attn.gla.gla_fwd import _gla_precompute_g_kernel

__all__ = ["GLABwdKernel"]

LOG2_E = 1.44269504


# ---------------------------------------------------------------------------
# Pass 1: compute dh per chunk (reverse order, sequential)
# ---------------------------------------------------------------------------

def _gla_bwd_dh_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    has_initial_state: bool,
    dtype: str,
    num_v_partitions: int = 1,
) -> Callable:
    """Accumulate dh in reverse chunk order, store per-chunk dh.

    Sequential (B*H*Vp blocks) because dh has inter-chunk dependency.
    V-partition parallelism splits the V dimension across thread blocks.
    Stores dh_out[i_c] = dh after adding chunk i_c's contribution (before decay).
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    dim_v_part = dim_v // num_v_partitions

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _dh_func(num_stages, threads=128):
        q_shape = [batch, seq_len, heads, dim_k]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        do_shape = [batch, seq_len, heads, dim_v]
        dht_shape = [batch, heads, dim_k, dim_v]
        dh_out_shape = [batch, num_chunks, heads, dim_k, dim_v]
        dh0_shape = [batch, heads, dim_k, dim_v]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            do: T.Tensor(do_shape, dtype),
            dht: T.Tensor(dht_shape, accum_dtype),
            dh_out: T.Tensor(dh_out_shape, accum_dtype),
            dh0: T.Tensor(dh0_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_v_partitions,
                          threads=threads) as bx:
                i_b = bx // (heads * num_v_partitions)
                i_h = (bx // num_v_partitions) % heads
                i_vp = bx % num_v_partitions
                v_offset = i_vp * dim_v_part

                dh_s = T.alloc_shared([dim_k, dim_v_part], accum_dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                q_s = T.alloc_shared([chunk_size, dim_k], dtype)
                do_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                q_gated_s = T.alloc_shared([chunk_size, dim_k], dtype)

                # Load dht V-slice
                for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                    dh_s[i_k, i_v] = dht[i_b, i_h, i_k, v_offset + i_v]

                for t in T.Serial(num_chunks):
                    i_c = num_chunks - 1 - t
                    chunk_start = i_c * chunk_size

                    T.copy(q[i_b, chunk_start:chunk_start + chunk_size,
                             i_h, :],
                           q_s, disable_tma=True)
                    T.copy(do[i_b, chunk_start:chunk_start + chunk_size,
                              i_h, v_offset:v_offset + dim_v_part],
                           do_s, disable_tma=True)
                    T.copy(g_cumsum[i_b,
                                    chunk_start:chunk_start + chunk_size,
                                    i_h, :],
                           g_cumsum_s, disable_tma=True)

                    g_last = T.alloc_fragment([dim_k], accum_dtype)
                    for i_k in T.Parallel(dim_k):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]

                    # q_gated (redundant across V-partitions)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k):
                        q_gated_s[i_t, i_k] = T.cast(
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.exp2(g_cumsum_s[i_t, i_k] * LOG2_E),
                            dtype)

                    # dh += scale * q_gated^T @ do_slice
                    dh_delta = T.alloc_fragment([dim_k, dim_v_part],
                                               accum_dtype)
                    T.fill(dh_delta, 0.0)
                    T.gemm(q_gated_s, do_s, dh_delta, transpose_A=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh_s[i_k, i_v] = (dh_s[i_k, i_v]
                                          + scale * dh_delta[i_k, i_v])

                    # Store dh BEFORE decay
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh_out[i_b, i_c, i_h, i_k,
                               v_offset + i_v] = dh_s[i_k, i_v]

                    # Decay for next (earlier) chunk
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh_s[i_k, i_v] = (dh_s[i_k, i_v]
                                          * T.exp2(g_last[i_k] * LOG2_E))

                # Write dh0
                if has_initial_state:
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh0[i_b, i_h, i_k, v_offset + i_v] = dh_s[i_k, i_v]
                else:
                    for i_k, i_v in T.Parallel(dim_k, dim_v_part):
                        dh0[i_b, i_h, i_k, v_offset + i_v] = 0.0

        return _main

    return _dh_func


# ---------------------------------------------------------------------------
# Pass 2: fused intra+inter kernel (eliminates global memory round-trip)
# ---------------------------------------------------------------------------

def _gla_bwd_fused_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    scale: float,
    dtype: str,
    sub_chunk_size: int = 16,
) -> Callable:
    """Fused intra+inter backward kernel.

    Phase A: Compute dq_intra, dk_intra, dv_intra using sub-chunk GEMM tiling
             (kept in registers — no global memory write).
    Phase B: Add inter-chunk contributions via GEMM, compute dg, write final outputs.

    This avoids the global memory round-trip of the split intra/inter approach.
    """
    accum_dtype = "float32"
    num_chunks = seq_len // chunk_size
    BT = chunk_size
    BC = sub_chunk_size
    NS = BT // BC

    @tilelang.jit(
        out_idx=[-4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _fused_func(num_stages, threads=256):
        q_shape = [batch, seq_len, heads, dim_k]
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        do_shape = [batch, seq_len, heads, dim_v]
        h_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]
        dh_shape = [batch, num_chunks, heads, dim_k, dim_v]
        dq_shape = [batch, seq_len, heads, dim_k]
        dk_shape = [batch, seq_len, heads, dim_k]
        dv_shape = [batch, seq_len, heads, dim_v]
        dg_shape = [batch, seq_len, heads, dim_k]

        @T.prim_func
        def _main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(k_shape, dtype),
            v: T.Tensor(v_shape, dtype),
            g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype),
            do: T.Tensor(do_shape, dtype),
            h: T.Tensor(h_shape, accum_dtype),
            dh: T.Tensor(dh_shape, accum_dtype),
            dq_out: T.Tensor(dq_shape, accum_dtype),
            dk_out: T.Tensor(dk_shape, accum_dtype),
            dv_out: T.Tensor(dv_shape, accum_dtype),
            dg_out: T.Tensor(dg_shape, accum_dtype),
        ):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = (bx // num_chunks) % heads
                i_c = bx % num_chunks
                chunk_start = i_c * BT

                q_s = T.alloc_shared([BT, dim_k], dtype)
                k_s = T.alloc_shared([BT, dim_k], dtype)
                v_s = T.alloc_shared([BT, dim_v], dtype)
                do_s = T.alloc_shared([BT, dim_v], dtype)
                g_cumsum_s = T.alloc_shared([BT, dim_k], accum_dtype)
                A_s = T.alloc_shared([BT, BT], dtype)

                T.copy(q[i_b, chunk_start:chunk_start + BT, i_h, :],
                       q_s, disable_tma=True)
                T.copy(k[i_b, chunk_start:chunk_start + BT, i_h, :],
                       k_s, disable_tma=True)
                T.copy(v[i_b, chunk_start:chunk_start + BT, i_h, :],
                       v_s, disable_tma=True)
                T.copy(do[i_b, chunk_start:chunk_start + BT, i_h, :],
                       do_s, disable_tma=True)
                T.copy(g_cumsum[i_b, chunk_start:chunk_start + BT, i_h, :],
                       g_cumsum_s, disable_tma=True)

                # ============================================================
                # PHASE A: Intra-chunk (results kept in fragments)
                # ============================================================

                # ---- A[i,j] = scale * sum_k q*k*exp(g_i - g_j), causal ----
                A_frag = T.alloc_fragment([BT, BT], accum_dtype)
                T.fill(A_frag, 0.0)
                for i_k in T.Serial(dim_k):
                    for i_t, i_j in T.Parallel(BT, BT):
                        A_frag[i_t, i_j] = A_frag[i_t, i_j] + (
                            T.cast(q_s[i_t, i_k], accum_dtype)
                            * T.cast(k_s[i_j, i_k], accum_dtype)
                            * T.exp2((g_cumsum_s[i_t, i_k]
                                      - g_cumsum_s[i_j, i_k]) * LOG2_E))
                for i_t, i_j in T.Parallel(BT, BT):
                    A_s[i_t, i_j] = T.cast(
                        T.if_then_else(
                            i_j <= i_t,
                            A_frag[i_t, i_j] * scale,
                            0.0),
                        dtype)

                # dv_intra = A^T @ do (keep in fragment for phase B)
                dv_frag = T.alloc_fragment([BT, dim_v], accum_dtype)
                T.fill(dv_frag, 0.0)
                T.gemm(A_s, do_s, dv_frag, transpose_A=True,
                       policy=T.GemmWarpPolicy.FullRow)

                # dA = scale * do @ v^T, causal (overwrite A_s)
                dA_frag = T.alloc_fragment([BT, BT], accum_dtype)
                T.fill(dA_frag, 0.0)
                T.gemm(do_s, v_s, dA_frag, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_j in T.Parallel(BT, BT):
                    A_s[i_t, i_j] = T.cast(T.if_then_else(
                        i_j <= i_t,
                        scale * dA_frag[i_t, i_j],
                        0.0,
                    ), dtype)

                # Sub-chunk tiled dq_intra (kept in fragment)
                dq_frag = T.alloc_fragment([BT, dim_k], accum_dtype)
                T.fill(dq_frag, 0.0)
                dA_sub = T.alloc_shared([BC, BC], dtype)
                k_shifted_sub = T.alloc_shared([BC, dim_k], dtype)

                for s_i in T.Serial(NS):
                    dq_sub = T.alloc_fragment([BC, dim_k], accum_dtype)
                    T.fill(dq_sub, 0.0)

                    for s_j in T.Serial(NS):
                        if s_j < s_i:
                            for i_t, i_j in T.Parallel(BC, BC):
                                dA_sub[i_t, i_j] = A_s[
                                    s_i * BC + i_t, s_j * BC + i_j]
                            for i_t, i_k in T.Parallel(BC, dim_k):
                                k_shifted_sub[i_t, i_k] = T.cast(
                                    T.cast(k_s[s_j * BC + i_t, i_k],
                                           accum_dtype)
                                    * T.exp2((g_cumsum_s[s_i * BC, i_k]
                                              - g_cumsum_s[
                                                  s_j * BC + i_t, i_k])
                                             * LOG2_E),
                                    dtype)
                            T.gemm(dA_sub, k_shifted_sub, dq_sub,
                                   policy=T.GemmWarpPolicy.FullRow)

                    for i_local, i_k in T.Parallel(BC, dim_k):
                        dq_sub[i_local, i_k] = (
                            dq_sub[i_local, i_k]
                            * T.exp2((g_cumsum_s[
                                s_i * BC + i_local, i_k]
                                - g_cumsum_s[s_i * BC, i_k])
                                * LOG2_E))

                    # Diagonal: fragment-based to avoid bf16 smem codegen bug
                    for j_local in T.Serial(BC):
                        dA_col = T.alloc_fragment([BC], accum_dtype)
                        k_row = T.alloc_fragment([dim_k], accum_dtype)
                        g_j = T.alloc_fragment([dim_k], accum_dtype)
                        for i_local in T.Parallel(BC):
                            dA_col[i_local] = T.if_then_else(
                                j_local <= i_local,
                                T.cast(A_s[s_i * BC + i_local,
                                           s_i * BC + j_local],
                                       accum_dtype),
                                0.0)
                        for i_k in T.Parallel(dim_k):
                            k_row[i_k] = T.cast(
                                k_s[s_i * BC + j_local, i_k], accum_dtype)
                            g_j[i_k] = g_cumsum_s[s_i * BC + j_local, i_k]
                        for i_local, i_k in T.Parallel(BC, dim_k):
                            dq_sub[i_local, i_k] = (
                                dq_sub[i_local, i_k]
                                + dA_col[i_local] * k_row[i_k]
                                * T.exp2((g_cumsum_s[
                                    s_i * BC + i_local, i_k]
                                    - g_j[i_k]) * LOG2_E))

                    for i_local, i_k in T.Parallel(BC, dim_k):
                        dq_frag[s_i * BC + i_local, i_k] = (
                            dq_sub[i_local, i_k])

                # Sub-chunk tiled dk_intra (kept in fragment)
                dk_frag = T.alloc_fragment([BT, dim_k], accum_dtype)
                T.fill(dk_frag, 0.0)
                q_shifted_sub = T.alloc_shared([BC, dim_k], dtype)

                for s_j in T.Serial(NS):
                    dk_sub = T.alloc_fragment([BC, dim_k], accum_dtype)
                    T.fill(dk_sub, 0.0)

                    for s_i in T.Serial(NS):
                        if s_i > s_j:
                            for i_i, i_j in T.Parallel(BC, BC):
                                dA_sub[i_i, i_j] = A_s[
                                    s_i * BC + i_i, s_j * BC + i_j]
                            for i_t, i_k in T.Parallel(BC, dim_k):
                                q_shifted_sub[i_t, i_k] = T.cast(
                                    T.cast(q_s[s_i * BC + i_t, i_k],
                                           accum_dtype)
                                    * T.exp2((g_cumsum_s[
                                        s_i * BC + i_t, i_k]
                                        - g_cumsum_s[
                                            (s_j + 1) * BC - 1, i_k])
                                        * LOG2_E),
                                    dtype)
                            T.gemm(dA_sub, q_shifted_sub, dk_sub,
                                   transpose_A=True,
                                   policy=T.GemmWarpPolicy.FullRow)

                    for j_local, i_k in T.Parallel(BC, dim_k):
                        dk_sub[j_local, i_k] = (
                            dk_sub[j_local, i_k]
                            * T.exp2((g_cumsum_s[
                                (s_j + 1) * BC - 1, i_k]
                                - g_cumsum_s[
                                    s_j * BC + j_local, i_k])
                                * LOG2_E))

                    # Diagonal: fragment-based
                    for i_local in T.Serial(BC):
                        dA_row = T.alloc_fragment([BC], accum_dtype)
                        q_row = T.alloc_fragment([dim_k], accum_dtype)
                        g_i = T.alloc_fragment([dim_k], accum_dtype)
                        for j_local in T.Parallel(BC):
                            dA_row[j_local] = T.if_then_else(
                                j_local <= i_local,
                                T.cast(A_s[s_j * BC + i_local,
                                           s_j * BC + j_local],
                                       accum_dtype),
                                0.0)
                        for i_k in T.Parallel(dim_k):
                            q_row[i_k] = T.cast(
                                q_s[s_j * BC + i_local, i_k], accum_dtype)
                            g_i[i_k] = g_cumsum_s[s_j * BC + i_local, i_k]
                        for j_local, i_k in T.Parallel(BC, dim_k):
                            dk_sub[j_local, i_k] = (
                                dk_sub[j_local, i_k]
                                + dA_row[j_local] * q_row[i_k]
                                * T.exp2((g_i[i_k]
                                    - g_cumsum_s[
                                        s_j * BC + j_local, i_k])
                                    * LOG2_E))

                    for j_local, i_k in T.Parallel(BC, dim_k):
                        dk_frag[s_j * BC + j_local, i_k] = (
                            dk_sub[j_local, i_k])

                # ============================================================
                # PHASE B: Inter-chunk gradients + combine + dg
                # ============================================================

                h_cast_s = T.alloc_shared([dim_k, dim_v], dtype)
                dh_cast_s = T.alloc_shared([dim_k, dim_v], dtype)

                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    h_cast_s[i_k, i_v] = T.cast(
                        h[i_b, i_c, i_h, i_k, i_v], dtype)
                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    dh_cast_s[i_k, i_v] = T.cast(
                        dh[i_b, i_c, i_h, i_k, i_v], dtype)

                g_last = T.alloc_fragment([dim_k], accum_dtype)
                for i_k in T.Parallel(dim_k):
                    g_last[i_k] = g_cumsum_s[BT - 1, i_k]

                # dv_inter = k_adj @ dh
                k_gated_s = T.alloc_shared([BT, dim_k], dtype)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    k_gated_s[i_t, i_k] = T.cast(
                        T.cast(k_s[i_t, i_k], accum_dtype)
                        * T.exp2((g_last[i_k]
                                  - g_cumsum_s[i_t, i_k]) * LOG2_E),
                        dtype)

                T.gemm(k_gated_s, dh_cast_s, dv_frag,
                       policy=T.GemmWarpPolicy.FullRow)
                # dv_frag now = dv_intra + dv_inter (accumulated)
                for i_t, i_v in T.Parallel(BT, dim_v):
                    dv_out[i_b, chunk_start + i_t, i_h, i_v] = (
                        dv_frag[i_t, i_v])

                # dq_inter = do @ h^T → write to shared to avoid layout conflict
                dq_inter_s = T.alloc_shared([BT, dim_k], accum_dtype)
                dq_inter_frag = T.alloc_fragment([BT, dim_k], accum_dtype)
                T.fill(dq_inter_frag, 0.0)
                T.gemm(do_s, h_cast_s, dq_inter_frag, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dq_inter_s[i_t, i_k] = dq_inter_frag[i_t, i_k]

                # dq = dq_intra + scale * dq_inter * exp(g_cumsum)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dq_frag[i_t, i_k] = (
                        dq_frag[i_t, i_k]
                        + scale * dq_inter_s[i_t, i_k]
                        * T.exp2(g_cumsum_s[i_t, i_k] * LOG2_E))
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dq_out[i_b, chunk_start + i_t, i_h, i_k] = (
                        dq_frag[i_t, i_k])

                # dk_inter = v @ dh^T → write to shared to avoid layout conflict
                dk_inter_s = T.alloc_shared([BT, dim_k], accum_dtype)
                dk_inter_frag = T.alloc_fragment([BT, dim_k], accum_dtype)
                T.fill(dk_inter_frag, 0.0)
                T.gemm(v_s, dh_cast_s, dk_inter_frag, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dk_inter_s[i_t, i_k] = dk_inter_frag[i_t, i_k]

                # dk = dk_intra + dk_inter * exp(g_last - g_cumsum)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dk_frag[i_t, i_k] = (
                        dk_frag[i_t, i_k]
                        + dk_inter_s[i_t, i_k]
                        * T.exp2((g_last[i_k]
                                  - g_cumsum_s[i_t, i_k]) * LOG2_E))
                for i_t, i_k in T.Parallel(BT, dim_k):
                    dk_out[i_b, chunk_start + i_t, i_h, i_k] = (
                        dk_frag[i_t, i_k])

                # ==== dg ====
                dg_inter = T.alloc_shared([dim_k], accum_dtype)
                for i_k in T.Parallel(dim_k):
                    dg_inter[i_k] = 0.0
                for i_v2 in T.Serial(dim_v):
                    for i_k in T.Parallel(dim_k):
                        dg_inter[i_k] = dg_inter[i_k] + (
                            h[i_b, i_c, i_h, i_k, i_v2]
                            * T.cast(dh[i_b, i_c, i_h, i_k, i_v2],
                                     accum_dtype))
                for i_k in T.Parallel(dim_k):
                    dg_inter[i_k] = (dg_inter[i_k]
                                     * T.exp2(g_last[i_k] * LOG2_E))

                # Correction: k * dk_inter_gated
                corr_s = T.alloc_shared([BT, dim_k], accum_dtype)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    corr_s[i_t, i_k] = (
                        T.cast(k_s[i_t, i_k], accum_dtype)
                        * dk_inter_s[i_t, i_k]
                        * T.exp2((g_last[i_k]
                                  - g_cumsum_s[i_t, i_k]) * LOG2_E))
                for i_t in T.Serial(BT):
                    for i_k in T.Parallel(dim_k):
                        dg_inter[i_k] = dg_inter[i_k] + corr_s[i_t, i_k]

                # dg_local = q * dq - k * dk (using final combined values)
                for i_t, i_k in T.Parallel(BT, dim_k):
                    g_cumsum_s[i_t, i_k] = (
                        T.cast(q_s[i_t, i_k], accum_dtype)
                        * dq_frag[i_t, i_k]
                        - T.cast(k_s[i_t, i_k], accum_dtype)
                        * dk_frag[i_t, i_k])

                # Reverse cumsum
                for s in T.Serial(BT - 1):
                    i_t_rev = BT - 2 - s
                    for i_k in T.Parallel(dim_k):
                        g_cumsum_s[i_t_rev, i_k] = (
                            g_cumsum_s[i_t_rev, i_k]
                            + g_cumsum_s[i_t_rev + 1, i_k])

                for i_t, i_k in T.Parallel(BT, dim_k):
                    dg_out[i_b, chunk_start + i_t, i_h, i_k] = (
                        g_cumsum_s[i_t, i_k] + dg_inter[i_k])

        return _main

    return _fused_func


class GLABwdKernel(Kernel):
    """GLA backward kernel — two-pass architecture.

    Pass 1 (sequential reverse, B*H blocks): Accumulate dh per chunk.
    Pass 2 (parallel, B*H*NC blocks): Fused intra+inter kernel computes
        dq, dk, dv, dg in a single pass using sub-chunk GEMM tiling.

    h is read from forward's h_out (no recomputation needed).
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale if scale > 0 else dim_k**-0.5
        self.dtype = dtype
        self.dtype_name = str(dtype).split('.')[-1]
        self.init_config(config, tune)
        if not tune:
            self._build_kernels(self.config)

    @property
    def default_config(self) -> dict:
        return {"num_stages": 1, "threads_par": 128, "threads_seq": 256,
                "num_v_partitions": 4}

    @property
    def autotune_configs(self) -> list[dict]:
        configs = []
        for ns in [1, 2, 3]:
            for t_par in [64, 128, 256]:
                for t_seq in [64, 128, 256]:
                    for nvp in [2, 4, 8]:
                        configs.append({
                            "num_stages": ns,
                            "threads_par": t_par,
                            "threads_seq": t_seq,
                            "num_v_partitions": nvp,
                        })
        return configs

    def _build_kernels(self, config: dict) -> None:
        """Rebuild all sub-kernels from a config dict."""
        ns = config.get("num_stages", 2)
        thr_seq = config.get("threads_seq", config.get("threads", 256))
        thr_par = config.get("threads_par", config.get("threads", 256))
        num_vp = config.get("num_v_partitions", 4)
        self._g_fn = _gla_precompute_g_kernel(
            self.batch, self.seq_len, self.heads, self.dim_k,
            self.chunk_size, self.dtype_name,
        )(ns, thr_par)
        self._dh_fn = _gla_bwd_dh_kernel(
            self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v,
            self.chunk_size, self.scale, False, self.dtype_name,
            num_v_partitions=num_vp,
        )(1, thr_seq)
        self._dh_fn_with_init = _gla_bwd_dh_kernel(
            self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v,
            self.chunk_size, self.scale, True, self.dtype_name,
            num_v_partitions=num_vp,
        )(1, thr_seq)
        self._fused_fn = _gla_bwd_fused_kernel(
            self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v,
            self.chunk_size, self.scale, self.dtype_name,
        )(ns, thr_par)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Custom autotuning for multi-kernel backward pass."""
        if self.autotune_configs is None:
            return
        print(f'Start autotuning {self.__class__.__name__} '
              f'({len(self.autotune_configs)} configs)...')

        B, T, H, K, V = (self.batch, self.seq_len, self.heads,
                          self.dim_k, self.dim_v)
        BT = self.chunk_size
        NT = T // BT
        dtype_torch = self.dtype

        # Generate representative inputs
        q = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=dtype_torch) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype_torch).abs()
        h = torch.randn(B, NT + 1, H, K, V, device="cuda",
                         dtype=torch.float32) * 0.01
        do = torch.randn(B, T, H, V, device="cuda", dtype=dtype_torch) * 0.1
        dht = torch.zeros(B, H, K, V, dtype=torch.float32, device="cuda")

        best_lat = float('inf')
        best_cfg = None

        for cfg in self.autotune_configs:
            try:
                self._build_kernels(cfg)

                # Warmup run
                self.forward(q, k, v, g, h, do, dht)
                torch.cuda.synchronize()

                lat = do_bench(
                    lambda: self.forward(q, k, v, g, h, do, dht),
                    warmup=warmup, rep=rep,
                )
                print(f'  config={cfg} -> {lat:.3f}ms')
                if lat < best_lat:
                    best_lat = lat
                    best_cfg = cfg
            except Exception as e:
                print(f'  config={cfg} -> FAILED: {e}')
                continue

        if best_cfg is not None:
            self.config = best_cfg
            self._build_kernels(best_cfg)
            print(f'Best config: {best_cfg} ({best_lat:.3f}ms)')
        else:
            print('Autotuning failed, using default config')
            self.config = self.default_config
            self._build_kernels(self.config)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        has_initial_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype_torch = self.dtype

        # Pre-compute g_cumsum (parallel, fast)
        g_cumsum = self._g_fn(g.to(dtype_torch))

        # Pass 1: compute dh per chunk (sequential reverse)
        dh_fn = self._dh_fn_with_init if has_initial_state else self._dh_fn
        dh_out, dh0 = dh_fn(
            q.to(dtype_torch), g_cumsum, do.to(dtype_torch), dht,
        )

        # Pass 2: fused intra+inter (dq, dk, dv, dg in one kernel)
        dq, dk, dv, dg = self._fused_fn(
            q.to(dtype_torch), k.to(dtype_torch), v.to(dtype_torch),
            g_cumsum, do.to(dtype_torch),
            h.to(torch.float32), dh_out,
        )

        return dq, dk, dv, dg
