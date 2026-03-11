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
                # Shared buffers in native dtype for tensor-core gemm
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
                # Gradients (chunk)
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
                dh_buf = T.alloc_shared([dim_k, dim_v], dtype)
                k_scaled = T.alloc_shared([block_C, dim_k], dtype)
                # Scalar/vector accumulators keep fp32 for precision
                d_g_last_buf = T.alloc_shared([1], accum_dtype)
                # Fragments for T.gemm (fp32 accumulation)
                ws_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_v_new_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                d_attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_q_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                d_k_c_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dh_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                dh_h_tmp = T.alloc_shared([dim_k, dim_v], dtype)  # temp for dh_buf * h_c

                # Zero dh_buf (gradient from "next" chunk; zero for last chunk)
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_buf[i, j] = T.float32(0.0)

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    # Load chunk inputs via T.copy (same dtype)
                    T.copy(q[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], q_c, disable_tma=True)
                    T.copy(k[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], k_c, disable_tma=True)
                    T.copy(g[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C], g_c, disable_tma=True)
                    T.copy(w[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], w_c, disable_tma=True)
                    T.copy(u[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], u_c, disable_tma=True)
                    T.copy(do[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], do_c, disable_tma=True)
                    T.copy(S[bid, hid, t_bwd, :, :], h_c, disable_tma=True)

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

                    # dh_frag = dh_from_next (from dh_buf)
                    T.copy(dh_buf, dh_frag)

                    # Step 2: d_v_new_c = attn^T @ do_c, d_attn = do_c @ v_new_c^T (causal)
                    T.clear(d_v_new_frag)
                    T.gemm(attn, do_c, d_v_new_frag, transpose_A=True)
                    T.copy(d_v_new_frag, d_v_new_c)

                    T.clear(d_attn_frag)
                    T.gemm(do_c, v_new_c, d_attn_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        d_attn[i, j] = T.if_then_else(i >= j, d_attn_frag[i, j], T.float32(0.0))

                    # Step 3: dg_c, d_q_c, dh_frag
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

                    # Step 4: d_q_c += d_attn @ k_c, d_k_c = d_attn^T @ q_c
                    T.gemm(d_attn, k_c, d_q_c_frag)
                    T.copy(d_q_c_frag, d_q_c)

                    T.clear(d_k_c_frag)
                    T.gemm(d_attn, q_c, d_k_c_frag, transpose_A=True)
                    T.copy(d_k_c_frag, d_k_c)

                    # Step 5: P = w_c*exp_g, dP = -d_v_new_c @ h_c^T, dh_frag -= P^T @ d_v_new_c
                    for i, j in T.Parallel(block_C, dim_k):
                        P[i, j] = w_c[i, j] * exp_g[i]
                    T.clear(dP_frag)
                    T.gemm(d_v_new_c, h_c, dP_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, dim_k):
                        dP[i, j] = -dP_frag[i, j]
                    dh_sub_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                    T.clear(dh_sub_frag)
                    T.gemm(P, d_v_new_c, dh_sub_frag, transpose_A=True)
                    for i, j in T.Parallel(dim_k, dim_v):
                        dh_frag[i, j] -= dh_sub_frag[i, j]
                    for i, j in T.Parallel(block_C, dim_k):
                        d_w_c[i, j] = dP[i, j] * exp_g[i]
                    for i, j in T.Parallel(block_C, dim_k):
                        P[i, j] = P[i, j] * dP[i, j]
                    dg_step5_tmp = T.alloc_shared([block_C], dtype)
                    T.reduce_sum(P, dg_step5_tmp, dim=1)
                    for i in T.Parallel(block_C):
                        dg_c[i] += dg_step5_tmp[i]

                    # Step 6: h_next = h*exp(g_last)+k_scaled^T@v_new_c
                    exp_g_last = T.exp2(g_last * _LOG2E)
                    for i, j in T.Parallel(dim_k, dim_v):
                        dh_frag[i, j] = dh_frag[i, j] + dh_buf[i, j] * exp_g_last
                    T.copy(d_v_new_c, d_v_new_frag)
                    T.gemm(k_scaled, dh_buf, d_v_new_frag)
                    T.copy(d_v_new_frag, d_v_new_c)

                    T.clear(dP_frag)
                    T.gemm(v_new_c, dh_buf, dP_frag, transpose_B=True)
                    T.copy(dP_frag, dP)
                    for n, kk in T.Parallel(block_C, dim_k):
                        d_k_c[n, kk] = d_k_c[n, kk] + dP[n, kk] * T.exp2((g_last - g_c[n]) * _LOG2E)
                    for n, kk in T.Parallel(block_C, dim_k):
                        dP[n, kk] = dP[n, kk] * k_scaled[n, kk]
                    d_g_last_partial2 = T.alloc_shared([block_C], dtype)
                    T.reduce_sum(dP, d_g_last_partial2, dim=1)
                    for n in T.Parallel(block_C):
                        dg_c[n] = dg_c[n] - d_g_last_partial2[n]

                    # d_g_last part 1: sum(dh_buf * h_c) * exp_g_last
                    for i, j in T.Parallel(dim_k, dim_v):
                        dh_h_tmp[i, j] = dh_buf[i, j] * h_c[i, j]
                    d_g_last_partial = T.alloc_shared([dim_k], dtype)
                    T.reduce_sum(dh_h_tmp, d_g_last_partial, dim=1)
                    d_g_last_buf[0] = T.float32(0.0)
                    for kk in T.Serial(dim_k):
                        d_g_last_buf[0] += d_g_last_partial[kk] * exp_g_last

                    # d_g_last part 2: sum_n(d_g_last_partial2)
                    for s6n in T.Serial(block_C):
                        d_g_last_buf[0] = d_g_last_buf[0] + d_g_last_partial2[s6n]
                    dg_c[block_C - 1] = dg_c[block_C - 1] + d_g_last_buf[0]

                    # Write outputs via T.copy
                    T.copy(d_v_new_c, du[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], disable_tma=True)
                    T.copy(d_q_c, dq[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], disable_tma=True)
                    T.copy(d_k_c, dk[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], disable_tma=True)
                    for i in T.Parallel(block_C):
                        dg[bid, hid, t_bwd * block_C + i] = dg_c[i]
                    T.copy(d_w_c, dw[bid, hid, t_bwd * block_C : (t_bwd + 1) * block_C, :], disable_tma=True)

                    # Carry dh to next iteration
                    T.copy(dh_frag, dh_buf)

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


def _gated_deltanet_bwd_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    S: torch.Tensor,
    chunk_size: int,
    num_stages: int = 2,
    threads: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full backward using TileLang kernels: fused_prepare_compute_w_u + kernel2_bwd + compute_w_u_bwd."""
    from .compute_w_u_bwd import compute_w_u_bwd_tl
    from .fused_prepare_compute_w_u import fused_prepare_compute_w_u_tl

    B, H, S_len, DK = k.shape
    _, _, _, DV = v.shape
    BC = chunk_size
    dtype_str = str(q.dtype).split('.')[-1]

    # Step 1: Fused prepare_wy + compute_w_u (1 kernel instead of 2)
    fused_k = fused_prepare_compute_w_u_tl(B, H, S_len, BC, DK, DV, dtype_str)(num_stages, threads)
    Aw, Au, w, u = fused_k(k, v, g, beta)

    # Step 2: Run backward kernel
    k2_bwd = _gated_deltanet_kernel2_bwd_tl(B, H, S_len, BC, DK, DV, dtype_str)(num_stages, threads)
    dq, dk, dg, dw, du = k2_bwd(do, q, k, g, w, u, S)

    # Step 4: compute_w_u backward (step 9) via TileLang
    wu_bwd = compute_w_u_bwd_tl(B, H, S_len, BC, DK, DV, dtype_str)(num_stages, threads)
    _dAw, _dAu, dk_wu, dv, dbeta = wu_bwd(
        dw, du, Aw, Au, k, v, beta,
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
    S: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _gated_deltanet_bwd_impl(
        do, q, k, v, g, beta, S, chunk_size,
    )


@_gated_deltanet_bwd_wrapped_kernel.register_fake
def _gated_deltanet_bwd_wrapped_kernel_fake(
    batch: int, head: int, seq_len: int, chunk_size: int,
    dim_k: int, dim_v: int,
    do: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor,
    S: torch.Tensor,
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
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _gated_deltanet_bwd_wrapped_kernel(
            self.batch, self.head, self.seq_len, self.chunk_size,
            self.dim_k, self.dim_v,
            do, q, k, v, g, beta, S,
        )
