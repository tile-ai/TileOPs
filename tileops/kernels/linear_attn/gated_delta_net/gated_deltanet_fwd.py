"""
Gated DeltaNet forward: (q, k, v, g, beta) -> output o.

  w, u: from compute_w_u (Aw, Au, k, v, beta) -> (w, u)
  Kernel 2: (q, k, g, w, u, S_0) -> (S, o); v_new and S updated iteratively inside.
"""
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

from .compute_w_u import (
    compute_w_u_tl as _gated_deltanet_kernel1_tl,
)

__all__ = ["GatedDeltaNetFwdKernel"]

_LOG2E = 1.4426950408889634


def _gated_deltanet_kernel2_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """TileLang Kernel 2: (q, k, g, w, u, S_0) -> (S, o) by chunk iteration."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _kernel2_func(num_stages, threads=128):
        @T.macro
        def _kernel2_body(
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S_0: T.Tensor([batch, head, dim_k, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                q_c = T.alloc_shared([block_C, dim_k], accum_dtype)
                k_c = T.alloc_shared([block_C, dim_k], accum_dtype)
                g_c = T.alloc_shared([block_C], accum_dtype)
                h_c = T.alloc_shared([dim_k, dim_v], accum_dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], accum_dtype)
                attn = T.alloc_shared([block_C, block_C], accum_dtype)
                h_next = T.alloc_shared([dim_k, dim_v], accum_dtype)
                u_shared = T.alloc_shared([block_C, dim_v], accum_dtype)
                w_c = T.alloc_shared([block_C, dim_k], accum_dtype)
                k_scaled_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                # Fragments for T.gemm outputs
                ws_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                o_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                h_next_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)

                for i, j in T.Parallel(dim_k, dim_v):
                    S[bid, hid, 0, i, j] = S_0[bid, hid, i, j]

                for t in T.Serial(num_chunks):
                    for i, j in T.Parallel(block_C, dim_k):
                        q_c[i, j] = q[bid, hid, t * block_C + i, j]
                    for i, j in T.Parallel(block_C, dim_k):
                        k_c[i, j] = k[bid, hid, t * block_C + i, j]
                    for i, j in T.Parallel(block_C, dim_k):
                        w_c[i, j] = w[bid, hid, t * block_C + i, j]
                    for i in T.Parallel(block_C):
                        g_c[i] = g[bid, hid, t * block_C + i]
                    for i, j in T.Parallel(block_C, dim_v):
                        u_shared[i, j] = u[bid, hid, t * block_C + i, j]
                    for i, j in T.Parallel(dim_k, dim_v):
                        h_c[i, j] = S[bid, hid, t, i, j]

                    # v_new_c = u_c - (w_c @ h_c) * exp(g_c)
                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, dim_v):
                        v_new_c[i, j] = u_shared[i, j] - ws_frag[i, j] * T.exp2(g_c[i] * _LOG2E)

                    # o_frag = (q_c @ h_c) * exp(g_c), then o_frag += attn @ v_new_c
                    T.clear(o_frag)
                    T.gemm(q_c, h_c, o_frag)
                    for i, j in T.Parallel(block_C, dim_v):
                        o_frag[i, j] = o_frag[i, j] * T.exp2(g_c[i] * _LOG2E)

                    # attn = causal(q_c @ k_c^T)
                    T.clear(attn_frag)
                    T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        attn[i, j] = T.if_then_else(i >= j, attn_frag[i, j], T.float32(0.0))

                    # o = o_frag + attn @ v_new_c
                    T.gemm(attn, v_new_c, o_frag)
                    T.copy(o_frag, o[bid, hid, t * block_C : (t + 1) * block_C, :], disable_tma=True)

                    # h_next = h_c * exp(g_last) + k_scaled^T @ v_new_c
                    g_last = g_c[block_C - 1]
                    for n, kk in T.Parallel(block_C, dim_k):
                        k_scaled_s[n, kk] = k_c[n, kk] * T.exp2((g_last - g_c[n]) * _LOG2E)
                    for i, j in T.Parallel(dim_k, dim_v):
                        h_next_frag[i, j] = h_c[i, j] * T.exp2(g_last * _LOG2E)
                    T.gemm(k_scaled_s, v_new_c, h_next_frag, transpose_A=True)
                    T.copy(h_next_frag, h_next)
                    for i, j in T.Parallel(dim_k, dim_v):
                        S[bid, hid, t + 1, i, j] = h_next[i, j]

        @T.prim_func
        def gated_deltanet_kernel2(
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S_0: T.Tensor([batch, head, dim_k, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            _kernel2_body(q, k, g, w, u, S_0, S, o)

        return gated_deltanet_kernel2

    return _kernel2_func


def _gated_deltanet_fwd_from_aw_au(
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
) -> torch.Tensor:
    """Full forward pipeline using TileLang: Kernel1 (compute_w_u) -> Kernel2 -> o."""
    B, H, S, DK = k.shape
    _, _, _, DV = v.shape

    S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
    k1 = _gated_deltanet_kernel1_tl(B, H, S, chunk_size, DK, DV, "float32")(num_stages, threads)
    k2 = _gated_deltanet_kernel2_tl(B, H, S, chunk_size, DK, DV, "float32")(num_stages, threads)
    w, u = k1(Aw, Au, k.float(), v.float(), beta.float())
    _S_buf, o = k2(q.float(), k.float(), g.float(), w, u, S_0)
    return o.to(v.dtype)


# ---- Custom op and Kernel class ----

@torch.library.custom_op("tileops::gated_deltanet_fwd_kernel", mutates_args=())
def _gated_deltanet_fwd_wrapped_kernel(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str, num_stages: int, threads: int,
    Aw: torch.Tensor, Au: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    return _gated_deltanet_fwd_from_aw_au(
        q, k, v, g, beta, Aw, Au, chunk_size,
        num_stages=num_stages, threads=threads,
    )


@_gated_deltanet_fwd_wrapped_kernel.register_fake
def _gated_deltanet_fwd_wrapped_kernel_fake(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str, num_stages: int, threads: int,
    Aw: torch.Tensor, Au: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(batch, head, seq_len, dim_v, dtype=q.dtype, device=q.device)


class GatedDeltaNetFwdKernel(Kernel):
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
        return {"num_stages": 2, "threads": 128}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        Aw: torch.Tensor,
        Au: torch.Tensor,
    ) -> torch.Tensor:
        return _gated_deltanet_fwd_wrapped_kernel(
            self.batch, self.head, self.seq_len, self.chunk_size,
            self.dim_k, self.dim_v, self.dtype_str,
            self.config["num_stages"], self.config["threads"],
            Aw, Au, q, k, v, g, beta,
        )
