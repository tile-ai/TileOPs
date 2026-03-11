"""
Gated DeltaNet forward: (q, k, v, g, beta) -> output o.

  w, u: from compute_w_u (Aw, Au, k, v, beta) -> (w, u)
  Kernel 2: (q, k, g, w, u, S_0) -> (S, o); v_new and S updated iteratively inside.
"""
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

from .compute_w_u import (
    compute_w_u_tl as _gated_deltanet_kernel1_tl,
)
from .fused_prepare_compute_w_u import fused_prepare_compute_w_u_tl

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
                # Shared buffers in native dtype for tensor-core gemm (half×half→fp32)
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                u_shared = T.alloc_shared([block_C, dim_v], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                k_scaled_s = T.alloc_shared([block_C, dim_k], dtype)
                # Fragments for T.gemm outputs (fp32 accumulation)
                ws_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                o_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                h_next_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)

                # Initialize h_c from S_0 (avoid global read inside loop)
                T.copy(S_0[bid, hid, :, :], h_c, disable_tma=True)
                for i, j in T.Parallel(dim_k, dim_v):
                    S[bid, hid, 0, i, j] = h_c[i, j]

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(q[bid, hid, t * block_C : (t + 1) * block_C, :], q_c, disable_tma=True)
                    T.copy(k[bid, hid, t * block_C : (t + 1) * block_C, :], k_c, disable_tma=True)
                    T.copy(w[bid, hid, t * block_C : (t + 1) * block_C, :], w_c, disable_tma=True)
                    T.copy(g[bid, hid, t * block_C : (t + 1) * block_C], g_c, disable_tma=True)
                    T.copy(u[bid, hid, t * block_C : (t + 1) * block_C, :], u_shared, disable_tma=True)

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
                    # Write h_next to S for backward, then reuse as h_c for next iteration
                    T.copy(h_next_frag, h_c)
                    for i, j in T.Parallel(dim_k, dim_v):
                        S[bid, hid, t + 1, i, j] = h_c[i, j]

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
    dtype_str = str(q.dtype).split('.')[-1]

    S_0 = torch.zeros(B, H, DK, DV, dtype=q.dtype, device=q.device)
    k1 = _gated_deltanet_kernel1_tl(B, H, S, chunk_size, DK, DV, dtype_str)(num_stages, threads)
    k2 = _gated_deltanet_kernel2_tl(B, H, S, chunk_size, DK, DV, dtype_str)(num_stages, threads)
    w, u = k1(Aw, Au, k, v, beta)
    S_buf, o = k2(q, k, g, w, u, S_0)
    return o, S_buf


def _gated_deltanet_fwd_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    fused_num_stages: int = 2,
    fused_threads: int = 128,
    k2_num_stages: int = 4,
    k2_threads: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused forward: prepare_wy + compute_w_u + kernel2, returns (o, S, Aw, Au)."""
    B, H, S, DK = k.shape
    _, _, _, DV = v.shape
    dtype_str = str(q.dtype).split('.')[-1]

    S_0 = torch.zeros(B, H, DK, DV, dtype=q.dtype, device=q.device)
    fused_k = fused_prepare_compute_w_u_tl(B, H, S, chunk_size, DK, DV, dtype_str)(fused_num_stages, fused_threads)
    k2 = _gated_deltanet_kernel2_tl(B, H, S, chunk_size, DK, DV, dtype_str)(k2_num_stages, k2_threads)
    Aw, Au, w, u = fused_k(k, v, g, beta)
    S_buf, o = k2(q, k, g, w, u, S_0)
    return o, S_buf, Aw, Au


# ---- Custom op and Kernel class ----

@torch.library.custom_op("tileops::gated_deltanet_fwd_kernel", mutates_args=())
def _gated_deltanet_fwd_wrapped_kernel(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    fused_num_stages: int, fused_threads: int,
    k2_num_stages: int, k2_threads: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _gated_deltanet_fwd_fused(
        q, k, v, g, beta, chunk_size,
        fused_num_stages=fused_num_stages, fused_threads=fused_threads,
        k2_num_stages=k2_num_stages, k2_threads=k2_threads,
    )


@_gated_deltanet_fwd_wrapped_kernel.register_fake
def _gated_deltanet_fwd_wrapped_kernel_fake(
    batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int,
    dtype: str,
    fused_num_stages: int, fused_threads: int,
    k2_num_stages: int, k2_threads: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_chunks = seq_len // chunk_size
    o = torch.empty(batch, head, seq_len, dim_v, dtype=q.dtype, device=q.device)
    S = torch.empty(batch, head, num_chunks + 1, dim_k, dim_v, dtype=q.dtype, device=q.device)
    Aw = torch.empty(batch, head, seq_len, chunk_size, dtype=q.dtype, device=q.device)
    Au = torch.empty_like(Aw)
    return o, S, Aw, Au


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
        # Cache JIT-compiled kernels to avoid re-creation overhead
        self._fused_fn = fused_prepare_compute_w_u_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, self.dtype_str,
        )(self.config["fused_num_stages"], self.config["fused_threads"])
        self._k2_fn = _gated_deltanet_kernel2_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, self.dtype_str,
        )(self.config["k2_num_stages"], self.config["k2_threads"])
        self._S_0 = torch.zeros(
            batch, head, dim_k, dim_v,
            dtype={"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[self.dtype_str],
            device="cuda",
        )

    @property
    def default_config(self) -> dict:
        return {
            "fused_num_stages": 2,
            "fused_threads": 128,
            "k2_num_stages": 4,
            "k2_threads": 256,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Aw, Au, w, u = self._fused_fn(k, v, g, beta)
        S_buf, o = self._k2_fn(q, k, g, w, u, self._S_0)
        return o, S_buf, Aw, Au
