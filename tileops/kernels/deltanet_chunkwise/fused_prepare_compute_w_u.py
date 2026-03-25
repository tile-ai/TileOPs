"""
Fused prepare_wy_repr + compute_w_u kernel for DeltaNet (ungated).

Combines:
  prepare_wy_repr: (k, beta) -> Aw, Au  (Neumann series inversion)
  compute_w_u:     (Aw, Au, k, v, beta) -> w, u  (matmul)

Into a single kernel where Aw/Au stay in shared memory:
  fused: (k, v, beta) -> (Aw, Au, w, u)

This eliminates the Aw/Au global memory round-trip between the two kernels.
Unlike gated DeltaNet, there is no gate parameter g and no Gamma decay matrix.
"""
import functools
import math

import tilelang
import tilelang.language as T

__all__ = ["fused_prepare_compute_w_u_tl"]


@functools.lru_cache(maxsize=32)
def fused_prepare_compute_w_u_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Fused TileLang kernel: (k, v, beta) -> (Aw, Au, w, u) per chunk."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0

    @tilelang.jit(
        out_idx=[-4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _fused_func(num_stages, threads=128):
        @T.macro
        def _fused_body(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                # Shared buffers
                k_shared = T.alloc_shared([block_C, dim_k], dtype)
                v_shared = T.alloc_shared([block_C, dim_v], dtype)
                beta_shared = T.alloc_shared([block_C], dtype)
                k_beta_shared = T.alloc_shared([block_C, dim_k], dtype)
                v_beta_shared = T.alloc_shared([block_C, dim_v], dtype)
                S_shared = T.alloc_shared([block_C, block_C], dtype)
                P_shared = T.alloc_shared([block_C, block_C], dtype)
                # Fragments (fp32 accumulators)
                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                w_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                u_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)

                # Load inputs
                T.copy(k[bid, hid, by * block_C : (by + 1) * block_C, :], k_shared, disable_tma=True)
                T.copy(v[bid, hid, by * block_C : (by + 1) * block_C, :], v_shared, disable_tma=True)
                T.copy(beta[bid, hid, by * block_C : (by + 1) * block_C], beta_shared, disable_tma=True)

                # KKT = k @ k^T
                T.clear(gram_frag)
                T.gemm(k_shared, k_shared, gram_frag, transpose_B=True)

                # --- Compute A = (I + strictLower(diag(beta) * KK^T))^{-1} ---
                # No gate: P = -strictLower(diag(beta) * KK^T)
                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(
                        i > j,
                        -gram_frag[i, j] * beta_shared[i],
                        T.float32(0.0))
                for i, j in T.Parallel(block_C, block_C):
                    S_shared[i, j] = T.if_then_else(
                        i == j, T.float32(1.0), T.float32(0.0))

                for _r in T.Serial(num_rounds):
                    T.clear(temp_frag)
                    T.gemm(P_shared, S_shared, temp_frag)
                    for i, j in T.Parallel(block_C, block_C):
                        S_shared[i, j] = S_shared[i, j] + temp_frag[i, j]
                    T.clear(temp_frag)
                    T.gemm(P_shared, P_shared, temp_frag)
                    T.copy(temp_frag, P_shared)

                # S_shared = A^{-1}; write to both Aw and Au (same matrix)
                T.copy(S_shared, temp_frag)
                T.copy(temp_frag, Aw[bid, hid, by * block_C : (by + 1) * block_C, :], disable_tma=True)
                T.copy(temp_frag, Au[bid, hid, by * block_C : (by + 1) * block_C, :], disable_tma=True)

                # k_beta = k * beta
                for i_s, i_k in T.Parallel(block_C, dim_k):
                    k_beta_shared[i_s, i_k] = k_shared[i_s, i_k] * beta_shared[i_s]

                # w = A^{-1} @ k_beta
                T.clear(w_frag)
                T.gemm(S_shared, k_beta_shared, w_frag)
                T.copy(w_frag, w[bid, hid, by * block_C : (by + 1) * block_C, :], disable_tma=True)

                # v_beta = v * beta
                for i, j in T.Parallel(block_C, dim_v):
                    v_beta_shared[i, j] = v_shared[i, j] * beta_shared[i]

                # u = A^{-1} @ v_beta
                T.clear(u_frag)
                T.gemm(S_shared, v_beta_shared, u_frag)
                T.copy(u_frag, u[bid, hid, by * block_C : (by + 1) * block_C, :], disable_tma=True)

        @T.prim_func
        def fused_prepare_compute_w_u(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            _fused_body(k, v, beta, Aw, Au, w, u)

        return fused_prepare_compute_w_u

    return _fused_func
