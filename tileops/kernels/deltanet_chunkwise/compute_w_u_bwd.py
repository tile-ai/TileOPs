"""
Backward of compute_w_u with fused A_inv backward.

Forward: w = A_inv @ (k*beta), u = A_inv @ (v*beta) per chunk,
where A = I + strictLower(diag(beta) * KK^T) and A_inv via Neumann series.

This kernel computes dk, dv, dbeta INCLUDING the gradient through A_inv
(which depends on k, beta via the Gram matrix). Previously dAw/dAu were
output and the A_inv backward was done in Python; now it's fused here.

Backward:
  d(k*beta) = Aw^T @ dw  ->  dk_direct = d(k*beta) * beta
  d(v*beta) = Au^T @ du  ->  dv = d(v*beta) * beta
  dA_inv = dw @ (k*beta)^T + du @ (v*beta)^T
  dA = -(A_inv^T @ dA_inv @ A_inv^T)
  dP = strictLower(dA)   where P = diag(beta) * KK^T
  dk_A = diag(beta) @ dP @ k + dP^T @ diag(beta) @ k
  dbeta_A = (dP * KK^T).sum(-1)
  dk = dk_direct + dk_A
  dbeta = dbeta_direct + dbeta_A
"""

import functools

import tilelang
import tilelang.language as T

__all__ = ["compute_w_u_bwd_tl"]


@functools.lru_cache(maxsize=32)
def compute_w_u_bwd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """TileLang: backward of compute_w_u with fused A_inv backward."""
    accum_dtype = "float32"
    block_C = chunk_size

    @tilelang.jit(
        out_idx=[-3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _kernel_func(num_stages, threads=128):
        @T.prim_func
        def compute_w_u_bwd(
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du: T.Tensor([batch, head, seq_len, dim_v], dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            # Outputs: dk, dv, dbeta (A_inv backward fused in)
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dv: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dbeta: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                # --- Shared memory ---
                Aw_s = T.alloc_shared([block_C, block_C], accum_dtype)
                Au_s = T.alloc_shared([block_C, block_C], accum_dtype)
                dw_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                du_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                k_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                v_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                beta_s = T.alloc_shared([block_C], accum_dtype)
                k_beta_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                v_beta_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                d_k_beta_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                d_v_beta_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                dbeta_s = T.alloc_shared([block_C], accum_dtype)
                # A_inv backward shared
                dA_inv_s = T.alloc_shared([block_C, block_C], accum_dtype)
                dP_s = T.alloc_shared([block_C, block_C], accum_dtype)
                # --- Fragments ---
                dAw_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                dAu_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_k_beta_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                d_v_beta_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                dA_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                dk_A_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)

                # Load inputs
                T.copy(Aw[bid, hid, by * block_C : (by + 1) * block_C, :], Aw_s, disable_tma=True)
                T.copy(Au[bid, hid, by * block_C : (by + 1) * block_C, :], Au_s, disable_tma=True)
                T.copy(dw[bid, hid, by * block_C : (by + 1) * block_C, :], dw_s, disable_tma=True)
                T.copy(du[bid, hid, by * block_C : (by + 1) * block_C, :], du_s, disable_tma=True)
                T.copy(k[bid, hid, by * block_C : (by + 1) * block_C, :], k_s, disable_tma=True)
                T.copy(v[bid, hid, by * block_C : (by + 1) * block_C, :], v_s, disable_tma=True)
                T.copy(beta[bid, hid, by * block_C : (by + 1) * block_C], beta_s, disable_tma=True)

                # k_beta = k * beta, v_beta = v * beta
                for i, j in T.Parallel(block_C, dim_k):
                    k_beta_s[i, j] = k_s[i, j] * beta_s[i]
                for i, j in T.Parallel(block_C, dim_v):
                    v_beta_s[i, j] = v_s[i, j] * beta_s[i]

                # ===== Part 1: direct gradients (same as before) =====

                # dAw = dw @ k_beta^T: [BC,DK] @ [DK,BC] -> [BC,BC]
                T.clear(dAw_frag)
                T.gemm(dw_s, k_beta_s, dAw_frag, transpose_B=True)

                # d_k_beta = Aw^T @ dw: [BC,BC]^T @ [BC,DK] -> [BC,DK]
                T.clear(d_k_beta_frag)
                T.gemm(Aw_s, dw_s, d_k_beta_frag, transpose_A=True)
                T.copy(d_k_beta_frag, d_k_beta_s)

                # dAu = du @ v_beta^T: [BC,DV] @ [DV,BC] -> [BC,BC]
                T.clear(dAu_frag)
                T.gemm(du_s, v_beta_s, dAu_frag, transpose_B=True)

                # d_v_beta = Au^T @ du: [BC,BC]^T @ [BC,DV] -> [BC,DV]
                T.clear(d_v_beta_frag)
                T.gemm(Au_s, du_s, d_v_beta_frag, transpose_A=True)
                T.copy(d_v_beta_frag, d_v_beta_s)

                # dv = d_v_beta * beta
                for i, j in T.Parallel(block_C, dim_v):
                    dv[bid, hid, by * block_C + i, j] = d_v_beta_s[i, j] * beta_s[i]

                # dbeta_direct = (d_k_beta * k).sum(-1) + (d_v_beta * v).sum(-1)
                for i, j in T.Parallel(block_C, dim_k):
                    d_k_beta_s[i, j] = d_k_beta_s[i, j] * k_s[i, j]
                T.reduce_sum(d_k_beta_s, dbeta_s, dim=1)

                dbeta_v_tmp = T.alloc_shared([block_C], accum_dtype)
                for i, j in T.Parallel(block_C, dim_v):
                    d_v_beta_s[i, j] = d_v_beta_s[i, j] * v_s[i, j]
                T.reduce_sum(d_v_beta_s, dbeta_v_tmp, dim=1)

                for i in T.Parallel(block_C):
                    dbeta_s[i] = dbeta_s[i] + dbeta_v_tmp[i]

                # ===== Part 2: A_inv backward (fused) =====
                # dA_inv = dAw + dAu  [BC, BC]
                for i, j in T.Parallel(block_C, block_C):
                    dA_inv_s[i, j] = dAw_frag[i, j] + dAu_frag[i, j]

                # dA = -(Aw^T @ dA_inv @ Aw^T)  [BC, BC]
                # Step 1: tmp = dA_inv @ Aw^T -> dP_s
                T.clear(dA_frag)
                T.gemm(dA_inv_s, Aw_s, dA_frag, transpose_B=True)
                T.copy(dA_frag, dP_s)
                # Step 2: dA = Aw^T @ tmp -> use a fresh fragment
                dA_frag2 = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.clear(dA_frag2)
                T.gemm(Aw_s, dP_s, dA_frag2, transpose_A=True)

                # dP = strictLower(-dA)  (negate and mask)
                for i, j in T.Parallel(block_C, block_C):
                    dP_s[i, j] = T.if_then_else(
                        i > j,
                        -dA_frag2[i, j],
                        T.float32(0.0))

                # dk_A = diag(beta) @ dP @ k + (dP^T * diag(beta)) @ k
                # Term 1: beta * (dP @ k)  [BC, DK]
                T.clear(dk_A_frag)
                T.gemm(dP_s, k_s, dk_A_frag)
                # Reload d_k_beta for reuse as dk_A accumulator
                d_k_beta_frag2 = T.alloc_fragment([block_C, dim_k], accum_dtype)
                for i, j in T.Parallel(block_C, dim_k):
                    d_k_beta_frag2[i, j] = dk_A_frag[i, j] * beta_s[i]

                # Term 2: (dP^T * diag(beta)) @ k = dP^T @ diag(beta) @ k
                # = dP^T @ k_beta  [BC, DK]
                T.clear(dk_A_frag)
                T.gemm(dP_s, k_beta_s, dk_A_frag, transpose_A=True)
                for i, j in T.Parallel(block_C, dim_k):
                    d_k_beta_frag2[i, j] = d_k_beta_frag2[i, j] + dk_A_frag[i, j]

                # dk_direct = (Aw^T @ dw) * beta (reload from Part 1)
                # We already consumed d_k_beta_s in dbeta computation, recompute
                T.clear(d_k_beta_frag)
                T.gemm(Aw_s, dw_s, d_k_beta_frag, transpose_A=True)

                # dk = dk_direct * beta + dk_A
                for i, j in T.Parallel(block_C, dim_k):
                    dk[bid, hid, by * block_C + i, j] = (
                        d_k_beta_frag[i, j] * beta_s[i] + d_k_beta_frag2[i, j]
                    )

                # dbeta_A = (dP * KK^T).sum(-1)
                # KK^T = k @ k^T  [BC, BC]
                kkt_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.clear(kkt_frag)
                T.gemm(k_s, k_s, kkt_frag, transpose_B=True)
                # dP * KK^T element-wise, then sum rows
                for i, j in T.Parallel(block_C, block_C):
                    dP_s[i, j] = dP_s[i, j] * kkt_frag[i, j]
                dbeta_A_tmp = T.alloc_shared([block_C], accum_dtype)
                T.reduce_sum(dP_s, dbeta_A_tmp, dim=1)

                # dbeta = dbeta_direct + dbeta_A
                for i in T.Parallel(block_C):
                    dbeta[bid, hid, by * block_C + i] = dbeta_s[i] + dbeta_A_tmp[i]

        return compute_w_u_bwd

    return _kernel_func
