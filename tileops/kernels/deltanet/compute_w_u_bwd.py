"""
Fused backward of compute_w_u + A_inv backward + dw_corr + dk merge.

This kernel absorbs all post-recurrence operations into a single parallel
TileLang kernel, eliminating Python-level batched matmuls and tensor adds.

Inputs (all pre-computed by bwd_parallel and dh_recurrence):
  dw, du_partial, du_corr, S, Aw, Au, k, v, beta, dk_partial, dk_corr

Internal computations per chunk:
  1. du = du_partial + du_corr
  2. dw_corr = -(du_corr @ S^T);  dw_total = dw + dw_corr
  3. wu_bwd:  dk_wu = (Aw^T @ dw_total) * beta,  dv = (Au^T @ du) * beta
  4. A_inv backward:  dAw + dAu → dA → dP → dk_A, dbeta_A
  5. dk = dk_partial + dk_corr + dk_wu + dk_A
  6. dbeta = dbeta_direct + dbeta_A

Outputs: dk, dv, dbeta
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
    """TileLang: fused wu_bwd + A_inv backward + dw_corr + dk merge."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

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
            du_partial: T.Tensor([batch, head, seq_len, dim_v], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], accum_dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            dk_partial: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dk_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
            # Outputs
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dv: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dbeta: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(batch, head, num_chunks, threads=threads) as (bid, hid, by):
                # --- Shared memory ---
                Aw_s = T.alloc_shared([block_C, block_C], accum_dtype)
                Au_s = T.alloc_shared([block_C, block_C], accum_dtype)
                dw_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                du_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                du_corr_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                S_s = T.alloc_shared([dim_k, dim_v], accum_dtype)
                k_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                v_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                beta_s = T.alloc_shared([block_C], accum_dtype)
                k_beta_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                v_beta_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                dbeta_s = T.alloc_shared([block_C], accum_dtype)
                dP_s = T.alloc_shared([block_C, block_C], accum_dtype)
                dk_partial_s = T.alloc_shared([block_C, dim_k], dtype)
                dk_corr_s = T.alloc_shared([block_C, dim_k], dtype)
                # --- Fragments ---
                dAw_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                dAu_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_k_beta_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                d_v_beta_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                dk_A_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dw_corr_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)

                # Load inputs
                T.copy(Aw[bid, hid, by * block_C : (by + 1) * block_C, :], Aw_s, disable_tma=True)
                T.copy(Au[bid, hid, by * block_C : (by + 1) * block_C, :], Au_s, disable_tma=True)
                T.copy(dw[bid, hid, by * block_C : (by + 1) * block_C, :], dw_s, disable_tma=True)
                T.copy(du_partial[bid, hid, by * block_C : (by + 1) * block_C, :], du_s, disable_tma=True)
                T.copy(du_corr[bid, hid, by * block_C : (by + 1) * block_C, :], du_corr_s, disable_tma=True)
                T.copy(S[bid, hid, by, :, :], S_s, disable_tma=True)
                T.copy(k[bid, hid, by * block_C : (by + 1) * block_C, :], k_s, disable_tma=True)
                T.copy(v[bid, hid, by * block_C : (by + 1) * block_C, :], v_s, disable_tma=True)
                T.copy(beta[bid, hid, by * block_C : (by + 1) * block_C], beta_s, disable_tma=True)
                T.copy(dk_partial[bid, hid, by * block_C : (by + 1) * block_C, :], dk_partial_s, disable_tma=True)
                T.copy(dk_corr[bid, hid, by * block_C : (by + 1) * block_C, :], dk_corr_s, disable_tma=True)

                # Step 1: du = du_partial + du_corr
                for i, j in T.Parallel(block_C, dim_v):
                    du_s[i, j] = du_s[i, j] + du_corr_s[i, j]

                # Step 2: dw_corr = -(du_corr @ S^T), dw_total = dw + dw_corr
                T.clear(dw_corr_frag)
                T.gemm(du_corr_s, S_s, dw_corr_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, dim_k):
                    dw_s[i, j] = dw_s[i, j] - dw_corr_frag[i, j]

                # Prepare k_beta, v_beta
                for i, j in T.Parallel(block_C, dim_k):
                    k_beta_s[i, j] = k_s[i, j] * beta_s[i]
                for i, j in T.Parallel(block_C, dim_v):
                    v_beta_s[i, j] = v_s[i, j] * beta_s[i]

                # ===== wu_bwd: direct gradients =====
                # dAw = dw @ k_beta^T
                T.clear(dAw_frag)
                T.gemm(dw_s, k_beta_s, dAw_frag, transpose_B=True)

                # d_k_beta = Aw^T @ dw  (for dk_direct and dbeta)
                T.clear(d_k_beta_frag)
                T.gemm(Aw_s, dw_s, d_k_beta_frag, transpose_A=True)

                # dAu = du @ v_beta^T
                T.clear(dAu_frag)
                T.gemm(du_s, v_beta_s, dAu_frag, transpose_B=True)

                # d_v_beta = Au^T @ du
                T.clear(d_v_beta_frag)
                T.gemm(Au_s, du_s, d_v_beta_frag, transpose_A=True)

                # dv = d_v_beta * beta
                for i, j in T.Parallel(block_C, dim_v):
                    dv[bid, hid, by * block_C + i, j] = d_v_beta_frag[i, j] * beta_s[i]

                # dbeta_direct = (d_k_beta * k).sum(-1) + (d_v_beta * v).sum(-1)
                d_k_beta_s = T.alloc_shared([block_C, dim_k], accum_dtype)
                T.copy(d_k_beta_frag, d_k_beta_s)
                for i, j in T.Parallel(block_C, dim_k):
                    d_k_beta_s[i, j] = d_k_beta_s[i, j] * k_s[i, j]
                T.reduce_sum(d_k_beta_s, dbeta_s, dim=1)

                d_v_beta_s = T.alloc_shared([block_C, dim_v], accum_dtype)
                T.copy(d_v_beta_frag, d_v_beta_s)
                for i, j in T.Parallel(block_C, dim_v):
                    d_v_beta_s[i, j] = d_v_beta_s[i, j] * v_s[i, j]
                dbeta_v_tmp = T.alloc_shared([block_C], accum_dtype)
                T.reduce_sum(d_v_beta_s, dbeta_v_tmp, dim=1)
                for i in T.Parallel(block_C):
                    dbeta_s[i] = dbeta_s[i] + dbeta_v_tmp[i]

                # ===== A_inv backward =====
                dA_inv_s = T.alloc_shared([block_C, block_C], accum_dtype)
                for i, j in T.Parallel(block_C, block_C):
                    dA_inv_s[i, j] = dAw_frag[i, j] + dAu_frag[i, j]

                # dA = -(Aw^T @ dA_inv @ Aw^T)
                dA_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.clear(dA_frag)
                T.gemm(dA_inv_s, Aw_s, dA_frag, transpose_B=True)
                T.copy(dA_frag, dP_s)
                dA_frag2 = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.clear(dA_frag2)
                T.gemm(Aw_s, dP_s, dA_frag2, transpose_A=True)

                for i, j in T.Parallel(block_C, block_C):
                    dP_s[i, j] = T.if_then_else(i > j, -dA_frag2[i, j], T.float32(0.0))

                # dk_A = beta * (dP @ k) + dP^T @ k_beta
                T.clear(dk_A_frag)
                T.gemm(dP_s, k_s, dk_A_frag)
                dk_A_frag2 = T.alloc_fragment([block_C, dim_k], accum_dtype)
                for i, j in T.Parallel(block_C, dim_k):
                    dk_A_frag2[i, j] = dk_A_frag[i, j] * beta_s[i]
                T.clear(dk_A_frag)
                T.gemm(dP_s, k_beta_s, dk_A_frag, transpose_A=True)
                for i, j in T.Parallel(block_C, dim_k):
                    dk_A_frag2[i, j] = dk_A_frag2[i, j] + dk_A_frag[i, j]

                # dk = dk_partial + dk_corr + dk_direct*beta + dk_A
                for i, j in T.Parallel(block_C, dim_k):
                    dk[bid, hid, by * block_C + i, j] = (
                        dk_partial_s[i, j] + dk_corr_s[i, j]
                        + d_k_beta_frag[i, j] * beta_s[i]
                        + dk_A_frag2[i, j]
                    )

                # dbeta_A = (dP * KK^T).sum(-1)
                kkt_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.clear(kkt_frag)
                T.gemm(k_s, k_s, kkt_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    dP_s[i, j] = dP_s[i, j] * kkt_frag[i, j]
                dbeta_A_tmp = T.alloc_shared([block_C], accum_dtype)
                T.reduce_sum(dP_s, dbeta_A_tmp, dim=1)
                for i in T.Parallel(block_C):
                    dbeta[bid, hid, by * block_C + i] = dbeta_s[i] + dbeta_A_tmp[i]

        return compute_w_u_bwd

    return _kernel_func
