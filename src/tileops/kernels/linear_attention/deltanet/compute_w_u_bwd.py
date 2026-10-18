"""
Backward of compute_w_u and the A_inv backward, from the walked state gradient.

One CTA a (batch, head, chunk). Inputs: dh_t (the transposed state gradient the chunk hands
on, from the walk), v_new, du_partial, S, Aw, Au, k, v, beta and dk_partial.

Internal computations per chunk:
  1. du = du_partial + k @ dh;  dk_corr = v_new @ dh^T;  dw = -(du @ S^T)
  2. wu_bwd:  dk_wu = (Aw^T @ dw) * beta,  dv = (Au^T @ du) * beta
  3. A_inv backward:  dAw + dAu -> dA -> dP -> dk_A, dbeta_A
  4. dk = dk_partial + dk_corr + dk_wu + dk_A
  5. dbeta = dbeta_direct + dbeta_A

Matmul operands are in the input dtype and accumulate in float32.

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
    """TileLang: the w/u backward and the A_inv backward of one chunk, from the walked dh."""
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
    def _kernel_func(threads=128):
        @T.prim_func
        def compute_w_u_bwd(
            dh_t: T.Tensor([batch, head, num_chunks, dim_v, dim_k], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            du_partial: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], accum_dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            dk_partial: T.Tensor([batch, head, seq_len, dim_k], dtype),
            # Outputs
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dv: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dbeta: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (by, bid, hid):
                r0 = by * block_C
                beta_s = T.alloc_shared([block_C], accum_dtype)
                k_s = T.alloc_shared([block_C, dim_k], dtype)
                dh_t_s = T.alloc_shared([dim_v, dim_k], dtype)
                v_new_s = T.alloc_shared([block_C, dim_v], dtype)
                T.copy(beta[bid, hid, r0 : r0 + block_C], beta_s, disable_tma=True)
                T.copy(k[bid, hid, r0 : r0 + block_C, :], k_s, disable_tma=True)
                T.copy(dh_t[bid, hid, by, :, :], dh_t_s, disable_tma=True)
                T.copy(v_new[bid, hid, r0 : r0 + block_C, :], v_new_s, disable_tma=True)

                # du = du_partial + k dh; the key gradient starts from v_new dh^T.
                du_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                T.copy(du_partial[bid, hid, r0 : r0 + block_C, :], du_frag, disable_tma=True)
                T.gemm(k_s, dh_t_s, du_frag, transpose_B=True)
                dk_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                T.clear(dk_frag)
                T.gemm(v_new_s, dh_t_s, dk_frag)
                du_s = T.alloc_shared([block_C, dim_v], dtype)
                T.copy(du_frag, du_s)

                # dw = -(du S^T), S the state the chunk starts from.
                S_s = T.alloc_shared([dim_k, dim_v], dtype)
                T.copy(S[bid, hid, by, :, :], S_s, disable_tma=True)
                dw_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                T.clear(dw_frag)
                T.gemm(du_s, S_s, dw_frag, transpose_B=True)
                dw_s = T.alloc_shared([block_C, dim_k], dtype)
                for i, j in T.Parallel(block_C, dim_k):
                    dw_s[i, j] = -dw_frag[i, j]

                # dA_inv = du (beta v)^T + dw (beta k)^T
                v_s = T.alloc_shared([block_C, dim_v], dtype)
                T.copy(v[bid, hid, r0 : r0 + block_C, :], v_s, disable_tma=True)
                v_beta_s = T.alloc_shared([block_C, dim_v], dtype)
                k_beta_s = T.alloc_shared([block_C, dim_k], dtype)
                for i, j in T.Parallel(block_C, dim_v):
                    v_beta_s[i, j] = v_s[i, j] * beta_s[i]
                for i, j in T.Parallel(block_C, dim_k):
                    k_beta_s[i, j] = k_s[i, j] * beta_s[i]
                dA_inv_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.clear(dA_inv_frag)
                T.gemm(du_s, v_beta_s, dA_inv_frag, transpose_B=True)
                T.gemm(dw_s, k_beta_s, dA_inv_frag, transpose_B=True)

                # dv = beta (Au^T du); dbeta gets rowsum((Au^T du) * v).
                Au_s = T.alloc_shared([block_C, block_C], dtype)
                T.copy(Au[bid, hid, r0 : r0 + block_C, :], Au_s, disable_tma=True)
                d_v_beta_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                T.clear(d_v_beta_frag)
                T.gemm(Au_s, du_s, d_v_beta_frag, transpose_A=True)
                for i, j in T.Parallel(block_C, dim_v):
                    dv[bid, hid, r0 + i, j] = d_v_beta_frag[i, j] * beta_s[i]
                dbeta_frag = T.alloc_fragment([block_C], accum_dtype)
                for i, j in T.Parallel(block_C, dim_v):
                    d_v_beta_frag[i, j] = d_v_beta_frag[i, j] * v_s[i, j]
                T.reduce_sum(d_v_beta_frag, dbeta_frag, dim=1)
                # Each row sum leaves its fragment through shared memory: the three come from
                # fragments of different widths, whose row layouts need not agree.
                dbeta_s = T.alloc_shared([block_C], accum_dtype)
                T.copy(dbeta_frag, dbeta_s)

                # dk += beta (Aw^T dw); dbeta gets rowsum((Aw^T dw) * k).
                Aw_s = T.alloc_shared([block_C, block_C], dtype)
                T.copy(Aw[bid, hid, r0 : r0 + block_C, :], Aw_s, disable_tma=True)
                d_k_beta_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                T.clear(d_k_beta_frag)
                T.gemm(Aw_s, dw_s, d_k_beta_frag, transpose_A=True)
                for i, j in T.Parallel(block_C, dim_k):
                    dk_frag[i, j] = dk_frag[i, j] + d_k_beta_frag[i, j] * beta_s[i]
                dbeta_k_frag = T.alloc_fragment([block_C], accum_dtype)
                for i, j in T.Parallel(block_C, dim_k):
                    d_k_beta_frag[i, j] = d_k_beta_frag[i, j] * k_s[i, j]
                T.reduce_sum(d_k_beta_frag, dbeta_k_frag, dim=1)
                dbeta_k_s = T.alloc_shared([block_C], accum_dtype)
                T.copy(dbeta_k_frag, dbeta_k_s)

                # ===== A_inv backward: dP = tril(-(Aw^T dA_inv Aw^T), -1) =====
                dA_inv_s = T.alloc_shared([block_C, block_C], dtype)
                T.copy(dA_inv_frag, dA_inv_s)
                dA_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.clear(dA_frag)
                T.gemm(dA_inv_s, Aw_s, dA_frag, transpose_B=True)
                dA_s = T.alloc_shared([block_C, block_C], dtype)
                T.copy(dA_frag, dA_s)
                dP_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.clear(dP_frag)
                T.gemm(Aw_s, dA_s, dP_frag, transpose_A=True)
                dP_s = T.alloc_shared([block_C, block_C], dtype)
                for i, j in T.Parallel(block_C, block_C):
                    dP_frag[i, j] = T.if_then_else(i > j, -dP_frag[i, j], T.float32(0.0))
                T.copy(dP_frag, dP_s)

                # dbeta_A = rowsum(dP * k k^T)
                kkt_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.clear(kkt_frag)
                T.gemm(k_s, k_s, kkt_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    kkt_frag[i, j] = dP_frag[i, j] * kkt_frag[i, j]
                dbeta_A_frag = T.alloc_fragment([block_C], accum_dtype)
                T.reduce_sum(kkt_frag, dbeta_A_frag, dim=1)
                for i in T.Parallel(block_C):
                    dbeta_s[i] = dbeta_s[i] + dbeta_k_s[i] + dbeta_A_frag[i]
                T.copy(dbeta_s, dbeta[bid, hid, r0 : r0 + block_C], disable_tma=True)

                # dk_A = beta (dP k) + dP^T (beta k)
                dk_A_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                T.clear(dk_A_frag)
                T.gemm(dP_s, k_s, dk_A_frag)
                for i, j in T.Parallel(block_C, dim_k):
                    dk_frag[i, j] = dk_frag[i, j] + dk_A_frag[i, j] * beta_s[i]
                T.clear(dk_A_frag)
                T.gemm(dP_s, k_beta_s, dk_A_frag, transpose_A=True)

                for i, j in T.Parallel(block_C, dim_k):
                    dk[bid, hid, r0 + i, j] = (
                        dk_frag[i, j] + dk_A_frag[i, j] + dk_partial[bid, hid, r0 + i, j]
                    )

        return compute_w_u_bwd

    return _kernel_func
