"""
Backward of compute_w_u: given dw, du, compute dk, dv, dbeta (and optionally dAw, dAu).

Forward: w = Aw @ (k*beta), u = Au @ (v*beta) per chunk.
Backward:
  dAw = dw @ (k*beta)^T
  d(k*beta) = Aw^T @ dw   -> dk = d(k*beta) * beta
  dAu = du @ (v*beta)^T
  d(v*beta) = Au^T @ du   -> dv = d(v*beta) * beta
  dbeta = (d(k*beta) * k).sum(-1) + (d(v*beta) * v).sum(-1)
"""

import functools

import tilelang
import tilelang.language as T

from .fused_prepare_compute_w_u import _LOG2E

__all__ = ["compute_w_u_bwd_full_tl"]


@functools.lru_cache(maxsize=32)
def compute_w_u_bwd_full_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Complete Gated DeltaNet prepare backward for one independent chunk."""
    accum_dtype = "float32"
    block_C = chunk_size

    @tilelang.jit(
        out_idx=[-4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _kernel_func(num_stages, threads=128):
        @T.prim_func
        def compute_w_u_bwd_full(
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dw_corr: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du_partial: T.Tensor([batch, head, seq_len, dim_v], dtype),
            du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype),
            A: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dv: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dbeta: T.Tensor([batch, head, seq_len], dtype),
            dg: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                offset = by * block_C
                A_s = T.alloc_shared([block_C, block_C], dtype)
                dw_s = T.alloc_shared([block_C, dim_k], dtype)
                du_s = T.alloc_shared([block_C, dim_v], dtype)
                k_s = T.alloc_shared([block_C, dim_k], dtype)
                v_s = T.alloc_shared([block_C, dim_v], dtype)
                g_s = T.alloc_shared([block_C], dtype)
                beta_s = T.alloc_shared([block_C], dtype)
                k_work_s = T.alloc_shared([block_C, dim_k], dtype)
                v_work_s = T.alloc_shared([block_C, dim_v], dtype)
                matrix_a_s = T.alloc_shared([block_C, block_C], dtype)
                matrix_b_s = T.alloc_shared([block_C, block_C], dtype)
                row_s = T.alloc_shared([block_C], accum_dtype)
                dbeta_v_s = T.alloc_shared([block_C], accum_dtype)
                dbeta_A_s = T.alloc_shared([block_C], accum_dtype)
                dg_row_s = T.alloc_shared([block_C], accum_dtype)
                dg_col_s = T.alloc_shared([block_C], accum_dtype)

                matrix_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                vector_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dk_A_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                dv_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)

                T.copy(A[bid, hid, offset : offset + block_C, :], A_s, disable_tma=True)
                T.copy(k[bid, hid, offset : offset + block_C, :], k_s, disable_tma=True)
                T.copy(v[bid, hid, offset : offset + block_C, :], v_s, disable_tma=True)
                T.copy(g[bid, hid, offset : offset + block_C], g_s, disable_tma=True)
                T.copy(beta[bid, hid, offset : offset + block_C], beta_s, disable_tma=True)
                for i, j in T.Parallel(block_C, dim_k):
                    dw_s[i, j] = dw[bid, hid, offset + i, j] + dw_corr[bid, hid, offset + i, j]
                    k_work_s[i, j] = k_s[i, j] * beta_s[i]
                for i, j in T.Parallel(block_C, dim_v):
                    du_s[i, j] = (
                        du_partial[bid, hid, offset + i, j] + du_corr[bid, hid, offset + i, j]
                    )
                    v_work_s[i, j] = v_s[i, j] * beta_s[i]

                T.clear(matrix_frag)
                T.gemm(dw_s, k_work_s, matrix_frag, transpose_B=True)
                T.gemm(du_s, v_work_s, matrix_frag, transpose_B=True)
                T.copy(matrix_frag, matrix_a_s)

                T.clear(vector_frag)
                T.gemm(A_s, dw_s, vector_frag, transpose_A=True)
                T.copy(vector_frag, k_work_s)
                T.clear(dv_frag)
                T.gemm(A_s, du_s, dv_frag, transpose_A=True)
                T.copy(dv_frag, v_work_s)

                for i, j in T.Parallel(block_C, dim_k):
                    dw_s[i, j] = k_work_s[i, j] * k_s[i, j]
                T.reduce_sum(dw_s, row_s, dim=1)
                for i, j in T.Parallel(block_C, dim_v):
                    du_s[i, j] = v_work_s[i, j] * v_s[i, j]
                T.reduce_sum(du_s, dbeta_v_s, dim=1)

                # If A = (I + L)^-1, then dL = -A^T @ dA @ A^T.
                T.clear(matrix_frag)
                T.gemm(matrix_a_s, A_s, matrix_frag, transpose_B=True)
                T.copy(matrix_frag, matrix_b_s)
                T.clear(matrix_frag)
                T.gemm(A_s, matrix_b_s, matrix_frag, transpose_A=True)
                for i, j in T.Parallel(block_C, block_C):
                    matrix_a_s[i, j] = T.if_then_else(i > j, -matrix_frag[i, j], T.float32(0.0))

                # dGram = dL * beta_i * exp(g_i - g_j).
                for i, j in T.Parallel(block_C, block_C):
                    matrix_b_s[i, j] = (
                        matrix_a_s[i, j] * beta_s[i] * T.exp2((g_s[i] - g_s[j]) * _LOG2E)
                    )
                T.clear(dk_A_frag)
                T.gemm(matrix_b_s, k_s, dk_A_frag)
                T.clear(vector_frag)
                T.gemm(matrix_b_s, k_s, vector_frag, transpose_A=True)
                for i, j in T.Parallel(block_C, dim_k):
                    dk[bid, hid, offset + i, j] = (
                        k_work_s[i, j] * beta_s[i] + dk_A_frag[i, j] + vector_frag[i, j]
                    )
                for i, j in T.Parallel(block_C, dim_v):
                    dv[bid, hid, offset + i, j] = v_work_s[i, j] * beta_s[i]

                # dbeta_A and dg_A use dL * exp(g_i-g_j) * <k_i,k_j>.
                T.clear(matrix_frag)
                T.gemm(k_s, k_s, matrix_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    matrix_b_s[i, j] = (
                        matrix_a_s[i, j] * T.exp2((g_s[i] - g_s[j]) * _LOG2E) * matrix_frag[i, j]
                    )
                    matrix_a_s[i, j] = matrix_b_s[i, j] * beta_s[i]

                T.reduce_sum(matrix_b_s, dbeta_A_s, dim=1)
                T.reduce_sum(matrix_a_s, dg_row_s, dim=1)
                T.reduce_sum(matrix_a_s, dg_col_s, dim=0)
                for i in T.Parallel(block_C):
                    dbeta[bid, hid, offset + i] = row_s[i] + dbeta_v_s[i] + dbeta_A_s[i]
                    dg[bid, hid, offset + i] = dg_row_s[i] - dg_col_s[i]

        return compute_w_u_bwd_full

    return _kernel_func
