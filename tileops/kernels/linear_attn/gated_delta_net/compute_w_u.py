"""
Compute w, u for Gated DeltaNet: (Aw, Au, k, v, beta) -> (w, u).

  w = Aw @ (k * beta)   per chunk [Aw: (C,C), k*beta: (C,DK) -> w: (C,DK)]
  u = Au @ (v * beta)   per chunk [Au: (C,C), v*beta: (C,DV) -> u: (C,DV)]

Used by gated_deltanet_fwd as the first step before recurrence and Kernel 2.
"""

import tilelang
import tilelang.language as T

__all__ = ["compute_w_u_tl"]


def compute_w_u_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """TileLang: w = Aw@(k*beta), u = Au@(v*beta) per chunk."""
    accum_dtype = "float32"
    block_C = chunk_size

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _kernel_func(num_stages, threads=128):
        @T.macro
        def _body(
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                Aw_s = T.alloc_shared([block_C, block_C], dtype)
                Au_s = T.alloc_shared([block_C, block_C], dtype)
                k_s = T.alloc_shared([block_C, dim_k], dtype)
                v_s = T.alloc_shared([block_C, dim_v], dtype)
                beta_s = T.alloc_shared([block_C], dtype)
                k_beta_s = T.alloc_shared([block_C, dim_k], dtype)
                v_beta_s = T.alloc_shared([block_C, dim_v], dtype)
                w_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                u_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)

                T.copy(Aw[bid, hid, by * block_C : (by + 1) * block_C, :], Aw_s, disable_tma=True)
                T.copy(Au[bid, hid, by * block_C : (by + 1) * block_C, :], Au_s, disable_tma=True)
                T.copy(beta[bid, hid, by * block_C : (by + 1) * block_C], beta_s, disable_tma=True)
                T.copy(k[bid, hid, by * block_C : (by + 1) * block_C, :], k_s, disable_tma=True)
                T.copy(v[bid, hid, by * block_C : (by + 1) * block_C, :], v_s, disable_tma=True)

                # w = Aw @ (k * beta)
                for i, j in T.Parallel(block_C, dim_k):
                    k_beta_s[i, j] = k_s[i, j] * beta_s[i]
                T.clear(w_frag)
                T.gemm(Aw_s, k_beta_s, w_frag)
                T.copy(w_frag, w[bid, hid, by * block_C : (by + 1) * block_C, :], disable_tma=True)

                # u = Au @ (v * beta)
                for i, j in T.Parallel(block_C, dim_v):
                    v_beta_s[i, j] = v_s[i, j] * beta_s[i]
                T.clear(u_frag)
                T.gemm(Au_s, v_beta_s, u_frag)
                T.copy(u_frag, u[bid, hid, by * block_C : (by + 1) * block_C, :], disable_tma=True)

        @T.prim_func
        def compute_w_u(
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            _body(Aw, Au, k, v, beta, w, u)

        return compute_w_u

    return _kernel_func
