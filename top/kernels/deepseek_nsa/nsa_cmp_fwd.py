import torch
from typing import Optional, Any, Callable, Tuple

import tilelang
from tilelang import language as T

from top.kernels.kernel import Kernel


def _nsa_cmp_fwd_varlen_kernel(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_num: int,
    group: int,
    scale: float,
    bc: int,
    bs: int,
    bk: int,
    bv: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    LOG2_E = 1.44269504
    scale_log2 = scale * LOG2_E
    head_kv = heads // group
    
    q_shape = [c_seq_len, heads, dim_k]
    k_cmp_shape = [chunk_num, head_kv, dim_k]
    v_cmp_shape = [chunk_num, head_kv, dim_v]
    lse_shape = [c_seq_len, heads]
    offsets_shape = [seq_num + 1]
    token_indices_shape = [c_seq_len, 2]
    chunk_offsets_shape = [seq_num + 1]
    o_shape = [c_seq_len, heads, dim_v]

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _nsa_cmp_fwd_varlen_func(threads: int):
        @T.prim_func
        def _parallel_nsa_cmp_fwd_varlen_main(
            q: T.Tensor(q_shape, dtype),
            k_cmp: T.Tensor(k_cmp_shape, dtype),
            v_cmp: T.Tensor(v_cmp_shape, dtype),
            offsets: T.Tensor(offsets_shape, T.int32),
            chunk_offsets: T.Tensor(chunk_offsets_shape, T.int32),
            token_indices: T.Tensor(token_indices_shape, T.int32),
            output: T.Tensor(o_shape, dtype),
            temp_lse: T.Tensor(lse_shape, dtype),
        ):
            with T.Kernel(c_seq_len, head_kv, threads=threads) as (bx, by):
                q_shared = T.alloc_shared([group, bk], dtype)
                k_shared = T.alloc_shared([bc, bk], dtype)
                v_shared = T.alloc_shared([bc, bv], dtype)
                

                i_c, i_h = bx, by
                i_n, i_t = token_indices[i_c, 0], token_indices[i_c, 1]

                bos, eos = offsets[i_n], offsets[i_n + 1]
                boc = chunk_offsets[i_n]
                nc = (i_t + 1) // bs

                T.copy(q[bos + i_t, i_h * group : (i_h + 1) * group, :bk], q_shared)

                b_o = T.alloc_fragment([group, bv], dtype)
                b_lse = T.alloc_fragment([group], dtype)
                acc_s = T.alloc_fragment([group, bc], accum_dtype)
                acc_s_cast = T.alloc_fragment([group, bc], dtype)
                scores_max = T.alloc_fragment([group], accum_dtype)
                scores_max_prev = T.alloc_fragment([group], accum_dtype)
                scores_scale = T.alloc_fragment([group], accum_dtype)
                scores_sum = T.alloc_fragment([group], accum_dtype)
                logsum = T.alloc_fragment([group], accum_dtype)
                
                T.fill(b_o, 0.0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.fill(logsum, 0.0)
            
                for i_loop in T.Pipelined(T.ceildiv(nc, bc), num_stages=3):
                    curr_bc = T.min(bc, nc - i_loop * bc)
                    T.copy(k_cmp[boc + i_loop * bc : boc + i_loop * bc + curr_bc, i_h, :bk], k_shared[:curr_bc, :bk])
                    T.copy(v_cmp[boc + i_loop * bc : boc + i_loop * bc + curr_bc, i_h, :bv], v_shared[:curr_bc, :bv])
                    
                    for g_m, c_m in T.Parallel(group, bc):
                        acc_s[g_m, c_m] = T.if_then_else(c_m < curr_bc, 0.0, -T.infinity(accum_dtype))

                    T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for i in T.Parallel(group):
                        scores_scale[i] = T.if_then_else(scores_max[i] > -T.infinity(accum_dtype), 
                                                     T.exp2(scores_max_prev[i] * scale_log2 - scores_max[i] * scale_log2), 0.0)
                
                    for i, j in T.Parallel(group, bc):
                        acc_s[i, j] = T.if_then_else(acc_s[i, j] > -T.infinity(accum_dtype), 
                                                    T.exp2(acc_s[i, j] * scale_log2 - scores_max[i] * scale_log2), 0.0)
                    
                    for i, k_idx in T.Parallel(group, bv):
                        b_o[i, k_idx] *= scores_scale[i]
                    
                    T.copy(acc_s, acc_s_cast)
                    
                    T.gemm(acc_s_cast, v_shared, b_o, policy=T.GemmWarpPolicy.FullRow)

                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(group):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                
                for i, k_idx in T.Parallel(group, bv):
                    if nc > 0 and logsum[i] > 0:
                        b_o[i, k_idx] /= logsum[i]

                for i in T.Parallel(group):
                    if nc == 0 or logsum[i] <= 0:
                        b_lse[i] = 0.0
                    else:
                        b_lse[i] = (scores_max[i] * scale_log2 + T.log2(logsum[i])) / LOG2_E
                
                T.copy(b_o, output[bos + i_t, i_h * group : (i_h + 1) * group, :dim_v])
                T.copy(b_lse, temp_lse[bos + i_t, i_h * group : (i_h + 1) * group])

        return _parallel_nsa_cmp_fwd_varlen_main

    return _nsa_cmp_fwd_varlen_func


@torch.library.custom_op("top::nsa_cmp_fwd_varlen_wrapped_kernel", mutates_args=())
def _nsa_cmp_fwd_varlen_wrapped_kernel(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_num: int,
    group: int,
    scale: float,
    bc: int,
    bs: int,
    bk: int,
    bv: int,
    dtype: str,
    accum_dtype: str,
    threads: int,
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    v_cmp: torch.Tensor,
    offsets: torch.Tensor,
    chunk_offsets: torch.Tensor,
    token_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _nsa_cmp_fwd_varlen_kernel(seq_num, c_seq_len, heads, dim_k, dim_v, chunk_num, group, scale,
                                   bc, bs, bk, bv, dtype, accum_dtype)(threads)(q, k_cmp, v_cmp, offsets, chunk_offsets,
                                                                                 token_indices)


@_nsa_cmp_fwd_varlen_wrapped_kernel.register_fake
def _(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_num: int,
    group: int,
    scale: float,
    bc: int,
    bs: int,
    bk: int,
    bv: int,
    dtype: str,
    accum_dtype: str,
    threads: int,
    *inputs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _ = (seq_num, dim_k, dim_v, chunk_num, group, scale, bc, bs, bk, bv, dtype, accum_dtype, threads)
    return (torch.empty([c_seq_len, heads, dim_v],
                       dtype=inputs[0].dtype,
                       device=inputs[0].device),
            torch.empty([c_seq_len, heads],
                       dtype=inputs[0].dtype,
                       device=inputs[0].device))


class NSACmpFwdVarlenKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 seq_num: int,
                 c_seq_len: int,
                 heads: int,
                 dim_k: int,
                 dim_v: int,
                 chunk_num: int,
                 group: int,
                 scale: float,
                 bc: int,
                 bs: int,
                 bk: int,
                 bv: int,
                 dtype: torch.dtype,
                 accum_dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_num = chunk_num
        self.group = group
        self.scale = scale
        self.bc = bc
        self.bs = bs
        self.bk = bk
        self.bv = bv
        self.dtype_name = str(dtype).split('.')[-1]
        self.accum_dtype_name = str(accum_dtype).split('.')[-1]
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "threads": 32,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        threads = [32]
        return [{"threads": t} for t in threads]

    def forward(self, q: torch.Tensor, k_cmp: torch.Tensor, v_cmp: torch.Tensor,
                offsets: torch.Tensor, chunk_offsets: torch.Tensor,
                token_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _nsa_cmp_fwd_varlen_wrapped_kernel(self.seq_num, self.c_seq_len, self.heads, self.dim_k, self.dim_v,
                                               self.chunk_num, self.group, self.scale,
                                               self.bc, self.bs, self.bk, self.bv,
                                               self.dtype_name, self.accum_dtype_name,
                                               self.config["threads"],
                                               q.to(getattr(torch, self.dtype_name)),
                                               k_cmp.to(getattr(torch, self.dtype_name)),
                                               v_cmp.to(getattr(torch, self.dtype_name)),
                                               offsets.to(torch.int32),
                                               chunk_offsets.to(torch.int32),
                                               token_indices.to(torch.int32))


