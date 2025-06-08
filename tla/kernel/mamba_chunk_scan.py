# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, repeat
import itertools
import torch.nn as nn
from tilelang.utils.tensor import torch_assert_close

def get_configs():
    block_M = [64, 128, 256]
    block_N = [32, 64]
    block_K = [64, 128, 256]
    block_Dstate = [128]
    num_stages = [1, 2, 3, 4, 5]
    _configs = list(itertools.product(block_M, block_N, block_K, block_Dstate, num_stages))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'block_K': c[2],
        'block_Dstate': c[3],
        'num_stages': c[4],
        'threads': c[0] * 2
    } for c in _configs]
    return configs


def _chunk_scan_fwd(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, tune=False):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504

    def kernel_func(block_M, block_N, block_K, block_Dstate, num_stages, threads):

        @T.prim_func
        def _chunk_scan_fwd_main(
            cb: T.Tensor((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),
            x: T.Tensor((batch, seqlen, nheads, headdim), dtype), 
            dt: T.Tensor((batch, nheads, nchunks, chunk_size), dtype), 
            dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype), 
            C: T.Tensor((batch, seqlen, ngroups, dstate), dtype), 
            prev_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), dtype), 
            D: T.Tensor((nheads), dtype), 
            Output: T.Tensor((batch, seqlen, nheads, headdim), dtype)
        ):
            with T.Kernel(
                    nheads,
                    T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N),
                    batch * nchunks,
                    threads=threads) as (bz, bx, by):
                acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
                acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
                cb_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.dyn")
                cb_local = T.alloc_fragment((block_M, block_K), dtype)
                dA_cs_k_shared = T.alloc_shared((block_K), dtype, scope="shared")
                dA_cs_k_local = T.alloc_fragment((block_K), accum_dtype)
                dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
                dt_shared = T.alloc_shared((block_K), dtype, scope="shared")
                dt_local = T.alloc_fragment((block_K), accum_dtype)
                x_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.dyn")
                dA_cs_m_shared = T.alloc_shared((block_M), dtype, scope="shared")
                scale_m_local = T.alloc_fragment((block_M), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
                prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)
                D_local = T.alloc_fragment((1), accum_dtype)
                x_residual_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared.dyn")
                x_residual_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                batch_idx = by % batch
                chunk_idx = by // batch
                # m: chunk_size
                # n : headdim
                m_idx = bx // T.ceildiv(headdim, block_N)
                n_idx = bx % T.ceildiv(headdim, block_N)

                T.annotate_layout({
                    acc_o_shared: tilelang.layout.make_swizzled_layout(acc_o_shared),
                    cb_shared: tilelang.layout.make_swizzled_layout(cb_shared),
                    x_residual_shared: tilelang.layout.make_swizzled_layout(x_residual_shared)
                })

                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M:(m_idx + 1) * block_M],
                       dA_cs_m_shared)
                T.copy(dA_cs_m_shared, dA_cs_m_local)
                T.clear(acc_o)

                for i in T.Parallel(block_M):
                    scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
                T.copy(
                    C[batch_idx, chunk_idx * chunk_size + m_idx * block_M:chunk_idx * chunk_size +
                      (m_idx + 1) * block_M, bz // (nheads // ngroups), 0:block_Dstate], C_shared)
                T.copy(
                    prev_states[batch_idx, chunk_idx, bz, n_idx * block_N:(n_idx + 1) * block_N,
                                0:block_Dstate], prev_state_shared)
                T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
                for i, j in T.Parallel(block_M, block_N):
                    acc_o[i, j] *= scale_m_local[i]

                loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        cb[batch_idx, chunk_idx, bz // (nheads // ngroups),
                           m_idx * block_M:(m_idx + 1) * block_M, k * block_K:(k + 1) * block_K],
                        cb_shared)
                    T.copy(cb_shared, cb_local)
                    T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K:(k + 1) * block_K],
                           dA_cs_k_shared)
                    T.copy(dA_cs_k_shared, dA_cs_k_local)
                    for i, j in T.Parallel(block_M, block_K):
                        cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p -
                                                                 dA_cs_k_local[j] * p)
                    T.copy(dt[batch_idx, bz, chunk_idx, k * block_K:(k + 1) * block_K], dt_shared)
                    T.copy(dt_shared, dt_local)
                    for i, j in T.Parallel(block_M, block_K):
                        cb_local[i, j] *= dt_local[j]
                    for i, j in T.Parallel(block_M, block_K):
                        cb_local[i, j] = T.if_then_else(m_idx * block_M + i >= k * block_K + j,
                                                        cb_local[i, j], 0)
                    T.copy(
                        x[batch_idx, chunk_idx * chunk_size + k * block_K:chunk_idx * chunk_size +
                          (k + 1) * block_K, bz, n_idx * block_N:(n_idx + 1) * block_N], x_shared)
                    T.gemm(cb_local, x_shared, acc_o)

                D_local[0] = D[bz]
                T.copy(
                    x[batch_idx, chunk_idx * chunk_size + m_idx * block_M:chunk_idx * chunk_size +
                      (m_idx + 1) * block_M, bz, n_idx * block_N:(n_idx + 1) * block_N],
                    x_residual_shared)
                T.copy(x_residual_shared, x_residual_local)
                for i, j in T.Parallel(block_M, block_N):
                    acc_o[i, j] += x_residual_local[i, j] * D_local[0]

                T.copy(acc_o, acc_o_shared)
                T.copy(
                    acc_o_shared,
                    Output[batch_idx, chunk_idx * chunk_size +
                           m_idx * block_M:chunk_idx * chunk_size + (m_idx + 1) * block_M, bz,
                           n_idx * block_N:(n_idx + 1) * block_N])

        return _chunk_scan_fwd_main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tilelang.jit(out_idx=[7])
        def kernel(block_M=None,
                   block_N=None,
                   block_K=None,
                   block_Dstate=None,
                   num_stages=None,
                   threads=None):
            return kernel_func(block_M, block_N, block_K, block_Dstate, num_stages, threads)

        return kernel()
    else:
        @tilelang.jit(out_idx=[7])
        def kernel(block_M, block_N, block_K, block_Dstate, num_stages, threads):
            return kernel_func(block_M, block_N, block_K, block_Dstate, num_stages, threads)

        return kernel


@torch.compile
class _MAMBA_CHUNK_SCAN_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, cb, x, dt, dA_cumsum, C, prev_states, D, config):
        BATCH, _, NGROUPS, CHUNK_SIZE, _ = cb.shape
        _, SEQLEN, NHEADS, HEADDIM = x.shape
        DSTATE = C.shape[-1]
        mod = _chunk_scan_fwd(BATCH, SEQLEN, CHUNK_SIZE, NGROUPS, NHEADS, HEADDIM, DSTATE)(**config)
        o = mod(cb, x, dt, dA_cumsum, C, prev_states, D)
        return o

    @staticmethod
    def backward(ctx, do):
        pass

MAMBA_CHUNK_SCAN_attention = _MAMBA_CHUNK_SCAN_attention.apply



class MAMBA_CHUNK_SCAN_kernel(nn.Module):

    def __init__(self,
                 batch, 
                 heads, 
                 groups, 
                 seq_len, 
                 chunk_size, 
                 dim,
                 dstate,
                 block_M = None, 
                 block_N = None, 
                 block_K = None, 
                 block_Dstate = None, 
                 num_stages = None, 
                 threads = None,
                 tune = False,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = MAMBA_CHUNK_SCAN_attention
        self.batch = batch
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.groups = groups
        self.heads = heads
        self.dim = dim
        self.dstate = dstate
        self.block_M = block_M if block_M else 64
        self.block_N = block_N if block_N else 64
        self.block_K = block_K if block_K else 64
        self.block_Dstate = block_Dstate if block_Dstate else 128
        self.num_stages = num_stages if num_stages else 2
        self.threads = threads if threads else 128
        self.config = {
            "block_M": self.block_M,
            "block_N": self.block_N,
            "block_K": self.block_K,
            "block_Dstate": self.block_Dstate,
            "num_stages": self.num_stages,
            "threads": self.threads
        }
        self.tune = tune
        self.tune_config = None
        self.total_flops = 2 * batch * seq_len * chunk_size * heads * dim * 0.5 \
                            + 2 * batch * seq_len * heads * dim * dstate
        self.fwd_program = _chunk_scan_fwd(batch, seq_len, chunk_size, groups, heads, dim, dstate)(**self.config)
        self.fwd_profiler = self.fwd_program.get_profiler(tilelang.TensorSupplyType.Normal)

    def forward(self, cb, x, dt, dA_cumsum, C, prev_states, D):
        if self.tune_config is None and self.tune:
            self.autotune()
        if self.tune_config:
            o = self.attention(cb, x, dt, dA_cumsum, C, prev_states, D, self.tune_config)
            return o
        o = self.attention(cb, x, dt, dA_cumsum, C, prev_states, D, self.config)
        return o

    def backward(self, cb, x, dt, dA_cumsum, C, prev_states, do):
        pass

    def autotune(self):
        best_result = _chunk_scan_fwd(self.batch, self.seq_len, self.chunk_size, self.groups, self.heads, self.dim, self.dstate, tune=True)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {self.total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")
        if best_result.config:
            self.config = best_result.config

    def profile(self, warmup=500):
        if self.tune_config is None and self.tune:
            self.autotune()
        if self.tune_config:
            self.fwd_program = _chunk_scan_fwd(self.batch, self.seq_len, self.chunk_size, self.groups, self.heads, self.dim, self.dstate)(**self.tune_config)
            self.fwd_profiler = self.fwd_program.get_profiler(tilelang.TensorSupplyType.Normal)
        latency = self.fwd_profiler.do_bench(warmup=warmup)
        return latency

    def ref_program(self, cb, x, dt, dA_cumsum, C, prev_states, D):
        """
        Argument:
            cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
            x: (batch, seqlen, nheads, headdim)
            dt: (batch, nheads, nchunks, chunk_size)
            dA_cumsum: (batch, nheads, nchunks, chunk_size)
            C: (batch, seqlen, ngroups, dstate)
            prev_states: (batch, nchunks, nheads, headdim, dstate)
            D: (nheads, headdim) or (nheads,)
            z: (batch, seqlen, nheads, headdim)
        Return:
            out: (batch, seqlen, nheads, headdim)
        """
        _, _, ngroups, _, _ = cb.shape
        batch, seqlen, nheads, headdim = x.shape
        # _, _, ngroups, dstate = B.shape
        # assert B.shape == (batch, seqlen, ngroups, dstate)
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen == nchunks * chunk_size
        # assert C.shape == B.shape
        # B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
        C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
        cb = repeat(cb, "b c g l s -> b c (g h) l s", h=nheads // ngroups)
        # CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
        #                   rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
        # (batch, nheads, nchunks, chunksize, chunksize)
        dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
        decay = torch.exp(dt_segment_sum)
        scores_decay = cb * rearrange(decay, "b h c l s -> b c h l s")
        causal_mask = torch.tril(
            torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
        scores_decay = scores_decay.masked_fill(~causal_mask, 0)
        out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                        rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
        state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
        out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(
            C, "b (c l) h n -> b c l h n", c=nchunks), prev_states.to(C.dtype)) * state_decay_out
        out = out + out_prev
        out = rearrange(out, "b c l h p -> b (c l) h p")
        if D is not None:
            if D.dim() == 1:
                D = rearrange(D, "h -> h 1")
            out = out + x * D
        return out

    def check(self, cb, x, dt, dA_cumsum, C, prev_states, D):
        if self.tune_config:
            o = self.attention(cb, x, dt, dA_cumsum, C, prev_states, D, self.tune_config)
        else:
            o = self.attention(cb, x, dt, dA_cumsum, C, prev_states, D, self.config)
        o_ref = self.ref_program(cb, x, dt, dA_cumsum, C, prev_states, D)
        torch_assert_close(o, o_ref, rtol=1e-2, atol=1e-2,max_mismatched_ratio=0.01)
        print("MAMBA CHUNK SCAN kernel check passed!")
