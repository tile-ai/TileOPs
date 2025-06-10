# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
from einops import rearrange, repeat
import itertools
import torch.nn as nn
from tilelang.utils.tensor import torch_assert_close


def get_configs():
    block_M = [64, 128]
    block_N = [32, 64, 128]
    block_K = [32, 64]
    num_stages = [1, 2, 3, 4, 5]
    _configs = list(itertools.product(block_M, block_N, block_K, num_stages))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'block_K': c[2],
        'num_stages': c[3],
        'threads': c[0] * 2
    } for c in _configs]
    return configs


def _chunk_state_fwd(batch,
                     seqlen,
                     chunk_size,
                     ngroups,
                     nheads,
                     headdim,
                     dstate,
                     tune=False):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504

    def kernel_func(block_M, block_N, block_K, num_stages, threads):

        @T.prim_func
        def main(
            B: T.Tensor((batch, seqlen, ngroups, dstate), dtype), # type: ignore
            x: T.Tensor((batch, seqlen, nheads, headdim), dtype), # type: ignore
            dt: T.Tensor((batch, nheads, nchunks, chunk_size), dtype), # type: ignore
            dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size),dtype), # type: ignore
            Output: T.Tensor((batch, nchunks, nheads, headdim, dstate),dtype) # type: ignore
        ):
            with T.Kernel(nheads,
                          T.ceildiv(headdim, block_M) *
                          T.ceildiv(dstate, block_N),
                          batch * nchunks,
                          threads=threads) as (bz, bx, by):
                x_shared = T.alloc_shared((block_K, block_M), dtype)
                x_local = T.alloc_fragment((block_K, block_M), dtype)
                xt_local = T.alloc_fragment((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                dt_shared = T.alloc_shared((block_K), dtype)
                dA_cumsum_shared = T.alloc_shared((block_K), dtype)
                acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
                acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
                scale = T.alloc_fragment((block_K), accum_dtype)
                dA_cs_last = T.alloc_fragment((1), accum_dtype)
                dA_cumsum_local = T.alloc_fragment((block_K), accum_dtype)
                dt_local = T.alloc_fragment((block_K), accum_dtype)

                loop_range = T.ceildiv(chunk_size, block_K)

                batch_idx = by % batch
                chunk_idx = by // batch
                m_idx = bx // T.ceildiv(dstate, block_N)
                n_idx = bx % T.ceildiv(dstate, block_N)

                T.annotate_layout({
                    x_shared:
                    tilelang.layout.make_swizzled_layout(x_shared),
                    acc_o_shared:
                    tilelang.layout.make_swizzled_layout(acc_o_shared)
                })

                dA_cs_last[0] = dA_cumsum[batch_idx, bz, chunk_idx,
                                          chunk_size - 1]
                T.clear(acc_o)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        x[batch_idx, chunk_idx * chunk_size +
                          k * block_K:chunk_idx * chunk_size +
                          (k + 1) * block_K, bz,
                          m_idx * block_M:(m_idx + 1) * block_M], x_shared)
                    T.copy(
                        dA_cumsum[batch_idx, bz, chunk_idx,
                                  k * block_K:(k + 1) * block_K],
                        dA_cumsum_shared)
                    T.copy(
                        dt[batch_idx, bz, chunk_idx,
                           k * block_K:(k + 1) * block_K], dt_shared)
                    T.copy(dA_cumsum_shared, dA_cumsum_local)
                    T.copy(dt_shared, dt_local)
                    for i in T.Parallel(block_K):
                        scale[i] = T.exp2(dA_cs_last[0] * p -
                                          dA_cumsum_local[i] * p) * dt_local[i]
                    T.copy(x_shared, x_local)
                    for i, j in T.Parallel(block_M, block_K):
                        xt_local[i, j] = x_local[j, i] * scale[j]
                    T.copy(
                        B[batch_idx, chunk_idx * chunk_size +
                          k * block_K:chunk_idx * chunk_size +
                          (k + 1) * block_K, bz // (nheads // ngroups),
                          n_idx * block_N:(n_idx + 1) * block_N], B_shared)
                    T.gemm(xt_local, B_shared, acc_o)
                T.copy(acc_o, acc_o_shared)
                T.copy(
                    acc_o_shared,
                    Output[batch_idx, chunk_idx, bz,
                           m_idx * block_M:(m_idx + 1) * block_M,
                           n_idx * block_N:(n_idx + 1) * block_N])

        return main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tilelang.jit(out_idx=[4])
        def kernel(block_M=None,
                   block_N=None,
                   block_K=None,
                   num_stages=None,
                   threads=None):
            return kernel_func(block_M, block_N, block_K, num_stages, threads)

        return kernel()
    else:

        @tilelang.jit(out_idx=[4])
        def kernel(block_M, block_N, block_K, num_stages, threads):
            return kernel_func(block_M, block_N, block_K, num_stages, threads)

        return kernel


@torch.compile
class _MAMBA_CHUNK_STATE_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, x, dt, dA_cumsum, config):
        BATCH, SEQLEN, NGROUPS, DSTATE = B.shape
        _, _, NHEADS, HEADDIM = x.shape
        CHUNK_SIZE = dt.shape[-1]
        mod = _chunk_state_fwd(BATCH, SEQLEN, CHUNK_SIZE, NGROUPS, NHEADS,
                               HEADDIM, DSTATE)(**config)
        o = mod(B, x, dt, dA_cumsum)
        return o

    @staticmethod
    def backward(ctx, do):
        pass


MAMBA_CHUNK_STATE_attention = _MAMBA_CHUNK_STATE_attention.apply


class MAMBA_CHUNK_STATE_kernel(nn.Module):

    def __init__(self,
                 batch,
                 heads,
                 groups,
                 seq_len,
                 chunk_size,
                 dim,
                 dstate,
                 block_M=None,
                 block_N=None,
                 block_K=None,
                 num_stages=None,
                 threads=None,
                 tune=False,
                 dtype=torch.float16,
                 device="cuda"):
        super().__init__()
        self.attention = MAMBA_CHUNK_STATE_attention
        self.batch = batch
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.groups = groups
        self.heads = heads
        self.dim = dim
        self.dstate = dstate
        self.block_M = block_M if block_M else 64
        self.block_N = block_N if block_N else 128
        self.block_K = block_K if block_K else 64
        self.num_stages = num_stages if num_stages else 4
        self.threads = threads if threads else 128
        self.config = {
            "block_M": self.block_M,
            "block_N": self.block_N,
            "block_K": self.block_K,
            "num_stages": self.num_stages,
            "threads": self.threads
        }
        self.tune = tune
        self.tune_config = None
        self.total_flops = 2 * batch * seq_len * heads * dim * dstate
        self.fwd_program = _chunk_state_fwd(batch, seq_len, chunk_size, groups,
                                            heads, dim, dstate)(**self.config)
        self.fwd_profiler = self.fwd_program.get_profiler(
            tilelang.TensorSupplyType.Normal)

    def forward(self, B, x, dt, dA_cumsum):
        if self.tune_config is None and self.tune:
            self.autotune()
        if self.tune_config:
            o = self.attention(B, x, dt, dA_cumsum, self.tune_config)
            return o
        o = self.attention(B, x, dt, dA_cumsum, self.config)
        return o

    def backward(self, B, x, dt, dA_cumsum, do):
        pass

    def autotune(self):
        best_result = _chunk_state_fwd(self.batch,
                                       self.seq_len,
                                       self.chunk_size,
                                       self.groups,
                                       self.heads,
                                       self.dim,
                                       self.dstate,
                                       tune=True)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {self.total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")
        if best_result.config:
            self.tune_config = best_result.config

    def profile(self, warmup=500):
        if self.tune_config is None and self.tune:
            self.autotune()
        if self.tune_config:
            self.fwd_program = _chunk_state_fwd(
                self.batch, self.seq_len, self.chunk_size, self.groups,
                self.heads, self.dim, self.dstate)(**self.tune_config)
            self.fwd_profiler = self.fwd_program.get_profiler(
                tilelang.TensorSupplyType.Normal)
        latency = self.fwd_profiler.do_bench(warmup=warmup)
        return latency

    def ref_program(self, B, x, dt, dA_cumsum):
        """
        Argument:
            B: (batch, seqlen, ngroups, headdim)
            x: (batch, seqlen, nheads, headdim)
            dt: (batch, nheads, nchunks, chunk_size)
            dA_cumsum: (batch, nheads, nchunks, chunk_size)
        Return:
            states: (batch, nchunks, nheads, headdim, dstate)
        """
        # Check constraints.
        batch, seqlen, nheads, headdim = x.shape
        dstate = B.shape[-1]
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen <= nchunks * chunk_size
        assert x.shape == (batch, seqlen, nheads, headdim)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        ngroups = B.shape[2]
        assert nheads % ngroups == 0
        assert B.shape == (batch, seqlen, ngroups, dstate)
        B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        if seqlen < nchunks * chunk_size:
            x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
            B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
        B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
        decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
        return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype),
                            decay_states.to(x.dtype), dt.to(x.dtype), x)

    def check(self, B, x, dt, dA_cumsum):
        if self.tune_config:
            o = self.attention(B, x, dt, dA_cumsum, self.tune_config)
        else:
            o = self.attention(B, x, dt, dA_cumsum, self.config)
        o_ref = self.ref_program(B, x, dt, dA_cumsum)
        torch_assert_close(o,
                           o_ref,
                           rtol=1e-2,
                           atol=1e-2,
                           max_mismatched_ratio=0.01)
        print("MAMBA CHUNK STATE kernel check passed!")
