import torch
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import autotune
from typing import Optional
from .kernel import Kernel
import itertools


__all__ = ['mha_fwd_kernel_sm80']


def _mha_fwd_kernel_sm80(batch, heads, seq_len, dim, is_causal, tune=False):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    def _mha_fwd_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def _mha_fwd_main(
                Q: T.Tensor(shape, dtype),  # type: ignore
                K: T.Tensor(shape, dtype),  # type: ignore
                V: T.Tensor(shape, dtype),  # type: ignore
                Output: T.Tensor(shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))
                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                         -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M:(bx + 1) * block_M])

        return _mha_fwd_main

    #TODO: refactor this
    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tl.jit(out_idx=[3, 4])
        def _mha_fwd_kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return _mha_fwd_func(block_M, block_N, num_stages, threads)

        return _mha_fwd_kernel()
    else:

        @tl.jit(out_idx=[3, 4])
        def _mha_fwd_kernel(block_M, block_N, num_stages, threads):
            return _mha_fwd_func(block_M, block_N, num_stages, threads)

        return _mha_fwd_kernel

    
class mha_fwd_kernel_sm80(Kernel):
    def __init__(self, batch, heads, seq_len, dim, is_causal, config: Optional[dict] = None, tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        
        if tune:
            assert config is None, "config should be None when tune is True"
            # TODO: use autotune to get the best config
        else:
            if config is not None:
                for k, v in self.default_config.items():
                    self.config[k] = config[k] if config.get(k) is not None else v
            else:
                self.config = self.default_config

            self.mod = _mha_fwd_kernel_sm80(self.batch, self.heads, self.seq_len, self.dim, self.is_causal)(**self.config)

        print(f"mha_fwd_kernel_sm80 initialized with config: {self.config}")

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 64,
            "block_N":64,
            "num_stages": 1,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128]
        block_N = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]
        return configs

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        # TODO: Enhance handling of return_lse
        o, lse = self.mod(Q, K, V)
        return o, lse