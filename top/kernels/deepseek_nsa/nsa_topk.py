import itertools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = ["nsa_topk_fwd_kernel"]


def _topk_kernel(M, N, topk, dtype="float32"):

    @tilelang.jit(
        out_idx=[1, 2], 
        compile_flags=[
            "--use_fast_math", "-O3", "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "--ptxas-options=-v,--register-usage-level=10", "-DNDEBUG",
            "-DENABLE_BF16", "-DCUDA_FP8_ENABLED=0"
        ],)
    def _topk_func(blk_m, threads):

        @T.prim_func
        def _topk_main(
                logits: T.Tensor([M, N], dtype),
                topk_gates: T.Tensor([M, topk], dtype),
                topk_indices: T.Tensor([M, topk], "int32"),
        ):
            with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
                logits_frag = T.alloc_fragment([blk_m, N], dtype=dtype)
                max_val = T.alloc_fragment([blk_m], dtype=dtype)
                expand_max_idx = T.alloc_fragment([blk_m, N], "int32")
                max_idx = T.alloc_fragment([blk_m], "int32")

                T.copy(logits[bx * blk_m, 0], logits_frag)

                for k in T.serial(topk):
                    T.fill(expand_max_idx, -1)
                    T.reduce_max(logits_frag, max_val, dim=1, clear=True)

                    for i, j in T.Parallel(blk_m, N):
                        expand_max_idx[i, j] = T.if_then_else(max_val[i] == logits_frag[i, j], j,
                                                              expand_max_idx[i, j])

                    T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)

                    min_val = T.cast(-1e10, dtype)
                    for i, j in T.Parallel(blk_m, N):
                        logits_frag[i, j] = T.if_then_else(j == max_idx[i],
                                                           min_val, logits_frag[i, j])

                    for i in T.Parallel(blk_m):
                        topk_gates[bx * blk_m + i, k] = max_val[i]
                        topk_indices[bx * blk_m + i, k] = max_idx[i]

        return _topk_main

    return _topk_func


@torch.library.custom_op("top::topk_kernel", mutates_args=())
def _topk_wrapped_kernel(
    M: int,
    N: int,
    topk: int,
    dtype: str,
    blk_m: int,
    threads: int,
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _topk_kernel(M, N, topk, dtype)(blk_m, threads)(logits)


@_topk_wrapped_kernel.register_fake
def _(M, N, topk, dtype, blk_m, threads, *inputs):
    fake_gates = torch.empty([M, topk])
    fake_indices = torch.empty([M, topk])
    return fake_gates, fake_indices


class nsa_topk_fwd_kernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self, M, N, topk, dtype, config: Optional[dict] = None, tune=False):
        self.M = M
        self.N = N
        self.topk = topk
        self.dtype = dtype
        self.kernel = _topk_kernel(self.M, self.N, self.topk, self.dtype)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"blk_m": 64, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        blk_m = [32, 64, 128]
        threads = [128, 256, 512]
        _configs = list(itertools.product(blk_m, threads))
        configs = [{"blk_m": c[0], "threads": c[1]} for c in _configs]
        return configs

    def forward(self, logits: torch.Tensor):
        return _topk_wrapped_kernel(self.M, self.N, self.topk, self.dtype, self.config["blk_m"],
                                    self.config["threads"], logits)
