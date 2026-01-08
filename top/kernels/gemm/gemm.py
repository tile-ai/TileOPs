import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel
from top.utils import get_sm_version

__all__ = [
    'gemm_kernel',
]


def _gemm_kernel(M, N, K, trans_A, trans_B, dtype='float16'):
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_func(block_M, block_N, block_K, threads, num_stages, enable_rasteration):

        A_shape = (K, M) if trans_A else (M, K)
        B_shape = (N, K) if trans_B else (K, N)
        A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
        B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

        @T.prim_func
        def _gemm_main(
                A: T.Tensor(A_shape, dtype),  # type: ignore
                B: T.Tensor(B_shape, dtype),  # type: ignore
                C: T.Tensor((M, N), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                          threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, dtype)
                B_shared = T.alloc_shared(B_shared_shape, dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)

                T.annotate_layout({
                    C_shared: tilelang.layout.make_swizzled_layout(C_shared),
                })
                T.use_swizzle(10, enable=enable_rasteration)

                T.clear(C_local)

                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    if not trans_A:
                        # A: (M, K)
                        T.copy(A[by * block_M, k * block_K], A_shared)  # [block_M, block_K]
                    else:
                        # A: (K, M)
                        T.copy(A[k * block_K, by * block_M], A_shared)  # [block_K, block_M]

                    if not trans_B:
                        # B: (K, N)
                        T.copy(B[k * block_K, bx * block_N], B_shared)  # [block_K, block_N]
                    else:
                        # B: (N, K)
                        T.copy(B[bx * block_N, k * block_K], B_shared)  # [block_N, block_K]
                    T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)

                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return _gemm_main

    return _gemm_func


class gemm_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 m,
                 n,
                 k,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False,
                 trans_a=False,
                 trans_b=False):
        super().__init__()
        self.M = m
        self.N = n
        self.K = k
        self.dtype = dtype

        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # From tilelang/examples/gemm/example_gemm_autotune.py
        sm_version = get_sm_version()
        if sm_version in {80}:
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 32,
                "num_stages": 2,
                "threads": 128,
                "enable_rasteration": True
            }
        elif sm_version in {90}:
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 64,
                "num_stages": 3,
                "threads": 256,
                "enable_rasteration": True
            }
        else:
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 32,
                "num_stages": 0,
                "threads": 128,
                "enable_rasteration": True
            }

    @property
    def autotune_configs(self) -> list[dict]:
        # From tilelang/examples/gemm/example_gemm_autotune.py
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        enable_rasteration = [True, False]
        _configs = list(
            itertools.product(block_M, block_N, block_K, num_stages, threads, enable_rasteration))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'block_K': c[2],
            'num_stages': c[3],
            'threads': c[4],
            'enable_rasteration': c[5]
        } for c in _configs]
        return configs

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return self.kernel(**self.config)(a, b)


# TODO: add persistent, split-k, steam-k...
