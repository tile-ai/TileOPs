import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = [
    "grouped_gemm_kernel",
]

# Default configs per layout variant (transpose_a, transpose_b)
_DEFAULT_CONFIGS = {
    (False, True): {"block_m": 64, "block_n": 256, "block_k": 64, "num_stages": 2, "threads": 128},
    (False, False): {
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "num_stages": 2,
        "threads": 128
    },
    (True, False): {
        "block_m": 32,
        "block_n": 128,
        "block_k": 128,
        "num_stages": 2,
        "threads": 128
    },
    (True, True): {"block_m": 64, "block_n": 256, "block_k": 64, "num_stages": 2, "threads": 128},
}


# flake8: noqa
def _grouped_gemm_kernel(batch_sum, batch_count, N, K, transpose_a, transpose_b, dtype='float16'):
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _grouped_gemm_func(block_m, block_n, block_k, num_stages, threads):
        if not transpose_a:
            # NT / NN pattern: iterate over K, batch-offset lookup
            A_shape = (batch_sum, K)
            C_shape = (batch_sum, N)
            A_shared_shape = (block_m, block_k)
            if transpose_b:
                B_shape = (batch_count, N, K)
                B_shared_shape = (block_n, block_k)
            else:
                B_shape = (batch_count, K, N)
                B_shared_shape = (block_k, block_n)

            @T.prim_func
            def _grouped_gemm_main(
                    A: T.Tensor(A_shape, dtype),  # type: ignore
                    B: T.Tensor(B_shape, dtype),  # type: ignore
                    C: T.Tensor(C_shape, dtype),  # type: ignore
                    batch_sizes: T.Tensor([batch_count], "int32"),  # noqa: F821
                    batch_offsets: T.Tensor([batch_count], "int32"),  # noqa: F821
                    batch_padded_offsets: T.Tensor([batch_count], "int32"),  # noqa: F821
            ):
                with T.Kernel(
                        T.ceildiv(batch_sum, block_m), T.ceildiv(N, block_n),
                        threads=threads) as (bx, by):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_m, block_n], accum_dtype)
                    cur_batch_idx = T.alloc_local([1], "int32")
                    m_start = bx * block_m
                    n_start = by * block_n

                    cur_batch_idx[0] = 0
                    for i in range(batch_count):
                        batch_start = batch_offsets[i]
                        batch_end = batch_start + batch_sizes[i]
                        in_batch = (m_start >= batch_start) & (m_start < batch_end)
                        cur_batch_idx[0] = T.if_then_else(in_batch, i, cur_batch_idx[0])
                    batch_start = batch_offsets[cur_batch_idx[0]]
                    batch_size = batch_sizes[cur_batch_idx[0]]
                    batch_end = batch_start + batch_size
                    actual_rows = T.min(block_m, batch_end - m_start)
                    actual_cols = T.min(block_n, N - n_start)
                    T.clear(C_local)

                    for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                        # Load A block (same for NT and NN)
                        for i, j in T.Parallel(block_m, block_k):
                            A_shared[i, j] = T.if_then_else(
                                i < actual_rows and j < K - k * block_k,
                                A[m_start + i, k * block_k + j], 0)
                        # Load B block
                        if transpose_b:
                            for i, j in T.Parallel(block_n, block_k):
                                B_shared[i, j] = T.if_then_else(
                                    j < K - k * block_k and i < actual_cols,
                                    B[cur_batch_idx[0], n_start + i, k * block_k + j], 0)
                        else:
                            for i, j in T.Parallel(block_k, block_n):
                                B_shared[i, j] = T.if_then_else(
                                    i < K - k * block_k and j < actual_cols,
                                    B[cur_batch_idx[0], k * block_k + i, n_start + j], 0)
                        T.gemm(A_shared, B_shared, C_local, transpose_B=transpose_b)
                    # Store result
                    for i, j in T.Parallel(block_m, block_n):
                        if i < actual_rows and j < actual_cols:
                            C[m_start + i, n_start + j] = C_local[i, j]

        else:
            # TN / TT pattern: iterate per batch, outputs 3D tensor
            A_shape = (batch_sum, N)
            C_shape = (batch_count, N, K)
            A_shared_shape = (block_m, block_n)
            if transpose_b:
                B_shape = (K, batch_sum)
                B_shared_shape = (block_k, block_m)
            else:
                B_shape = (batch_sum, K)
                B_shared_shape = (block_m, block_k)

            @T.prim_func
            def _grouped_gemm_main(
                    A: T.Tensor(A_shape, dtype),  # type: ignore
                    B: T.Tensor(B_shape, dtype),  # type: ignore
                    C: T.Tensor(C_shape, dtype),  # type: ignore
                    batch_sizes: T.Tensor([batch_count], "int32"),  # noqa: F821
                    batch_offsets: T.Tensor([batch_count], "int32"),  # noqa: F821
                    batch_padded_offsets: T.Tensor([batch_count], "int32"),  # noqa: F821
            ):
                with T.Kernel(
                        batch_count, T.ceildiv(N, block_n) * T.ceildiv(K, block_k),
                        threads=threads) as (bx, by):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_n, block_k], accum_dtype)

                    n_block_idx = by // T.ceildiv(K, block_k)
                    k_block_idx = by % T.ceildiv(K, block_k)
                    n_start = n_block_idx * block_n
                    k_start = k_block_idx * block_k
                    actual_N = T.min(block_n, N - n_start)
                    actual_K = T.min(block_k, K - k_start)
                    T.clear(C_local)

                    batch_start = batch_offsets[bx]
                    batch_size = batch_sizes[bx]

                    for m in T.Pipelined(T.ceildiv(batch_size, block_m), num_stages=num_stages):
                        m_start = batch_start + m * block_m
                        actual_rows = T.min(block_m, batch_size - m * block_m)
                        # Load A block (same for TN and TT)
                        for i, j in T.Parallel(block_m, block_n):
                            A_shared[i, j] = T.if_then_else(i < actual_rows and j < actual_N,
                                                            A[m_start + i, n_start + j], 0)
                        # Load B block
                        if transpose_b:
                            for i, j in T.Parallel(block_k, block_m):
                                B_shared[i, j] = T.if_then_else(
                                    i < actual_K and j < actual_rows,
                                    B[k_start + i, m_start + j], 0)
                        else:
                            for i, j in T.Parallel(block_m, block_k):
                                B_shared[i, j] = T.if_then_else(
                                    i < actual_rows and j < actual_K,
                                    B[m_start + i, k_start + j], 0)
                        T.gemm(A_shared, B_shared, C_local, transpose_A=True, transpose_B=transpose_b)
                    # Store result
                    for i, j in T.Parallel(block_n, block_k):
                        if i < actual_N and j < actual_K:
                            C[bx, n_start + i, k_start + j] = C_local[i, j]

        return _grouped_gemm_main

    return _grouped_gemm_func


class grouped_gemm_kernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(self,
                 batch_sum,
                 batch_count,
                 N,
                 K,
                 dtype,
                 transpose_a: bool = False,
                 transpose_b: bool = True,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

        self.kernel = _grouped_gemm_kernel(self.batch_sum, self.batch_count, self.N, self.K,
                                           self.transpose_a, self.transpose_b, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return _DEFAULT_CONFIGS[(self.transpose_a, self.transpose_b)]

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [32, 64, 128, 256]
        block_n = [32, 64, 128, 256]
        block_k = [32, 64, 128, 256]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, block_n, block_k, num_stages, threads))

        return [{
            'block_m': c[0],
            'block_n': c[1],
            'block_k': c[2],
            'num_stages': c[3],
            'threads': c[4]
        } for c in _configs]

    def forward(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        kernel = _grouped_gemm_kernel(self.batch_sum, self.batch_count, self.N, self.K,
                                      self.transpose_a, self.transpose_b,
                                      self.dtype_str)(self.config["block_m"],
                                                      self.config["block_n"],
                                                      self.config["block_k"],
                                                      self.config["num_stages"],
                                                      self.config["threads"])
        return kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
