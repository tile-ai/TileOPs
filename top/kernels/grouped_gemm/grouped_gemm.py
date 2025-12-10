import tilelang
import tilelang.language as T
from typing import Optional, Tuple
from top.kernels.kernel import Kernel
import itertools
import torch

__all__ = ['grouped_gemm_nt_kernel', 'grouped_gemm_nn_kernel', 'grouped_gemm_tn_kernel', 'grouped_gemm_tt_kernel']

def _grouped_gemm_nt_kernel(batch_sum, batch_count, N, K, dtype='float16'):
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _grouped_gemm_nt_func(block_M, block_N, block_K, num_stages, threads):
        A_shape = (batch_sum, K)
        B_shape = (batch_count, N, K)
        C_shape = (batch_sum, N)
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K)
        @T.prim_func
        def _grouped_gemm_nt_main(
            A: T.Tensor(A_shape, dtype),  # type: ignore
            B: T.Tensor(B_shape, dtype),  # type: ignore
            C: T.Tensor(C_shape, dtype),  # type: ignore
            batch_sizes: T.Tensor([batch_count], "int32"),  # type: ignore
            batch_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
            batch_padded_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(batch_sum, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, dtype)
                B_shared = T.alloc_shared(B_shared_shape, dtype)
                C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
                cur_batch_idx = T.alloc_local([1], "int32")
                m_start = bx * block_M

                cur_batch_idx[0] = 0
                for i in range(batch_count):
                    batch_start = batch_offsets[i]
                    batch_end = batch_start + batch_sizes[i]
                    in_batch = (m_start >= batch_start) & (m_start < batch_end)
                    cur_batch_idx[0] = T.if_then_else(in_batch, i, cur_batch_idx[0])
                batch_start = batch_offsets[cur_batch_idx[0]]
                batch_size = batch_sizes[cur_batch_idx[0]]
                batch_end = batch_start + batch_size
                actual_rows = T.min(block_M, batch_end - m_start)
                T.clear(C_local)
                
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    # Load A block
                    for i, j in T.Parallel(block_M, block_K):
                        A_shared[i, j] = T.if_then_else(
                            i < actual_rows and j < K - k * block_K,
                            A[m_start + i, k * block_K + j], 0)
                    # Load B block
                    for i, j in T.Parallel(block_N, block_K):
                        B_shared[i, j] = T.if_then_else(
                            j < K - k * block_K and i < N - by * block_N,
                            B[cur_batch_idx[0], by * block_N + i, k * block_K + j], 0)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Store result
                for i, j in T.Parallel(block_M, block_N):
                    if i < actual_rows and j < N - by * block_N:
                        C[m_start + i, by * block_N + j] = C_local[i, j]
        return _grouped_gemm_nt_main
    
    return _grouped_gemm_nt_func

class grouped_gemm_nt_kernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(self,
                 batch_sum,
                 batch_count,
                 N,
                 K,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype

        self.kernel = _grouped_gemm_nt_kernel(self.batch_sum, self.batch_count, self.N, self.K, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 64,
            "block_N": 256,
            "block_K": 64,
            "num_stages": 2,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128, 256]
        block_N = [32, 64, 128, 256]
        block_K = [32, 64, 128, 256]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, block_K, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'block_K': c[2],
            'num_stages': c[3],
            'threads': c[4]
        } for c in _configs]
        return configs
    
    def forward(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        kernel = _grouped_gemm_nt_kernel(
            self.batch_sum, self.batch_count, self.N, self.K, self.dtype_str
        )(
            self.config["block_M"], 
            self.config["block_N"], 
            self.config["block_K"],
            self.config["num_stages"], 
            self.config["threads"]
        )
        return kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)


def _grouped_gemm_nn_kernel(batch_sum, batch_count, N, K, dtype='float16'):
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _grouped_gemm_nn_func(block_M, block_N, block_K, num_stages, threads):
        A_shape = (batch_sum, K)
        B_shape = (batch_count, K, N)
        C_shape = (batch_sum, N)
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_K, block_N)
        @T.prim_func
        def _grouped_gemm_nn_main(
            A: T.Tensor(A_shape, dtype),  # type: ignore
            B: T.Tensor(B_shape, dtype),  # type: ignore
            C: T.Tensor(C_shape, dtype),  # type: ignore
            batch_sizes: T.Tensor([batch_count], "int32"),  # type: ignore
            batch_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
            batch_padded_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(batch_sum, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, dtype)
                B_shared = T.alloc_shared(B_shared_shape, dtype)
                C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
                cur_batch_idx = T.alloc_local([1], "int32")
                m_start = bx * block_M
                n_start = by * block_N

                cur_batch_idx[0] = 0
                for i in range(batch_count):
                    batch_start = batch_offsets[i]
                    batch_end = batch_start + batch_sizes[i]
                    in_batch = (m_start >= batch_start) & (m_start < batch_end)
                    cur_batch_idx[0] = T.if_then_else(in_batch, i, cur_batch_idx[0])
                batch_start = batch_offsets[cur_batch_idx[0]]
                batch_size = batch_sizes[cur_batch_idx[0]]
                batch_end = batch_start + batch_size
                actual_rows = T.min(block_M, batch_end - m_start)
                actual_cols = T.min(block_N, N - n_start)
                T.clear(C_local)
                
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    # Load A block
                    for i, j in T.Parallel(block_M, block_K):
                        A_shared[i, j] = T.if_then_else(
                            i < actual_rows and j < K - k * block_K,
                            A[m_start + i, k * block_K + j], 0)
                    # Load B block
                    for i, j in T.Parallel(block_K, block_N):
                        B_shared[i, j] = T.if_then_else(
                            i < K - k * block_K and j < actual_cols,
                            B[cur_batch_idx[0], k * block_K + i, n_start + j], 0)
                    T.gemm(A_shared, B_shared, C_local)
                # Store result
                for i, j in T.Parallel(block_M, block_N):
                    if i < actual_rows and j < actual_cols:
                        C[m_start + i, n_start + j] = C_local[i, j]
        return _grouped_gemm_nn_main

    return _grouped_gemm_nn_func

class grouped_gemm_nn_kernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(self,
                 batch_sum,
                 batch_count,
                 N,
                 K,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype

        self.kernel = _grouped_gemm_nn_kernel(self.batch_sum, self.batch_count, self.N, self.K, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 128,
            "block_N": 128,
            "block_K": 32,
            "num_stages": 2,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128, 256]
        block_N = [32, 64, 128, 256]
        block_K = [32, 64, 128, 256]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, block_K, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'block_K': c[2],
            'num_stages': c[3],
            'threads': c[4]
        } for c in _configs]
        return configs
    
    def forward(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        kernel = _grouped_gemm_nn_kernel(
            self.batch_sum, self.batch_count, self.N, self.K, self.dtype_str
        )(
            self.config["block_M"], 
            self.config["block_N"], 
            self.config["block_K"],
            self.config["num_stages"],
            self.config["threads"]
        )
        return kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)


def _grouped_gemm_tn_kernel(batch_sum, batch_count, N, K, dtype='float16'):
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _grouped_gemm_tn_func(block_M, block_N, block_K, num_stages, threads):
        A_shape = (batch_sum, N)
        B_shape = (batch_sum, K)
        C_shape = (batch_count, N, K)
        A_shared_shape = (block_M, block_N)
        B_shared_shape = (block_M, block_K)
        @T.prim_func
        def _grouped_gemm_tn_main(
            A: T.Tensor(A_shape, dtype),  # type: ignore
            B: T.Tensor(B_shape, dtype),  # type: ignore
            C: T.Tensor(C_shape, dtype),  # type: ignore
            batch_sizes: T.Tensor([batch_count], "int32"),  # type: ignore
            batch_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
            batch_padded_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        ):
            with T.Kernel(batch_count, T.ceildiv(N, block_N) * T.ceildiv(K, block_K), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, dtype)
                B_shared = T.alloc_shared(B_shared_shape, dtype)
                C_local = T.alloc_fragment([block_N, block_K], accum_dtype)

                n_block_idx = by // T.ceildiv(K, block_K)
                k_block_idx = by % T.ceildiv(K, block_K)
                n_start = n_block_idx * block_N
                k_start = k_block_idx * block_K
                actual_N = T.min(block_N, N - n_start)
                actual_K = T.min(block_K, K - k_start)
                T.clear(C_local)

                batch_start = batch_offsets[bx]
                batch_size = batch_sizes[bx]
                
                for m in T.Pipelined(T.ceildiv(batch_size, block_M), num_stages=num_stages):
                    m_start = batch_start + m * block_M
                    actual_rows = T.min(block_M, batch_size - m * block_M)
                    for i, j in T.Parallel(block_M, block_N):
                        A_shared[i, j] = T.if_then_else(
                            i < actual_rows and j < actual_N,
                            A[m_start + i, n_start + j], 0)
                    for i, j in T.Parallel(block_M, block_K):
                        B_shared[i, j] = T.if_then_else(
                            i < actual_rows and j < actual_K,
                            B[m_start + i, k_start + j], 0)
                    T.gemm(A_shared, B_shared, C_local, transpose_A=True)
                for i, j in T.Parallel(block_N, block_K):
                    if i < actual_N and j < actual_K:
                        C[bx, n_start + i, k_start + j] = C_local[i, j]
        return _grouped_gemm_tn_main

    return _grouped_gemm_tn_func

class grouped_gemm_tn_kernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(self,
                 batch_sum,
                 batch_count,
                 N,
                 K,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype

        self.kernel = _grouped_gemm_tn_kernel(self.batch_sum, self.batch_count, self.N, self.K, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 32,
            "block_N": 128,
            "block_K": 128,
            "num_stages": 2,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128, 256]
        block_N = [32, 64, 128, 256]
        block_K = [32, 64, 128, 256]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, block_K, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'block_K': c[2],
            'num_stages': c[3],
            'threads': c[4]
        } for c in _configs]
        return configs
    
    def forward(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        kernel = _grouped_gemm_tn_kernel(
            self.batch_sum, self.batch_count, self.N, self.K, self.dtype_str
        )(
            self.config["block_M"], 
            self.config["block_N"], 
            self.config["block_K"],
            self.config["num_stages"], 
            self.config["threads"]
        )
        return kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)


def _grouped_gemm_tt_kernel(batch_sum, batch_count, N, K, dtype='float16'):
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _grouped_gemm_tt_func(block_M, block_N, block_K, num_stages, threads):
        A_shape = (batch_sum, N)
        B_shape = (K, batch_sum)
        C_shape = (batch_count, N, K)
        A_shared_shape = (block_M, block_N)
        B_shared_shape = (block_K, block_M)
        
        @T.prim_func
        def _grouped_gemm_tt_main(
            A: T.Tensor(A_shape, dtype),  # type: ignore
            B: T.Tensor(B_shape, dtype),  # type: ignore
            C: T.Tensor(C_shape, dtype),  # type: ignore
            batch_sizes: T.Tensor([batch_count], "int32"),  # type: ignore
            batch_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
            batch_padded_offsets: T.Tensor([batch_count], "int32"),  # type: ignore
        ):
            with T.Kernel(batch_count, T.ceildiv(N, block_N) * T.ceildiv(K, block_K), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, dtype)
                B_shared = T.alloc_shared(B_shared_shape, dtype)
                C_local = T.alloc_fragment([block_N, block_K], accum_dtype)

                n_block_idx = by // T.ceildiv(K, block_K)
                k_block_idx = by % T.ceildiv(K, block_K)
                n_start = n_block_idx * block_N
                k_start = k_block_idx * block_K
                actual_N = T.min(block_N, N - n_start)
                actual_K = T.min(block_K, K - k_start)
                T.clear(C_local)

                batch_start = batch_offsets[bx]
                batch_size = batch_sizes[bx]
                
                for m in T.Pipelined(T.ceildiv(batch_size, block_M), num_stages=num_stages):
                    m_start = batch_start + m * block_M
                    actual_rows = T.min(block_M, batch_size - m * block_M)
                    for i, j in T.Parallel(block_M, block_N):
                        A_shared[i, j] = T.if_then_else(
                            i < actual_rows and j < actual_N,
                            A[m_start + i, n_start + j], 0)
                    for i, j in T.Parallel(block_K, block_M):
                        B_shared[i, j] = T.if_then_else(
                            i < actual_K and j < actual_rows,
                            B[k_start + i, m_start + j], 0)
                    T.gemm(A_shared, B_shared, C_local, transpose_A=True, transpose_B=True)
                for i, j in T.Parallel(block_N, block_K):
                    if i < actual_N and j < actual_K:
                        C[bx, n_start + i, k_start + j] = C_local[i, j]
        return _grouped_gemm_tt_main

    return _grouped_gemm_tt_func


class grouped_gemm_tt_kernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(self,
                 batch_sum,
                 batch_count,
                 N,
                 K,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype

        self.kernel = _grouped_gemm_tt_kernel(self.batch_sum, self.batch_count, self.N, self.K, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 64,
            "block_N": 256,
            "block_K": 64,
            "num_stages": 2,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128, 256]
        block_N = [32, 64, 128, 256]
        block_K = [32, 64, 128, 256]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, block_K, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'block_K': c[2],
            'num_stages': c[3],
            'threads': c[4]
        } for c in _configs]
        return configs
    
    def forward(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        kernel = _grouped_gemm_tt_kernel(
            self.batch_sum, self.batch_count, self.N, self.K, self.dtype_str
        )(
            self.config["block_M"], 
            self.config["block_N"], 
            self.config["block_K"],
            self.config["num_stages"], 
            self.config["threads"]
        )
        return kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
