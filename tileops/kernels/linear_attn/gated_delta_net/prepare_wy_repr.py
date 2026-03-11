import itertools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["PrepareWYReprKernel"]


def _prepare_wy_repr_kernel(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dtype: str = 'float32'):

    import math
    accum_dtype = "float32"
    _LOG2E = 1.4426950408889634
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _prepare_wy_repr_func(num_stages, threads=128):
        block_C = chunk_size

        @T.macro
        def _prepare_wy_aw_au(
                k: T.Tensor([batch, head, seq_len, dim_k], dtype),
                g: T.Tensor([batch, head, seq_len], dtype),
                beta: T.Tensor([batch, head, seq_len], dtype),
                Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
                Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                k_shared = T.alloc_shared([block_C, dim_k], dtype)
                g_shared = T.alloc_shared([block_C], dtype)
                beta_shared = T.alloc_shared([block_C], dtype)
                k_beta_shared = T.alloc_shared([block_C, dim_k], dtype)
                S_shared = T.alloc_shared([block_C, block_C], dtype)
                P_shared = T.alloc_shared([block_C, block_C], dtype)
                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)

                # Load inputs
                T.copy(g[bid, hid, by * block_C: (by + 1) * block_C], g_shared, disable_tma=True)
                T.copy(beta[bid, hid, by * block_C: (by + 1) * block_C], beta_shared, disable_tma=True)
                T.copy(k[bid, hid, by * block_C: (by + 1) * block_C, :], k_shared, disable_tma=True)

                # k_beta = k * beta
                for i_s, i_k in T.Parallel(block_C, dim_k):
                    k_beta_shared[i_s, i_k] = k_shared[i_s, i_k] * beta_shared[i_s]

                # Gram = k_beta @ k_beta^T (single gemm)
                T.clear(gram_frag)
                T.gemm(k_beta_shared, k_beta_shared, gram_frag, transpose_B=True)

                # --- Compute Aw = (I - tril(Gram,-1))^{-1} ---
                # Neumann series: L^{-1} = I + N + N^2 + ... + N^{C-1}
                # where N = tril(Gram, -1) (strictly lower triangular)
                # Computed via repeated squaring in log2(C) rounds.

                # P = N = tril(Gram, -1)
                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(
                        i > j, gram_frag[i, j], T.float32(0.0))
                # S = I
                for i, j in T.Parallel(block_C, block_C):
                    S_shared[i, j] = T.if_then_else(
                        i == j, T.float32(1.0), T.float32(0.0))

                for _r in T.Serial(num_rounds):
                    # S = S + P @ S
                    T.clear(temp_frag)
                    T.gemm(P_shared, S_shared, temp_frag)
                    for i, j in T.Parallel(block_C, block_C):
                        S_shared[i, j] = S_shared[i, j] + temp_frag[i, j]
                    # P = P @ P
                    T.clear(temp_frag)
                    T.gemm(P_shared, P_shared, temp_frag)
                    T.copy(temp_frag, P_shared)

                T.copy(S_shared, temp_frag)
                T.copy(temp_frag, Aw[bid, hid, by * block_C: (by + 1) * block_C, :], disable_tma=True)

                # --- Compute Au = (I - tril(Gram_g,-1))^{-1} ---
                # N_g = tril(Gram * exp(g_i - g_j), -1)
                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(
                        i > j,
                        gram_frag[i, j] * T.exp2((g_shared[i] - g_shared[j]) * _LOG2E),
                        T.float32(0.0))
                # S = I
                for i, j in T.Parallel(block_C, block_C):
                    S_shared[i, j] = T.if_then_else(
                        i == j, T.float32(1.0), T.float32(0.0))

                for _r in T.Serial(num_rounds):
                    T.clear(temp_frag)
                    T.gemm(P_shared, S_shared, temp_frag)
                    for i, j in T.Parallel(block_C, block_C):
                        S_shared[i, j] = S_shared[i, j] + temp_frag[i, j]
                    T.clear(temp_frag)
                    T.gemm(P_shared, P_shared, temp_frag)
                    T.copy(temp_frag, P_shared)

                T.copy(S_shared, temp_frag)
                T.copy(temp_frag, Au[bid, hid, by * block_C: (by + 1) * block_C, :], disable_tma=True)

        @T.prim_func
        def prepare_wy_repr(
                k: T.Tensor([batch, head, seq_len, dim_k], dtype),
                g: T.Tensor([batch, head, seq_len], dtype),
                beta: T.Tensor([batch, head, seq_len], dtype),
                Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
                Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
        ):
            _prepare_wy_aw_au(k, g, beta, Aw, Au)

        return prepare_wy_repr

    return _prepare_wy_repr_func


@torch.library.custom_op("tileops::prepare_wy_repr_wrapped_kernel", mutates_args=())
def _prepare_wy_repr_wrapped_kernel(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dtype: str,
    num_stages: int,
    threads: int,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel_fn = _prepare_wy_repr_kernel(
        batch, head, seq_len, chunk_size, dim_k, dtype
    )(num_stages, threads)
    Aw, Au = kernel_fn(k, g, beta)
    return Aw, Au


@_prepare_wy_repr_wrapped_kernel.register_fake
def _(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dtype: str,
    num_stages: int,
    threads: int,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Aw = torch.empty(
        batch, head, seq_len, chunk_size,
        dtype=torch.float32, device=k.device
    )
    Au = torch.empty_like(Aw)
    return Aw, Au


class PrepareWYReprKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 head,
                 seq_len,
                 chunk_size,
                 dim_k,
                 dtype: str = 'float32',
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.head = head
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dtype = dtype
        self.weights_dtype = torch.float32
        self.kernel = _prepare_wy_repr_kernel(self.batch, self.head, self.seq_len, self.chunk_size, self.dim_k, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"num_stages": 2, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        num_stages = [2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(num_stages, threads))

        configs = [{
            'num_stages': c[0],
            'threads': c[1]
        } for c in _configs]
        return configs

    def forward(self, k, g, beta):
        return _prepare_wy_repr_wrapped_kernel(
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dtype_str,
            self.config["num_stages"],
            self.config["threads"],
            k,
            g,
            beta,
        )
