import itertools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["PrepareWYReprKernel"]


def _prepare_wy_repr_kernel(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dtype: str = 'float32'):

    accum_dtype = "float32"

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

                T.copy(g[bid, hid, by * block_C: (by + 1) * block_C], g_shared, disable_tma=True)
                T.copy(beta[bid, hid, by * block_C: (by + 1) * block_C], beta_shared, disable_tma=True)
                for i, j in T.Parallel(block_C, dim_k):
                    k_shared[i, j] = k[bid, hid, by * block_C + i, j]

                k_beta_shared = T.alloc_shared([block_C, dim_k], accum_dtype)
                for i_s, i_k in T.Parallel(block_C, dim_k):
                    k_beta_shared[i_s, i_k] = k_shared[i_s, i_k] * beta_shared[i_s]

                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.clear(gram_frag)
                for m in T.Serial(dim_k):
                    for j, kk in T.Parallel(block_C, block_C):
                        gram_frag[j, kk] += k_beta_shared[j, m] * k_beta_shared[kk, m]

                L_shared = T.alloc_shared([block_C, block_C], accum_dtype)
                A_inv = T.alloc_fragment([block_C, block_C], accum_dtype)
                acc_inv = T.alloc_fragment([block_C], accum_dtype)

                for i, j in T.Parallel(block_C, block_C):
                    L_shared[i, j] = T.if_then_else(
                        i < j,
                        T.float32(0.0),
                        T.if_then_else(i == j, T.float32(1.0), -gram_frag[i, j]),
                    )

                for i, j in T.Parallel(block_C, block_C):
                    A_inv[i, j] = 0
                    if i == j:
                        A_inv[i, j] = 1.0

                for i in T.serial(block_C):
                    for j in T.Serial(block_C):
                        if i > j:
                            acc_inv[j] = 0
                            for kk in T.serial(block_C):
                                acc_inv[j] += T.if_then_else(
                                    kk > i - 1,
                                    T.float32(0.0),
                                    L_shared[i, kk] * A_inv[kk, j],
                                )
                            A_inv[i, j] = -acc_inv[j]
                T.copy(A_inv, Aw[bid, hid, by * block_C: (by + 1) * block_C, :], disable_tma=True)

                gram_g_shared = T.alloc_shared([block_C, block_C], accum_dtype)
                for i, j in T.Parallel(block_C, block_C):
                    gram_g_shared[i, j] = gram_frag[i, j] * T.exp2(
                        (g_shared[i] - g_shared[j]) * 1.4426950408889634
                    )
                for i, j in T.Parallel(block_C, block_C):
                    L_shared[i, j] = T.if_then_else(
                        i < j,
                        T.float32(0.0),
                        T.if_then_else(i == j, T.float32(1.0), -gram_g_shared[i, j]),
                    )
                for i, j in T.Parallel(block_C, block_C):
                    A_inv[i, j] = 0
                    if i == j:
                        A_inv[i, j] = 1.0
                for i in T.serial(block_C):
                    for j in T.Serial(block_C):
                        if i > j:
                            acc_inv[j] = 0
                            for kk in T.serial(block_C):
                                acc_inv[j] += T.if_then_else(
                                    kk > i - 1,
                                    T.float32(0.0),
                                    L_shared[i, kk] * A_inv[kk, j],
                                )
                            A_inv[i, j] = -acc_inv[j]
                T.copy(A_inv, Au[bid, hid, by * block_C: (by + 1) * block_C, :], disable_tma=True)

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
