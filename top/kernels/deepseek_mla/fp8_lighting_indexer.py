# ruff: noqa
import itertools
from typing import Optional

import tilelang
from tilelang import language as T
import torch
from tilelang.autotuner import autotune

from top.kernels.kernel import Kernel
# from utils import generate_random_cu_seqlens, per_custom_dims_cast_to_fp8

__all__ = ["fp8_lighting_indexer_kernel"]


def _fp8_lighting_indexer_kernel(seq_len,
                                 heads,
                                 index_dim,
                                 seq_len_kv,
                                 clean_logits=True):

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },)
    def _fp8_lighting_indexer_func(
        # heads,
        # index_dim,
        block_N=256,
        num_stages=3,
        threads=512,
        block_Q=None,
    ):
        if block_Q is None:
            block_Q = 128 // heads
        dtype = T.float8_e4m3fn
        accum_dtype = T.float32
        index_dtype = T.int32

        seq_len = T.dynamic("seq_len")
        seq_len_kv = T.dynamic("seq_len_kv")

        index_q_shape = [seq_len * heads, index_dim]
        index_k_shape = [seq_len_kv, index_dim]
        index_k_scale_shape = [seq_len_kv]
        logits_shape = [seq_len, seq_len_kv]

        @T.prim_func
        # def mqa_attn_return_logits_kernel(
        def _fp8_lighting_indexer_main(
                IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
                IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
                IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype),  # type: ignore
                Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
                Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
                CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
                CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
                index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
                index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
                index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
                s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
                s_reshaped = T.reshape(s, (block_N, block_Q, heads))
                logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
                weights = T.alloc_fragment([block_Q, heads], accum_dtype)

                seq_len_i = bx * block_Q

                cu_k_s_min = T.alloc_var(index_dtype)
                cu_k_e_max = T.alloc_var(index_dtype)

                cu_k_s_min = 2147483647
                cu_k_e_max = -2147483648

                for bq_i in T.serial(block_Q):
                    cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
                for bq_i in T.serial(block_Q):
                    cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

                T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
                T.copy(Weights[seq_len_i, 0], weights)

                for nbn_i in T.Pipelined(
                        T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                    T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                    T.copy(IndexKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                        s_reshaped[bn_i, bq_i,
                                   h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) *
                                           weights[bq_i, h_i]) * index_k_scale_fragment[bn_i]

                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    for bq_i, bn_i in T.Parallel(block_Q, block_N):
                        Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i,
                                                                                               bq_i]

        # Return the kernel function handle
        return _fp8_lighting_indexer_main

    return _fp8_lighting_indexer_func


@tilelang.jit
def clean_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
            Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
            CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
            CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel


@torch.library.custom_op("top::fp8_lighting_indexer_wrapped_kernel", mutates_args=())
def fp8_lighting_indexer_wrapped_kernel(seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                                        clean_logits: bool, block_N: int, num_stages: int,
                                        threads: int, block_Q: int, IndexQ: torch.Tensor,
                                        IndexK: torch.Tensor, IndexKScale: torch.Tensor,
                                        Logits: torch.Tensor, Weights: torch.Tensor,
                                        CuSeqLenKS: torch.Tensor,
                                        CuSeqLenKE: torch.Tensor) -> torch.Tensor:
    print("seq_len:", seq_len)
    print("heads:", heads)
    print("index_dim:", index_dim)
    _fp8_lighting_indexer_kernel(seq_len, heads, index_dim,
                                 seq_len_kv)(block_N, num_stages, threads,
                                             block_Q)(IndexQ.view(seq_len * heads,
                                                                  index_dim), IndexK, IndexKScale,
                                                      Logits, Weights, CuSeqLenKS, CuSeqLenKE)
    if clean_logits:
        clean_logits_()(Logits, CuSeqLenKS, CuSeqLenKE)
    return Logits.clone()


@fp8_lighting_indexer_wrapped_kernel.register_fake
def _(
        seq_len: int,
        heads: int,
        index_dim: int,
        seq_len_kv: int,
        clean_logits: bool,
        block_N: int,
        num_stages: int,
        threads: int,
        block_Q: int,
        IndexQ: torch.Tensor,
        IndexK: torch.Tensor,
        IndexKScale: torch.Tensor,
        Logits: torch.Tensor,
        Weights: torch.Tensor,
        CuSeqLenKS: torch.Tensor,
        CuSeqLenKE: torch.Tensor) -> torch.Tensor:
    fake_o = torch.empty([seq_len, seq_len_kv], dtype=IndexKScale.dtype, device=IndexKScale.device)

    return fake_o


class fp8_lighting_indexer_kernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 seq_len,
                 heads,
                 index_dim,
                 seq_len_kv,
                 clean_logits=True,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.clean_logits = clean_logits
        self.config = config

        self.kernel = _fp8_lighting_indexer_kernel(self.seq_len, self.heads, self.index_dim,
                                                   self.seq_len_kv, self.clean_logits)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_N": 64, "num_stages": 0, "threads": 128, "block_Q": 1}

    @property
    def autotune_configs(self) -> list[dict]:
        block_N = [32, 64, 128]
        num_stages = [0, 1, 2]
        threads = [128, 256]
        block_Q = [1, 2, 4]
        _configs = list(itertools.product(block_N, num_stages, threads, block_Q))

        configs = [{
            'block_N': c[0],
            'num_stages': c[1],
            'threads': c[2],
            'block_Q': c[3],
        } for c in _configs]
        return configs

    def forward(
            self,
            IndexQ: torch.Tensor,  # type: ignore
            IndexK: torch.Tensor,  # type: ignore
            IndexKScale: torch.Tensor,  # type: ignore
            Weights: torch.Tensor,  # type: ignore
            CuSeqLenKS: torch.Tensor,  # type: ignore
            CuSeqLenKE: torch.Tensor,  # type: ignore
    ) -> torch.Tensor:
        Logits = torch.empty([self.seq_len, self.seq_len_kv],
                             device=IndexQ.device,
                             dtype=torch.float32)
        return fp8_lighting_indexer_wrapped_kernel(
            self.seq_len, self.heads, self.index_dim, self.seq_len_kv, self.clean_logits,
            self.config["block_N"], self.config["num_stages"], self.config["threads"],
            self.config["block_Q"], IndexQ, IndexK, IndexKScale, Logits, Weights, CuSeqLenKS,
            CuSeqLenKE)

    def supply_prog(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        dtype=torch.float8_e4m3fn,
        accum_dtype=torch.float32,
        index_dtype=torch.int32,
        params=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.q = q
        self.kv = kv
        seq_len, heads, index_dim = q.shape
        seq_len_kv = kv.shape[0]
        IndexQ = torch.randn(seq_len * heads, index_dim, device='cuda', dtype=torch.float8_e4m3fn)
        IndexK = torch.randn(seq_len_kv, index_dim, device='cuda', dtype=self.dtype)
        IndexKScale = torch.randn(seq_len_kv, device='cuda', dtype=accum_dtype)
        Weights = torch.randn(seq_len, heads, device='cuda', dtype=accum_dtype)
        CuSeqLenKS = torch.zeros(seq_len, device='cuda', dtype=index_dtype)
        CuSeqLenKE = torch.full((seq_len,),
                                fill_value=seq_len_kv - 1,
                                device='cuda',
                                dtype=index_dtype)

        return IndexQ, IndexK, IndexKScale, Weights, CuSeqLenKS, CuSeqLenKE

    def autotune(self, warmup=10, rep=10):  # Removed supply_prog parameter
        if self.autotune_configs is None:
            return  # kernel doesn't support autotuning
        print(f'Start autotuning {self.__class__.__name__}...')

        # Apply autotune decorator to the kernel function
        autotuned_kernel_fn = autotune(
            configs=self.autotune_configs, warmup=warmup, rep=rep, supply_prog=self.supply_prog)(
                self.kernel)

        # Call without config parameters to trigger autotuning, returns the tuned kernel
        tuned_kernel = autotuned_kernel_fn()

        # Extract and store the best config
        self.config = tuned_kernel.config
        print(f'Best config: {self.config}')
