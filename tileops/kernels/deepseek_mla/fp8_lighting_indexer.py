# ruff: noqa
import itertools
from typing import Optional

import tilelang
from tilelang import language as T
import torch
from tilelang.autotuner import autotune

from tileops.kernels.kernel import Kernel
# from utils import generate_random_cu_seqlens, per_custom_dims_cast_to_fp8

__all__ = ["Fp8LightingIndexerKernel"]


def _fp8_lighting_indexer_kernel(batch,
                                 seq_len,
                                 heads,
                                 index_dim,
                                 seq_len_kv,
                                 kv_group,
                                 clean_logits=True):

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },)
    def _fp8_lighting_indexer_func(
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

        index_q_shape = [batch, seq_len * heads, index_dim]
        index_k_shape = [batch, seq_len_kv, kv_group, index_dim]
        index_k_scale_shape = [batch, seq_len_kv, kv_group]
        logits_shape = [batch, seq_len, seq_len_kv, kv_group]

        @T.prim_func
        def _fp8_lighting_indexer_main(
                IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
                IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
                IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype),  # type: ignore
                Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
                Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
                CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
                CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
        ):
            heads_per_group = heads // kv_group
            with T.Kernel(T.ceildiv(seq_len, block_Q), batch, threads=threads) as (bx, by):
                index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
                index_q_group_shared = T.alloc_shared([block_Q * heads_per_group, index_dim], dtype)
                index_k_shared = T.alloc_shared([block_N, kv_group, index_dim], dtype)
                index_k_group_shared = T.alloc_shared([block_N, index_dim], dtype)
                index_k_scale_fragment = T.alloc_fragment([block_N, kv_group], accum_dtype)
                s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)  #
                s_reshaped = T.reshape(s, (block_N, block_Q, heads_per_group, kv_group))
                s_tmp = T.alloc_fragment([block_N, heads_per_group], accum_dtype)
                logits = T.alloc_fragment([block_N, block_Q, kv_group], accum_dtype)
                weights = T.alloc_fragment([block_Q, heads], accum_dtype)

                seq_len_i = bx * block_Q
                b_i = by

                cu_k_s_min = T.alloc_var(index_dtype)
                cu_k_e_max = T.alloc_var(index_dtype)

                cu_k_s_min = 2147483647
                cu_k_e_max = -2147483648

                for bq_i in T.serial(block_Q):
                    cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
                for bq_i in T.serial(block_Q):
                    cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

                T.copy(IndexQ[b_i, seq_len_i * heads, 0], index_q_shared)
                T.copy(Weights[seq_len_i, 0], weights)

                for nbn_i in T.Pipelined(
                        T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                    T.copy(IndexK[b_i, cu_k_s_min + nbn_i * block_N, 0, 0], index_k_shared)
                    T.copy(IndexKScale[b_i, cu_k_s_min + nbn_i * block_N, 0],
                           index_k_scale_fragment)

                    for g in T.Serial(kv_group):
                        for bn_i, d_i in T.Parallel(block_N, index_dim):
                            index_k_group_shared[bn_i, d_i] = index_k_shared[bn_i, g, d_i]  #
                        for i, d in T.Parallel(block_Q * heads_per_group, index_dim):
                            index_q_group_shared[i, d] = index_q_shared[g * heads_per_group + i,
                                                                        d]  #
                        T.gemm(
                            index_k_group_shared,
                            index_q_group_shared,
                            s_tmp,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for bn_i, h_i in T.Parallel(block_N, heads_per_group):
                            s[bn_i, g * heads_per_group + h_i] = s_tmp[bn_i, h_i]

                    for bn_i, bq_i, h_i, g in T.Parallel(block_N, block_Q, heads_per_group,
                                                         kv_group):
                        s_reshaped[bn_i, bq_i, h_i, g] = (T.max(s_reshaped[bn_i, bq_i, h_i, g], 0) *
                                                          weights[bq_i, g * heads_per_group + h_i]
                                                         ) * index_k_scale_fragment[bn_i, g]

                    T.reduce_sum(s_reshaped, logits, dim=-2, clear=True)

                    for bq_i, bn_i, g in T.Parallel(block_Q, block_N, kv_group):
                        Logits[b_i, seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i,
                               g] = logits[bn_i, bq_i, g]

        # Return the kernel function handle
        return _fp8_lighting_indexer_main

    return _fp8_lighting_indexer_func


@tilelang.jit
def clean_logits_(block_K: int = 4096,):
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    kv_group = T.dynamic("kv_group")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
            Logits: T.Tensor([batch, seq_len, seq_len_kv, kv_group], dtype),  # type: ignore
            CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
            CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, batch, threads=512) as (bx, by):
            tx = T.get_thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // 512):
                    idx = n_i * block_K + k_i * 512 + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        for g in T.serial(kv_group):
                            Logits[by, bx, idx, g] = -T.infinity(dtype)

    return clean_logits_kernel


@torch.library.custom_op("top::fp8_lighting_indexer_wrapped_kernel", mutates_args=())
def fp8_lighting_indexer_wrapped_kernel(batch: int, seq_len: int, heads: int, index_dim: int,
                                        seq_len_kv: int, kv_group: int, clean_logits: bool,
                                        block_N: int, num_stages: int, threads: int, block_Q: int,
                                        IndexQ: torch.Tensor, IndexK: torch.Tensor,
                                        IndexKScale: torch.Tensor, Logits: torch.Tensor,
                                        Weights: torch.Tensor, CuSeqLenKS: torch.Tensor,
                                        CuSeqLenKE: torch.Tensor) -> torch.Tensor:

    _fp8_lighting_indexer_kernel(batch, seq_len, heads, index_dim, seq_len_kv,
                                 kv_group)(block_N, num_stages, threads,
                                           block_Q)(IndexQ.view(batch, seq_len * heads,
                                                                index_dim), IndexK, IndexKScale,
                                                    Logits, Weights, CuSeqLenKS, CuSeqLenKE)
    if clean_logits:
        clean_logits_()(Logits, CuSeqLenKS, CuSeqLenKE)
    return Logits.clone()


@fp8_lighting_indexer_wrapped_kernel.register_fake
def _(
        batch: int,
        seq_len: int,
        heads: int,
        index_dim: int,
        seq_len_kv: int,
        kv_group: int,
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


class Fp8LightingIndexerKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch,
                 seq_len,
                 heads,
                 index_dim,
                 seq_len_kv,
                 kv_group,
                 clean_logits=True,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.clean_logits = clean_logits
        self.config = config

        self.kernel = _fp8_lighting_indexer_kernel(self.batch, self.seq_len, self.heads,
                                                   self.index_dim, self.seq_len_kv, self.kv_group,
                                                   self.clean_logits)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_N": 64, "num_stages": 2, "threads": 128, "block_Q": 1}

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
        Logits = torch.empty([self.batch, self.seq_len, self.seq_len_kv, self.kv_group],
                             device=IndexQ.device,
                             dtype=torch.float32)
        return fp8_lighting_indexer_wrapped_kernel(
            self.batch, self.seq_len, self.heads, self.index_dim, self.seq_len_kv, self.kv_group,
            self.clean_logits, self.config["block_N"], self.config["num_stages"],
            self.config["threads"], self.config["block_Q"], IndexQ, IndexK, IndexKScale, Logits,
            Weights, CuSeqLenKS, CuSeqLenKE)

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
        batch, seq_len, heads, index_dim = q.shape
        seq_len_kv = kv.shape[0]
        IndexQ = torch.randn(
            batch, seq_len * heads, index_dim, device='cuda', dtype=torch.float8_e4m3fn)
        IndexK = torch.randn(
            batch, seq_len_kv, self.kv_group, index_dim, device='cuda', dtype=self.dtype)
        IndexKScale = torch.randn(batch, seq_len_kv, kv_group, device='cuda', dtype=accum_dtype)
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
