import functools
import itertools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.online_softmax import (
    make_log2e_scale,
    make_online_softmax,
    make_online_softmax_with_mask_guard,
    make_rescale,
)

__all__ = [
    'MhaFwdKernel', 'MhaFwdWgmmaPipelinedKernel', 'GqaFwdKernel', 'GqaFwdWgmmaPipelinedKernel',
    'GqaFwdWsKernel', 'GqaFwdWsPersistentKernel',
]

# MHA


@functools.lru_cache(maxsize=32)
def _mha_fwd_kernel(batch: int,
                    heads: int,
                    seq_len: int,
                    dim: int,
                    is_causal: bool,
                    dtype: str = 'float16') -> Callable:
    scale = make_log2e_scale(dim)  # log2(e)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mha_fwd_func(block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:
        shape = (batch, seq_len, heads, dim)
        online_softmax = make_online_softmax(scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def _mha_fwd_main(
                q: T.Tensor(shape, dtype),  # type: ignore
                k: T.Tensor(shape, dtype),  # type: ignore
                v: T.Tensor(shape, dtype),  # type: ignore
                output: T.Tensor(shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(seq_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.copy(q[bz, bx * block_m:(bx + 1) * block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_m, block_n) if is_causal else T.ceildiv(seq_len, block_n))

                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(k[bz, k_idx * block_n:(k_idx + 1) * block_n, by, :], k_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.if_then_else(bx * block_m + i >= k_idx * block_n + j, 0,
                                                         -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    T.copy(v[bz, k_idx * block_n:(k_idx + 1) * block_n, by, :], v_shared)
                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum,
                                   logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, output[bz, bx * block_m:(bx + 1) * block_m, by, :])
                for i in T.Parallel(block_m):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_m:(bx + 1) * block_m])

        return _mha_fwd_main

    return _mha_fwd_func


@torch.library.custom_op("top::mha_fwd_wrapped_kernel", mutates_args=())
def _mha_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _mha_fwd_kernel(batch, heads, seq_len, dim, is_causal,
                           dtype)(block_m, block_n, num_stages, threads)(q, k, v)


@_mha_fwd_wrapped_kernel.register_fake
def _(batch: int, heads: int, seq_len:
      int, dim: int, is_causal: bool, dtype: str,
      block_m: int, block_n: int, num_stages: int, hreads: int,
      *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class MhaFwdKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _mha_fwd_kernel(self.batch, self.heads, self.seq_len, self.dim,
                                      self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 64 if self.dim <= 128 else 32,
            "num_stages": 1,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [32, 64, 128]
        block_n = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, block_n, num_stages, threads))

        return [{
            'block_m': c[0],
            'block_n': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _mha_fwd_wrapped_kernel(self.batch, self.heads, self.seq_len, self.dim,
                                       self.is_causal, self.dtype_str, self.config["block_m"],
                                       self.config["block_n"], self.config["num_stages"],
                                       self.config["threads"], q, k, v)


@functools.lru_cache(maxsize=32)
def _mha_fwd_wgmma_pipelined_kernel(batch: int,
                                    heads: int,
                                    seq_len: int,
                                    dim: int,
                                    is_causal: bool,
                                    dtype: str = "float16") -> Callable:
    scale = make_log2e_scale(dim)  # log2(e)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mha_fwd_wgmma_pipelined_func(block_m: int, block_n: int, num_stages: int,
                                      threads: int) -> Callable:

        shape = (batch, seq_len, heads, dim)

        @T.macro
        def mma0(
            k: T.Tensor(shape, dtype),
            q_shared: T.SharedBuffer([block_m, dim], dtype),
            k_shared: T.SharedBuffer([block_n, dim], dtype),
            acc_s: T.FragmentBuffer([block_m, block_n], accum_dtype),
            k_idx: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ) -> None:
            T.copy(k[bz, k_idx * block_n:(k_idx + 1) * block_n, by, :], k_shared)
            if is_causal:
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.if_then_else(bx * block_m + i >= k_idx * block_n + j, 0,
                                                 -T.infinity(acc_s.dtype))
            else:
                T.clear(acc_s)
            T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def mma1(
            v: T.Tensor(shape, dtype),
            v_shared: T.SharedBuffer([block_n, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_m, block_n], dtype),
            acc_o: T.FragmentBuffer([block_m, dim], accum_dtype),
            k_idx: T.int32,
            by: T.int32,
            bz: T.int32,
        ) -> None:
            T.copy(v[bz, k_idx * block_n:(k_idx + 1) * block_n, by, :], v_shared)
            T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        online_softmax = make_online_softmax(scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def _mha_fwd_wgmma_pipelined_main(
                q: T.Tensor(shape, dtype),  # type: ignore
                k: T.Tensor(shape, dtype),  # type: ignore
                v: T.Tensor(shape, dtype),  # type: ignore
                output: T.Tensor(shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(seq_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                o_shared = T.alloc_shared([block_m, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.annotate_layout({o_shared: tilelang.layout.make_swizzled_layout(o_shared)})
                T.copy(q[bz, bx * block_m:(bx + 1) * block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_m, block_n) if is_causal else T.ceildiv(seq_len, block_n))

                for k_idx in T.Pipelined(
                        loop_range,
                        num_stages=num_stages,
                        order=[-1, 0, 3, 1, -1, 2],
                        stage=[-1, 0, 0, 1, -1, 1],
                        group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]):
                    mma0(k, q_shared, k_shared, acc_s, k_idx, bx, by, bz)
                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum,
                                   logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    mma1(v, v_shared, acc_s_cast, acc_o, k_idx, by, bz)
                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, output[bz, bx * block_m:(bx + 1) * block_m, by, :])
                for i in T.Parallel(block_m):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_m:(bx + 1) * block_m])

        return _mha_fwd_wgmma_pipelined_main

    return _mha_fwd_wgmma_pipelined_func


@torch.library.custom_op("top::mha_fwd_wgmma_pipelined_wrapped_kernel", mutates_args=())
def _mha_fwd_wgmma_pipelined_wrapped_kernel(batch: int, heads: int, seq_len: int, dim: int,
                                            is_causal: bool, dtype: str, block_m: int, block_n: int,
                                            num_stages: int, threads: int, q: torch.Tensor,
                                            k: torch.Tensor,
                                            v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _mha_fwd_wgmma_pipelined_kernel(batch, heads, seq_len, dim, is_causal,
                                           dtype)(block_m, block_n, num_stages, threads)(q, k, v)


@_mha_fwd_wgmma_pipelined_wrapped_kernel.register_fake
def _(batch: int, heads: int, seq_len: int,
      dim: int, is_causal: bool, dtype: str,
      block_m: int, block_n: int, num_stages: int, threads: int,
      *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class MhaFwdWgmmaPipelinedKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _mha_fwd_wgmma_pipelined_kernel(self.batch, self.heads, self.seq_len,
                                                      self.dim, self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_m": 128, "block_n": 128, "num_stages": 2, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [32, 64, 128]
        block_n = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, block_n, num_stages, threads))
        return [{
            'block_m': c[0],
            'block_n': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _mha_fwd_wgmma_pipelined_wrapped_kernel(self.batch, self.heads, self.seq_len,
                                                       self.dim, self.is_causal, self.dtype_str,
                                                       self.config["block_m"],
                                                       self.config["block_n"],
                                                       self.config["num_stages"],
                                                       self.config["threads"], q, k, v)


# GQA


@functools.lru_cache(maxsize=32)
def _gqa_fwd_kernel(batch: int,
                    heads: int,
                    heads_kv: int,
                    seq_len: int,
                    dim: int,
                    is_causal: bool,
                    dtype: str = 'float16') -> Callable:
    scale = make_log2e_scale(dim)  # log2(e)
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_fwd_func(block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:

        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        online_softmax = make_online_softmax(scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def _gqa_fwd_main(
                q: T.Tensor(q_shape, dtype),  # type: ignore
                k: T.Tensor(kv_shape, dtype),  # type: ignore
                v: T.Tensor(kv_shape, dtype),  # type: ignore
                output: T.Tensor(q_shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(seq_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.copy(q[bz, bx * block_m:(bx + 1) * block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_m, block_n) if is_causal else T.ceildiv(seq_len, block_n))

                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(k[bz, k_idx * block_n:(k_idx + 1) * block_n, by // groups, :], k_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.if_then_else(bx * block_m + i >= k_idx * block_n + j, 0,
                                                         -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    T.copy(v[bz, k_idx * block_n:(k_idx + 1) * block_n, by // groups, :], v_shared)
                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum,
                                   logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, output[bz, bx * block_m:(bx + 1) * block_m, by, :])
                for i in T.Parallel(block_m):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_m:(bx + 1) * block_m])

        return _gqa_fwd_main

    return _gqa_fwd_func


@torch.library.custom_op("top::gqa_fwd_wrapped_kernel", mutates_args=())
def _gqa_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal,
                           dtype)(block_m, block_n, num_stages, threads)(q, k, v)


@_gqa_fwd_wrapped_kernel.register_fake
def _(batch: int, heads: int,
      heads_kv: int, seq_len: int, dim: int, is_causal: bool,
      dtype: str, block_m: int, block_n: int, num_stages: int, threads: int,
      *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class GqaFwdKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _gqa_fwd_kernel(self.batch, self.heads, self.heads_kv, self.seq_len, self.dim,
                                      self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 64 if self.dim <= 128 else 32,
            "num_stages": 1,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [32, 64, 128]
        block_n = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, block_n, num_stages, threads))

        return [{
            'block_m': c[0],
            'block_n': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gqa_fwd_wrapped_kernel(self.batch, self.heads, self.heads_kv, self.seq_len,
                                       self.dim, self.is_causal, self.dtype_str,
                                       self.config["block_m"], self.config["block_n"],
                                       self.config["num_stages"], self.config["threads"], q, k, v)


@functools.lru_cache(maxsize=32)
def _gqa_fwd_wgmma_pipelined_kernel(batch: int,
                                    heads: int,
                                    heads_kv: int,
                                    seq_len: int,
                                    dim: int,
                                    is_causal: bool,
                                    dtype: str = "float16") -> Callable:
    scale = make_log2e_scale(dim)  # log2(e)
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_fwd_wgmma_pipelined_func(block_m: int, block_n: int, num_stages: int,
                                      threads: int) -> Callable:

        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        @T.macro
        def mma0(
            k: T.Tensor(kv_shape, dtype),
            q_shared: T.SharedBuffer([block_m, dim], dtype),
            k_shared: T.SharedBuffer([block_n, dim], dtype),
            acc_s: T.FragmentBuffer([block_m, block_n], accum_dtype),
            k_idx: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ) -> None:
            T.copy(k[bz, k_idx * block_n:(k_idx + 1) * block_n, by // groups, :], k_shared)
            if is_causal:
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.if_then_else(bx * block_m + i >= k_idx * block_n + j, 0,
                                                 -T.infinity(acc_s.dtype))
            else:
                T.clear(acc_s)
            T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def mma1(
            v: T.Tensor(kv_shape, dtype),
            v_shared: T.SharedBuffer([block_n, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_m, block_n], dtype),
            acc_o: T.FragmentBuffer([block_m, dim], accum_dtype),
            k_idx: T.int32,
            by: T.int32,
            bz: T.int32,
        ) -> None:
            T.copy(v[bz, k_idx * block_n:(k_idx + 1) * block_n, by // groups, :], v_shared)
            T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        online_softmax = make_online_softmax(scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def _gqa_fwd_wgmma_pipelined_main(
                q: T.Tensor(q_shape, dtype),  # type: ignore
                k: T.Tensor(kv_shape, dtype),  # type: ignore
                v: T.Tensor(kv_shape, dtype),  # type: ignore
                output: T.Tensor(q_shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(seq_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                o_shared = T.alloc_shared([block_m, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.annotate_layout({o_shared: tilelang.layout.make_swizzled_layout(o_shared)})
                T.copy(q[bz, bx * block_m:(bx + 1) * block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_m, block_n) if is_causal else T.ceildiv(seq_len, block_n))

                for k_idx in T.Pipelined(
                        loop_range,
                        num_stages=num_stages,
                        order=[-1, 0, 3, 1, -1, 2],
                        stage=[-1, 0, 0, 1, -1, 1],
                        group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]):
                    mma0(k, q_shared, k_shared, acc_s, k_idx, bx, by, bz)
                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum,
                                   logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    mma1(v, v_shared, acc_s_cast, acc_o, k_idx, by, bz)
                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, output[bz, bx * block_m:(bx + 1) * block_m, by, :])
                for i in T.Parallel(block_m):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_m:(bx + 1) * block_m])

        return _gqa_fwd_wgmma_pipelined_main

    return _gqa_fwd_wgmma_pipelined_func


@torch.library.custom_op("top::gqa_fwd_wgmma_pipelined_wrapped_kernel", mutates_args=())
def _gqa_fwd_wgmma_pipelined_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_wgmma_pipelined_kernel(batch, heads, heads_kv, seq_len, dim, is_causal,
                                           dtype)(block_m, block_n, num_stages, threads)(q, k, v)


@_gqa_fwd_wgmma_pipelined_wrapped_kernel.register_fake
def _(batch: int, heads: int, heads_kv: int,
      seq_len: int, dim: int, is_causal: bool,
      dtype: str, block_m: int, block_n: int, num_stages: int, threads: int,
      *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class GqaFwdWgmmaPipelinedKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _gqa_fwd_wgmma_pipelined_kernel(self.batch, self.heads, self.heads_kv,
                                                      self.seq_len, self.dim, self.is_causal,
                                                      self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_m": 128, "block_n": 128, "num_stages": 2, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [32, 64, 128]
        block_n = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, block_n, num_stages, threads))
        return [{
            'block_m': c[0],
            'block_n': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gqa_fwd_wgmma_pipelined_wrapped_kernel(self.batch, self.heads, self.heads_kv,
                                                       self.seq_len, self.dim, self.is_causal,
                                                       self.dtype_str, self.config["block_m"],
                                                       self.config["block_n"],
                                                       self.config["num_stages"],
                                                       self.config["threads"], q, k, v)

# ---------------------------------------------------------------------------
# GQA Forward: warp-specialized variant (FA3-aligned, Hopper TMA + barriers)
# 3-WG design: WG0=producer, WG1=consumer(rows 0..half_m), WG2=consumer(rows half_m..block_m)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _gqa_fwd_ws_kernel(batch: int,
                       heads: int,
                       heads_kv: int,
                       seq_len: int,
                       dim: int,
                       is_causal: bool,
                       dtype: str = "float16") -> Callable:
    """FA3-aligned warp-specialized GQA forward.

    3-WG design with double-buffered K/V, raw thread-binding (no T.ws blocks),
    mbarrier-based ping-pong scheduler between consumer warpgroups.

    Causal IntraWGOverlap fix (post-wgmma mask): the causal mask is applied
    AFTER ``wait_wgmma`` (i.e., once the wgmma has drained ``acc_s``), not
    before the next wgmma issue.  Placing a conditional non-wgmma write to
    ``acc_s`` inside the K loop body would force TileLang's data-flow
    analysis to insert ``wait_group<0>`` instead of ``<1>``, destroying
    IntraWGOverlap.  Full SASS-level analysis is documented in
    `tile-ai/TileOPs#872`.

    No out-of-tree TileLang patches required.  Both the mbarrier scheduler
    and the post-wgmma mask use only first-class TileLang APIs
    (``T.alloc_barrier``, ``T.barrier_arrive``, ``T.barrier_wait``,
    ``T.tma_copy``, ``T.wgmma_gemm``).  See ``tile-ai/TileOPs#872`` for the
    upstream tilelang gaps that motivated this design and the perf headroom
    that would be unlocked by addressing them.
    """
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError(
            f"GqaFwdWsKernel currently requires dim==128, got dim={dim}")
    scale = make_log2e_scale(dim)
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_fwd_ws_func(block_m: int, block_n: int) -> Callable:
        if block_m % 2 != 0:
            raise ValueError(f"block_m must be even, got block_m={block_m}")
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        half_m = block_m // 2
        softmax_1 = make_online_softmax_with_mask_guard(
            scale, accum_dtype, half_m, block_n)
        softmax_2 = make_online_softmax_with_mask_guard(
            scale, accum_dtype, half_m, block_n)
        rescale_1 = make_rescale(half_m, dim)
        rescale_2 = make_rescale(half_m, dim)

        @T.prim_func
        def _gqa_fwd_ws_main(
                q: T.Tensor(q_shape, dtype),  # type: ignore
                k: T.Tensor(kv_shape, dtype),  # type: ignore
                v: T.Tensor(kv_shape, dtype),  # type: ignore
                output: T.Tensor(q_shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(seq_len, block_m), heads, batch,
                    threads=384) as (bx, by, bz):
                # ---- Shared memory ----
                q_shared_1 = T.alloc_shared([half_m, dim], dtype)
                q_shared_2 = T.alloc_shared([half_m, dim], dtype)
                # Double-buffered K and V
                k_smem_0 = T.alloc_shared([block_n, dim], dtype)
                k_smem_1 = T.alloc_shared([block_n, dim], dtype)
                v_smem_0 = T.alloc_shared([block_n, dim], dtype)
                v_smem_1 = T.alloc_shared([block_n, dim], dtype)

                # ---- Consumer 1 fragments (rows 0..half_m) ----
                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_1 = T.alloc_fragment([half_m, block_n], dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                # ---- Consumer 2 fragments (rows half_m..block_m) ----
                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_2 = T.alloc_fragment([half_m, block_n], dtype)
                acc_o_2 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_2 = T.alloc_fragment([half_m], accum_dtype)
                smp_2 = T.alloc_fragment([half_m], accum_dtype)
                ss_2 = T.alloc_fragment([half_m], accum_dtype)
                ssum_2 = T.alloc_fragment([half_m], accum_dtype)
                ls_2 = T.alloc_fragment([half_m], accum_dtype)

                # ---- Pipeline barriers (FA3-aligned) ----
                # K pipeline: producer -> consumer (data ready)
                k_full = T.alloc_barrier(arrive_count=128)
                # K pipeline: consumer -> producer (buffer free)
                # arrive_count=256: both WG1(128) + WG2(128) must arrive
                k_empty = T.alloc_barrier(arrive_count=256)
                # V pipeline
                v_full = T.alloc_barrier(arrive_count=128)
                v_empty = T.alloc_barrier(arrive_count=256)
                # WG1↔WG2 ping-pong scheduler (mbarrier-based, replaces
                # named-bar bar.arrive helper that needed an out-of-tree
                # tilelang patch).  Each direction has arrive_count=128
                # because only one consumer arrives per phase.
                wg_sched_12 = T.alloc_barrier(arrive_count=128)
                wg_sched_21 = T.alloc_barrier(arrive_count=128)

                T.annotate_layout({
                    q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                    q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                })

                T.sync_threads()  # after barrier init

                head_kv = by // groups
                row_base = bx * block_m
                loop_range = (
                    T.ceildiv((bx + 1) * block_m, block_n)
                    if is_causal else T.ceildiv(seq_len, block_n))

                T.copy(q[bz, row_base:row_base + half_m, by, :], q_shared_1)
                T.copy(q[bz, row_base + half_m:row_base + block_m, by, :],
                       q_shared_2)

                T.sync_threads()  # after Q loads

                # =============================================
                # FA3-aligned per-warpgroup body layout (THREAD-BIND variant)
                # =============================================
                # tx is the raw threadIdx.x. Python if/elif/else lowers via
                # the TIR parser into nested T.If/T.Then/T.Else, which ptxas
                # sees as a true if/elseif/else tree (mutually exclusive
                # branches). No need for shfl_transform / if_else_chain
                # post-process hacks.
                #
                # FA3-style register reallocation (setmaxnreg):
                #   Producer dec:  128 * (168 - 24)  = 18432 regs released
                #   Consumer inc:  256 * (240 - 168) = 18432 regs claimed
                # 24/240 are the only numbers that match for 1+2 WG split.
                tx = T.get_thread_binding()

                # ===== WG0 (producer, tx < 128) =====
                if tx < 128:
                    T.dec_max_nreg(24)
                    for n_idx in T.Pipelined(loop_range, num_stages=0):
                        # Acquire K stage: wait for consumers to free it
                        T.barrier_wait(k_empty, (n_idx + 1) % 2)
                        if n_idx % 2 == 0:
                            T.tma_copy(
                                k[bz, n_idx * block_n:(n_idx + 1) * block_n,
                                  head_kv, :],
                                k_smem_0, barrier=k_full)
                        else:
                            T.tma_copy(
                                k[bz, n_idx * block_n:(n_idx + 1) * block_n,
                                  head_kv, :],
                                k_smem_1, barrier=k_full)
                        T.barrier_arrive(k_full)
                        # Load V[n-1] into v_smem[(n-1)%2]
                        # Wait-phase invariant: this kernel is correct only
                        # because the V pipeline lags K by exactly one
                        # iteration (V[n-1] is loaded in iter n).  The wait
                        # parity ``n_idx % 2`` here is asymmetric vs the K
                        # pipeline's ``(n_idx + 1) % 2`` — it works because
                        # the consumer's iter n-1 arrive on v_empty has
                        # already advanced v_empty's phase by the time the
                        # producer reaches iter n's wait.  If pipeline depth
                        # changes (e.g., V is also loaded in iter 0), this
                        # formula must be revisited.  See tile-ai/TileOPs#871
                        # review (Gabbering) for the analysis.
                        if n_idx > 0:
                            T.barrier_wait(v_empty, n_idx % 2)
                            if (n_idx - 1) % 2 == 0:
                                T.tma_copy(
                                    v[bz,
                                      (n_idx - 1) * block_n:n_idx * block_n,
                                      head_kv, :],
                                    v_smem_0, barrier=v_full)
                            else:
                                T.tma_copy(
                                    v[bz,
                                      (n_idx - 1) * block_n:n_idx * block_n,
                                      head_kv, :],
                                    v_smem_1, barrier=v_full)
                            T.barrier_arrive(v_full)
                    # Producer epilogue: tail load V[loop_range-1]
                    T.barrier_wait(v_empty, loop_range % 2)
                    if (loop_range - 1) % 2 == 0:
                        T.tma_copy(
                            v[bz,
                              (loop_range - 1) * block_n:loop_range * block_n,
                              head_kv, :],
                            v_smem_0, barrier=v_full)
                    else:
                        T.tma_copy(
                            v[bz,
                              (loop_range - 1) * block_n:loop_range * block_n,
                              head_kv, :],
                            v_smem_1, barrier=v_full)
                    T.barrier_arrive(v_full)

                # ===== WG1 (consumer 1, 128 <= tx < 256) =====
                elif tx < 256:
                    T.inc_max_nreg(240)
                    # Asymmetric bootstrap: only WG1 pre-fires its incoming
                    # scheduler mbarrier (wg_sched_21).  WG2 has NO bootstrap
                    # on wg_sched_12 — instead, the WG1↔WG2 ordering
                    # invariant is that WG1 reaches its first barrier_arrive
                    # on wg_sched_12 (inside the n_idx==0 body, after the
                    # first wgmma) BEFORE WG2 reaches its first
                    # barrier_wait on wg_sched_12.  This is enforced by the
                    # barrier_wait(k_full) at the top of both consumers'
                    # K loops: both WGs serialize behind the producer there,
                    # and WG1's post-wgmma signal happens within the same
                    # iteration body before WG2 can advance.
                    T.barrier_arrive(wg_sched_21)  # bootstrap WG1→WG2 sched
                    T.clear(acc_o_1)
                    T.clear(ls_1)
                    T.fill(sm_1, -T.infinity(accum_dtype))
                    for n_idx in T.Pipelined(loop_range, num_stages=0):
                        T.barrier_wait(k_full, n_idx % 2)
                        T.barrier_wait(wg_sched_21, n_idx % 2)
                        # K loop body: ALWAYS clear, no inline mask.  The
                        # causal mask is applied AFTER wait_wgmma (see
                        # function docstring for the IntraWGOverlap rationale).
                        T.clear(acc_s_1)
                        if n_idx == 0:
                            T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1,
                                         transpose_B=True,
                                         policy=T.GemmWarpPolicy.FullRow)
                            T.barrier_arrive(wg_sched_12)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_s_1, num_regs=64)
                            T.barrier_arrive(k_empty)
                            # Post-wgmma mask: only diagonal block.
                            if is_causal:
                                if n_idx == loop_range - 1:
                                    for i, j in T.Parallel(half_m, block_n):
                                        acc_s_1[i, j] = T.if_then_else(
                                            row_base + i
                                            >= n_idx * block_n + j,
                                            acc_s_1[i, j],
                                            -T.infinity(accum_dtype))
                            softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                            T.copy(acc_s_1, acc_s_cast_1)
                        else:
                            if n_idx % 2 == 0:
                                T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow)
                            rescale_1(acc_o_1, ss_1)
                            T.barrier_wait(v_full, (n_idx - 1) % 2)
                            if (n_idx - 1) % 2 == 0:
                                T.wgmma_gemm(acc_s_cast_1, v_smem_0, acc_o_1,
                                             policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(acc_s_cast_1, v_smem_1, acc_o_1,
                                             policy=T.GemmWarpPolicy.FullRow)
                            T.barrier_arrive(wg_sched_12)
                            T.wait_wgmma(1)
                            T.warpgroup_fence_operand(acc_s_1, num_regs=64)
                            T.barrier_arrive(k_empty)
                            # Post-wgmma mask: only diagonal block.
                            if is_causal:
                                if n_idx == loop_range - 1:
                                    for i, j in T.Parallel(half_m, block_n):
                                        acc_s_1[i, j] = T.if_then_else(
                                            row_base + i
                                            >= n_idx * block_n + j,
                                            acc_s_1[i, j],
                                            -T.infinity(accum_dtype))
                            softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                            T.barrier_arrive(v_empty)
                            T.copy(acc_s_1, acc_s_cast_1)
                    # Consumer 1 epilogue: rescale + last PV
                    rescale_1(acc_o_1, ss_1)
                    T.barrier_wait(v_full, (loop_range - 1) % 2)
                    if (loop_range - 1) % 2 == 0:
                        T.wgmma_gemm(acc_s_cast_1, v_smem_0, acc_o_1,
                                     policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.wgmma_gemm(acc_s_cast_1, v_smem_1, acc_o_1,
                                     policy=T.GemmWarpPolicy.FullRow)
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                    # Output write for half 1
                    for i, j in T.Parallel(half_m, dim):
                        acc_o_1[i, j] /= ls_1[i]
                    T.copy(acc_o_1, q_shared_1)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=3, arrive_count=128)
                    T.copy(q_shared_1,
                           output[bz, row_base:row_base + half_m, by, :])
                    for i in T.Parallel(half_m):
                        ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * scale
                    T.copy(ls_1,
                           lse[bz, by, row_base:row_base + half_m])

                # ===== WG2 (consumer 2, tx >= 256) =====
                else:
                    T.inc_max_nreg(240)
                    T.clear(acc_o_2)
                    T.clear(ls_2)
                    T.fill(sm_2, -T.infinity(accum_dtype))
                    for n_idx in T.Pipelined(loop_range, num_stages=0):
                        T.barrier_wait(k_full, n_idx % 2)
                        T.barrier_wait(wg_sched_12, n_idx % 2)
                        # K loop body: ALWAYS clear (mask applied post-wgmma).
                        T.clear(acc_s_2)
                        if n_idx == 0:
                            T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2,
                                         transpose_B=True,
                                         policy=T.GemmWarpPolicy.FullRow)
                            T.barrier_arrive(wg_sched_21)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_s_2, num_regs=64)
                            T.barrier_arrive(k_empty)
                            # Post-wgmma mask: only diagonal block.
                            if is_causal:
                                if n_idx == loop_range - 1:
                                    for i, j in T.Parallel(half_m, block_n):
                                        acc_s_2[i, j] = T.if_then_else(
                                            row_base + half_m + i
                                            >= n_idx * block_n + j,
                                            acc_s_2[i, j],
                                            -T.infinity(accum_dtype))
                            softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                            T.copy(acc_s_2, acc_s_cast_2)
                        else:
                            if n_idx % 2 == 0:
                                T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow)
                            rescale_2(acc_o_2, ss_2)
                            T.barrier_wait(v_full, (n_idx - 1) % 2)
                            if (n_idx - 1) % 2 == 0:
                                T.wgmma_gemm(acc_s_cast_2, v_smem_0, acc_o_2,
                                             policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(acc_s_cast_2, v_smem_1, acc_o_2,
                                             policy=T.GemmWarpPolicy.FullRow)
                            T.barrier_arrive(wg_sched_21)
                            T.wait_wgmma(1)
                            T.warpgroup_fence_operand(acc_s_2, num_regs=64)
                            T.barrier_arrive(k_empty)
                            # Post-wgmma mask: only diagonal block.
                            if is_causal:
                                if n_idx == loop_range - 1:
                                    for i, j in T.Parallel(half_m, block_n):
                                        acc_s_2[i, j] = T.if_then_else(
                                            row_base + half_m + i
                                            >= n_idx * block_n + j,
                                            acc_s_2[i, j],
                                            -T.infinity(accum_dtype))
                            softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                            T.barrier_arrive(v_empty)
                            T.copy(acc_s_2, acc_s_cast_2)
                    # Consumer 2 epilogue: rescale + last PV
                    rescale_2(acc_o_2, ss_2)
                    T.barrier_wait(v_full, (loop_range - 1) % 2)
                    if (loop_range - 1) % 2 == 0:
                        T.wgmma_gemm(acc_s_cast_2, v_smem_0, acc_o_2,
                                     policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.wgmma_gemm(acc_s_cast_2, v_smem_1, acc_o_2,
                                     policy=T.GemmWarpPolicy.FullRow)
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                    # Output write for half 2
                    for i, j in T.Parallel(half_m, dim):
                        acc_o_2[i, j] /= ls_2[i]
                    T.copy(acc_o_2, q_shared_2)
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=4, arrive_count=128)
                    T.copy(q_shared_2,
                           output[bz, row_base + half_m:row_base + block_m,
                                  by, :])
                    for i in T.Parallel(half_m):
                        ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                    T.copy(ls_2,
                           lse[bz, by,
                               row_base + half_m:row_base + block_m])

        return _gqa_fwd_ws_main

    return _gqa_fwd_ws_func


@torch.library.custom_op("top::gqa_fwd_ws_wrapped_kernel", mutates_args=())
def _gqa_fwd_ws_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_ws_kernel(batch, heads, heads_kv, seq_len, dim, is_causal,
                              dtype)(block_m, block_n)(q, k, v)


@_gqa_fwd_ws_wrapped_kernel.register_fake
def _(batch: int, heads: int, heads_kv: int,
      seq_len: int, dim: int, is_causal: bool,
      dtype: str, block_m: int, block_n: int,
      *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class GqaFwdWsKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        # The producer's epilogue tail V load reads
        # ``v[bz, (loop_range-1)*block_n : loop_range*block_n, ...]``
        # which can read past ``seq_len`` if ``seq_len < block_n``.
        # The default block_n is 128; require seq_len to be at least
        # 128 to avoid out-of-bounds TMA reads in the epilogue.
        if seq_len < 128:
            raise ValueError(
                f"GqaFwdWsKernel requires seq_len >= 128 to avoid "
                f"out-of-bounds V loads in the producer epilogue.  "
                f"Got seq_len={seq_len}.")
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _gqa_fwd_ws_kernel(self.batch, self.heads, self.heads_kv, self.seq_len,
                                          self.dim, self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_m": 128, "block_n": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [64, 128]
        block_n = [64, 128]
        _configs = list(itertools.product(block_m, block_n))
        return [{
            'block_m': c[0],
            'block_n': c[1],
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gqa_fwd_ws_wrapped_kernel(self.batch, self.heads, self.heads_kv, self.seq_len,
                                           self.dim, self.is_causal, self.dtype_str,
                                           self.config["block_m"], self.config["block_n"],
                                           q, k, v)


# ---------------------------------------------------------------------------
# GQA Forward: persistent + paired warp-specialized variant
# Causal-only. Persistent CTA over (B, H, M_blocks/2) pairs with FA3 tile
# pairing for causal load balance.  Postmask trick from GqaFwdWsKernel.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _gqa_fwd_ws_persistent_kernel(batch: int,
                                   heads: int,
                                   heads_kv: int,
                                   seq_len: int,
                                   dim: int,
                                   is_causal: bool,
                                   dtype: str = "float16") -> Callable:
    """Persistent CTA + causal tile pairing + post-wgmma mask GQA forward.

    Built on top of ``GqaFwdWsKernel`` (same 3-WG thread-bind structure +
    post-wgmma mask).  Two architectural additions:

    1. **Persistent CTA**: grid is ``min(num_sms, total_pairs)`` instead of
       the natural ``(M_blocks, H, B)``.  Each CTA loops over its share of
       ``(tile_b, tile_h, pair_idx)`` triples via ``T.Persistent``.  Per-WG
       global iteration counters (``gi_kp / gi_vp / gi_kc1 / gi_vc1 / ...``)
       track the cumulative mbarrier phase across tiles, replacing per-tile
       ``n_idx % 2``.  This lets the persistent loop reuse smem buffers
       across tile boundaries without resetting barriers.

    2. **Causal tile pairing**: ``tile_m=k`` is paired with ``tile_m=M-1-k``
       inside each CTA's persistent stream.  Each pair has constant total
       work ``M+1`` K-iters, eliminating the long-tail load imbalance that
       drags vanilla causal to ~50% FA3.  Combined with the postmask
       trick, causal reaches ~84% FA3 on H200.

    Reference shape (B=4 S=4096 H=64 Hkv=8 D=128 fp16) on locked H200:
      causal block_m=128 block_n=128: ~84% FA3 (vs ~80% for GqaFwdWsKernel).

    Hardware lock-in:
      ``num_sms`` is detected from the current device at kernel build time
      and baked into the kernel.  Each ``GqaFwdWsPersistentKernel`` instance
      is therefore locked to one GPU model — running it on a GPU with a
      different SM count would either underutilize SMs (more SMs than
      ``num_sms``) or hang (fewer SMs would leave persistent CTAs unscheduled
      and their consumer barriers never released).  Use the standard
      ``GqaFwdWsKernel`` for hardware-portable Hopper code.

    Restrictions:
      - ``is_causal=True`` only (pairing is causal-only).
      - ``dim==128`` (FA3-aligned 3-WG layout).
      - ``ceil(seq_len / block_m)`` must be a positive even integer
        (pairing requirement).

    No out-of-tree TileLang patches required.  Same upstream tilelang
    gaps as ``GqaFwdWsKernel`` — see ``tile-ai/TileOPs#872`` for the
    perf headroom that would be unlocked by addressing them.
    """
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError(
            f"GqaFwdWsPersistentKernel currently requires dim==128, got dim={dim}")
    if not is_causal:
        raise ValueError(
            "GqaFwdWsPersistentKernel only supports is_causal=True. "
            "For non-causal use GqaFwdWsKernel.")

    # Detect SM count from the *currently selected* CUDA device (respects
    # torch.cuda.set_device() and CUDA_VISIBLE_DEVICES).  Locked into the
    # kernel at build time — see "Hardware lock-in" in the docstring.
    _device = torch.cuda.current_device()
    num_sms = torch.cuda.get_device_properties(_device).multi_processor_count

    scale = make_log2e_scale(dim)
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_fwd_ws_persistent_func(block_m: int, block_n: int) -> Callable:
        if block_m % 2 != 0:
            raise ValueError(f"block_m must be even, got block_m={block_m}")
        half_m = block_m // 2
        M_blocks = (seq_len + block_m - 1) // block_m
        if M_blocks % 2 != 0:
            raise ValueError(
                f"M_blocks={M_blocks} (seq_len={seq_len}, block_m={block_m}) "
                f"must be even for causal tile pairing")
        half_M_blocks = M_blocks // 2
        total_pairs = batch * heads * half_M_blocks
        # Clamp num_sms to total_pairs to avoid idle CTAs in single-wave
        # case.  T.Persistent doesn't emit loop_break when waves==1, so
        # idle CTAs would leak out-of-range pair_idx → negative tile_m →
        # negative loop_range → CUDA_ERROR_ILLEGAL_INSTRUCTION inside wgmma.
        effective_num_sms = min(num_sms, total_pairs)

        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        softmax_1 = make_online_softmax_with_mask_guard(
            scale, accum_dtype, half_m, block_n)
        softmax_2 = make_online_softmax_with_mask_guard(
            scale, accum_dtype, half_m, block_n)
        rescale_1 = make_rescale(half_m, dim)
        rescale_2 = make_rescale(half_m, dim)

        @T.prim_func
        def _gqa_fwd_ws_persistent_main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(kv_shape, dtype),
            v: T.Tensor(kv_shape, dtype),
            output: T.Tensor(q_shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ) -> None:
            with T.Kernel(
                effective_num_sms, 1, 1, threads=384,
            ) as (bx, _by, _bz):
                # ---- Shared memory ----
                q_shared_1 = T.alloc_shared([half_m, dim], dtype)
                q_shared_2 = T.alloc_shared([half_m, dim], dtype)
                k_smem_0 = T.alloc_shared([block_n, dim], dtype)
                k_smem_1 = T.alloc_shared([block_n, dim], dtype)
                v_smem_0 = T.alloc_shared([block_n, dim], dtype)
                v_smem_1 = T.alloc_shared([block_n, dim], dtype)

                # ---- Fragments ----
                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_1 = T.alloc_fragment(
                    [half_m, block_n], dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_2 = T.alloc_fragment(
                    [half_m, block_n], dtype)
                acc_o_2 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_2 = T.alloc_fragment([half_m], accum_dtype)
                smp_2 = T.alloc_fragment([half_m], accum_dtype)
                ss_2 = T.alloc_fragment([half_m], accum_dtype)
                ssum_2 = T.alloc_fragment([half_m], accum_dtype)
                ls_2 = T.alloc_fragment([half_m], accum_dtype)

                # ---- Pipeline barriers ----
                k_full = T.alloc_barrier(arrive_count=128)
                k_empty = T.alloc_barrier(arrive_count=256)
                v_full = T.alloc_barrier(arrive_count=128)
                v_empty = T.alloc_barrier(arrive_count=256)
                # WG1↔WG2 ping-pong scheduler (mbarrier-based, replaces
                # named-bar bar.arrive helper that needed an out-of-tree
                # tilelang patch).  Each direction has arrive_count=128
                # because only one consumer arrives per phase.
                wg_sched_12 = T.alloc_barrier(arrive_count=128)
                wg_sched_21 = T.alloc_barrier(arrive_count=128)
                q_full_1 = T.alloc_barrier(arrive_count=128)
                q_full_2 = T.alloc_barrier(arrive_count=128)

                T.annotate_layout({
                    q_shared_1:
                        tilelang.layout.make_swizzled_layout(q_shared_1),
                    q_shared_2:
                        tilelang.layout.make_swizzled_layout(q_shared_2),
                })

                T.sync_threads()  # after barrier init

                # ---- Per-WG global iter counters (Approach A) ----
                gi_kp = T.alloc_var("int32", init=0)
                gi_vp = T.alloc_var("int32", init=0)
                gi_kc1 = T.alloc_var("int32", init=0)
                gi_vc1 = T.alloc_var("int32", init=0)
                gi_kc2 = T.alloc_var("int32", init=0)
                gi_vc2 = T.alloc_var("int32", init=0)
                gi_q1 = T.alloc_var("int32", init=0)
                gi_q2 = T.alloc_var("int32", init=0)

                tx = T.get_thread_binding()

                # ===== WG0 (producer, tx < 128) =====
                if tx < 128:
                    T.dec_max_nreg(24)
                    for tile_b, tile_h, pair_idx in T.Persistent(
                        [batch, heads, half_M_blocks],
                        wave_size=effective_num_sms,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_h // groups
                        # Inner Python loop unrolls into 2 sub-tile bodies.
                        # sub_idx=0: short side (tile_m = pair_idx)
                        # sub_idx=1: long  side (tile_m = M-1-pair_idx)
                        for sub_idx in range(2):
                            # tile_m as a single TIR expression (no Python
                            # if-frame). sub_idx is Python int 0 or 1:
                            #   sub_idx=0 → tile_m = pair_idx
                            #   sub_idx=1 → tile_m = M_blocks - 1 - pair_idx
                            tile_m = (
                                pair_idx
                                + sub_idx * (M_blocks - 1 - 2 * pair_idx))
                            loop_range = T.ceildiv(
                                (tile_m + 1) * block_m, block_n)

                            for n_idx in T.Pipelined(
                                    loop_range, num_stages=0):
                                T.barrier_wait(k_empty, (gi_kp + 1) % 2)
                                if gi_kp % 2 == 0:
                                    T.tma_copy(
                                        k[tile_b,
                                          n_idx * block_n:
                                          (n_idx + 1) * block_n,
                                          head_kv, :],
                                        k_smem_0, barrier=k_full)
                                else:
                                    T.tma_copy(
                                        k[tile_b,
                                          n_idx * block_n:
                                          (n_idx + 1) * block_n,
                                          head_kv, :],
                                        k_smem_1, barrier=k_full)
                                T.barrier_arrive(k_full)
                                if n_idx > 0:
                                    T.barrier_wait(
                                        v_empty, (gi_vp + 1) % 2)
                                    if gi_vp % 2 == 0:
                                        T.tma_copy(
                                            v[tile_b,
                                              (n_idx - 1) * block_n:
                                              n_idx * block_n,
                                              head_kv, :],
                                            v_smem_0, barrier=v_full)
                                    else:
                                        T.tma_copy(
                                            v[tile_b,
                                              (n_idx - 1) * block_n:
                                              n_idx * block_n,
                                              head_kv, :],
                                            v_smem_1, barrier=v_full)
                                    T.barrier_arrive(v_full)
                                    gi_vp = gi_vp + 1
                                gi_kp = gi_kp + 1
                            # Producer epilogue: tail load V[loop_range-1]
                            T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                            if gi_vp % 2 == 0:
                                T.tma_copy(
                                    v[tile_b,
                                      (loop_range - 1) * block_n:
                                      loop_range * block_n,
                                      head_kv, :],
                                    v_smem_0, barrier=v_full)
                            else:
                                T.tma_copy(
                                    v[tile_b,
                                      (loop_range - 1) * block_n:
                                      loop_range * block_n,
                                      head_kv, :],
                                    v_smem_1, barrier=v_full)
                            T.barrier_arrive(v_full)
                            gi_vp = gi_vp + 1

                # ===== WG1 (consumer 1, 128 <= tx < 256) =====
                elif tx < 256:
                    T.inc_max_nreg(240)
                    # Bootstrap: ONCE per CTA, OUTSIDE persistent loop
                    T.barrier_arrive(wg_sched_21)  # bootstrap WG1→WG2 sched
                    for tile_b, tile_h, pair_idx in T.Persistent(
                        [batch, heads, half_M_blocks],
                        wave_size=effective_num_sms,
                        index=bx,
                        group_size=8,
                    ):
                        for sub_idx in range(2):
                            # tile_m as a single TIR expression (no Python
                            # if-frame). sub_idx is Python int 0 or 1:
                            #   sub_idx=0 → tile_m = pair_idx
                            #   sub_idx=1 → tile_m = M_blocks - 1 - pair_idx
                            tile_m = (
                                pair_idx
                                + sub_idx * (M_blocks - 1 - 2 * pair_idx))
                            row_base = tile_m * block_m
                            loop_range = T.ceildiv(
                                (tile_m + 1) * block_m, block_n)

                            # Per-sub-tile Q load (per-WG ownership)
                            T.tma_copy(
                                q[tile_b,
                                  row_base:row_base + half_m,
                                  tile_h, :],
                                q_shared_1, barrier=q_full_1)
                            T.barrier_arrive(q_full_1)
                            T.barrier_wait(q_full_1, gi_q1 % 2)
                            gi_q1 = gi_q1 + 1

                            # Per-sub-tile state reset
                            T.clear(acc_o_1)
                            T.clear(ls_1)
                            T.fill(sm_1, -T.infinity(accum_dtype))

                            for n_idx in T.Pipelined(
                                    loop_range, num_stages=0):
                                T.barrier_wait(k_full, gi_kc1 % 2)
                                T.barrier_wait(wg_sched_21, gi_kc1 % 2)
                                # ALWAYS clear, no in-loop mask. Mask is
                                # applied AFTER wgmma to preserve
                                # IntraWGOverlap (see postmask root cause).
                                T.clear(acc_s_1)
                                if n_idx == 0:
                                    if gi_kc1 % 2 == 0:
                                        T.wgmma_gemm(
                                            q_shared_1, k_smem_0, acc_s_1,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            q_shared_1, k_smem_1, acc_s_1,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    T.barrier_arrive(wg_sched_12)
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(
                                        acc_s_1, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    if n_idx == loop_range - 1:
                                        for i, j in T.Parallel(
                                                half_m, block_n):
                                            acc_s_1[i, j] = T.if_then_else(
                                                row_base + i
                                                >= n_idx * block_n + j,
                                                acc_s_1[i, j],
                                                -T.infinity(accum_dtype))
                                    softmax_1(acc_s_1, sm_1, smp_1,
                                              ss_1, ssum_1, ls_1)
                                    T.copy(acc_s_1, acc_s_cast_1)
                                else:
                                    if gi_kc1 % 2 == 0:
                                        T.wgmma_gemm(
                                            q_shared_1, k_smem_0, acc_s_1,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            q_shared_1, k_smem_1, acc_s_1,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    rescale_1(acc_o_1, ss_1)
                                    T.barrier_wait(v_full, gi_vc1 % 2)
                                    if gi_vc1 % 2 == 0:
                                        T.wgmma_gemm(
                                            acc_s_cast_1, v_smem_0,
                                            acc_o_1,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            acc_s_cast_1, v_smem_1,
                                            acc_o_1,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    T.barrier_arrive(wg_sched_12)
                                    T.wait_wgmma(1)
                                    T.warpgroup_fence_operand(
                                        acc_s_1, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    if n_idx == loop_range - 1:
                                        for i, j in T.Parallel(
                                                half_m, block_n):
                                            acc_s_1[i, j] = T.if_then_else(
                                                row_base + i
                                                >= n_idx * block_n + j,
                                                acc_s_1[i, j],
                                                -T.infinity(accum_dtype))
                                    softmax_1(acc_s_1, sm_1, smp_1,
                                              ss_1, ssum_1, ls_1)
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(
                                        acc_o_1, num_regs=64)
                                    T.barrier_arrive(v_empty)
                                    T.copy(acc_s_1, acc_s_cast_1)
                                    gi_vc1 = gi_vc1 + 1
                                gi_kc1 = gi_kc1 + 1
                            # Consumer 1 epilogue: rescale + last PV
                            rescale_1(acc_o_1, ss_1)
                            T.barrier_wait(v_full, gi_vc1 % 2)
                            if gi_vc1 % 2 == 0:
                                T.wgmma_gemm(
                                    acc_s_cast_1, v_smem_0, acc_o_1,
                                    policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(
                                    acc_s_cast_1, v_smem_1, acc_o_1,
                                    policy=T.GemmWarpPolicy.FullRow)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(
                                acc_o_1, num_regs=64)
                            T.barrier_arrive(v_empty)
                            gi_vc1 = gi_vc1 + 1
                            # Output write for half 1
                            for i, j in T.Parallel(half_m, dim):
                                acc_o_1[i, j] /= ls_1[i]
                            T.copy(acc_o_1, q_shared_1)
                            T.fence_proxy_async()
                            T.sync_threads(
                                barrier_id=3, arrive_count=128)
                            T.copy(q_shared_1,
                                   output[tile_b,
                                          row_base:row_base + half_m,
                                          tile_h, :])
                            for i in T.Parallel(half_m):
                                ls_1[i] = (T.log2(ls_1[i])
                                           + sm_1[i] * scale)
                            T.copy(ls_1,
                                   lse[tile_b, tile_h,
                                       row_base:row_base + half_m])

                # ===== WG2 (consumer 2, tx >= 256) =====
                else:
                    T.inc_max_nreg(240)
                    for tile_b, tile_h, pair_idx in T.Persistent(
                        [batch, heads, half_M_blocks],
                        wave_size=effective_num_sms,
                        index=bx,
                        group_size=8,
                    ):
                        for sub_idx in range(2):
                            # tile_m as a single TIR expression (no Python
                            # if-frame). sub_idx is Python int 0 or 1:
                            #   sub_idx=0 → tile_m = pair_idx
                            #   sub_idx=1 → tile_m = M_blocks - 1 - pair_idx
                            tile_m = (
                                pair_idx
                                + sub_idx * (M_blocks - 1 - 2 * pair_idx))
                            row_base = tile_m * block_m
                            loop_range = T.ceildiv(
                                (tile_m + 1) * block_m, block_n)

                            T.tma_copy(
                                q[tile_b,
                                  row_base + half_m:
                                  row_base + block_m,
                                  tile_h, :],
                                q_shared_2, barrier=q_full_2)
                            T.barrier_arrive(q_full_2)
                            T.barrier_wait(q_full_2, gi_q2 % 2)
                            gi_q2 = gi_q2 + 1

                            T.clear(acc_o_2)
                            T.clear(ls_2)
                            T.fill(sm_2, -T.infinity(accum_dtype))

                            for n_idx in T.Pipelined(
                                    loop_range, num_stages=0):
                                T.barrier_wait(k_full, gi_kc2 % 2)
                                T.barrier_wait(wg_sched_12, gi_kc2 % 2)
                                T.clear(acc_s_2)
                                if n_idx == 0:
                                    if gi_kc2 % 2 == 0:
                                        T.wgmma_gemm(
                                            q_shared_2, k_smem_0, acc_s_2,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            q_shared_2, k_smem_1, acc_s_2,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    T.barrier_arrive(wg_sched_21)
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(
                                        acc_s_2, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    if n_idx == loop_range - 1:
                                        for i, j in T.Parallel(
                                                half_m, block_n):
                                            acc_s_2[i, j] = T.if_then_else(
                                                row_base + half_m + i
                                                >= n_idx * block_n + j,
                                                acc_s_2[i, j],
                                                -T.infinity(accum_dtype))
                                    softmax_2(acc_s_2, sm_2, smp_2,
                                              ss_2, ssum_2, ls_2)
                                    T.copy(acc_s_2, acc_s_cast_2)
                                else:
                                    if gi_kc2 % 2 == 0:
                                        T.wgmma_gemm(
                                            q_shared_2, k_smem_0, acc_s_2,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            q_shared_2, k_smem_1, acc_s_2,
                                            transpose_B=True,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    rescale_2(acc_o_2, ss_2)
                                    T.barrier_wait(v_full, gi_vc2 % 2)
                                    if gi_vc2 % 2 == 0:
                                        T.wgmma_gemm(
                                            acc_s_cast_2, v_smem_0,
                                            acc_o_2,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(
                                            acc_s_cast_2, v_smem_1,
                                            acc_o_2,
                                            policy=T.GemmWarpPolicy.FullRow)
                                    T.barrier_arrive(wg_sched_21)
                                    T.wait_wgmma(1)
                                    T.warpgroup_fence_operand(
                                        acc_s_2, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    if n_idx == loop_range - 1:
                                        for i, j in T.Parallel(
                                                half_m, block_n):
                                            acc_s_2[i, j] = T.if_then_else(
                                                row_base + half_m + i
                                                >= n_idx * block_n + j,
                                                acc_s_2[i, j],
                                                -T.infinity(accum_dtype))
                                    softmax_2(acc_s_2, sm_2, smp_2,
                                              ss_2, ssum_2, ls_2)
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(
                                        acc_o_2, num_regs=64)
                                    T.barrier_arrive(v_empty)
                                    T.copy(acc_s_2, acc_s_cast_2)
                                    gi_vc2 = gi_vc2 + 1
                                gi_kc2 = gi_kc2 + 1
                            # Consumer 2 epilogue
                            rescale_2(acc_o_2, ss_2)
                            T.barrier_wait(v_full, gi_vc2 % 2)
                            if gi_vc2 % 2 == 0:
                                T.wgmma_gemm(
                                    acc_s_cast_2, v_smem_0, acc_o_2,
                                    policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(
                                    acc_s_cast_2, v_smem_1, acc_o_2,
                                    policy=T.GemmWarpPolicy.FullRow)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(
                                acc_o_2, num_regs=64)
                            T.barrier_arrive(v_empty)
                            gi_vc2 = gi_vc2 + 1
                            # Output write for half 2
                            for i, j in T.Parallel(half_m, dim):
                                acc_o_2[i, j] /= ls_2[i]
                            T.copy(acc_o_2, q_shared_2)
                            T.fence_proxy_async()
                            T.sync_threads(
                                barrier_id=4, arrive_count=128)
                            T.copy(q_shared_2,
                                   output[tile_b,
                                          row_base + half_m:
                                          row_base + block_m,
                                          tile_h, :])
                            for i in T.Parallel(half_m):
                                ls_2[i] = (T.log2(ls_2[i])
                                           + sm_2[i] * scale)
                            T.copy(ls_2,
                                   lse[tile_b, tile_h,
                                       row_base + half_m:
                                       row_base + block_m])

        return _gqa_fwd_ws_persistent_main

    return _gqa_fwd_ws_persistent_func


@torch.library.custom_op("top::gqa_fwd_ws_persistent_wrapped_kernel", mutates_args=())
def _gqa_fwd_ws_persistent_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_ws_persistent_kernel(batch, heads, heads_kv, seq_len, dim, is_causal,
                                          dtype)(block_m, block_n)(q, k, v)


@_gqa_fwd_ws_persistent_wrapped_kernel.register_fake
def _(batch: int, heads: int, heads_kv: int,
      seq_len: int, dim: int, is_causal: bool,
      dtype: str, block_m: int, block_n: int,
      *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class GqaFwdWsPersistentKernel(Kernel):
    """Persistent CTA + causal tile pairing + post-wgmma mask GQA forward.

    Causal-only Hopper specialization. ~84% FA3 on H200 reference shape
    (B=4 S=4096 H=64 Hkv=8 D=128 fp16, block_m=128 block_n=128).

    Hardware-locked: ``num_sms`` is detected from the current device at
    build time. Use ``GqaFwdWsKernel`` for hardware-portable code.

    Constraints (validated at construction):
      - is_causal=True
      - dim==128
      - heads % heads_kv == 0
      - (seq_len // block_m) % 2 == 0  (default block_m=128 → seq_len % 256 == 0)
    """
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if dim != 128:
            raise ValueError(
                f"GqaFwdWsPersistentKernel currently requires dim==128, got dim={dim}")
        if not is_causal:
            raise ValueError(
                "GqaFwdWsPersistentKernel only supports is_causal=True. "
                "For non-causal use GqaFwdWsKernel.")
        # The producer's epilogue tail V load reads
        # ``v[..., (loop_range-1)*block_n : loop_range*block_n, ...]``
        # which can read past ``seq_len`` if ``seq_len < block_n``.
        # Default block_n is 128; require seq_len >= 128.
        if seq_len < 128:
            raise ValueError(
                f"GqaFwdWsPersistentKernel requires seq_len >= 128 to "
                f"avoid out-of-bounds V loads in the producer epilogue.  "
                f"Got seq_len={seq_len}.")
        # Default block_m is 128; validate seq_len divisibility for the
        # default config using the SAME ceil-div formula as the JIT-time
        # check inside _gqa_fwd_ws_persistent_func.  If user overrides
        # block_m via tune, the JIT-time check re-validates with the
        # actual block_m.
        default_block_m = 128
        m_blocks_default = (seq_len + default_block_m - 1) // default_block_m
        if m_blocks_default == 0 or m_blocks_default % 2 != 0:
            raise ValueError(
                f"GqaFwdWsPersistentKernel requires ceil(seq_len / block_m) "
                f"to be a positive even integer for tile pairing.  Got "
                f"seq_len={seq_len}, default block_m={default_block_m}, "
                f"M_blocks={m_blocks_default}.")
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _gqa_fwd_ws_persistent_kernel(self.batch, self.heads, self.heads_kv,
                                                     self.seq_len, self.dim, self.is_causal,
                                                     self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_m": 128, "block_n": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        # Only block_m=128 (with M_blocks even shape constraint).  block_m=64
        # would double M_blocks but most LLM seq_lens stay even at /128.
        block_m = [128]
        block_n = [64, 128]
        _configs = list(itertools.product(block_m, block_n))
        return [{
            'block_m': c[0],
            'block_n': c[1],
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gqa_fwd_ws_persistent_wrapped_kernel(self.batch, self.heads, self.heads_kv,
                                                      self.seq_len, self.dim, self.is_causal,
                                                      self.dtype_str, self.config["block_m"],
                                                      self.config["block_n"],
                                                      q, k, v)
