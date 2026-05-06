import functools
import itertools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.online_softmax import (
    LOG2E,
    make_log2e_scale,
    make_online_softmax,
    make_online_softmax_with_mask_guard,
    make_rescale,
)

__all__ = [
    'GQAFwdKernel',
    'GQAFwdWgmmaPipelinedKernel',
    'GQAPrefillFwdKernel',
    'GQAPrefillPagedWithKVCacheFwdKernel',
    'GQAPrefillWithKVCacheFwdKernel',
    'MHAFwdKernel',
    'MHAFwdWgmmaPipelinedKernel'
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


class MHAFwdKernel(Kernel):
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


class MHAFwdWgmmaPipelinedKernel(Kernel):
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


class GQAFwdKernel(Kernel):
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


class GQAFwdWgmmaPipelinedKernel(Kernel):
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


# GQA prefill, allowing seq_len_q != seq_len_kv.


@functools.lru_cache(maxsize=32)
def _gqa_prefill_fwd_kernel(batch: int,
                            heads: int,
                            heads_kv: int,
                            seq_len_q: int,
                            seq_len_kv: int,
                            dim: int,
                            is_causal: bool,
                            sm_scale: Optional[float] = None,
                            dtype: str = 'float16') -> Callable:
    scale = (dim**-0.5 if sm_scale is None else sm_scale) * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if is_causal and seq_len_q > seq_len_kv:
        raise ValueError("causal prefill requires seq_len_q <= seq_len_kv")
    groups = heads // heads_kv
    causal_offset = seq_len_kv - seq_len_q
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_prefill_fwd_func(block_m: int, block_n: int, num_stages: int,
                              threads: int) -> Callable:
        q_shape = (batch, seq_len_q, heads, dim)
        kv_shape = (batch, seq_len_kv, heads_kv, dim)
        o_shape = (batch, seq_len_q, heads, dim)
        online_softmax = make_online_softmax_with_mask_guard(
            scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def _gqa_prefill_fwd_main(
                q: T.Tensor(q_shape, dtype),  # type: ignore
                k: T.Tensor(kv_shape, dtype),  # type: ignore
                v: T.Tensor(kv_shape, dtype),  # type: ignore
                output: T.Tensor(o_shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len_q], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(seq_len_q, block_m), heads, batch, threads=threads) as (bx, by, bz):
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

                if (bx + 1) * block_m <= seq_len_q:
                    T.copy(
                        q[bz, bx * block_m:(bx + 1) * block_m, by, :],
                        q_shared,
                        disable_tma=True)
                else:
                    for i, d in T.Parallel(block_m, dim):
                        q_pos = bx * block_m + i
                        if q_pos < seq_len_q:
                            q_shared[i, d] = q[bz, q_pos, by, d]
                        else:
                            q_shared[i, d] = T.cast(0, dtype)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv((bx + 1) * block_m + causal_offset, block_n)
                    if is_causal else T.ceildiv(seq_len_kv, block_n))

                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    if (k_idx + 1) * block_n <= seq_len_kv:
                        T.copy(
                            k[bz, k_idx * block_n:(k_idx + 1) * block_n, by // groups, :],
                            k_shared,
                            disable_tma=True)
                        T.copy(
                            v[bz, k_idx * block_n:(k_idx + 1) * block_n, by // groups, :],
                            v_shared,
                            disable_tma=True)
                    else:
                        for j, d in T.Parallel(block_n, dim):
                            kv_pos = k_idx * block_n + j
                            if kv_pos < seq_len_kv:
                                k_shared[j, d] = k[bz, kv_pos, by // groups, d]
                                v_shared[j, d] = v[bz, kv_pos, by // groups, d]
                            else:
                                k_shared[j, d] = T.cast(0, dtype)
                                v_shared[j, d] = T.cast(0, dtype)
                    for i, j in T.Parallel(block_m, block_n):
                        q_pos = bx * block_m + i
                        kv_pos = k_idx * block_n + j
                        if is_causal:
                            valid = (
                                (q_pos < seq_len_q)
                                & (kv_pos < seq_len_kv)
                                & (q_pos + causal_offset >= kv_pos)
                            )
                            acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                        else:
                            valid = (q_pos < seq_len_q) & (kv_pos < seq_len_kv)
                            acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum,
                                   logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                if (bx + 1) * block_m <= seq_len_q:
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] /= logsum[i]
                    T.copy(
                        acc_o,
                        output[bz, bx * block_m:(bx + 1) * block_m, by, :],
                        disable_tma=True)
                    for i in T.Parallel(block_m):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                    T.copy(logsum, lse[bz, by, bx * block_m:(bx + 1) * block_m],
                           disable_tma=True)
                else:
                    for i, j in T.Parallel(block_m, dim):
                        q_pos = bx * block_m + i
                        if q_pos < seq_len_q:
                            output[bz, q_pos, by, j] = acc_o[i, j] / logsum[i]
                    for i in T.Parallel(block_m):
                        q_pos = bx * block_m + i
                        if q_pos < seq_len_q:
                            lse[bz, by, q_pos] = T.log2(logsum[i]) + scores_max[i] * scale

        return _gqa_prefill_fwd_main

    return _gqa_prefill_fwd_func


@torch.library.custom_op("top::gqa_prefill_fwd_wrapped_kernel", mutates_args=())
def _gqa_prefill_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_prefill_fwd_kernel(batch, heads, heads_kv, seq_len_q, seq_len_kv, dim,
                                   is_causal, sm_scale, dtype)(
                                       block_m, block_n, num_stages, threads)(q, k, v)


@_gqa_prefill_fwd_wrapped_kernel.register_fake
def _(batch: int, heads: int,
      heads_kv: int, seq_len_q: int, seq_len_kv: int, dim: int, is_causal: bool,
      sm_scale: float, dtype: str, block_m: int, block_n: int, num_stages: int,
      threads: int, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len_q])
    return fake_o, fake_lse


class GQAPrefillFwdKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len_q: int,
                 seq_len_kv: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 sm_scale: Optional[float] = None,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if is_causal and seq_len_q > seq_len_kv:
            raise ValueError("causal prefill requires seq_len_q <= seq_len_kv")
        self.heads_kv = heads_kv
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale

        self.kernel = _gqa_prefill_fwd_kernel(self.batch, self.heads, self.heads_kv,
                                              self.seq_len_q, self.seq_len_kv, self.dim,
                                              self.is_causal, self.sm_scale, self.dtype_str)

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
        return _gqa_prefill_fwd_wrapped_kernel(
            self.batch, self.heads, self.heads_kv, self.seq_len_q, self.seq_len_kv, self.dim,
            self.is_causal, self.sm_scale, self.dtype_str, self.config["block_m"],
            self.config["block_n"], self.config["num_stages"], self.config["threads"], q, k, v)


# GQA prefill with contiguous KV cache. The attention path reads old KV from
# cache and current chunk KV directly from k_new/v_new, then appends k_new/v_new.


@functools.lru_cache(maxsize=32)
def _gqa_prefill_with_kv_cache_fwd_kernel(batch: int,
                                          heads: int,
                                          heads_kv: int,
                                          seq_len_new: int,
                                          seqlen_kv: int,
                                          dim: int,
                                          is_causal: bool,
                                          sm_scale: Optional[float] = None,
                                          dtype: str = 'float16') -> Callable:
    scale = (dim**-0.5 if sm_scale is None else sm_scale) * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[6, 7],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_prefill_with_kv_cache_fwd_func(block_m: int, block_n: int, num_stages: int,
                                            threads: int) -> Callable:
        q_shape = (batch, seq_len_new, heads, dim)
        kv_new_shape = (batch, seq_len_new, heads_kv, dim)
        kv_cache_shape = (batch, seqlen_kv, heads_kv, dim)
        o_shape = (batch, seq_len_new, heads, dim)
        online_softmax = make_online_softmax_with_mask_guard(scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def _gqa_prefill_with_kv_cache_fwd_main(
                q: T.Tensor(q_shape, dtype),  # type: ignore
                k_new: T.Tensor(kv_new_shape, dtype),  # type: ignore
                v_new: T.Tensor(kv_new_shape, dtype),  # type: ignore
                k_cache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
                v_cache: T.Tensor(kv_cache_shape, dtype),  # type: ignore
                cache_seqlens: T.Tensor([batch], T.int32),  # type: ignore
                output: T.Tensor(o_shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len_new], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(seq_len_new, block_m), heads, batch, threads=threads) as (bx, by, bz):
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

                old_len = cache_seqlens[bz]
                total_len = old_len + seq_len_new
                cur_kv_head = by // groups

                if (bx + 1) * block_m <= seq_len_new:
                    T.copy(
                        q[bz, bx * block_m:(bx + 1) * block_m, by, :],
                        q_shared,
                        disable_tma=True)
                else:
                    for i, d in T.Parallel(block_m, dim):
                        new_pos = bx * block_m + i
                        if new_pos < seq_len_new:
                            q_shared[i, d] = q[bz, new_pos, by, d]
                        else:
                            q_shared[i, d] = T.cast(0, dtype)

                # The first heads_kv CTAs append one KV head each; in this branch
                # by is the KV-head index, not the query-head-to-KV mapping.
                if by < heads_kv:
                    if (bx + 1) * block_m <= seq_len_new and old_len + (
                            bx + 1) * block_m <= seqlen_kv:
                        T.copy(
                            k_new[bz, bx * block_m:(bx + 1) * block_m, by, :],
                            k_cache[bz, old_len + bx * block_m:old_len + (bx + 1) * block_m,
                                    by, :],
                            disable_tma=True)
                        T.copy(
                            v_new[bz, bx * block_m:(bx + 1) * block_m, by, :],
                            v_cache[bz, old_len + bx * block_m:old_len + (bx + 1) * block_m,
                                    by, :],
                            disable_tma=True)
                    else:
                        for i, d in T.Parallel(block_m, dim):
                            new_pos = bx * block_m + i
                            cache_pos = old_len + new_pos
                            if new_pos < seq_len_new and cache_pos < seqlen_kv:
                                k_cache[bz, cache_pos, by, d] = k_new[bz, new_pos, by, d]
                                v_cache[bz, cache_pos, by, d] = v_new[bz, new_pos, by, d]

                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(old_len + (bx + 1) * block_m, block_n)
                    if is_causal else T.ceildiv(total_len, block_n))

                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    tile_start = k_idx * block_n
                    tile_end = (k_idx + 1) * block_n
                    if tile_end <= old_len:
                        T.copy(
                            k_cache[bz, tile_start:tile_end, cur_kv_head, :],
                            k_shared,
                            disable_tma=True)
                        T.copy(
                            v_cache[bz, tile_start:tile_end, cur_kv_head, :],
                            v_shared,
                            disable_tma=True)
                    elif tile_start >= old_len and tile_end <= total_len:
                        new_start = tile_start - old_len
                        T.copy(
                            k_new[bz, new_start:new_start + block_n, cur_kv_head, :],
                            k_shared,
                            disable_tma=True)
                        T.copy(
                            v_new[bz, new_start:new_start + block_n, cur_kv_head, :],
                            v_shared,
                            disable_tma=True)
                    else:
                        for j, d in T.Parallel(block_n, dim):
                            kv_pos = tile_start + j
                            new_pos = kv_pos - old_len
                            if kv_pos < old_len:
                                k_shared[j, d] = k_cache[bz, kv_pos, cur_kv_head, d]
                                v_shared[j, d] = v_cache[bz, kv_pos, cur_kv_head, d]
                            elif kv_pos < total_len:
                                k_shared[j, d] = k_new[bz, new_pos, cur_kv_head, d]
                                v_shared[j, d] = v_new[bz, new_pos, cur_kv_head, d]
                            else:
                                k_shared[j, d] = T.cast(0, dtype)
                                v_shared[j, d] = T.cast(0, dtype)
                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            kv_pos = k_idx * block_n + j
                            q_abs_pos = old_len + bx * block_m + i
                            valid = (bx * block_m + i < seq_len_new) & (kv_pos < total_len) & (
                                kv_pos <= q_abs_pos)
                            acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            kv_pos = k_idx * block_n + j
                            valid = (bx * block_m + i < seq_len_new) & (kv_pos < total_len)
                            acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum,
                                   logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                if (bx + 1) * block_m <= seq_len_new:
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] /= logsum[i]
                    T.copy(
                        acc_o,
                        output[bz, bx * block_m:(bx + 1) * block_m, by, :],
                        disable_tma=True)
                    for i in T.Parallel(block_m):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                    T.copy(logsum, lse[bz, by, bx * block_m:(bx + 1) * block_m],
                           disable_tma=True)
                else:
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] = T.if_then_else(
                            bx * block_m + i < seq_len_new,
                            acc_o[i, j] / logsum[i],
                            T.cast(0, accum_dtype),
                        )
                    for i, j in T.Parallel(block_m, dim):
                        new_pos = bx * block_m + i
                        if new_pos < seq_len_new:
                            output[bz, new_pos, by, j] = acc_o[i, j]
                    for i in T.Parallel(block_m):
                        logsum[i] = T.if_then_else(
                            bx * block_m + i < seq_len_new,
                            T.log2(logsum[i]) + scores_max[i] * scale,
                            T.cast(0, accum_dtype),
                        )
                    for i in T.Parallel(block_m):
                        new_pos = bx * block_m + i
                        if new_pos < seq_len_new:
                            lse[bz, by, new_pos] = logsum[i]

        return _gqa_prefill_with_kv_cache_fwd_main

    return _gqa_prefill_with_kv_cache_fwd_func


@torch.library.custom_op(
    "top::gqa_prefill_with_kv_cache_fwd_wrapped_kernel",
    mutates_args=("k_cache", "v_cache"),
)
def _gqa_prefill_with_kv_cache_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_new: int,
    seqlen_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_prefill_with_kv_cache_fwd_kernel(
        batch, heads, heads_kv, seq_len_new, seqlen_kv, dim, is_causal, sm_scale,
        dtype)(block_m, block_n, num_stages, threads)(q, k_new, v_new, k_cache, v_cache,
                                                      cache_seqlens)


@_gqa_prefill_with_kv_cache_fwd_wrapped_kernel.register_fake
def _(batch: int, heads: int,
      heads_kv: int, seq_len_new: int, seqlen_kv: int, dim: int, is_causal: bool,
      sm_scale: float, dtype: str, block_m: int, block_n: int, num_stages: int,
      threads: int, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len_new])
    return fake_o, fake_lse


class GQAPrefillWithKVCacheFwdKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len_new: int,
                 seqlen_kv: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 sm_scale: Optional[float] = None,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.heads_kv = heads_kv
        self.seq_len_new = seq_len_new
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale

        self.kernel = _gqa_prefill_with_kv_cache_fwd_kernel(
            self.batch, self.heads, self.heads_kv, self.seq_len_new, self.seqlen_kv, self.dim,
            self.is_causal, self.sm_scale, self.dtype_str)

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

    def forward(self, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
                k_cache: torch.Tensor, v_cache: torch.Tensor,
                cache_seqlens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gqa_prefill_with_kv_cache_fwd_wrapped_kernel(
            self.batch, self.heads, self.heads_kv, self.seq_len_new, self.seqlen_kv, self.dim,
            self.is_causal, self.sm_scale, self.dtype_str, self.config["block_m"],
            self.config["block_n"], self.config["num_stages"], self.config["threads"], q, k_new,
            v_new, k_cache, v_cache, cache_seqlens)


# GQA packed prefill with paged KV cache. Current chunk is packed THD and
# old KV is addressed by block_table. The kernel reads current KV directly from
# k_new/v_new and appends it into k_pages/v_pages in-place.


@functools.lru_cache(maxsize=32)
def _gqa_prefill_paged_with_kv_cache_fwd_kernel(batch: int,
                                                heads: int,
                                                heads_kv: int,
                                                total_q: int,
                                                physical_tokens: int,
                                                max_pages_per_req: int,
                                                page_size: int,
                                                dim: int,
                                                is_causal: bool,
                                                sm_scale: Optional[float] = None,
                                                dtype: str = 'float16') -> Callable:
    scale = (dim**-0.5 if sm_scale is None else sm_scale) * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if page_size <= 0 or page_size & (page_size - 1) != 0:
        raise ValueError("page_size must be a positive power of two")
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[8, 9],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_prefill_paged_with_kv_cache_fwd_func(
            block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:

        q_shape = (total_q, heads, dim)
        kv_new_shape = (total_q, heads_kv, dim)
        kv_pages_shape = (physical_tokens, heads_kv, dim)
        block_table_shape = (batch, max_pages_per_req)
        o_shape = (total_q, heads, dim)
        online_softmax = make_online_softmax_with_mask_guard(
            scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)
        page_size_log2 = page_size.bit_length() - 1

        @T.prim_func
        def _gqa_prefill_paged_with_kv_cache_fwd_main(
                q: T.Tensor(q_shape, dtype),  # type: ignore
                k_new: T.Tensor(kv_new_shape, dtype),  # type: ignore
                v_new: T.Tensor(kv_new_shape, dtype),  # type: ignore
                k_pages: T.Tensor(kv_pages_shape, dtype),  # type: ignore
                v_pages: T.Tensor(kv_pages_shape, dtype),  # type: ignore
                cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
                cache_seqlens: T.Tensor([batch], T.int32),  # type: ignore
                block_table: T.Tensor(block_table_shape, T.int32),  # type: ignore
                output: T.Tensor(o_shape, dtype),  # type: ignore
                lse: T.Tensor([heads, total_q], accum_dtype),  # type: ignore
                max_seqlen_q: T.int32,  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(max_seqlen_q, block_m), heads, batch, threads=threads) as (
                        bx, by, bz):
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

                q_start = cu_seqlens_q[bz]
                q_len = cu_seqlens_q[bz + 1] - q_start
                old_len = cache_seqlens[bz]
                total_len = old_len + q_len
                cur_kv_head = by // groups

                if bx * block_m + block_m <= q_len:
                    T.copy(
                        q[q_start + bx * block_m:q_start + (bx + 1) * block_m, by, :],
                        q_shared,
                        disable_tma=True)
                else:
                    for i, d in T.Parallel(block_m, dim):
                        new_pos = bx * block_m + i
                        if new_pos < q_len:
                            q_shared[i, d] = q[q_start + new_pos, by, d]
                        else:
                            q_shared[i, d] = T.cast(0, dtype)

                if by < heads_kv:
                    append_start = old_len + bx * block_m
                    append_end = append_start + block_m
                    if bx * block_m + block_m <= q_len:
                        if append_start >> T.int32(page_size_log2) == (
                                append_end - 1) >> T.int32(page_size_log2):
                            page_idx = append_start >> T.int32(page_size_log2)
                            page_offset = append_start - page_idx * page_size
                            physical_start = block_table[bz, page_idx] * page_size + page_offset
                            for i, d in T.Parallel(block_m, dim):
                                k_pages[physical_start + i, by, d] = k_new[
                                    q_start + bx * block_m + i, by, d]
                                v_pages[physical_start + i, by, d] = v_new[
                                    q_start + bx * block_m + i, by, d]
                        else:
                            for i, d in T.Parallel(block_m, dim):
                                new_pos = bx * block_m + i
                                logical_pos = old_len + new_pos
                                page_idx = logical_pos >> T.int32(page_size_log2)
                                page_offset = logical_pos - page_idx * page_size
                                physical_pos = block_table[bz, page_idx] * page_size + page_offset
                                k_pages[physical_pos, by, d] = k_new[q_start + new_pos, by, d]
                                v_pages[physical_pos, by, d] = v_new[q_start + new_pos, by, d]
                    else:
                        for i, d in T.Parallel(block_m, dim):
                            new_pos = bx * block_m + i
                            safe_new_pos = T.if_then_else(new_pos < q_len, new_pos, 0)
                            logical_pos = old_len + safe_new_pos
                            page_idx = logical_pos >> T.int32(page_size_log2)
                            page_offset = logical_pos - page_idx * page_size
                            if new_pos < q_len:
                                physical_pos = block_table[bz, page_idx] * page_size + page_offset
                                k_pages[physical_pos, by, d] = k_new[q_start + new_pos, by, d]
                                v_pages[physical_pos, by, d] = v_new[q_start + new_pos, by, d]

                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(old_len + (bx + 1) * block_m, block_n)
                    if is_causal else T.ceildiv(total_len, block_n))

                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    tile_start = k_idx * block_n
                    tile_end = tile_start + block_n
                    if tile_end <= old_len:
                        if page_size % block_n == 0:
                            page_idx = tile_start >> T.int32(page_size_log2)
                            page_offset = tile_start - page_idx * page_size
                            physical_start = block_table[bz, page_idx] * page_size + page_offset
                            for j, d in T.Parallel(block_n, dim):
                                k_shared[j, d] = k_pages[physical_start + j, cur_kv_head, d]
                                v_shared[j, d] = v_pages[physical_start + j, cur_kv_head, d]
                        elif block_n % page_size == 0:
                            tile_page_start = tile_start >> T.int32(page_size_log2)
                            for p in range(block_n // page_size):
                                segment_physical_start = block_table[
                                    bz, tile_page_start + p] * page_size
                                for off, d in T.Parallel(page_size, dim):
                                    shared_row = p * page_size + off
                                    k_shared[shared_row, d] = k_pages[
                                        segment_physical_start + off, cur_kv_head, d]
                                    v_shared[shared_row, d] = v_pages[
                                        segment_physical_start + off, cur_kv_head, d]
                        else:
                            for j, d in T.Parallel(block_n, dim):
                                kv_pos = tile_start + j
                                page_idx = kv_pos >> T.int32(page_size_log2)
                                page_offset = kv_pos - page_idx * page_size
                                physical_pos = block_table[bz, page_idx] * page_size + page_offset
                                k_shared[j, d] = k_pages[physical_pos, cur_kv_head, d]
                                v_shared[j, d] = v_pages[physical_pos, cur_kv_head, d]
                    elif tile_start >= old_len and tile_end <= total_len:
                        new_start = tile_start - old_len
                        for j, d in T.Parallel(block_n, dim):
                            k_shared[j, d] = k_new[q_start + new_start + j, cur_kv_head, d]
                            v_shared[j, d] = v_new[q_start + new_start + j, cur_kv_head, d]
                    else:
                        for j, d in T.Parallel(block_n, dim):
                            kv_pos = tile_start + j
                            new_pos = kv_pos - old_len
                            safe_kv_pos = T.if_then_else(kv_pos < old_len, kv_pos, 0)
                            page_idx = safe_kv_pos >> T.int32(page_size_log2)
                            page_offset = safe_kv_pos - page_idx * page_size
                            physical_pos = block_table[bz, page_idx] * page_size + page_offset
                            if kv_pos < old_len:
                                k_shared[j, d] = k_pages[physical_pos, cur_kv_head, d]
                                v_shared[j, d] = v_pages[physical_pos, cur_kv_head, d]
                            elif kv_pos < total_len:
                                k_shared[j, d] = k_new[q_start + new_pos, cur_kv_head, d]
                                v_shared[j, d] = v_new[q_start + new_pos, cur_kv_head, d]
                            else:
                                k_shared[j, d] = T.cast(0, dtype)
                                v_shared[j, d] = T.cast(0, dtype)
                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            kv_pos = k_idx * block_n + j
                            q_abs_pos = old_len + bx * block_m + i
                            valid = (bx * block_m + i < q_len) & (kv_pos < total_len) & (
                                kv_pos <= q_abs_pos)
                            acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            kv_pos = k_idx * block_n + j
                            valid = (bx * block_m + i < q_len) & (kv_pos < total_len)
                            acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale,
                                   scores_sum, logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_m, dim):
                    if bx * block_m + i < q_len:
                        output[q_start + bx * block_m + i, by, j] = acc_o[i, j] / logsum[i]
                for i in T.Parallel(block_m):
                    if bx * block_m + i < q_len:
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                        lse[by, q_start + bx * block_m + i] = logsum[i]

        return _gqa_prefill_paged_with_kv_cache_fwd_main

    return _gqa_prefill_paged_with_kv_cache_fwd_func


@torch.library.custom_op(
    "top::gqa_prefill_paged_with_kv_cache_fwd_wrapped_kernel",
    mutates_args=("k_pages", "v_pages"),
)
def _gqa_prefill_paged_with_kv_cache_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    physical_tokens: int,
    max_pages_per_req: int,
    page_size: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    max_seqlen_q: int,
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_prefill_paged_with_kv_cache_fwd_kernel(
        batch, heads, heads_kv, total_q, physical_tokens, max_pages_per_req, page_size, dim,
        is_causal, sm_scale, dtype)(block_m, block_n, num_stages, threads)(
            q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table,
            max_seqlen_q)


@_gqa_prefill_paged_with_kv_cache_fwd_wrapped_kernel.register_fake
def _(batch: int, heads: int, heads_kv: int, total_q: int, physical_tokens: int,
      max_pages_per_req: int, page_size: int, dim: int, is_causal: bool, sm_scale: float,
      dtype: str, block_m: int, block_n: int, num_stages: int, threads: int,
      max_seqlen_q: int, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([heads, total_q])
    return fake_o, fake_lse


class GQAPrefillPagedWithKVCacheFwdKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 max_pages_per_req: int,
                 page_size: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 sm_scale: Optional[float] = None,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.heads_kv = heads_kv
        self.max_pages_per_req = max_pages_per_req
        self.page_size = page_size
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale

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

    def forward(self, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
                k_pages: torch.Tensor, v_pages: torch.Tensor, cu_seqlens_q: torch.Tensor,
                cache_seqlens: torch.Tensor, block_table: torch.Tensor,
                max_seqlen_q: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gqa_prefill_paged_with_kv_cache_fwd_wrapped_kernel(
            self.batch, self.heads, self.heads_kv, q.shape[0], k_pages.shape[0],
            self.max_pages_per_req, self.page_size, self.dim, self.is_causal, self.sm_scale,
            self.dtype_str, self.config["block_m"], self.config["block_n"],
            self.config["num_stages"], self.config["threads"], max_seqlen_q, q, k_new, v_new,
            k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table)
