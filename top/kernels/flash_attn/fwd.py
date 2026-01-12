import itertools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = [
    'MhaFwdKernel', 'MhaFwdWgmmaPipelinedKernel', 'GqaFwdKernel', 'GqaFwdWgmmaPipelinedKernel'
]

# MHA


def _mha_fwd_kernel(batch: int,
                    heads: int,
                    seq_len: int,
                    dim: int,
                    is_causal: bool,
                    dtype: str = 'float16') -> Callable:
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mha_fwd_func(block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:
        shape = (batch, seq_len, heads, dim)

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
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
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
      int, dim: int, is_causal: bool, dtype: str,  # noqa: U100
      block_m: int, block_n: int, num_stages: int, hreads: int,  # noqa: U100
      *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: U100
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
            "block_M": 64,
            "block_N": 64 if self.dim <= 128 else 32,
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
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _mha_fwd_wrapped_kernel(self.batch, self.heads, self.seq_len, self.dim,
                                       self.is_causal, self.dtype_str, self.config["block_M"],
                                       self.config["block_N"], self.config["num_stages"],
                                       self.config["threads"], q, k, v)


def _mha_fwd_wgmma_pipelined_kernel(batch: int,
                                    heads: int,
                                    seq_len: int,
                                    dim: int,
                                    is_causal: bool,
                                    dtype: str = "float16") -> Callable:
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
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

        @T.macro
        def softmax(
                acc_s: T.FragmentBuffer([block_m, block_n], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_m, block_n], dtype),
                scores_max: T.FragmentBuffer([block_m], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_m], accum_dtype),
                scores_scale: T.FragmentBuffer([block_m], accum_dtype),
                scores_sum: T.FragmentBuffer([block_m], accum_dtype),
                logsum: T.FragmentBuffer([block_m], accum_dtype),
        ) -> None:
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            for i in T.Parallel(block_m):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_m, block_n):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_m):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def rescale(
                acc_o: T.FragmentBuffer([block_m, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_m], accum_dtype),
        ) -> None:
            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] *= scores_scale[i]

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
                    softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
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
      dim: int, is_causal: bool, dtype: str,  # noqa: U100
      block_m: int, block_n: int, num_stages: int, threads: int,  # noqa: U100
      *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: U100
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
        return {"block_M": 128, "block_N": 128, "num_stages": 2, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [32, 64, 128]
        block_n = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, block_n, num_stages, threads))
        return [{
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _mha_fwd_wgmma_pipelined_wrapped_kernel(self.batch, self.heads, self.seq_len,
                                                       self.dim, self.is_causal, self.dtype_str,
                                                       self.config["block_M"],
                                                       self.config["block_N"],
                                                       self.config["num_stages"],
                                                       self.config["threads"], q, k, v)


# GQA


def _gqa_fwd_kernel(batch: int,
                    heads: int,
                    heads_kv: int,
                    seq_len: int,
                    dim: int,
                    is_causal: bool,
                    dtype: str = 'float16') -> Callable:
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
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
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
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
      heads_kv: int, seq_len: int, dim: int, is_causal: bool,  # noqa: U100
      dtype: str, block_m: int, block_n: int, num_stages: int, threads: int,  # noqa: U100
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
        assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
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
            "block_M": 64,
            "block_N": 64 if self.dim <= 128 else 32,
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
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gqa_fwd_wrapped_kernel(self.batch, self.heads, self.heads_kv, self.seq_len,
                                       self.dim, self.is_causal, self.dtype_str,
                                       self.config["block_M"], self.config["block_N"],
                                       self.config["num_stages"], self.config["threads"], q, k, v)


def _gqa_fwd_wgmma_pipelined_kernel(batch: int,
                                    heads: int,
                                    heads_kv: int,
                                    seq_len: int,
                                    dim: int,
                                    is_causal: bool,
                                    dtype: str = "float16") -> Callable:
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
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

        @T.macro
        def softmax(
                acc_s: T.FragmentBuffer([block_m, block_n], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_m, block_n], dtype),
                scores_max: T.FragmentBuffer([block_m], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_m], accum_dtype),
                scores_scale: T.FragmentBuffer([block_m], accum_dtype),
                scores_sum: T.FragmentBuffer([block_m], accum_dtype),
                logsum: T.FragmentBuffer([block_m], accum_dtype),
        ) -> None:
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            for i in T.Parallel(block_m):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_m, block_n):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_m):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def rescale(
                acc_o: T.FragmentBuffer([block_m, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_m], accum_dtype),
        ) -> None:
            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] *= scores_scale[i]

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
                    softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
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
def _(batch: int, heads: int, heads_kv: int,  # noqa: U100
      seq_len: int, dim: int, is_causal: bool,  # noqa: U100
      dtype: str, block_m: int, block_n: int, num_stages: int, threads: int,  # noqa: U100
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
        assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
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
        return {"block_M": 128, "block_N": 128, "num_stages": 2, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        block_m = [32, 64, 128]
        block_n = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, block_n, num_stages, threads))
        return [{
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gqa_fwd_wgmma_pipelined_wrapped_kernel(self.batch, self.heads, self.heads_kv,
                                                       self.seq_len, self.dim, self.is_causal,
                                                       self.dtype_str, self.config["block_M"],
                                                       self.config["block_N"],
                                                       self.config["num_stages"],
                                                       self.config["threads"], q, k, v)
