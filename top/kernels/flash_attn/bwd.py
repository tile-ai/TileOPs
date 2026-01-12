import itertools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = [
    'FlashAttnBwdPreprocessKernel', 'FlashAttnBwdPostprocessKernel', 'MhaBwdKernel',
    'MhaBwdWgmmaPipelinedKernel', 'GqaBwdKernel', 'GqaBwdWgmmaPipelinedKernel'
]

# pre/post process for mha/gqa bwd


@tilelang.jit(out_idx=[2])
def _flashattn_bwd_preprocess_kernel(batch: int, heads: int, seq_len: int, dim: int,
                                     dtype: str) -> Callable:
    accum_dtype = "float"
    shape = (batch, seq_len, heads, dim)
    blk = 256

    @T.prim_func
    def flash_bwd_prep(
            o: T.Tensor(shape, dtype),  # type: ignore
            do: T.Tensor(shape, dtype),  # noqa: VNE002  # d(out): gradient of output reciprocal
            delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ) -> None:
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o_frag = T.alloc_fragment([blk, blk], dtype)
            do_frag = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta_frag = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(o[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], o_frag)
                T.copy(do[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], do_frag)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o_frag[i, j] * do_frag[i, j]
            T.reduce_sum(acc, delta_frag, 1)
            T.copy(delta_frag, delta[bz, bx, by * blk:(by + 1) * blk])

    return flash_bwd_prep


class FlashAttnBwdPreprocessKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self, batch: int, heads: int, seq_len: int, dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = dtype

        self.kernel = _flashattn_bwd_preprocess_kernel(self.batch, self.heads, self.seq_len,
                                                       self.dim, self.dtype_str)

    def forward(
            self,
            o: torch.Tensor,
            do: torch.Tensor  # noqa: VNE002
    ) -> torch.Tensor:
        return self.kernel(o, do)


def make_dq_layout(dq: torch.Tensor) -> T.Layout:
    # atomicAdd cannot be vectorized on Ampere, need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(
        dq.shape, lambda b, length, h, d:
        [b, length // 8, h, d // 8, (d % 2), 4 * (length % 8) + (d % 8) // 2])


@tilelang.jit(out_idx=[1])
def _flashattn_bwd_postprocess_kernel(batch: int,
                                      heads: int,
                                      seq_len: int,
                                      dim: int,
                                      dtype: str = "float16") -> Callable:
    accum_dtype = "float"
    shape = (batch, seq_len, heads, dim)
    blk = 64

    @T.prim_func
    def flash_bwd_post(
            dq: T.Tensor(shape, accum_dtype),  # type: ignore
            dq_out: T.Tensor(shape, dtype),  # type: ignore
    ) -> None:
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dq: make_dq_layout(dq)})
            T.copy(
                dq[bz, bx * blk:(bx + 1) * blk, by, :],
                dq_out[bz, bx * blk:(bx + 1) * blk, by, :],
            )

    return flash_bwd_post


class FlashAttnBwdPostprocessKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self, batch: int, heads: int, seq_len: int, dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = dtype

        self.kernel = _flashattn_bwd_postprocess_kernel(self.batch, self.heads, self.seq_len,
                                                        self.dim, self.dtype_str)

    def forward(self, dq: torch.Tensor) -> torch.Tensor:
        return self.kernel(dq)


# MHA


def _mha_bwd_kernel(batch: int,
                    heads: int,
                    seq_len: int,
                    dim: int,
                    is_causal: bool,
                    dtype: str = "float16") -> Callable:
    sm_scale = (1.0 / dim)**0.5
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[7, 8],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mha_bwd_func(block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:

        shape = (batch, seq_len, heads, dim)

        @T.prim_func
        def _mha_bwd_main(
                q: T.Tensor(shape, dtype),  # type: ignore
                k: T.Tensor(shape, dtype),  # type: ignore
                v: T.Tensor(shape, dtype),  # type: ignore
                do: T.Tensor(shape, dtype),  # noqa: VNE002  # d(out): gradient of output reciprocal
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                dq: T.Tensor(shape, accum_dtype),  # type: ignore
                dk: T.Tensor(shape, dtype),  # type: ignore
                dv: T.Tensor(shape, dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    heads, T.ceildiv(seq_len, block_m), batch, threads=threads) as (bx, by, bz):
                k_shared = T.alloc_shared([block_m, dim], dtype)
                dst_shared = T.alloc_shared([block_m, block_n], dtype)
                # should not store k to local if dim is large
                # k_local = T.alloc_fragment([block_m, dim], dtype)
                # k_local_t = T.alloc_fragment([block_m, dim], dtype)
                # v_local = T.alloc_fragment([block_m, dim], dtype)
                q_frag = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_m, dim], dtype)
                qkt = T.alloc_fragment([block_m, block_n], accum_dtype)
                dst = T.alloc_fragment([block_m, block_n], accum_dtype)
                qkt_cast = T.alloc_fragment([block_m, block_n], dtype)
                dst_cast = T.alloc_fragment([block_m, block_n], dtype)
                lse_shared = T.alloc_shared([block_n], accum_dtype)
                delta_shared = T.alloc_shared([block_n], accum_dtype)
                do_shared = T.alloc_shared([block_n, dim], dtype)
                dv_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dk_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dq_frag = T.alloc_fragment([block_n, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_m, dim], dtype)
                dk_shared = T.alloc_shared([block_m, dim], dtype)

                T.annotate_layout({
                    dq: make_dq_layout(dq),
                    dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                })

                T.copy(k[bz, by * block_m:(by + 1) * block_m, bx, :], k_shared)
                T.copy(v[bz, by * block_m:(by + 1) * block_m, bx, :], v_shared)
                T.clear(dv_frag)
                T.clear(dk_frag)

                loop_st = T.floordiv(by * block_m, block_n) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_n)

                for k_idx in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(q[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], q_frag)
                    T.clear(qkt)
                    T.gemm(k_shared, q_frag, qkt, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(lse[bz, bx, k_idx * block_n:(k_idx + 1) * block_n], lse_shared)
                    for i, j in T.Parallel(block_m, block_n):
                        qkt[i, j] = T.exp2(qkt[i, j] * scale - lse_shared[j])
                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            qkt[i, j] = T.if_then_else(by * block_m + i <= k_idx * block_n + j,
                                                       qkt[i, j], 0)
                    T.copy(do[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], do_shared)
                    T.clear(dst)
                    T.gemm(
                        v_shared, do_shared, dst, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(qkt, qkt_cast)
                    T.gemm(qkt_cast, do_shared, dv_frag, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(delta[bz, bx, k_idx * block_n:(k_idx + 1) * block_n], delta_shared)

                    for i, j in T.Parallel(block_m, block_n):
                        dst_cast[i, j] = qkt[i, j] * (dst[i, j] - delta_shared[j]) * sm_scale
                    T.gemm(dst_cast, q_frag, dk_frag, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(dst_cast, dst_shared)
                    T.clear(dq_frag)
                    T.gemm(dst_shared, k_shared, dq_frag, transpose_A=True)
                    for i, j in T.Parallel(block_n, dim):
                        T.atomic_add(dq[bz, k_idx * block_n + i, bx, j], dq_frag[i, j])

                T.copy(dv_frag, dv_shared)
                T.copy(dk_frag, dk_shared)
                T.copy(dv_shared, dv[bz, by * block_m:(by + 1) * block_m, bx, :])
                T.copy(dk_shared, dk[bz, by * block_m:(by + 1) * block_m, bx, :])

        return _mha_bwd_main

    return _mha_bwd_func


class MhaBwdKernel(Kernel):
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

        self.kernel = _mha_bwd_kernel(self.batch, self.heads, self.seq_len, self.dim,
                                      self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 64,
            "block_N": 64 if self.dim <= 64 else 32,
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

    def forward(self, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return self.kernel(**self.config)(*inputs)


def _mha_bwd_wgmma_pipelined_kernel(batch: int,
                                    heads: int,
                                    seq_len: int,
                                    dim: int,
                                    is_causal: bool,
                                    dtype: str = "float16") -> Callable:
    sm_scale = (1.0 / dim)**0.5
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[7, 8],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mha_bwd_wgmma_pipelined_func(block_m: int, block_n: int, num_stages: int,
                                      threads: int) -> Callable:

        shape = (batch, seq_len, heads, dim)

        @T.prim_func
        def _mha_bwd_wgmma_pipelined_main(
                q: T.Tensor(shape, dtype),  # type: ignore
                k: T.Tensor(shape, dtype),  # type: ignore
                v: T.Tensor(shape, dtype),  # type: ignore
                do: T.Tensor(shape, dtype),  # noqa: VNE002  # d(out): gradient of output reciprocal
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                dq: T.Tensor(shape, accum_dtype),  # type: ignore
                dk: T.Tensor(shape, dtype),  # type: ignore
                dv: T.Tensor(shape, dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    heads, T.ceildiv(seq_len, block_m), batch, threads=threads) as (bx, by, bz):
                k_shared = T.alloc_shared([block_m, dim], dtype)
                dst_shared = T.alloc_shared([block_m, block_n], dtype)
                # should not store k to local if dim is large
                # k_local = T.alloc_fragment([block_m, dim], dtype)
                # k_local_t = T.alloc_fragment([block_m, dim], dtype)
                # v_local = T.alloc_fragment([block_m, dim], dtype)
                q_frag = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_m, dim], dtype)
                qkt = T.alloc_fragment([block_m, block_n], accum_dtype)
                dst = T.alloc_fragment([block_m, block_n], accum_dtype)
                qkt_cast = T.alloc_fragment([block_m, block_n], dtype)
                dst_cast = T.alloc_fragment([block_m, block_n], dtype)
                lse_shared = T.alloc_shared([block_n], accum_dtype)
                delta_shared = T.alloc_shared([block_n], accum_dtype)
                do_shared = T.alloc_shared([block_n, dim], dtype)
                dv_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dk_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dq_frag = T.alloc_fragment([block_n, dim], accum_dtype)
                dq_shared = T.alloc_shared([block_n, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_m, dim], dtype)
                dk_shared = T.alloc_shared([block_m, dim], dtype)

                T.annotate_layout({
                    dq_shared: tilelang.layout.make_swizzled_layout(dq_shared),
                    dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                })

                T.use_swizzle(10, enable=True)

                T.copy(k[bz, by * block_m:(by + 1) * block_m, bx, :], k_shared)
                T.copy(v[bz, by * block_m:(by + 1) * block_m, bx, :], v_shared)
                T.clear(dv_frag)
                T.clear(dk_frag)

                loop_st = T.floordiv(by * block_m, block_n) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_n)

                for k_idx in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(q[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], q_frag)
                    T.clear(qkt)
                    T.gemm(
                        k_shared,
                        q_frag,
                        qkt,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)
                    T.copy(do[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], do_shared)
                    T.clear(dst)
                    T.gemm(
                        v_shared,
                        do_shared,
                        dst,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)

                    T.copy(lse[bz, bx, k_idx * block_n:(k_idx + 1) * block_n], lse_shared)
                    T.wait_wgmma(1)
                    for i, j in T.Parallel(block_m, block_n):
                        qkt[i, j] = T.exp2(qkt[i, j] * scale - lse_shared[j])
                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            qkt[i, j] = T.if_then_else(by * block_m + i <= k_idx * block_n + j,
                                                       qkt[i, j], 0)
                    T.wait_wgmma(0)
                    T.copy(qkt, qkt_cast)
                    T.gemm(
                        qkt_cast, do_shared, dv_frag, policy=T.GemmWarpPolicy.FullRow, wg_wait=-1)

                    T.copy(delta[bz, bx, k_idx * block_n:(k_idx + 1) * block_n], delta_shared)

                    for i, j in T.Parallel(block_m, block_n):
                        dst_cast[i, j] = qkt[i, j] * (dst[i, j] - delta_shared[j]) * sm_scale
                    T.gemm(dst_cast, q_frag, dk_frag, policy=T.GemmWarpPolicy.FullRow, wg_wait=1)

                    T.copy(dst_cast, dst_shared)
                    T.clear(dq_frag)
                    T.gemm(dst_shared, k_shared, dq_frag, transpose_A=True, wg_wait=1)
                    T.wait_wgmma(0)
                    T.copy(dq_frag, dq_shared)
                    T.atomic_add(dq[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], dq_shared)

                T.copy(dv_frag, dv_shared)
                T.copy(dk_frag, dk_shared)
                T.copy(dv_shared, dv[bz, by * block_m:(by + 1) * block_m, bx, :])
                T.copy(dk_shared, dk[bz, by * block_m:(by + 1) * block_m, bx, :])

        return _mha_bwd_wgmma_pipelined_main

    return _mha_bwd_wgmma_pipelined_func


class MhaBwdWgmmaPipelinedKernel(Kernel):
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

        self.kernel = _mha_bwd_wgmma_pipelined_kernel(self.batch, self.heads, self.seq_len,
                                                      self.dim, self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 128,
            "block_N": 128 if self.dim <= 64 else 32,
            "num_stages": 2,
            "threads": 256
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

    def forward(self, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return self.kernel(**self.config)(*inputs)


# GQA


def _gqa_bwd_kernel(batch: int,
                    heads: int,
                    heads_kv: int,
                    seq_len: int,
                    dim: int,
                    is_causal: bool,
                    dtype: str = "float16") -> Callable:
    sm_scale = (1.0 / dim)**0.5
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_bwd_func(block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:

        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        @T.prim_func
        def _gqa_bwd_main(
                q: T.Tensor(q_shape, dtype),  # type: ignore
                k: T.Tensor(kv_shape, dtype),  # type: ignore
                v: T.Tensor(kv_shape, dtype),  # type: ignore
                do: T.Tensor(q_shape, dtype),  # noqa: VNE002
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                dq: T.Tensor(q_shape, accum_dtype),  # type: ignore
                dk: T.Tensor(kv_shape, accum_dtype),  # type: ignore
                dv: T.Tensor(kv_shape, accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    heads, T.ceildiv(seq_len, block_m), batch, threads=threads) as (bx, by, bz):
                k_shared = T.alloc_shared([block_m, dim], dtype)
                dst_shared = T.alloc_shared([block_m, block_n], dtype)
                q_frag = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_m, dim], dtype)
                qkt = T.alloc_fragment([block_m, block_n], accum_dtype)
                dst = T.alloc_fragment([block_m, block_n], accum_dtype)
                qkt_cast = T.alloc_fragment([block_m, block_n], dtype)
                dst_cast = T.alloc_fragment([block_m, block_n], dtype)
                lse_shared = T.alloc_shared([block_n], accum_dtype)
                delta_shared = T.alloc_shared([block_n], accum_dtype)
                do_shared = T.alloc_shared([block_n, dim], dtype)
                dv_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dk_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dq_frag = T.alloc_fragment([block_n, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_m, dim], accum_dtype)
                dk_shared = T.alloc_shared([block_m, dim], accum_dtype)

                T.annotate_layout({
                    dq: make_dq_layout(dq),
                    dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                })

                T.copy(k[bz, by * block_m:(by + 1) * block_m, bx // groups, :], k_shared)
                T.copy(v[bz, by * block_m:(by + 1) * block_m, bx // groups, :], v_shared)
                T.clear(dv_frag)
                T.clear(dk_frag)

                loop_st = T.floordiv(by * block_m, block_n) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_n)

                for k_idx in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(q[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], q_frag)
                    T.clear(qkt)
                    T.gemm(k_shared, q_frag, qkt, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(lse[bz, bx, k_idx * block_n:(k_idx + 1) * block_n], lse_shared)
                    for i, j in T.Parallel(block_m, block_n):
                        qkt[i, j] = T.exp2(qkt[i, j] * scale - lse_shared[j])
                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            qkt[i, j] = T.if_then_else(by * block_m + i <= k_idx * block_n + j,
                                                       qkt[i, j], 0)
                    T.copy(do[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], do_shared)
                    T.clear(dst)
                    T.gemm(
                        v_shared, do_shared, dst, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(qkt, qkt_cast)
                    T.gemm(qkt_cast, do_shared, dv_frag, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(delta[bz, bx, k_idx * block_n:(k_idx + 1) * block_n], delta_shared)

                    for i, j in T.Parallel(block_m, block_n):
                        dst_cast[i, j] = qkt[i, j] * (dst[i, j] - delta_shared[j]) * sm_scale
                    T.gemm(dst_cast, q_frag, dk_frag, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(dst_cast, dst_shared)
                    T.clear(dq_frag)
                    T.gemm(dst_shared, k_shared, dq_frag, transpose_A=True)
                    for i, j in T.Parallel(block_n, dim):
                        T.atomic_add(dq[bz, k_idx * block_n + i, bx, j], dq_frag[i, j])
                T.copy(dv_frag, dv_shared)
                T.atomic_add(dv[bz, by * block_m:(by + 1) * block_m, bx // groups, :], dv_shared)
                T.copy(dk_frag, dk_shared)
                T.atomic_add(dk[bz, by * block_m:(by + 1) * block_m, bx // groups, :], dk_shared)

        return _gqa_bwd_main

    return _gqa_bwd_func


class GqaBwdKernel(Kernel):
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
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _gqa_bwd_kernel(self.batch, self.heads, self.heads_kv, self.seq_len, self.dim,
                                      self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 64,
            "block_N": 64 if self.dim <= 64 else 32,
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

    def forward(self, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return self.kernel(**self.config)(*inputs)


def _gqa_bwd_wgmma_pipelined_kernel(batch: int,
                                    heads: int,
                                    heads_kv: int,
                                    seq_len: int,
                                    dim: int,
                                    is_causal: bool,
                                    dtype: str = "float16") -> Callable:
    sm_scale = (1.0 / dim)**0.5
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_bwd_wgmma_pipelined_func(block_m: int, block_n: int, num_stages: int,
                                      threads: int) -> Callable:

        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        @T.prim_func
        def _gqa_bwd_wgmma_pipelined_main(
                q: T.Tensor(q_shape, dtype),  # type: ignore
                k: T.Tensor(kv_shape, dtype),  # type: ignore
                v: T.Tensor(kv_shape, dtype),  # type: ignore
                do: T.Tensor(q_shape, dtype),  # noqa: VNE002
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                dq: T.Tensor(q_shape, accum_dtype),  # type: ignore
                dk: T.Tensor(kv_shape, accum_dtype),  # type: ignore
                dv: T.Tensor(kv_shape, accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    heads, T.ceildiv(seq_len, block_m), batch, threads=threads) as (bx, by, bz):
                k_shared = T.alloc_shared([block_m, dim], dtype)
                dst_shared = T.alloc_shared([block_m, block_n], dtype)
                q_frag = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_m, dim], dtype)
                qkt = T.alloc_fragment([block_m, block_n], accum_dtype)
                dst = T.alloc_fragment([block_m, block_n], accum_dtype)
                qkt_cast = T.alloc_fragment([block_m, block_n], dtype)
                dst_cast = T.alloc_fragment([block_m, block_n], dtype)
                lse_shared = T.alloc_shared([block_n], accum_dtype)
                delta_shared = T.alloc_shared([block_n], accum_dtype)
                do_shared = T.alloc_shared([block_n, dim], dtype)
                dv_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dk_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dq_frag = T.alloc_fragment([block_n, dim], accum_dtype)
                dq_shared = T.alloc_shared([block_n, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_m, dim], accum_dtype)
                dk_shared = T.alloc_shared([block_m, dim], accum_dtype)

                T.annotate_layout({
                    dq_shared: tilelang.layout.make_swizzled_layout(dq_shared),
                    dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                })

                T.copy(k[bz, by * block_m:(by + 1) * block_m, bx // groups, :], k_shared)
                T.copy(v[bz, by * block_m:(by + 1) * block_m, bx // groups, :], v_shared)
                T.clear(dv_frag)
                T.clear(dk_frag)

                loop_st = T.floordiv(by * block_m, block_n) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_n)

                for k_idx in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(q[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], q_frag)
                    T.clear(qkt)
                    T.gemm(
                        k_shared,
                        q_frag,
                        qkt,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)
                    T.copy(lse[bz, bx, k_idx * block_n:(k_idx + 1) * block_n], lse_shared)
                    for i, j in T.Parallel(block_m, block_n):
                        qkt[i, j] = T.exp2(qkt[i, j] * scale - lse_shared[j])
                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            qkt[i, j] = T.if_then_else(by * block_m + i <= k_idx * block_n + j,
                                                       qkt[i, j], 0)
                    T.copy(do[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], do_shared)
                    T.clear(dst)
                    T.gemm(
                        v_shared,
                        do_shared,
                        dst,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)
                    T.wait_wgmma(1)
                    T.copy(qkt, qkt_cast)
                    T.gemm(
                        qkt_cast, do_shared, dv_frag, policy=T.GemmWarpPolicy.FullRow, wg_wait=-1)

                    T.copy(delta[bz, bx, k_idx * block_n:(k_idx + 1) * block_n], delta_shared)

                    for i, j in T.Parallel(block_m, block_n):
                        dst_cast[i, j] = qkt[i, j] * (dst[i, j] - delta_shared[j]) * sm_scale
                    T.wait_wgmma(0)
                    T.gemm(dst_cast, q_frag, dk_frag, policy=T.GemmWarpPolicy.FullRow, wg_wait=1)

                    T.copy(dst_cast, dst_shared)
                    T.clear(dq_frag)
                    T.gemm(dst_shared, k_shared, dq_frag, transpose_A=True, wg_wait=1)
                    T.wait_wgmma(0)
                    T.copy(dq_frag, dq_shared)
                    T.atomic_add(dq[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], dq_shared)
                T.copy(dv_frag, dv_shared)
                T.atomic_add(dv[bz, by * block_m:(by + 1) * block_m, bx // groups, :], dv_shared)
                T.copy(dk_frag, dk_shared)
                T.atomic_add(dk[bz, by * block_m:(by + 1) * block_m, bx // groups, :], dk_shared)

        return _gqa_bwd_wgmma_pipelined_main

    return _gqa_bwd_wgmma_pipelined_func


class GqaBwdWgmmaPipelinedKernel(Kernel):
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
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _gqa_bwd_wgmma_pipelined_kernel(self.batch, self.heads, self.heads_kv,
                                                      self.seq_len, self.dim, self.is_causal,
                                                      self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 128,
            "block_N": 128 if self.dim <= 64 else 32,
            "num_stages": 2,
            "threads": 256
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

    def forward(self, *inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return self.kernel(**self.config)(*inputs)
