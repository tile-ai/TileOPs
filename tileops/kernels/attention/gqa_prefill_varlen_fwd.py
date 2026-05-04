"""Packed variable-length GQA prefill forward kernel.

Inputs use THD layout:
  q: [T_q, H, D]
  k/v: [T_kv, H_kv, D]

``cu_seqlens_q`` and ``cu_seqlens_kv`` describe per-request packed ranges.
Causal masking uses bottom-right alignment per request, matching the dense
prefill contract when q_len may be smaller than kv_len.
"""

import functools
import itertools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.online_softmax import (
    LOG2E,
    make_online_softmax_with_mask_guard,
    make_rescale,
)

__all__ = ["GQAPrefillVarlenFwdKernel"]


@functools.lru_cache(maxsize=32)
def _gqa_prefill_varlen_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    dtype: str = "float16",
) -> Callable:
    scale = (dim**-0.5 if sm_scale is None else sm_scale) * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[7, 8],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_prefill_varlen_fwd_func(
        block_m: int, block_n: int, num_stages: int, threads: int
    ) -> Callable:
        q_shape = (total_q, heads, dim)
        kv_shape = (total_kv, heads_kv, dim)
        online_softmax = make_online_softmax_with_mask_guard(
            scale, accum_dtype, block_m, block_n
        )
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def _gqa_prefill_varlen_fwd_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
            max_seqlen_q: T.int32,  # type: ignore
            max_seqlen_kv: T.int32,  # type: ignore
            output: T.Tensor(q_shape, dtype),  # type: ignore
            lse: T.Tensor([heads, total_q], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(max_seqlen_q, block_m), heads, batch, threads=threads
            ) as (bx, by, bz):
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
                inv_logsum = T.alloc_fragment([block_m], accum_dtype)

                q_start = cu_seqlens_q[bz]
                kv_start = cu_seqlens_kv[bz]
                q_len = cu_seqlens_q[bz + 1] - q_start
                kv_len = cu_seqlens_kv[bz + 1] - kv_start
                causal_offset = kv_len - q_len
                cur_kv_head = by // groups

                if (bx + 1) * block_m <= q_len:
                    T.copy(
                        q[q_start + bx * block_m:q_start + (bx + 1) * block_m, by, :],
                        q_shared,
                        disable_tma=True,
                    )
                else:
                    for i, d in T.Parallel(block_m, dim):
                        q_pos = bx * block_m + i
                        if q_pos < q_len:
                            q_shared[i, d] = q[q_start + q_pos, by, d]
                        else:
                            q_shared[i, d] = T.cast(0, dtype)

                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(T.min(kv_len, causal_offset + (bx + 1) * block_m), block_n)
                    if is_causal
                    else T.ceildiv(kv_len, block_n)
                )

                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    tile_start = k_idx * block_n
                    tile_end = (k_idx + 1) * block_n
                    if tile_end <= kv_len:
                        T.copy(
                            k[kv_start + tile_start:kv_start + tile_end, cur_kv_head, :],
                            k_shared,
                            disable_tma=True,
                        )
                        T.copy(
                            v[kv_start + tile_start:kv_start + tile_end, cur_kv_head, :],
                            v_shared,
                            disable_tma=True,
                        )
                    else:
                        for j, d in T.Parallel(block_n, dim):
                            kv_pos = tile_start + j
                            if kv_pos < kv_len:
                                k_shared[j, d] = k[kv_start + kv_pos, cur_kv_head, d]
                                v_shared[j, d] = v[kv_start + kv_pos, cur_kv_head, d]
                            else:
                                k_shared[j, d] = T.cast(0, dtype)
                                v_shared[j, d] = T.cast(0, dtype)

                    for i, j in T.Parallel(block_m, block_n):
                        q_pos = bx * block_m + i
                        kv_pos = tile_start + j
                        if is_causal:
                            valid = (
                                (q_pos < q_len)
                                & (kv_pos < kv_len)
                                & (kv_pos <= q_pos + causal_offset)
                            )
                            acc_s[i, j] = T.if_then_else(
                                valid, 0, -T.infinity(acc_s.dtype)
                            )
                        else:
                            valid = (q_pos < q_len) & (kv_pos < kv_len)
                            acc_s[i, j] = T.if_then_else(
                                valid, 0, -T.infinity(acc_s.dtype)
                            )
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    online_softmax(
                        acc_s,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                    )
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                if (bx + 1) * block_m <= q_len:
                    for i in T.Parallel(block_m):
                        inv_logsum[i] = T.cast(1, accum_dtype) / logsum[i]
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= inv_logsum[i]
                    T.copy(
                        acc_o,
                        output[q_start + bx * block_m:q_start + (bx + 1) * block_m, by, :],
                        disable_tma=True,
                    )
                    for i in T.Parallel(block_m):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                    T.copy(
                        logsum,
                        lse[by, q_start + bx * block_m:q_start + (bx + 1) * block_m],
                        disable_tma=True,
                    )
                else:
                    for i in T.Parallel(block_m):
                        q_pos = bx * block_m + i
                        if q_pos < q_len:
                            inv_logsum[i] = T.cast(1, accum_dtype) / logsum[i]
                    for i, j in T.Parallel(block_m, dim):
                        q_pos = bx * block_m + i
                        if q_pos < q_len:
                            output[q_start + q_pos, by, j] = acc_o[i, j] * inv_logsum[i]
                    for i in T.Parallel(block_m):
                        q_pos = bx * block_m + i
                        if q_pos < q_len:
                            lse[by, q_start + q_pos] = (
                                T.log2(logsum[i]) + scores_max[i] * scale
                            )

        return _gqa_prefill_varlen_fwd_main

    return _gqa_prefill_varlen_fwd_func


@torch.library.custom_op("top::gqa_prefill_varlen_fwd_wrapped_kernel", mutates_args=())
def _gqa_prefill_varlen_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_prefill_varlen_fwd_kernel(
        batch, heads, heads_kv, total_q, total_kv, dim, is_causal, sm_scale, dtype
    )(block_m, block_n, num_stages, threads)(
        q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
    )


@_gqa_prefill_varlen_fwd_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_o = torch.empty([total_q, heads, dim], dtype=q.dtype, device=q.device)
    fake_lse = fake_o.new_empty([heads, total_q])
    return fake_o, fake_lse


class GQAPrefillVarlenFwdKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
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
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = list(
            itertools.product([32, 64, 128], [32, 64, 128], [1, 2, 3], [128, 256])
        )
        return [
            {"block_m": c[0], "block_n": c[1], "num_stages": c[2], "threads": c[3]}
            for c in configs
        ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_q, total_kv = q.shape[0], k.shape[0]
        return _gqa_prefill_varlen_fwd_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            total_q,
            total_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
            max_seqlen_q,
            max_seqlen_kv,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
        )
