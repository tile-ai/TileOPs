"""Paged GQA prefill specializations built from one attention skeleton.

The public Paged Op has three physical policies today: matching-width cache,
storage-only FP8 cache, and fused RoPE with a separate append pass.  Their QK,
mask, online-softmax, and PV pipeline is identical; only Q/K loading and cache
append conversion differ.  Those differences are compile-time policy flags in
``_gqa_paged_attention_kernel`` rather than copied TileLang programs.
"""

import functools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.online_softmax import (
    LOG2E,
    make_apply_softcap,
    make_online_softmax_with_mask_guard,
    make_rescale,
)

from ._config import tile_stage_thread_configs
from .call_spec import fp8_dtype
from .fp8_prefill_core import make_native_fp8_prefill_tile_update
from .paged_prefill import PagedPrefillKernel
from .prefill_mask import make_bottom_right_attention_mask

__all__ = [
    "GQAPrefillPagedNativeFP8TensorCoreFwdKernel",
    "GQAPrefillPagedWithFP8KVCacheFwdKernel",
    "GQAPrefillPagedWithKVCacheFwdKernel",
    "GQAPrefillPagedWithKVCacheRopeFwdKernel",
]

_FAST_COMPILE_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DENABLE_BF16",
]


@functools.lru_cache(maxsize=64)
def _gqa_paged_attention_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    physical_tokens: int,
    max_pages_per_req: int,
    page_size: int,
    dim: int,
    is_causal: bool,
    sm_scale: Optional[float],
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    append_kv: bool,
    input_dtype: str,
    output_dtype: str,
    cache_dtype: str,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
) -> Callable:
    """Build one paged-attention program with compile-time loader policies."""
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if page_size <= 0 or page_size & (page_size - 1) != 0:
        raise ValueError("page_size must be a positive power of two")
    if fuse_rope and (
        rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > dim
    ):
        raise ValueError("rotary_dim must be positive, even, and <= dim")

    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    native_fp8 = input_dtype == "float8_e4m3fn"
    scale = LOG2E if native_fp8 or use_softcap else score_scale * LOG2E
    groups = heads // heads_kv
    accum_dtype = "float"
    fp8_cache = cache_dtype == "float8_e4m3fn"
    rope_rows = max_position if fuse_rope else 1
    rope_cols = rotary_dim // 2 if fuse_rope else 1
    page_size_log2 = page_size.bit_length() - 1

    @tilelang.jit(
        out_idx=[13],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
        compile_flags=_FAST_COMPILE_FLAGS,
    )
    def build(block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:
        q_shape = (total_q, heads, dim)
        kv_new_shape = (total_q, heads_kv, dim)
        kv_pages_shape = (physical_tokens, heads_kv, dim)
        block_table_shape = (batch, max_pages_per_req)
        rope_shape = (rope_rows, rope_cols)
        output_shape = (total_q, heads, dim)

        online_softmax = make_online_softmax_with_mask_guard(
            scale, accum_dtype, block_m, block_n
        )
        apply_softcap = (
            make_apply_softcap(
                1.0 if native_fp8 else score_scale,
                softcap,
                accum_dtype,
                block_m,
                block_n,
            )
            if use_softcap
            else None
        )
        rescale = make_rescale(block_m, dim)
        initialize_mask = make_bottom_right_attention_mask(
            is_causal,
            window_size_left,
            window_size_right,
            accum_dtype,
            block_m,
            block_n,
        )
        apply_transformed_mask = make_bottom_right_attention_mask(
            is_causal,
            window_size_left,
            window_size_right,
            accum_dtype,
            block_m,
            block_n,
            preserve_valid=True,
        )
        fp8_tile_update = make_native_fp8_prefill_tile_update(
            is_causal=is_causal,
            softcap=softcap,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            accum_dtype=accum_dtype,
            block_m=block_m,
            block_n=block_n,
            dim=dim,
        )

        @T.macro
        def quantize_cache(value, scale_value):
            if fp8_cache and not native_fp8:
                return T.clamp(
                    T.Cast("float32", value) / scale_value, -448.0, 448.0
                )
            return value

        @T.macro
        def load_cache(value, scale_value):
            if fp8_cache and not native_fp8:
                return T.Cast(input_dtype, T.Cast("float32", value) * scale_value)
            return T.Cast(input_dtype, value)

        @T.macro
        def rotate(value, paired_value, logical_pos, d, cos_table, sin_table):
            if fuse_rope:
                freq_idx = d % rope_cols
                c = cos_table[logical_pos, freq_idx]
                s = sin_table[logical_pos, freq_idx]
                rotated = T.if_then_else(d < rope_cols, -paired_value, paired_value)
                return T.if_then_else(
                    d < rotary_dim, value * c + rotated * s, value
                )
            return value

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, input_dtype),  # type: ignore
            k_new: T.Tensor(kv_new_shape, input_dtype),  # type: ignore
            v_new: T.Tensor(kv_new_shape, input_dtype),  # type: ignore
            k_pages: T.Tensor(kv_pages_shape, cache_dtype),  # type: ignore
            v_pages: T.Tensor(kv_pages_shape, cache_dtype),  # type: ignore
            q_scale: T.Tensor([batch, heads_kv], T.float32),  # type: ignore
            k_scale: T.Tensor([batch, heads_kv], T.float32),  # type: ignore
            v_scale: T.Tensor([batch, heads_kv], T.float32),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cache_seqlens: T.Tensor([batch], T.int32),  # type: ignore
            block_table: T.Tensor(block_table_shape, T.int32),  # type: ignore
            cos_table: T.Tensor(rope_shape, input_dtype),  # type: ignore
            sin_table: T.Tensor(rope_shape, input_dtype),  # type: ignore
            output: T.Tensor(output_shape, output_dtype),  # type: ignore
            max_seqlen_q: T.int32,  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(max_seqlen_q, block_m), heads, batch, threads=threads
            ) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, dim], input_dtype)
                k_shared = T.alloc_shared([block_n, dim], input_dtype)
                v_shared = T.alloc_shared([block_n, dim], input_dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                if native_fp8:
                    acc_s_cast = T.alloc_shared([block_m, block_n], input_dtype)
                else:
                    acc_s_cast = T.alloc_fragment([block_m, block_n], input_dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)
                inv_logsum = T.alloc_fragment([block_m], accum_dtype)

                q_start = cu_seqlens_q[bz]
                q_len = cu_seqlens_q[bz + 1] - q_start
                old_len = cache_seqlens[bz]
                total_len = old_len + q_len
                cur_kv_head = by // groups
                local_q_scale = q_scale[bz, cur_kv_head]
                local_k_scale = k_scale[bz, cur_kv_head]
                local_v_scale = v_scale[bz, cur_kv_head]

                if fuse_rope:
                    for i, d in T.Parallel(block_m, dim):
                        new_pos = bx * block_m + i
                        safe_new_pos = T.if_then_else(new_pos < q_len, new_pos, 0)
                        logical_pos = old_len + safe_new_pos
                        paired_d = T.if_then_else(
                            d < rope_cols,
                            d + rope_cols,
                            T.if_then_else(d < rotary_dim, d - rope_cols, d),
                        )
                        if new_pos < q_len:
                            value = q[q_start + new_pos, by, d]
                            paired_value = q[q_start + new_pos, by, paired_d]
                            q_shared[i, d] = rotate(
                                value, paired_value, logical_pos, d,
                                cos_table, sin_table,
                            )
                        else:
                            q_shared[i, d] = T.cast(0, input_dtype)
                elif bx * block_m + block_m <= q_len:
                    T.copy(
                        q[q_start + bx * block_m:q_start + (bx + 1) * block_m, by, :],
                        q_shared,
                        disable_tma=True,
                    )
                else:
                    for i, d in T.Parallel(block_m, dim):
                        new_pos = bx * block_m + i
                        if new_pos < q_len:
                            q_shared[i, d] = q[q_start + new_pos, by, d]
                        else:
                            q_shared[i, d] = T.cast(0, input_dtype)

                # RoPE owns a preceding rotate+append launch. Other policies
                # append here, converting only at the cache-storage boundary.
                if append_kv and not fuse_rope and by < heads_kv:
                    append_start = old_len + bx * block_m
                    append_end = append_start + block_m
                    if bx * block_m + block_m <= q_len:
                        if append_start >> T.int32(page_size_log2) == (
                            append_end - 1
                        ) >> T.int32(page_size_log2):
                            page_idx = append_start >> T.int32(page_size_log2)
                            page_offset = append_start - page_idx * page_size
                            physical_start = (
                                block_table[bz, page_idx] * page_size + page_offset
                            )
                            for i, d in T.Parallel(block_m, dim):
                                k_pages[physical_start + i, by, d] = quantize_cache(
                                    k_new[q_start + bx * block_m + i, by, d],
                                    k_scale[bz, by],
                                )
                                v_pages[physical_start + i, by, d] = quantize_cache(
                                    v_new[q_start + bx * block_m + i, by, d],
                                    v_scale[bz, by],
                                )
                        else:
                            for i, d in T.Parallel(block_m, dim):
                                new_pos = bx * block_m + i
                                logical_pos = old_len + new_pos
                                split_page_idx = logical_pos >> T.int32(page_size_log2)
                                split_page_offset = logical_pos - split_page_idx * page_size
                                physical_pos = (
                                    block_table[bz, split_page_idx] * page_size
                                    + split_page_offset
                                )
                                k_pages[physical_pos, by, d] = quantize_cache(
                                    k_new[q_start + new_pos, by, d], k_scale[bz, by]
                                )
                                v_pages[physical_pos, by, d] = quantize_cache(
                                    v_new[q_start + new_pos, by, d], v_scale[bz, by]
                                )
                    else:
                        for i, d in T.Parallel(block_m, dim):
                            new_pos = bx * block_m + i
                            safe_new_pos = T.if_then_else(new_pos < q_len, new_pos, 0)
                            logical_pos = old_len + safe_new_pos
                            page_idx = logical_pos >> T.int32(page_size_log2)
                            page_offset = logical_pos - page_idx * page_size
                            if new_pos < q_len:
                                physical_pos = (
                                    block_table[bz, page_idx] * page_size + page_offset
                                )
                                k_pages[physical_pos, by, d] = quantize_cache(
                                    k_new[q_start + new_pos, by, d], k_scale[bz, by]
                                )
                                v_pages[physical_pos, by, d] = quantize_cache(
                                    v_new[q_start + new_pos, by, d], v_scale[bz, by]
                                )

                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))
                loop_range = (
                    T.ceildiv(old_len + (bx + 1) * block_m, block_n)
                    if is_causal
                    else T.ceildiv(total_len, block_n)
                )

                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    tile_start = k_idx * block_n
                    tile_end = tile_start + block_n
                    if tile_end <= old_len:
                        if page_size % block_n == 0:
                            page_idx = tile_start >> T.int32(page_size_log2)
                            page_offset = tile_start - page_idx * page_size
                            physical_start = (
                                block_table[bz, page_idx] * page_size + page_offset
                            )
                            for j, d in T.Parallel(block_n, dim):
                                k_shared[j, d] = load_cache(
                                    k_pages[physical_start + j, cur_kv_head, d],
                                    local_k_scale,
                                )
                                v_shared[j, d] = load_cache(
                                    v_pages[physical_start + j, cur_kv_head, d],
                                    local_v_scale,
                                )
                        elif block_n % page_size == 0:
                            tile_page_start = tile_start >> T.int32(page_size_log2)
                            for p in range(block_n // page_size):
                                physical_start = (
                                    block_table[bz, tile_page_start + p] * page_size
                                )
                                for off, d in T.Parallel(page_size, dim):
                                    row = p * page_size + off
                                    k_shared[row, d] = load_cache(
                                        k_pages[physical_start + off, cur_kv_head, d],
                                        local_k_scale,
                                    )
                                    v_shared[row, d] = load_cache(
                                        v_pages[physical_start + off, cur_kv_head, d],
                                        local_v_scale,
                                    )
                        else:
                            for j, d in T.Parallel(block_n, dim):
                                kv_pos = tile_start + j
                                page_idx = kv_pos >> T.int32(page_size_log2)
                                page_offset = kv_pos - page_idx * page_size
                                physical_pos = (
                                    block_table[bz, page_idx] * page_size + page_offset
                                )
                                k_shared[j, d] = load_cache(
                                    k_pages[physical_pos, cur_kv_head, d], local_k_scale
                                )
                                v_shared[j, d] = load_cache(
                                    v_pages[physical_pos, cur_kv_head, d], local_v_scale
                                )
                    elif tile_start >= old_len and tile_end <= total_len:
                        new_start = tile_start - old_len
                        for j, d in T.Parallel(block_n, dim):
                            new_pos = new_start + j
                            if fuse_rope:
                                logical_pos = tile_start + j
                                paired_d = T.if_then_else(
                                    d < rope_cols,
                                    d + rope_cols,
                                    T.if_then_else(d < rotary_dim, d - rope_cols, d),
                                )
                                value = k_new[q_start + new_pos, cur_kv_head, d]
                                paired_value = k_new[
                                    q_start + new_pos, cur_kv_head, paired_d
                                ]
                                k_shared[j, d] = rotate(
                                    value, paired_value, logical_pos, d,
                                    cos_table, sin_table,
                                )
                            else:
                                k_shared[j, d] = k_new[
                                    q_start + new_pos, cur_kv_head, d
                                ]
                            v_shared[j, d] = v_new[
                                q_start + new_pos, cur_kv_head, d
                            ]
                    else:
                        for j, d in T.Parallel(block_n, dim):
                            kv_pos = tile_start + j
                            new_pos = kv_pos - old_len
                            safe_kv_pos = T.if_then_else(kv_pos < old_len, kv_pos, 0)
                            load_page_idx = safe_kv_pos >> T.int32(page_size_log2)
                            load_page_offset = safe_kv_pos - load_page_idx * page_size
                            physical_pos = (
                                block_table[bz, load_page_idx] * page_size
                                + load_page_offset
                            )
                            if kv_pos < old_len:
                                k_shared[j, d] = load_cache(
                                    k_pages[physical_pos, cur_kv_head, d], local_k_scale
                                )
                                v_shared[j, d] = load_cache(
                                    v_pages[physical_pos, cur_kv_head, d], local_v_scale
                                )
                            elif kv_pos < total_len:
                                if fuse_rope:
                                    paired_d = T.if_then_else(
                                        d < rope_cols,
                                        d + rope_cols,
                                        T.if_then_else(d < rotary_dim, d - rope_cols, d),
                                    )
                                    value = k_new[q_start + new_pos, cur_kv_head, d]
                                    paired_value = k_new[
                                        q_start + new_pos, cur_kv_head, paired_d
                                    ]
                                    k_shared[j, d] = rotate(
                                        value, paired_value, kv_pos, d,
                                        cos_table, sin_table,
                                    )
                                else:
                                    k_shared[j, d] = k_new[
                                        q_start + new_pos, cur_kv_head, d
                                    ]
                                v_shared[j, d] = v_new[
                                    q_start + new_pos, cur_kv_head, d
                                ]
                            else:
                                k_shared[j, d] = T.cast(0, input_dtype)
                                v_shared[j, d] = T.cast(0, input_dtype)

                    if native_fp8:
                        fp8_tile_update(
                            q_shared,
                            k_shared,
                            v_shared,
                            acc_s,
                            acc_s_cast,
                            acc_o,
                            scores_max,
                            scores_max_prev,
                            scores_scale,
                            scores_sum,
                            logsum,
                            k_idx,
                            bx,
                            q_len,
                            total_len,
                            old_len,
                            local_q_scale * local_k_scale * score_scale,
                        )
                    else:
                        if use_softcap:
                            T.clear(acc_s)
                        else:
                            initialize_mask(
                                acc_s, k_idx, bx, q_len, total_len, old_len
                            )

                        T.gemm(
                            q_shared,
                            k_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        if use_softcap:
                            apply_softcap(acc_s)
                            apply_transformed_mask(
                                acc_s, k_idx, bx, q_len, total_len, old_len
                            )

                        online_softmax(
                            acc_s, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum,
                        )
                        T.copy(acc_s, acc_s_cast, disable_tma=True)
                        rescale(acc_o, scores_scale)
                        T.gemm(
                            acc_s_cast, v_shared, acc_o,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                for i in T.Parallel(block_m):
                    if bx * block_m + i < q_len:
                        inv_logsum[i] = T.cast(1, accum_dtype) / logsum[i]
                for i, d in T.Parallel(block_m, dim):
                    if bx * block_m + i < q_len:
                        output[q_start + bx * block_m + i, by, d] = (
                            acc_o[i, d]
                            * inv_logsum[i]
                            * (local_v_scale if native_fp8 else 1.0)
                        )

        return main

    return build


@torch.library.custom_op(
    "top::gqa_paged_attention_wrapped_kernel",
    mutates_args=("k_pages", "v_pages"),
)
def _gqa_paged_attention_wrapped_kernel(
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
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    append_kv: bool,
    input_dtype: str,
    output_dtype: str,
    cache_dtype: str,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
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
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
) -> torch.Tensor:
    return _gqa_paged_attention_kernel(
        batch, heads, heads_kv, total_q, physical_tokens,
        max_pages_per_req, page_size, dim, is_causal, sm_scale, softcap,
        window_size_left, window_size_right, append_kv,
        input_dtype, output_dtype, cache_dtype, fuse_rope, max_position, rotary_dim,
    )(block_m, block_n, num_stages, threads)(
        q, k_new, v_new, k_pages, v_pages, q_scale, k_scale, v_scale,
        cu_seqlens_q, cache_seqlens, block_table, cos_table, sin_table,
        max_seqlen_q,
    )


@_gqa_paged_attention_wrapped_kernel.register_fake
def _(
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
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    append_kv: bool,
    input_dtype: str,
    output_dtype: str,
    cache_dtype: str,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    max_seqlen_q: int,
    *inputs: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    dtype = torch.float16 if output_dtype == "float16" else torch.bfloat16
    return torch.empty(inputs[0].shape, dtype=dtype, device=inputs[0].device)


@functools.lru_cache(maxsize=32)
def _gqa_prefill_paged_with_kv_cache_rope_append_kernel(batch: int,
                                                        heads_kv: int,
                                                        total_q: int,
                                                        physical_tokens: int,
                                                        max_pages_per_req: int,
                                                        page_size: int,
                                                        dim: int,
                                                        max_position: int,
                                                        rotary_dim: int,
                                                        dtype: str = 'float16') -> Callable:
    if page_size <= 0 or page_size & (page_size - 1) != 0:
        raise ValueError("page_size must be a positive power of two")
    if rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > dim:
        raise ValueError("rotary_dim must be positive, even, and <= dim")
    half = rotary_dim // 2
    page_size_log2 = page_size.bit_length() - 1

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_prefill_paged_with_kv_cache_rope_append_func(block_m: int,
                                                          threads: int) -> Callable:

        kv_new_shape = (total_q, heads_kv, dim)
        kv_pages_shape = (physical_tokens, heads_kv, dim)
        block_table_shape = (batch, max_pages_per_req)
        rope_shape = (max_position, half)

        @T.prim_func
        def _gqa_prefill_paged_with_kv_cache_rope_append_main(
                k_new: T.Tensor(kv_new_shape, dtype),  # type: ignore
                v_new: T.Tensor(kv_new_shape, dtype),  # type: ignore
                k_pages: T.Tensor(kv_pages_shape, dtype),  # type: ignore
                v_pages: T.Tensor(kv_pages_shape, dtype),  # type: ignore
                cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
                cache_seqlens: T.Tensor([batch], T.int32),  # type: ignore
                block_table: T.Tensor(block_table_shape, T.int32),  # type: ignore
                cos_table: T.Tensor(rope_shape, dtype),  # type: ignore
                sin_table: T.Tensor(rope_shape, dtype),  # type: ignore
                max_seqlen_q: T.int32,  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(max_seqlen_q, block_m), heads_kv, batch, threads=threads) as (
                        bx, by, bz):
                q_start = cu_seqlens_q[bz]
                q_len = cu_seqlens_q[bz + 1] - q_start
                old_len = cache_seqlens[bz]
                append_start = old_len + bx * block_m
                append_end = append_start + block_m

                if bx * block_m + block_m <= q_len:
                    if append_start >> T.int32(page_size_log2) == (
                            append_end - 1) >> T.int32(page_size_log2):
                        page_idx = append_start >> T.int32(page_size_log2)
                        page_offset = append_start - page_idx * page_size
                        physical_start = block_table[bz, page_idx] * page_size + page_offset
                        for i, d in T.Parallel(block_m, dim):
                            new_pos = bx * block_m + i
                            logical_pos = old_len + new_pos
                            freq_idx = d % half
                            paired_d = T.if_then_else(
                                d < half, d + half,
                                T.if_then_else(d < rotary_dim, d - half, d))
                            c = cos_table[logical_pos, freq_idx]
                            s = sin_table[logical_pos, freq_idx]
                            val = k_new[q_start + new_pos, by, d]
                            paired_val = k_new[q_start + new_pos, by, paired_d]
                            rotated = T.if_then_else(d < half, -paired_val, paired_val)
                            k_pages[physical_start + i, by, d] = T.if_then_else(
                                d < rotary_dim, val * c + rotated * s, val)
                            v_pages[physical_start + i, by, d] = v_new[q_start + new_pos, by, d]
                    else:
                        for i, d in T.Parallel(block_m, dim):
                            new_pos = bx * block_m + i
                            logical_pos = old_len + new_pos
                            append_page_idx = logical_pos >> T.int32(page_size_log2)
                            append_page_offset = logical_pos - append_page_idx * page_size
                            physical_pos = (
                                block_table[bz, append_page_idx] * page_size
                                + append_page_offset
                            )
                            freq_idx = d % half
                            paired_d = T.if_then_else(
                                d < half, d + half,
                                T.if_then_else(d < rotary_dim, d - half, d))
                            c = cos_table[logical_pos, freq_idx]
                            s = sin_table[logical_pos, freq_idx]
                            val = k_new[q_start + new_pos, by, d]
                            paired_val = k_new[q_start + new_pos, by, paired_d]
                            rotated = T.if_then_else(d < half, -paired_val, paired_val)
                            k_pages[physical_pos, by, d] = T.if_then_else(
                                d < rotary_dim, val * c + rotated * s, val)
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
                            freq_idx = d % half
                            paired_d = T.if_then_else(
                                d < half, d + half,
                                T.if_then_else(d < rotary_dim, d - half, d))
                            c = cos_table[logical_pos, freq_idx]
                            s = sin_table[logical_pos, freq_idx]
                            val = k_new[q_start + new_pos, by, d]
                            paired_val = k_new[q_start + new_pos, by, paired_d]
                            rotated = T.if_then_else(d < half, -paired_val, paired_val)
                            k_pages[physical_pos, by, d] = T.if_then_else(
                                d < rotary_dim, val * c + rotated * s, val)
                            v_pages[physical_pos, by, d] = v_new[q_start + new_pos, by, d]

        return _gqa_prefill_paged_with_kv_cache_rope_append_main

    return _gqa_prefill_paged_with_kv_cache_rope_append_func


class _GQAPrefillPagedWithKVCacheRopeAppendKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch: int,
                 heads_kv: int,
                 max_pages_per_req: int,
                 page_size: int,
                 dim: int,
                 max_position: int,
                 rotary_dim: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        if page_size <= 0 or page_size & (page_size - 1) != 0:
            raise ValueError("page_size must be a positive power of two")
        if rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > dim:
            raise ValueError("rotary_dim must be positive, even, and <= dim")
        self.batch = batch
        self.heads_kv = heads_kv
        self.max_pages_per_req = max_pages_per_req
        self.page_size = page_size
        self.dim = dim
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self.dtype = dtype
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_m": 64, "threads": 128}

    def forward(self, k_new: torch.Tensor, v_new: torch.Tensor, k_pages: torch.Tensor,
                v_pages: torch.Tensor, cu_seqlens_q: torch.Tensor, cache_seqlens: torch.Tensor,
                block_table: torch.Tensor, max_seqlen_q: int, cos_table: torch.Tensor,
                sin_table: torch.Tensor) -> None:
        kernel = _gqa_prefill_paged_with_kv_cache_rope_append_kernel(
            self.batch, self.heads_kv, k_new.shape[0], k_pages.shape[0],
            self.max_pages_per_req, self.page_size, self.dim, self.max_position,
            self.rotary_dim, self.dtype_str)
        kernel(self.config["block_m"], self.config["threads"])(
            k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table,
            cos_table, sin_table, max_seqlen_q)



class _GQAPagedAttentionKernel(PagedPrefillKernel):
    """Thin policy wrapper around the shared paged-attention program."""

    fuse_rope: bool = False
    fp8_cache: bool = False
    native_fp8: bool = False

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
        return tile_stage_thread_configs()

    def _cache_dtype_str(self) -> str:
        return "float8_e4m3fn" if self.fp8_cache else self.dtype_str

    def _input_dtype_str(self) -> str:
        return "float8_e4m3fn" if self.native_fp8 else self.dtype_str

    def forward(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
        cos_table: Optional[torch.Tensor] = None,
        sin_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q_scale is None or k_scale is None or v_scale is None:
            raise ValueError("paged prefill requires q_scale, k_scale, and v_scale tensors")
        if cos_table is None or sin_table is None:
            raise ValueError("paged prefill requires prepared RoPE or dummy tables")
        return _gqa_paged_attention_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            q.shape[0],
            k_pages.shape[0],
            self.max_pages_per_req,
            self.page_size,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.window_size_left,
            self.window_size_right,
            self.append_kv,
            self._input_dtype_str(),
            self.dtype_str,
            self._cache_dtype_str(),
            self.fuse_rope,
            self.max_position or 1,
            self.rotary_dim or 0,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
            max_seqlen_q,
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            q_scale,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
            cos_table,
            sin_table,
        )


class GQAPrefillPagedWithKVCacheFwdKernel(_GQAPagedAttentionKernel):
    """Paged prefill whose cache matches the attention element type."""

    supported_archs: list[int] = [80, 89, 90]

    @classmethod
    def applies(cls, call) -> bool:
        return not call.is_fp8 and not call.fuse_rope and call.cache_dtype == call.dtype


class GQAPrefillPagedWithFP8KVCacheFwdKernel(_GQAPagedAttentionKernel):
    """Paged prefill with storage-only FP8 pages and online dequantization."""

    supported_archs: list[int] = [89, 90]
    fp8_cache: bool = True

    @classmethod
    def applies(cls, call) -> bool:
        return not call.is_fp8 and not call.fuse_rope and call.cache_dtype == fp8_dtype()


class GQAPrefillPagedNativeFP8TensorCoreFwdKernel(_GQAPagedAttentionKernel):
    """Paged ragged prefill with native FP8 Q/K/V Tensor Core math."""

    supported_archs: list[int] = [90]
    fp8_cache: bool = True
    native_fp8: bool = True

    @classmethod
    def applies(cls, call) -> bool:
        return (
            call.is_fp8
            and not call.fuse_rope
            and call.cache_dtype == fp8_dtype()
            and call.dim == 128
        )


class GQAPrefillPagedWithKVCacheRopeFwdKernel(_GQAPagedAttentionKernel):
    """Rotate/append the new K, then use the shared paged-attention body."""

    supported_archs: list[int] = [80, 89, 90]
    fuse_rope: bool = True

    @classmethod
    def applies(cls, call) -> bool:
        return not call.is_fp8 and bool(call.fuse_rope) and call.cache_dtype == call.dtype

    def _build_program(self) -> None:
        if self.rotary_dim is None or self.max_position is None:
            raise ValueError(
                "GQAPrefillPagedWithKVCacheRopeFwdKernel requires "
                "max_position and rotary_dim"
            )
        self._append = _GQAPrefillPagedWithKVCacheRopeAppendKernel(
            batch=self.batch,
            heads_kv=self.heads_kv,
            max_pages_per_req=self.max_pages_per_req,
            page_size=self.page_size,
            dim=self.dim,
            max_position=self.max_position,
            rotary_dim=self.rotary_dim,
            dtype=self.dtype,
        )

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        super().autotune(warmup=warmup, rep=rep)
        self._append.autotune(warmup=warmup, rep=rep)

    def forward(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
        cos_table: Optional[torch.Tensor] = None,
        sin_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cos_table is None or sin_table is None:
            raise ValueError(
                "GQAPrefillPagedWithKVCacheRopeFwdKernel requires rotary tables"
            )
        if self.append_kv:
            self._append(
                k_new,
                v_new,
                k_pages,
                v_pages,
                cu_seqlens_q,
                cache_seqlens,
                block_table,
                max_seqlen_q,
                cos_table,
                sin_table,
            )
        return super().forward(
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            q_scale,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
            max_seqlen_q,
            cos_table,
            sin_table,
        )
