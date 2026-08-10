import functools
import os
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.online_softmax import (
    make_log2e_scale,
    make_online_softmax_with_score_scale,
)

from .call_spec import ATTENTION_DTYPES
from .fp8_prefill_core import make_native_fp8_prefill_tile_update
from .packed_prefill import PackedPrefillKernel
from .prefill_rope import make_prefill_rope_policy

__all__ = ["GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel"]
NUM_SMS = int(os.environ.get("V2P_NUM_SMS", "132"))
TMA_DTYPE_UINT8 = 0
TMA_INTERLEAVE_NONE = 0
TMA_SWIZZLE_128B = 3
TMA_L2_PROMOTION_128B = 2
TMA_OOB_FILL_NONE = 0
_FP8_GQA_HELPER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "_fp8_gqa_helper.h"))


def _make_fa3_pv_acc_fragment(dim: int, thread_offset: int) -> tilelang.layout.Fragment:
    col_phase = dim // 8

    def forward_fn(i, j):
        rv = j // 4
        thread = thread_offset + (i // 16) * 32 + (i % 8) * 4 + (j % 4)
        index = (rv % col_phase) * 4 + ((i % 16) // 8) * 2 + rv // col_phase
        return thread, index

    if dim != 128:
        raise ValueError("FA3 PV accumulator fragment annotation requires dim == 128.")
    return tilelang.layout.Fragment([64, dim], forward_fn=forward_fn)


def _make_fa3_qk_acc_fragment(block_n: int, thread_offset: int) -> tilelang.layout.Fragment:
    col_phase = block_n // 8

    def forward_fn(i, j):
        rv = j // 4
        thread = thread_offset + (i // 16) * 32 + (i % 8) * 4 + (j % 4)
        index = (rv % col_phase) * 4 + ((i % 16) // 8) * 2 + rv // col_phase
        return thread, index

    if block_n != 224:
        raise ValueError("FA3 QK accumulator fragment annotation requires block_n == 224.")
    return tilelang.layout.Fragment([64, block_n], forward_fn=forward_fn)


def _make_fa3_qk_row_fragment(thread_offset: int) -> tilelang.layout.Fragment:
    def forward_fn(i, rep):
        thread = thread_offset + (i // 16) * 32 + (i % 8) * 4 + rep
        index = (i % 16) // 8
        return thread, index

    return tilelang.layout.Fragment([64], forward_fn=forward_fn, replicate=4)


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_general_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    out_dtype: str,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
) -> Callable:
    """Native-FP8 dense attention for semantic/tail regions outside BN224."""
    groups = heads // heads_kv
    offset = seq_len_kv - seq_len_q
    accum_dtype = "float"
    fp8 = "float8_e4m3fn"
    if fuse_rope and (rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > dim):
        raise ValueError("rotary_dim must be positive, even, and <= dim")
    if rope_layout not in ("neox", "interleaved"):
        raise ValueError("rope_layout must be 'neox' or 'interleaved'")

    @tilelang.jit(
        out_idx=[8],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "--use_fast_math", "-DENABLE_BF16"],
    )
    def build(block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:
        q_shape = (batch, seq_len_q, heads, dim)
        kv_shape = (batch, seq_len_kv, heads_kv, dim)
        scale_shape = (batch, heads_kv)
        out_shape = (batch, seq_len_q, heads, dim)
        rope_cols, paired_dim, rotate = make_prefill_rope_policy(fuse_rope, rotary_dim, rope_layout)
        rope_shape = (max_position if fuse_rope else 1, rope_cols)
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

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, fp8),  # type: ignore
            k: T.Tensor(kv_shape, fp8),  # type: ignore
            v: T.Tensor(kv_shape, fp8),  # type: ignore
            q_scale: T.Tensor(scale_shape, T.float32),  # type: ignore
            k_scale: T.Tensor(scale_shape, T.float32),  # type: ignore
            v_scale: T.Tensor(scale_shape, T.float32),  # type: ignore
            cos_table: T.Tensor(rope_shape, out_dtype),  # type: ignore
            sin_table: T.Tensor(rope_shape, out_dtype),  # type: ignore
            output: T.Tensor(out_shape, out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(seq_len_q, block_m), heads, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                q_shared = T.alloc_shared([block_m, dim], fp8)
                k_shared = T.alloc_shared([block_n, dim], fp8)
                v_shared = T.alloc_shared([block_n, dim], fp8)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                prob_fp8 = T.alloc_shared([block_m, block_n], fp8)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                kv_head = by // groups
                score_descale = q_scale[bz, kv_head] * k_scale[bz, kv_head]
                value_descale = v_scale[bz, kv_head]
                for i, d in T.Parallel(block_m, dim):
                    q_pos = bx * block_m + i
                    if q_pos < seq_len_q:
                        if fuse_rope:
                            paired_d = paired_dim(d)
                            q_shared[i, d] = rotate(
                                q[bz, q_pos, by, d],
                                q[bz, q_pos, by, paired_d],
                                offset + q_pos,
                                d,
                                cos_table,
                                sin_table,
                            )
                        else:
                            q_shared[i, d] = q[bz, q_pos, by, d]
                    else:
                        q_shared[i, d] = T.cast(0, fp8)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                if is_causal:
                    k_end = T.ceildiv(T.min(seq_len_kv, (bx + 1) * block_m + offset), block_n)
                elif window_size_right >= 0:
                    k_end = T.ceildiv(
                        T.min(
                            seq_len_kv,
                            (bx + 1) * block_m + offset + window_size_right,
                        ),
                        block_n,
                    )
                else:
                    k_end = T.ceildiv(seq_len_kv, block_n)
                if window_size_left >= 0:
                    k_start = T.max(0, bx * block_m + offset - window_size_left) // block_n
                else:
                    k_start = 0

                for k_offset in T.Pipelined(T.max(k_end - k_start, 0), num_stages=num_stages):
                    k_idx = k_start + k_offset
                    for j, d in T.Parallel(block_n, dim):
                        kv_pos = k_idx * block_n + j
                        if kv_pos < seq_len_kv:
                            if fuse_rope:
                                paired_d = paired_dim(d)
                                k_shared[j, d] = rotate(
                                    k[bz, kv_pos, kv_head, d],
                                    k[bz, kv_pos, kv_head, paired_d],
                                    kv_pos,
                                    d,
                                    cos_table,
                                    sin_table,
                                )
                            else:
                                k_shared[j, d] = k[bz, kv_pos, kv_head, d]
                            v_shared[j, d] = v[bz, kv_pos, kv_head, d]
                        else:
                            k_shared[j, d] = T.cast(0, fp8)
                            v_shared[j, d] = T.cast(0, fp8)

                    fp8_tile_update(
                        q_shared,
                        k_shared,
                        v_shared,
                        acc_s,
                        prob_fp8,
                        acc_o,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                        k_idx,
                        bx,
                        seq_len_q,
                        seq_len_kv,
                        offset,
                        score_descale * sm_scale,
                    )

                for i, d in T.Parallel(block_m, dim):
                    if bx * block_m + i < seq_len_q:
                        output[bz, bx * block_m + i, by, d] = (
                            acc_o[i, d] * value_descale / logsum[i]
                        )

        return main

    return build


@torch.library.custom_op("top::gqa_fwd_fp8_general_wrapped_kernel", mutates_args=())
def _gqa_fwd_fp8_general_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    out_dtype: str,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    return _gqa_fwd_fp8_general_kernel(
        batch,
        heads,
        heads_kv,
        seq_len_q,
        seq_len_kv,
        dim,
        is_causal,
        sm_scale,
        softcap,
        window_size_left,
        window_size_right,
        out_dtype,
        fuse_rope,
        max_position,
        rotary_dim,
        rope_layout,
    )(block_m, block_n, num_stages, threads)(q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin)


@_gqa_fwd_fp8_general_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    out_dtype: str,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    *inputs: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    dtype = torch.float16 if out_dtype == "float16" else torch.bfloat16
    return torch.empty((batch, seq_len_q, heads, dim), dtype=dtype, device=inputs[0].device)


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_bn224_tma_v_kernel(
    batch: int, heads: int, heads_kv: int, seq_len: int, dim: int, out_dtype: str
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError(
            "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel currently requires dim == 128."
        )
    if seq_len % 224 != 0:
        raise ValueError(
            "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel currently requires seq_len % 224 == 0."
        )
    if seq_len % 128 != 0:
        raise ValueError(
            "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel currently requires seq_len % 128 == 0."
        )
    block_m = 128
    half_m = block_m // 2
    groups = heads // heads_kv
    accum_dtype = "float"
    fp8_dtype = "float8_e4m3fn"
    scale = make_log2e_scale(dim)
    defer_row_sum = (seq_len + 223) // 224 < 32

    @T.macro
    def online_softmax_with_partial_sum(
        acc_s,
        scores_max,
        scores_max_prev,
        scores_scale,
        scores_sum,
        logsum,
        score_scale,
    ):
        score_scale_softmax = score_scale * scale
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(half_m):
            scores_max[i] *= score_scale
        for i in T.Parallel(half_m):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(half_m, 224):
            acc_s[i, j] = T.exp2(acc_s[i, j] * score_scale_softmax - scores_max[i] * scale)
        # Accumulate lane-local row sums here; the quad reduction is deferred
        # until finalization instead of running once per K/V tile.
        T.call_extern(
            "handle",
            "tl::fp8_partial_row_sum_raw_acc_64x224",
            acc_s.data,
            scores_sum.data,
        )
        for i in T.Parallel(half_m):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    @tilelang.jit(
        out_idx=[6, 7],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-include",
            _FP8_GQA_HELPER_PATH,
        ],
    )
    def func():
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        descale_shape = (batch, heads_kv)
        # The 32-tile S7168 producer/consumer schedule still needs the original
        # per-tile quad reduction. Shorter schedules use the deferred reduction
        # without changing the public dispatch contract.
        if not defer_row_sum:
            online_softmax_1 = make_online_softmax_with_score_scale(scale, accum_dtype, half_m, 224)
            online_softmax_2 = make_online_softmax_with_score_scale(scale, accum_dtype, half_m, 224)
        else:
            online_softmax_1 = online_softmax_with_partial_sum
            online_softmax_2 = online_softmax_with_partial_sum
        pv_begin_accumulate_helper = "tl::fp8_pv_ptx_unit_begin_accumulate_fa3_raw_64x128x224"
        v_inplace_transform_helper = (
            "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm_barrier_each_iter"
        )

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, fp8_dtype),
            k: T.Tensor(kv_shape, fp8_dtype),
            v: T.Tensor(kv_shape, fp8_dtype),
            q_descale: T.Tensor(descale_shape, accum_dtype),
            k_descale: T.Tensor(descale_shape, accum_dtype),
            v_descale: T.Tensor(descale_shape, accum_dtype),
            output: T.Tensor(q_shape, out_dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ) -> None:
            with T.Kernel(NUM_SMS, 1, 1, threads=384) as (bx, _by, _bz):
                q_shared_1 = T.alloc_shared([half_m, dim], fp8_dtype)
                q_shared_2 = T.alloc_shared([half_m, dim], fp8_dtype)
                k_smem_0 = T.alloc_shared([224, dim], fp8_dtype)
                k_smem_1 = T.alloc_shared([224, dim], fp8_dtype)
                v_vt_smem_0 = T.alloc_shared([dim, 224], fp8_dtype)
                v_vt_smem_1 = T.alloc_shared([dim, 224], fp8_dtype)
                v_tc_smem_0 = v_vt_smem_0
                v_tc_smem_1 = v_vt_smem_1
                o_shared_1 = T.alloc_shared([half_m, dim], out_dtype)
                o_shared_2 = T.alloc_shared([half_m, dim], out_dtype)
                ss_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ss_shared_2 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_2 = T.alloc_shared([half_m], accum_dtype)
                # Valid sequence lengths make loop_range a multiple of four, so
                # K/V phases reset per work item; only the initial V warm-up persists.
                producer_warm_shared = T.alloc_shared([4], "int32")
                acc_s_1 = T.alloc_fragment([half_m, 224], accum_dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)
                acc_s_2 = T.alloc_fragment([half_m, 224], accum_dtype)
                acc_o_2 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_2 = T.alloc_fragment([half_m], accum_dtype)
                smp_2 = T.alloc_fragment([half_m], accum_dtype)
                ss_2 = T.alloc_fragment([half_m], accum_dtype)
                ssum_2 = T.alloc_fragment([half_m], accum_dtype)
                ls_2 = T.alloc_fragment([half_m], accum_dtype)
                k_full = T.alloc_barrier(arrive_count=128)
                k_empty = T.alloc_barrier(arrive_count=256)
                v_raw_full = T.alloc_barrier(arrive_count=128)
                v_full_0 = T.alloc_barrier(arrive_count=128)
                v_full_1 = T.alloc_barrier(arrive_count=128)
                v_empty_0 = T.alloc_barrier(arrive_count=256)
                v_empty_1 = T.alloc_barrier(arrive_count=256)
                q_full_1 = T.alloc_barrier(arrive_count=128)
                q_full_2 = T.alloc_barrier(arrive_count=128)
                T.annotate_layout(
                    {
                        q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                        q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                        k_smem_0: tilelang.layout.make_swizzled_layout(k_smem_0),
                        k_smem_1: tilelang.layout.make_swizzled_layout(k_smem_1),
                        acc_s_1: _make_fa3_qk_acc_fragment(224, 128),
                        acc_s_2: _make_fa3_qk_acc_fragment(224, 256),
                        sm_1: _make_fa3_qk_row_fragment(128),
                        smp_1: _make_fa3_qk_row_fragment(128),
                        ss_1: _make_fa3_qk_row_fragment(128),
                        ssum_1: _make_fa3_qk_row_fragment(128),
                        ls_1: _make_fa3_qk_row_fragment(128),
                        sm_2: _make_fa3_qk_row_fragment(256),
                        smp_2: _make_fa3_qk_row_fragment(256),
                        ss_2: _make_fa3_qk_row_fragment(256),
                        ssum_2: _make_fa3_qk_row_fragment(256),
                        ls_2: _make_fa3_qk_row_fragment(256),
                        acc_o_1: _make_fa3_pv_acc_fragment(dim, 128),
                        acc_o_2: _make_fa3_pv_acc_fragment(dim, 256),
                    }
                )
                T.clear(producer_warm_shared)
                T.sync_threads()
                gi_kc1 = T.alloc_var("int32", init=0)
                gi_vc1 = T.alloc_var("int32", init=0)
                gi_kc2 = T.alloc_var("int32", init=0)
                gi_vc2 = T.alloc_var("int32", init=0)
                gi_q1 = T.alloc_var("int32", init=0)
                gi_q2 = T.alloc_var("int32", init=0)
                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    for tile_b, tile_hkv, _tile_m, _tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len, block_m), groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_hkv
                        loop_range = T.ceildiv(seq_len, 224)
                        producer_warm = T.alloc_var("int32", init=producer_warm_shared[tx // 32])
                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            if n_idx > 0:
                                if producer_warm != 0 or n_idx >= 3:
                                    if (n_idx - 1) % 2 == 0:
                                        T.barrier_wait(v_empty_0, ((n_idx - 1) // 2 - 1) % 2)
                                    else:
                                        T.barrier_wait(v_empty_1, ((n_idx - 1) // 2 - 1) % 2)
                                if (n_idx - 1) % 2 == 0:
                                    if tx == 0:
                                        T.mbarrier_expect_tx(v_raw_full, dim * 224)
                                        v_desc = T.create_tma_descriptor(
                                            TMA_DTYPE_UINT8,
                                            4,
                                            v.data,
                                            dim,
                                            heads_kv,
                                            seq_len,
                                            batch,
                                            1,
                                            dim,
                                            heads_kv * dim,
                                            seq_len * heads_kv * dim,
                                            dim,
                                            1,
                                            224,
                                            1,
                                            1,
                                            1,
                                            1,
                                            1,
                                            TMA_INTERLEAVE_NONE,
                                            TMA_SWIZZLE_128B,
                                            TMA_L2_PROMOTION_128B,
                                            TMA_OOB_FILL_NONE,
                                        )
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_tma_load_4d_ptx",
                                            v_desc,
                                            v_raw_full[0],
                                            T.access_ptr(v_vt_smem_0, "w"),
                                            0,
                                            head_kv,
                                            (n_idx - 1) * 224,
                                            tile_b,
                                        )
                                    T.barrier_arrive(v_raw_full)
                                    T.barrier_wait(v_raw_full, (n_idx - 1) % 2)
                                    T.call_extern(
                                        "handle",
                                        v_inplace_transform_helper,
                                        v_vt_smem_0.access_ptr("rw"),
                                        v_vt_smem_0.access_ptr("rw"),
                                    )
                                else:
                                    if tx == 0:
                                        T.mbarrier_expect_tx(v_raw_full, dim * 224)
                                        v_desc = T.create_tma_descriptor(
                                            TMA_DTYPE_UINT8,
                                            4,
                                            v.data,
                                            dim,
                                            heads_kv,
                                            seq_len,
                                            batch,
                                            1,
                                            dim,
                                            heads_kv * dim,
                                            seq_len * heads_kv * dim,
                                            dim,
                                            1,
                                            224,
                                            1,
                                            1,
                                            1,
                                            1,
                                            1,
                                            TMA_INTERLEAVE_NONE,
                                            TMA_SWIZZLE_128B,
                                            TMA_L2_PROMOTION_128B,
                                            TMA_OOB_FILL_NONE,
                                        )
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_tma_load_4d_ptx",
                                            v_desc,
                                            v_raw_full[0],
                                            T.access_ptr(v_vt_smem_1, "w"),
                                            0,
                                            head_kv,
                                            (n_idx - 1) * 224,
                                            tile_b,
                                        )
                                    T.barrier_arrive(v_raw_full)
                                    T.barrier_wait(v_raw_full, (n_idx - 1) % 2)
                                    T.call_extern(
                                        "handle",
                                        v_inplace_transform_helper,
                                        v_vt_smem_1.access_ptr("rw"),
                                        v_vt_smem_1.access_ptr("rw"),
                                    )
                                if (n_idx - 1) % 2 == 0:
                                    T.barrier_arrive(v_full_0)
                                else:
                                    T.barrier_arrive(v_full_1)
                            T.barrier_wait(k_empty, (n_idx + 1) % 2)
                            if n_idx % 2 == 0:
                                T.tma_copy(
                                    k[tile_b, n_idx * 224 : (n_idx + 1) * 224, head_kv, :],
                                    k_smem_0,
                                    barrier=k_full,
                                )
                            else:
                                T.tma_copy(
                                    k[tile_b, n_idx * 224 : (n_idx + 1) * 224, head_kv, :],
                                    k_smem_1,
                                    barrier=k_full,
                                )
                            T.barrier_arrive(k_full)
                        if producer_warm != 0 or loop_range - 1 >= 2:
                            if (loop_range - 1) % 2 == 0:
                                T.barrier_wait(v_empty_0, ((loop_range - 1) // 2 - 1) % 2)
                            else:
                                T.barrier_wait(v_empty_1, ((loop_range - 1) // 2 - 1) % 2)
                        if (loop_range - 1) % 2 == 0:
                            if tx == 0:
                                T.mbarrier_expect_tx(v_raw_full, dim * 224)
                                v_desc_tail = T.create_tma_descriptor(
                                    TMA_DTYPE_UINT8,
                                    4,
                                    v.data,
                                    dim,
                                    heads_kv,
                                    seq_len,
                                    batch,
                                    1,
                                    dim,
                                    heads_kv * dim,
                                    seq_len * heads_kv * dim,
                                    dim,
                                    1,
                                    224,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    TMA_INTERLEAVE_NONE,
                                    TMA_SWIZZLE_128B,
                                    TMA_L2_PROMOTION_128B,
                                    TMA_OOB_FILL_NONE,
                                )
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_tma_load_4d_ptx",
                                    v_desc_tail,
                                    v_raw_full[0],
                                    T.access_ptr(v_vt_smem_0, "w"),
                                    0,
                                    head_kv,
                                    (loop_range - 1) * 224,
                                    tile_b,
                                )
                            T.barrier_arrive(v_raw_full)
                            T.barrier_wait(v_raw_full, (loop_range - 1) % 2)
                            T.call_extern(
                                "handle",
                                v_inplace_transform_helper,
                                v_vt_smem_0.access_ptr("rw"),
                                v_vt_smem_0.access_ptr("rw"),
                            )
                        else:
                            if tx == 0:
                                T.mbarrier_expect_tx(v_raw_full, dim * 224)
                                v_desc_tail = T.create_tma_descriptor(
                                    TMA_DTYPE_UINT8,
                                    4,
                                    v.data,
                                    dim,
                                    heads_kv,
                                    seq_len,
                                    batch,
                                    1,
                                    dim,
                                    heads_kv * dim,
                                    seq_len * heads_kv * dim,
                                    dim,
                                    1,
                                    224,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    TMA_INTERLEAVE_NONE,
                                    TMA_SWIZZLE_128B,
                                    TMA_L2_PROMOTION_128B,
                                    TMA_OOB_FILL_NONE,
                                )
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_tma_load_4d_ptx",
                                    v_desc_tail,
                                    v_raw_full[0],
                                    T.access_ptr(v_vt_smem_1, "w"),
                                    0,
                                    head_kv,
                                    (loop_range - 1) * 224,
                                    tile_b,
                                )
                            T.barrier_arrive(v_raw_full)
                            T.barrier_wait(v_raw_full, (loop_range - 1) % 2)
                            T.call_extern(
                                "handle",
                                v_inplace_transform_helper,
                                v_vt_smem_1.access_ptr("rw"),
                                v_vt_smem_1.access_ptr("rw"),
                            )
                        if (loop_range - 1) % 2 == 0:
                            T.barrier_arrive(v_full_0)
                        else:
                            T.barrier_arrive(v_full_1)
                        producer_warm_shared[tx // 32] = 1
                        if groups == 8 and defer_row_sum:
                            T.sync_threads(barrier_id=5, arrive_count=384)
                elif tx < 256:
                    T.inc_max_nreg(240)
                    for tile_b, tile_hkv, tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len, block_m), groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        tile_h = tile_hkv * groups + tile_g
                        head_kv = tile_hkv
                        row_base = tile_m * block_m
                        loop_range = T.ceildiv(seq_len, 224)
                        qk_descale = T.alloc_var(
                            accum_dtype,
                            init=q_descale[tile_b, head_kv] * k_descale[tile_b, head_kv],
                        )
                        value_descale = T.alloc_var(accum_dtype, init=v_descale[tile_b, head_kv])
                        T.tma_copy(
                            q[tile_b, row_base : row_base + half_m, tile_h, :],
                            q_shared_1,
                            barrier=q_full_1,
                        )
                        T.barrier_arrive(q_full_1)
                        T.barrier_wait(q_full_1, gi_q1 % 2)
                        gi_q1 = gi_q1 + 1
                        T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o_1.data)
                        T.clear(ls_1)
                        T.fill(sm_1, -T.infinity(accum_dtype))
                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc1 % 2)
                            if gi_kc1 % 2 == 0:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_qk_cute_grouped_fa3_raw_64x224x128",
                                    q_shared_1.access_ptr("r"),
                                    k_smem_0.access_ptr("r"),
                                    acc_s_1.data,
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_qk_cute_grouped_fa3_raw_64x224x128",
                                    q_shared_1.access_ptr("r"),
                                    k_smem_1.access_ptr("r"),
                                    acc_s_1.data,
                                )
                            if n_idx > 0:
                                T.wait_wgmma(1)
                                T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                if gi_vc1 % 2 == 0:
                                    T.barrier_arrive(v_empty_0)
                                else:
                                    T.barrier_arrive(v_empty_1)
                                gi_vc1 = gi_vc1 + 1
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                            T.barrier_arrive(k_empty)
                            gi_kc1 = gi_kc1 + 1
                            online_softmax_1(
                                acc_s_1,
                                sm_1,
                                smp_1,
                                ss_1,
                                ssum_1,
                                ls_1,
                                qk_descale,
                            )
                            T.copy(ss_1, ss_shared_1)
                            # The row-scale fragment is compacted through one
                            # lane per quad before the full consumer warpgroup
                            # reads it while rescaling the PV accumulator.
                            if groups == 8 and defer_row_sum:
                                T.sync_threads(barrier_id=6, arrive_count=128)
                            T.call_extern(
                                "handle",
                                "tl::fp8_fa3_raw_acc_rescale_keep_ptx_layout_64x128",
                                acc_o_1.data,
                                ss_shared_1.access_ptr("r"),
                            )
                            if gi_vc1 % 2 == 0:
                                T.barrier_wait(v_full_0, (gi_vc1 // 2) % 2)
                            else:
                                T.barrier_wait(v_full_1, (gi_vc1 // 2) % 2)
                            if gi_vc1 % 2 == 0:
                                T.call_extern(
                                    "handle",
                                    pv_begin_accumulate_helper,
                                    acc_s_1.data,
                                    v_tc_smem_0.access_ptr("r"),
                                    acc_o_1.data,
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    pv_begin_accumulate_helper,
                                    acc_s_1.data,
                                    v_tc_smem_1.access_ptr("r"),
                                    acc_o_1.data,
                                )
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                        if gi_vc1 % 2 == 0:
                            T.barrier_arrive(v_empty_0)
                        else:
                            T.barrier_arrive(v_empty_1)
                        gi_vc1 = gi_vc1 + 1
                        if defer_row_sum:
                            # Match FA3's reduction schedule: combine the four
                            # lane partials once after all tiles are consumed.
                            for i in T.Parallel(half_m):
                                ls_1[i] = ls_1[i] + T.shfl_xor(ls_1[i], 1)
                                ls_1[i] = ls_1[i] + T.shfl_xor(ls_1[i], 2)
                        T.copy(ls_1, ls_shared_1)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_finalize_store_smem_cute_64x128",
                            acc_o_1.data,
                            ls_shared_1.access_ptr("r"),
                            4,
                            value_descale,
                            o_shared_1.access_ptr("w"),
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=3, arrive_count=128)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_o_smem_store_global_cute_64x128",
                            o_shared_1.access_ptr("r"),
                            T.address_of(output[tile_b, row_base, tile_h, 0]),
                            heads * dim,
                        )
                        for i in T.Parallel(half_m):
                            ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * scale
                        T.copy(ls_1, lse[tile_b, tile_h, row_base : row_base + half_m])
                        if groups == 8 and defer_row_sum:
                            T.sync_threads(barrier_id=5, arrive_count=384)
                else:
                    T.inc_max_nreg(240)
                    for tile_b, tile_hkv, tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len, block_m), groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        tile_h = tile_hkv * groups + tile_g
                        head_kv = tile_hkv
                        row_base = tile_m * block_m
                        loop_range = T.ceildiv(seq_len, 224)
                        qk_descale = T.alloc_var(
                            accum_dtype,
                            init=q_descale[tile_b, head_kv] * k_descale[tile_b, head_kv],
                        )
                        value_descale = T.alloc_var(accum_dtype, init=v_descale[tile_b, head_kv])
                        T.tma_copy(
                            q[tile_b, row_base + half_m : row_base + block_m, tile_h, :],
                            q_shared_2,
                            barrier=q_full_2,
                        )
                        T.barrier_arrive(q_full_2)
                        T.barrier_wait(q_full_2, gi_q2 % 2)
                        gi_q2 = gi_q2 + 1
                        T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o_2.data)
                        T.clear(ls_2)
                        T.fill(sm_2, -T.infinity(accum_dtype))
                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc2 % 2)
                            if gi_kc2 % 2 == 0:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_qk_cute_grouped_fa3_raw_64x224x128",
                                    q_shared_2.access_ptr("r"),
                                    k_smem_0.access_ptr("r"),
                                    acc_s_2.data,
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_qk_cute_grouped_fa3_raw_64x224x128",
                                    q_shared_2.access_ptr("r"),
                                    k_smem_1.access_ptr("r"),
                                    acc_s_2.data,
                                )
                            if n_idx > 0:
                                T.wait_wgmma(1)
                                T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                if gi_vc2 % 2 == 0:
                                    T.barrier_arrive(v_empty_0)
                                else:
                                    T.barrier_arrive(v_empty_1)
                                gi_vc2 = gi_vc2 + 1
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                            T.barrier_arrive(k_empty)
                            gi_kc2 = gi_kc2 + 1
                            online_softmax_2(
                                acc_s_2,
                                sm_2,
                                smp_2,
                                ss_2,
                                ssum_2,
                                ls_2,
                                qk_descale,
                            )
                            T.copy(ss_2, ss_shared_2)
                            if groups == 8 and defer_row_sum:
                                T.sync_threads(barrier_id=7, arrive_count=128)
                            T.call_extern(
                                "handle",
                                "tl::fp8_fa3_raw_acc_rescale_keep_ptx_layout_64x128",
                                acc_o_2.data,
                                ss_shared_2.access_ptr("r"),
                            )
                            if gi_vc2 % 2 == 0:
                                T.barrier_wait(v_full_0, (gi_vc2 // 2) % 2)
                            else:
                                T.barrier_wait(v_full_1, (gi_vc2 // 2) % 2)
                            if gi_vc2 % 2 == 0:
                                T.call_extern(
                                    "handle",
                                    pv_begin_accumulate_helper,
                                    acc_s_2.data,
                                    v_tc_smem_0.access_ptr("r"),
                                    acc_o_2.data,
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    pv_begin_accumulate_helper,
                                    acc_s_2.data,
                                    v_tc_smem_1.access_ptr("r"),
                                    acc_o_2.data,
                                )
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                        if gi_vc2 % 2 == 0:
                            T.barrier_arrive(v_empty_0)
                        else:
                            T.barrier_arrive(v_empty_1)
                        gi_vc2 = gi_vc2 + 1
                        if defer_row_sum:
                            # Match FA3's reduction schedule: combine the four
                            # lane partials once after all tiles are consumed.
                            for i in T.Parallel(half_m):
                                ls_2[i] = ls_2[i] + T.shfl_xor(ls_2[i], 1)
                                ls_2[i] = ls_2[i] + T.shfl_xor(ls_2[i], 2)
                        T.copy(ls_2, ls_shared_2)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_finalize_store_smem_cute_64x128",
                            acc_o_2.data,
                            ls_shared_2.access_ptr("r"),
                            4,
                            value_descale,
                            o_shared_2.access_ptr("w"),
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=4, arrive_count=128)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_o_smem_store_global_cute_64x128",
                            o_shared_2.access_ptr("r"),
                            T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                            heads * dim,
                        )
                        for i in T.Parallel(half_m):
                            ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                        T.copy(ls_2, lse[tile_b, tile_h, row_base + half_m : row_base + block_m])
                        if groups == 8 and defer_row_sum:
                            T.sync_threads(barrier_id=5, arrive_count=384)

        return main

    return func


@torch.library.custom_op("top::gqa_fwd_fp8_bn224_tma_v_wrapped_kernel", mutates_args=())
def _gqa_fwd_fp8_bn224_tma_v_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_fp8_bn224_tma_v_kernel(batch, heads, heads_kv, seq_len, dim, out_dtype)()(
        q, k, v, q_descale, k_descale, v_descale
    )


@_gqa_fwd_fp8_bn224_tma_v_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    *inputs: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch_dtype = torch.float16 if out_dtype == "float16" else torch.bfloat16
    fake_o = torch.empty((batch, seq_len, heads, dim), dtype=torch_dtype, device=inputs[0].device)
    fake_lse = torch.empty((batch, heads, seq_len), dtype=torch.float32, device=inputs[0].device)
    return (fake_o, fake_lse)


def _validate_fa3_gqa_descales(
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    batch: int,
    heads_kv: int,
    device: torch.device,
) -> None:
    """Validate the direct FA3 ``[batch, heads_kv]`` descale contract."""
    expected_shape = (batch, heads_kv)
    for name, descale in (
        ("q_scale", q_descale),
        ("k_scale", k_descale),
        ("v_scale", v_descale),
    ):
        if tuple(descale.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(descale.shape)}."
            )
        if descale.dtype != torch.float32:
            raise ValueError(f"{name} must have dtype torch.float32, got {descale.dtype}.")
        if descale.device != device:
            raise ValueError(f"{name} must be on {device}, got {descale.device}.")
        if not descale.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")


class GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(PackedPrefillKernel):
    """Native-FP8 dense GQA family with a BN224 fast schedule.

    Query, key and value are always ``torch.float8_e4m3fn``; the only free dtype
    is the one the kernel writes, so ``dtype`` names the output element type.

    The block schedule is fixed by the PTX contract this kernel implements
    (BN224 warp specialization), so ``default_config`` is empty and
    ``autotune_configs`` is undefined: ``tune=True`` degrades to the default
    config with a warning from ``Kernel.init_config``.

    The BN224 producer/consumer schedule remains the fast region. Causal,
    windowed, softcap, rectangular, and tail shapes stay in this native-FP8
    family and use its general Tensor Core schedule; they never dequantize the
    full inputs into a 16-bit attention fallback.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        heads_kv: Number of key/value heads.
        max_seqlen_q: Query sequence length.
        max_seqlen_kv: Key/value sequence length.
        dim: Head dimension, which must be 128.
        is_causal: Whether to apply bottom-right causal masking.
        dtype: Output dtype, ``torch.float16`` or ``torch.bfloat16``.
        config: Optional config dict. This kernel exposes no tunable knobs.
        tune: Whether to autotune. No-op for this kernel; see above.
    """

    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call) -> bool:
        return call.is_fp8 and call.is_uniform and call.dim == 128

    def _validate_spec(self) -> None:
        if self.dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel currently requires dim == 128."
            )
        if self.is_causal and self.max_seqlen_q > self.max_seqlen_kv:
            raise ValueError("causal FP8 prefill requires max_seqlen_q <= max_seqlen_kv")
        if self.dtype not in ATTENTION_DTYPES:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel outputs float16 or bfloat16."
            )

    def _build_program(self) -> None:
        # Built inside the wrapped custom op; the schedule is fixed by contract.
        self.kernel = None

    def _uses_bn224_fast_schedule(self) -> bool:
        return (
            not self.fuse_rope
            and not self.is_causal
            and not (self.window_size_left != -1 or self.window_size_right != -1)
            and self.softcap == 0.0
            and self.max_seqlen_q == self.max_seqlen_kv
            and self.max_seqlen_q % 224 == 0
            and self.max_seqlen_q % 128 == 0
            and self.sm_scale == self.dim**-0.5
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fp8 = getattr(torch, "float8_e4m3fn", None)
        if fp8 is None:
            raise ValueError("torch.float8_e4m3fn is required for this kernel.")
        if q.dtype != fp8 or k.dtype != fp8 or v.dtype != fp8:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel expects q/k/v to be "
                "torch.float8_e4m3fn."
            )
        if q_scale is None or k_scale is None or v_scale is None:
            raise ValueError("GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel requires q/k/v descales.")
        if rope_cos is None or rope_sin is None:
            raise ValueError("native FP8 prefill requires prepared RoPE or dummy tables")
        q_bshd, k_bshd, v_bshd = self._bshd(q, k, v)
        if self._uses_bn224_fast_schedule():
            _validate_fa3_gqa_descales(
                q_scale,
                k_scale,
                v_scale,
                self.batch,
                self.heads_kv,
                q_bshd.device,
            )
            output, _ = _gqa_fwd_fp8_bn224_tma_v_wrapped_kernel(
                self.batch,
                self.heads,
                self.heads_kv,
                self.max_seqlen_q,
                self.dim,
                self.dtype_str,
                q_bshd,
                k_bshd,
                v_bshd,
                q_scale,
                k_scale,
                v_scale,
            )
        else:
            output = _gqa_fwd_fp8_general_wrapped_kernel(
                self.batch,
                self.heads,
                self.heads_kv,
                self.max_seqlen_q,
                self.max_seqlen_kv,
                self.dim,
                self.is_causal,
                self.sm_scale,
                self.softcap,
                self.window_size_left,
                self.window_size_right,
                self.dtype_str,
                self.fuse_rope,
                self.max_position or 1,
                self.rotary_dim or 0,
                self.rope_layout,
                64,
                64,
                1,
                128,
                q_bshd,
                k_bshd,
                v_bshd,
                q_scale,
                k_scale,
                v_scale,
                rope_cos,
                rope_sin,
            )
        return output.reshape(q.shape)
