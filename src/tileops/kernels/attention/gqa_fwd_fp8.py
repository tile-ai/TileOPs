import functools
import os
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from ..kernel_base import Kernel
from .call_spec import ATTENTION_DTYPES
from .gqa_dense import make_dense_qk_rope_preprocessor
from .online_softmax import (
    LOG2E,
    make_online_softmax_with_score_scale,
)

__all__ = ["GQADenseFP8Kernel"]
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
def _gqa_fwd_fp8_bn224_tma_v_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    out_dtype: str,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    write_lse: bool,
    num_sms: int,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("native-FP8 BN224 GQA currently requires dim == 128.")
    if is_causal and seq_len_q > seq_len_kv:
        raise ValueError("causal attention requires seq_len_q <= seq_len_kv")
    block_m = 128
    half_m = block_m // 2
    groups = heads // heads_kv
    accum_dtype = "float"
    fp8_dtype = "float8_e4m3fn"
    attention_scale = sm_scale
    scale = sm_scale * LOG2E
    use_softcap = softcap > 0.0
    capped_softmax_scale = softcap * LOG2E
    lse_scale = capped_softmax_scale if use_softcap else scale
    defer_row_sum = use_softcap or is_causal or (seq_len_kv + 223) // 224 < 32
    causal_offset = seq_len_kv - seq_len_q

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

    @T.macro
    def online_softmax_with_causal_partial_sum(
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
            scores_max[i] = T.max(scores_max[i] * score_scale, scores_max_prev[i])
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(half_m, 224):
            acc_s[i, j] = T.exp2(acc_s[i, j] * score_scale_softmax - scores_max[i] * scale)
        T.call_extern(
            "handle",
            "tl::fp8_partial_row_sum_raw_acc_64x224",
            acc_s.data,
            scores_sum.data,
        )
        for i in T.Parallel(half_m):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    @T.macro
    def online_softmax_with_softcap_partial_sum(
        acc_s,
        scores_max,
        scores_max_prev,
        scores_scale,
        scores_sum,
        logsum,
        score_scale,
    ):
        # The raw accumulator was capped before masking.  Keeping the tanh
        # transform in the raw PTX layout avoids a generic fragment loop.
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(half_m):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            scores_scale[i] = T.exp2(
                (scores_max_prev[i] - scores_max[i]) * T.cast(capped_softmax_scale, accum_dtype)
            )
        for i, j in T.Parallel(half_m, 224):
            acc_s[i, j] = T.exp2(
                (acc_s[i, j] - scores_max[i]) * T.cast(capped_softmax_scale, accum_dtype)
            )
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
        q_shape = (batch, seq_len_q, heads, dim)
        kv_shape = (batch, seq_len_kv, heads_kv, dim)
        descale_shape = (batch, heads_kv)
        # Preserve the original reduction schedule for long non-causal calls.
        # Causal calls use lane-local partial sums so their boundary masking
        # does not add a quad reduction to every visible KV tile.
        online_softmax_fast_1 = make_online_softmax_with_score_scale(
            scale, accum_dtype, half_m, 224
        )
        online_softmax_fast_2 = make_online_softmax_with_score_scale(
            scale, accum_dtype, half_m, 224
        )
        if use_softcap:
            online_softmax_fast_1 = online_softmax_with_softcap_partial_sum
            online_softmax_fast_2 = online_softmax_with_softcap_partial_sum
            online_softmax_1 = online_softmax_with_softcap_partial_sum
            online_softmax_2 = online_softmax_with_softcap_partial_sum
        elif is_causal:
            online_softmax_1 = online_softmax_with_causal_partial_sum
            online_softmax_2 = online_softmax_with_causal_partial_sum
        elif not defer_row_sum:
            online_softmax_1 = online_softmax_fast_1
            online_softmax_2 = online_softmax_fast_2
        else:
            online_softmax_1 = online_softmax_with_partial_sum
            online_softmax_2 = online_softmax_with_partial_sum
        pv_begin_accumulate_helper = "tl::fp8_pv_ptx_unit_begin_accumulate_fa3_raw_64x128x224"
        has_q_tail = seq_len_q % block_m != 0
        has_kv_tail = seq_len_kv % 224 != 0
        use_out_of_place_v = is_causal or has_kv_tail
        v_transform_helper = (
            "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm_out_of_place"
            if use_out_of_place_v
            else "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm_barrier_each_iter"
        )

        @T.macro
        def store_output_tile(o_shared, output, tile_b, row_start, tile_h):
            if not has_q_tail:
                T.call_extern(
                    "handle",
                    "tl::fp8_fa3_o_smem_store_global_cute_64x128",
                    o_shared.access_ptr("r"),
                    T.address_of(output[tile_b, row_start, tile_h, 0]),
                    heads * dim,
                )
            else:
                if row_start + half_m <= seq_len_q:
                    T.call_extern(
                        "handle",
                        "tl::fp8_fa3_o_smem_store_global_cute_64x128",
                        o_shared.access_ptr("r"),
                        T.address_of(output[tile_b, row_start, tile_h, 0]),
                        heads * dim,
                    )
                elif row_start < seq_len_q:
                    T.call_extern(
                        "handle",
                        "tl::fp8_fa3_o_smem_store_global_cute_64x128_tail",
                        o_shared.access_ptr("r"),
                        T.address_of(output[tile_b, row_start, tile_h, 0]),
                        heads * dim,
                        seq_len_q - row_start,
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
            lse: T.Tensor([batch, heads, seq_len_q], accum_dtype),
        ) -> None:
            with T.Kernel(num_sms, 1, 1, threads=384) as (bx, _by, _bz):
                q_shared_1 = T.alloc_shared([half_m, dim], fp8_dtype)
                q_shared_2 = T.alloc_shared([half_m, dim], fp8_dtype)
                k_smem_0 = T.alloc_shared([224, dim], fp8_dtype)
                k_smem_1 = T.alloc_shared([224, dim], fp8_dtype)
                v_vt_smem_0 = T.alloc_shared([dim, 224], fp8_dtype)
                v_vt_smem_1 = T.alloc_shared([dim, 224], fp8_dtype)
                if use_out_of_place_v:
                    v_tc_smem_0 = T.alloc_shared([dim, 224], fp8_dtype)
                    v_tc_smem_1 = T.alloc_shared([dim, 224], fp8_dtype)
                else:
                    v_tc_smem_0 = v_vt_smem_0
                    v_tc_smem_1 = v_vt_smem_1
                o_shared_1 = T.alloc_shared([half_m, dim], out_dtype)
                o_shared_2 = T.alloc_shared([half_m, dim], out_dtype)
                ss_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ss_shared_2 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_2 = T.alloc_shared([half_m], accum_dtype)
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
                T.sync_threads()
                gi_kc1 = T.alloc_var("int32", init=0)
                gi_vc1 = T.alloc_var("int32", init=0)
                gi_kc2 = T.alloc_var("int32", init=0)
                gi_vc2 = T.alloc_var("int32", init=0)
                gi_q1 = T.alloc_var("int32", init=0)
                gi_q2 = T.alloc_var("int32", init=0)
                gi_kp = T.alloc_var("int32", init=0)
                gi_vp = T.alloc_var("int32", init=0)
                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    for tile_b, tile_hkv, tile_m, _tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len_q, block_m), groups],
                        wave_size=num_sms,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_hkv
                        if is_causal:
                            visible_tiles = T.ceildiv(
                                T.min(seq_len_kv, causal_offset + (tile_m + 1) * block_m),
                                224,
                            )
                            loop_range = visible_tiles
                        else:
                            loop_range = T.ceildiv(seq_len_kv, 224)
                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            if n_idx > 0:
                                if gi_vp >= 2:
                                    if gi_vp % 2 == 0:
                                        T.barrier_wait(v_empty_0, (gi_vp // 2 - 1) % 2)
                                    else:
                                        T.barrier_wait(v_empty_1, (gi_vp // 2 - 1) % 2)
                                if gi_vp % 2 == 0:
                                    T.barrier_wait(v_raw_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        v_transform_helper,
                                        v_vt_smem_0.access_ptr("rw"),
                                        v_tc_smem_0.access_ptr("w"),
                                    )
                                else:
                                    T.barrier_wait(v_raw_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        v_transform_helper,
                                        v_vt_smem_1.access_ptr("rw"),
                                        v_tc_smem_1.access_ptr("w"),
                                    )
                                if gi_vp % 2 == 0:
                                    T.barrier_arrive(v_full_0)
                                else:
                                    T.barrier_arrive(v_full_1)
                                gi_vp = gi_vp + 1
                            if tx == 0:
                                T.mbarrier_expect_tx(v_raw_full, dim * 224)
                                v_desc = T.create_tma_descriptor(
                                    TMA_DTYPE_UINT8,
                                    4,
                                    v.data,
                                    dim,
                                    heads_kv,
                                    seq_len_kv,
                                    batch,
                                    1,
                                    dim,
                                    heads_kv * dim,
                                    seq_len_kv * heads_kv * dim,
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
                                if gi_vp % 2 == 0:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_tma_load_4d_ptx",
                                        v_desc,
                                        v_raw_full[0],
                                        T.access_ptr(v_vt_smem_0, "w"),
                                        0,
                                        head_kv,
                                        n_idx * 224,
                                        tile_b,
                                    )
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_tma_load_4d_ptx",
                                        v_desc,
                                        v_raw_full[0],
                                        T.access_ptr(v_vt_smem_1, "w"),
                                        0,
                                        head_kv,
                                        n_idx * 224,
                                        tile_b,
                                    )
                            T.barrier_arrive(v_raw_full)
                            T.barrier_wait(k_empty, (gi_kp + 1) % 2)
                            if gi_kp % 2 == 0:
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
                            gi_kp = gi_kp + 1
                        if gi_vp >= 2:
                            if gi_vp % 2 == 0:
                                T.barrier_wait(v_empty_0, (gi_vp // 2 - 1) % 2)
                            else:
                                T.barrier_wait(v_empty_1, (gi_vp // 2 - 1) % 2)
                        if gi_vp % 2 == 0:
                            T.barrier_wait(v_raw_full, gi_vp % 2)
                            T.call_extern(
                                "handle",
                                v_transform_helper,
                                v_vt_smem_0.access_ptr("rw"),
                                v_tc_smem_0.access_ptr("w"),
                            )
                        else:
                            T.barrier_wait(v_raw_full, gi_vp % 2)
                            T.call_extern(
                                "handle",
                                v_transform_helper,
                                v_vt_smem_1.access_ptr("rw"),
                                v_tc_smem_1.access_ptr("w"),
                            )
                        if gi_vp % 2 == 0:
                            T.barrier_arrive(v_full_0)
                        else:
                            T.barrier_arrive(v_full_1)
                        gi_vp = gi_vp + 1
                        if groups == 8 and defer_row_sum:
                            T.sync_threads(barrier_id=5, arrive_count=384)
                elif tx < 256:
                    T.inc_max_nreg(240)
                    for tile_b, tile_hkv, tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len_q, block_m), groups],
                        wave_size=num_sms,
                        index=bx,
                        group_size=8,
                    ):
                        tile_h = tile_hkv * groups + tile_g
                        head_kv = tile_hkv
                        row_base = tile_m * block_m
                        if is_causal:
                            visible_tiles = T.ceildiv(
                                T.min(seq_len_kv, causal_offset + row_base + block_m),
                                224,
                            )
                            loop_range = visible_tiles
                        else:
                            loop_range = T.ceildiv(seq_len_kv, 224)
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
                            if use_softcap:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_apply_softcap_raw_acc_64x224",
                                    acc_s_1.data,
                                    qk_descale * attention_scale / softcap,
                                )
                            if has_kv_tail and (n_idx + 1) * 224 > seq_len_kv:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_mask_columns_raw_acc_64x224",
                                    acc_s_1.data,
                                    seq_len_kv - n_idx * 224,
                                    -T.infinity(accum_dtype),
                                )
                            if is_causal and (n_idx + 1) * 224 > causal_offset + row_base + half_m:
                                for i, j in T.Parallel(half_m, 224):
                                    acc_s_1[i, j] = T.if_then_else(
                                        n_idx * 224 + j <= causal_offset + row_base + i,
                                        acc_s_1[i, j],
                                        -T.infinity(accum_dtype),
                                    )
                            if is_causal:
                                if (n_idx + 1) * 224 > causal_offset + row_base + half_m:
                                    online_softmax_1(
                                        acc_s_1,
                                        sm_1,
                                        smp_1,
                                        ss_1,
                                        ssum_1,
                                        ls_1,
                                        qk_descale,
                                    )
                                else:
                                    online_softmax_fast_1(
                                        acc_s_1,
                                        sm_1,
                                        smp_1,
                                        ss_1,
                                        ssum_1,
                                        ls_1,
                                        qk_descale,
                                    )
                            else:
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
                        store_output_tile(o_shared_1, output, tile_b, row_base, tile_h)
                        if write_lse:
                            for i in T.Parallel(half_m):
                                ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * lse_scale
                            T.copy(ls_1, lse[tile_b, tile_h, row_base : row_base + half_m])
                        if groups == 8 and defer_row_sum:
                            T.sync_threads(barrier_id=5, arrive_count=384)
                else:
                    T.inc_max_nreg(240)
                    for tile_b, tile_hkv, tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len_q, block_m), groups],
                        wave_size=num_sms,
                        index=bx,
                        group_size=8,
                    ):
                        tile_h = tile_hkv * groups + tile_g
                        head_kv = tile_hkv
                        row_base = tile_m * block_m
                        if is_causal:
                            visible_tiles = T.ceildiv(
                                T.min(seq_len_kv, causal_offset + row_base + block_m),
                                224,
                            )
                            loop_range = visible_tiles
                        else:
                            loop_range = T.ceildiv(seq_len_kv, 224)
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
                            if use_softcap:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_apply_softcap_raw_acc_64x224",
                                    acc_s_2.data,
                                    qk_descale * attention_scale / softcap,
                                )
                            if has_kv_tail and (n_idx + 1) * 224 > seq_len_kv:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_mask_columns_raw_acc_64x224",
                                    acc_s_2.data,
                                    seq_len_kv - n_idx * 224,
                                    -T.infinity(accum_dtype),
                                )
                            if is_causal and (n_idx + 1) * 224 > causal_offset + row_base + block_m:
                                for i, j in T.Parallel(half_m, 224):
                                    acc_s_2[i, j] = T.if_then_else(
                                        n_idx * 224 + j <= causal_offset + row_base + half_m + i,
                                        acc_s_2[i, j],
                                        -T.infinity(accum_dtype),
                                    )
                            if is_causal:
                                if (n_idx + 1) * 224 > causal_offset + row_base + block_m:
                                    online_softmax_2(
                                        acc_s_2,
                                        sm_2,
                                        smp_2,
                                        ss_2,
                                        ssum_2,
                                        ls_2,
                                        qk_descale,
                                    )
                                else:
                                    online_softmax_fast_2(
                                        acc_s_2,
                                        sm_2,
                                        smp_2,
                                        ss_2,
                                        ssum_2,
                                        ls_2,
                                        qk_descale,
                                    )
                            else:
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
                        store_output_tile(o_shared_2, output, tile_b, row_base + half_m, tile_h)
                        if write_lse:
                            for i in T.Parallel(half_m):
                                ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * lse_scale
                            T.copy(
                                ls_2,
                                lse[tile_b, tile_h, row_base + half_m : row_base + block_m],
                            )
                        if groups == 8 and defer_row_sum:
                            T.sync_threads(barrier_id=5, arrive_count=384)

        return main

    return func


@torch.library.custom_op("tileops::gqa_dense_fwd_fp8_wrapped_kernel", mutates_args=())
def _gqa_dense_fwd_fp8_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    out_dtype: str,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
) -> torch.Tensor:
    num_tasks = batch * heads * ((seq_len_q + 127) // 128)
    num_waves = (num_tasks + NUM_SMS - 1) // NUM_SMS
    grid_size = (num_tasks + num_waves - 1) // num_waves
    return _gqa_fwd_fp8_bn224_tma_v_kernel(
        batch,
        heads,
        heads_kv,
        seq_len_q,
        seq_len_kv,
        dim,
        out_dtype,
        is_causal,
        sm_scale,
        softcap,
        False,
        grid_size,
    )()(q, k, v, q_descale, k_descale, v_descale)[0]


@_gqa_dense_fwd_fp8_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    out_dtype: str,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    *inputs: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    del heads_kv, seq_len_kv, is_causal, sm_scale, softcap
    torch_dtype = torch.float16 if out_dtype == "float16" else torch.bfloat16
    return torch.empty((batch, seq_len_q, heads, dim), dtype=torch_dtype, device=inputs[0].device)


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


class GQADenseFP8Kernel(Kernel):
    """Native-FP8 Dense GQA main kernel using the BN224 schedule."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len_q: int,
        seq_len_kv: int,
        dim: int,
        is_causal: bool,
        window_size_left: int,
        window_size_right: int,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.fuse_rope = fuse_rope
        self._validate_spec()
        self.rope = make_dense_qk_rope_preprocessor(
            fuse_rope=fuse_rope,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            dim=dim,
            max_position=max_position,
            rotary_dim=rotary_dim,
            rope_layout=rope_layout,
            dtype="float8_e4m3fn",
            rope_dtype=self.dtype_str,
        )
        self.init_config(config, tune)

    def _validate_spec(self) -> None:
        if self.heads % self.heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if self.dim != 128:
            raise ValueError("native-FP8 Dense GQA currently requires dim == 128")
        if self.dtype not in ATTENTION_DTYPES:
            raise ValueError("native-FP8 Dense GQA outputs float16 or bfloat16")
        if self.is_causal and self.seq_len_q > self.seq_len_kv:
            raise ValueError("causal FP8 Dense GQA requires seq_len_q <= seq_len_kv")
        if self.fuse_rope and self.seq_len_q == 1:
            raise ValueError("FP8 Dense decode requires an in-kernel RoPE implementation")
        if self.window_size_left != -1 or self.window_size_right != -1:
            raise ValueError("native-FP8 Dense GQA does not support sliding windows")
        if not self.is_causal and self.seq_len_q != self.seq_len_kv:
            raise ValueError("non-causal native-FP8 Dense GQA requires Sq == Skv")
        if not self.is_causal and self.softcap != 0.0:
            raise ValueError("non-causal native-FP8 Dense GQA does not support softcap")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._require_cuda(
            q=q,
            k=k,
            v=v,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        fp8 = getattr(torch, "float8_e4m3fn", None)
        if fp8 is None or q.dtype != fp8 or k.dtype != fp8 or v.dtype != fp8:
            raise ValueError("GQADenseFP8Kernel requires float8_e4m3fn q, k, and v")
        if q_scale is None or k_scale is None or v_scale is None:
            raise ValueError("GQADenseFP8Kernel requires q_scale, k_scale, and v_scale")
        _validate_fa3_gqa_descales(
            q_scale,
            k_scale,
            v_scale,
            self.batch,
            self.heads_kv,
            q.device,
        )

        if self.rope is not None:
            q, k = self.rope(q, k, rope_cos, rope_sin)
        elif rope_cos is not None or rope_sin is not None:
            raise ValueError("native-FP8 Dense GQA does not accept RoPE tables")
        return _gqa_dense_fwd_fp8_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len_q,
            self.seq_len_kv,
            self.dim,
            self.dtype_str,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
        )
