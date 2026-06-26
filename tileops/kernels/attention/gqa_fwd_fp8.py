import functools
import os
from typing import Callable, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.online_softmax import (
    make_log2e_scale,
    make_online_softmax_with_score_scale,
)

__all__ = ["GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel"]
NUM_SMS = int(os.environ.get("V2P_NUM_SMS", "132"))
TMA_DTYPE_UINT8 = 0
TMA_INTERLEAVE_NONE = 0
TMA_SWIZZLE_128B = 3
TMA_L2_PROMOTION_128B = 2
TMA_OOB_FILL_NONE = 0
_ANCHOR_HELPER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "_anchor_helper.h"))
_FP8_GQA_HELPER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "_fp8_gqa_helper.h"))


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
    scale_block = 128
    scale_blocks = (seq_len + scale_block - 1) // scale_block

    @tilelang.jit(
        out_idx=[6, 7],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            "-include",
            _ANCHOR_HELPER_PATH,
            "-include",
            _FP8_GQA_HELPER_PATH,
        ],
    )
    def func():
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        q_scale_shape = (batch, heads, scale_blocks)
        kv_scale_shape = (batch, heads_kv, scale_blocks)
        online_softmax_1 = make_online_softmax_with_score_scale(scale, accum_dtype, half_m, 224)
        online_softmax_2 = make_online_softmax_with_score_scale(scale, accum_dtype, half_m, 224)
        pv_accumulate_helper = "tl::fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224"
        v_inplace_transform_helper = (
            "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm_barrier_each_iter"
        )

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, fp8_dtype),
            k: T.Tensor(kv_shape, fp8_dtype),
            v: T.Tensor(kv_shape, fp8_dtype),
            q_scale: T.Tensor(q_scale_shape, accum_dtype),
            k_scale: T.Tensor(kv_scale_shape, accum_dtype),
            v_scale: T.Tensor(kv_scale_shape, accum_dtype),
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
                acc_o_layout_seed_1 = T.alloc_shared([half_m, dim], out_dtype)
                acc_o_layout_seed_2 = T.alloc_shared([half_m, dim], out_dtype)
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
                v_full = T.alloc_barrier(arrive_count=128)
                v_empty_0 = T.alloc_barrier(arrive_count=256)
                v_empty_1 = T.alloc_barrier(arrive_count=256)
                q_full_1 = T.alloc_barrier(arrive_count=128)
                q_full_2 = T.alloc_barrier(arrive_count=128)
                T.annotate_layout(
                    {
                        q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                        q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                    }
                )
                T.sync_threads()
                gi_kp = T.alloc_var("int32", init=0)
                gi_vp = T.alloc_var("int32", init=0)
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
                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            if n_idx > 0:
                                if gi_vp >= 2:
                                    if gi_vp % 2 == 0:
                                        T.barrier_wait(v_empty_0, (gi_vp // 2 - 1) % 2)
                                    else:
                                        T.barrier_wait(v_empty_1, (gi_vp // 2 - 1) % 2)
                                if gi_vp % 2 == 0:
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
                                    T.barrier_wait(v_raw_full, gi_vp % 2)
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
                                    T.barrier_wait(v_raw_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        v_inplace_transform_helper,
                                        v_vt_smem_1.access_ptr("rw"),
                                        v_vt_smem_1.access_ptr("rw"),
                                    )
                                T.barrier_arrive(v_full)
                                gi_vp = gi_vp + 1
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
                            T.barrier_wait(v_raw_full, gi_vp % 2)
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
                            T.barrier_wait(v_raw_full, gi_vp % 2)
                            T.call_extern(
                                "handle",
                                v_inplace_transform_helper,
                                v_vt_smem_1.access_ptr("rw"),
                                v_vt_smem_1.access_ptr("rw"),
                            )
                        T.barrier_arrive(v_full)
                        gi_vp = gi_vp + 1
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
                        q_scale_idx = row_base // scale_block
                        loop_range = T.ceildiv(seq_len, 224)
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
                                T.wgmma_gemm(
                                    q_shared_1,
                                    k_smem_0,
                                    acc_s_1,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=True,
                                )
                            else:
                                T.wgmma_gemm(
                                    q_shared_1,
                                    k_smem_1,
                                    acc_s_1,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=True,
                                )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                            T.barrier_arrive(k_empty)
                            online_softmax_1(
                                acc_s_1,
                                sm_1,
                                smp_1,
                                ss_1,
                                ssum_1,
                                ls_1,
                                q_scale[tile_b, tile_h, q_scale_idx]
                                * k_scale[tile_b, head_kv, n_idx],
                            )
                            T.copy(ss_1, ss_shared_1)
                            T.barrier_wait(v_full, gi_vc1 % 2)
                            if gi_vc1 % 2 == 0:
                                T.call_extern(
                                    "handle",
                                    pv_accumulate_helper,
                                    acc_s_1.data,
                                    v_tc_smem_0.access_ptr("r"),
                                    4,
                                    ss_shared_1.access_ptr("r"),
                                    v_scale[tile_b, head_kv, n_idx],
                                    acc_o_1.data,
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    pv_accumulate_helper,
                                    acc_s_1.data,
                                    v_tc_smem_1.access_ptr("r"),
                                    4,
                                    ss_shared_1.access_ptr("r"),
                                    v_scale[tile_b, head_kv, n_idx],
                                    acc_o_1.data,
                                )
                            if gi_vc1 % 2 == 0:
                                T.barrier_arrive(v_empty_0)
                            else:
                                T.barrier_arrive(v_empty_1)
                            gi_vc1 = gi_vc1 + 1
                            gi_kc1 = gi_kc1 + 1
                        T.copy(ls_1, ls_shared_1)
                        T.copy(acc_o_1, acc_o_layout_seed_1)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_store_smem_cute_64x128",
                            acc_o_1.data,
                            ls_shared_1.access_ptr("r"),
                            4,
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
                        q_scale_idx = row_base // scale_block
                        loop_range = T.ceildiv(seq_len, 224)
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
                                T.wgmma_gemm(
                                    q_shared_2,
                                    k_smem_0,
                                    acc_s_2,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=True,
                                )
                            else:
                                T.wgmma_gemm(
                                    q_shared_2,
                                    k_smem_1,
                                    acc_s_2,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=True,
                                )
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                            T.barrier_arrive(k_empty)
                            online_softmax_2(
                                acc_s_2,
                                sm_2,
                                smp_2,
                                ss_2,
                                ssum_2,
                                ls_2,
                                q_scale[tile_b, tile_h, q_scale_idx]
                                * k_scale[tile_b, head_kv, n_idx],
                            )
                            T.copy(ss_2, ss_shared_2)
                            T.barrier_wait(v_full, gi_vc2 % 2)
                            if gi_vc2 % 2 == 0:
                                T.call_extern(
                                    "handle",
                                    pv_accumulate_helper,
                                    acc_s_2.data,
                                    v_tc_smem_0.access_ptr("r"),
                                    4,
                                    ss_shared_2.access_ptr("r"),
                                    v_scale[tile_b, head_kv, n_idx],
                                    acc_o_2.data,
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    pv_accumulate_helper,
                                    acc_s_2.data,
                                    v_tc_smem_1.access_ptr("r"),
                                    4,
                                    ss_shared_2.access_ptr("r"),
                                    v_scale[tile_b, head_kv, n_idx],
                                    acc_o_2.data,
                                )
                            if gi_vc2 % 2 == 0:
                                T.barrier_arrive(v_empty_0)
                            else:
                                T.barrier_arrive(v_empty_1)
                            gi_vc2 = gi_vc2 + 1
                            gi_kc2 = gi_kc2 + 1
                        T.copy(ls_2, ls_shared_2)
                        T.copy(acc_o_2, acc_o_layout_seed_2)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_store_smem_cute_64x128",
                            acc_o_2.data,
                            ls_shared_2.access_ptr("r"),
                            4,
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
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_fp8_bn224_tma_v_kernel(batch, heads, heads_kv, seq_len, dim, out_dtype)()(
        q, k, v, q_scale, k_scale, v_scale
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


def _expand_fa3_gqa_descales(
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept FA3-interface [batch, heads_kv] descales.

    The TileLang kernels below still index scale metadata as
    q: [batch, heads, scale_blocks] and k/v: [batch, heads_kv, scale_blocks].
    The public FA3 FP8 GQA binding exposes q/k/v descales as [batch, heads_kv],
    with q grouped by KV head.  This wrapper broadcasts that 2D contract to the
    internal per-block layout; direct kernel users may still pass the internal
    3D layout explicitly.
    """
    scale_blocks = (seq_len + 127) // 128
    group_size = heads // heads_kv

    def _record_stream(x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            x.record_stream(torch.cuda.current_stream(x.device))
        return x

    def _as_float_contiguous(x: torch.Tensor) -> torch.Tensor:
        return x if x.dtype == torch.float32 and x.is_contiguous() else x.float().contiguous()

    def _expand_q(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if tuple(x.shape) != (batch, heads, scale_blocks):
                raise ValueError(
                    f"q_descale/q_scale with 3 dimensions must have internal shape ({batch}, {heads}, {scale_blocks}), got {tuple(x.shape)}."
                )
            return _record_stream(_as_float_contiguous(x))
        if x.ndim != 2 or tuple(x.shape) != (batch, heads_kv):
            raise ValueError(
                f"q_descale must have FA3 shape ({batch}, {heads_kv}) or internal shape ({batch}, {heads}, {scale_blocks}), got {tuple(x.shape)}."
            )
        x = _as_float_contiguous(x).repeat_interleave(group_size, dim=1)
        return _record_stream(x[:, :, None].expand(batch, heads, scale_blocks).contiguous())

    def _expand_kv(name: str, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if tuple(x.shape) != (batch, heads_kv, scale_blocks):
                raise ValueError(
                    f"{name} with 3 dimensions must have internal shape ({batch}, {heads_kv}, {scale_blocks}), got {tuple(x.shape)}."
                )
            return _record_stream(_as_float_contiguous(x))
        if x.ndim != 2 or tuple(x.shape) != (batch, heads_kv):
            raise ValueError(
                f"{name} must have FA3 shape ({batch}, {heads_kv}) or internal shape ({batch}, {heads_kv}, {scale_blocks}), got {tuple(x.shape)}."
            )
        x = _as_float_contiguous(x)
        return _record_stream(x[:, :, None].expand(batch, heads_kv, scale_blocks).contiguous())

    return (
        _expand_q(q_descale),
        _expand_kv("k_descale", k_descale),
        _expand_kv("v_descale", v_descale),
    )


class GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(Kernel):
    """BN224 WS FP8 GQA kernel with FA3-compatible 2D descales and TMA-V layout."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        tune: bool = False,
    ) -> None:
        del tune
        super().__init__()
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
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel outputs float16 or bfloat16."
            )
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            raise ValueError("torch.float8_e4m3fn is required for this kernel.")
        if q.dtype != fp8_dtype or k.dtype != fp8_dtype or v.dtype != fp8_dtype:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        q_scale, k_scale, v_scale = _expand_fa3_gqa_descales(
            q_descale, k_descale, v_descale, self.batch, self.heads, self.heads_kv, self.seq_len
        )
        return _gqa_fwd_fp8_bn224_tma_v_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
            self.dim,
            self.dtype_str,
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
        )
