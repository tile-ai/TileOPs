import functools
import os
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch
from tilelang.intrinsics.mma_macro_generator import get_ldmatrix_offset
from tilelang.intrinsics.wgmma_macro_generator import TensorCoreIntrinEmitter
from tvm import tir

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.online_softmax import (
    make_log2e_scale,
    make_online_softmax,
    make_online_softmax_with_score_scale,
    make_rescale,
)

__all__ = [
    "GQAFwdFP8Fa3ContractKernel",
    "GQAFwdFP8Fa3ContractPtxAccDirectStoreKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224Kernel",
    "GQAFwdFP8Fa3ContractRawCudaSmokeKernel",
    "GQAFwdFP8Fa3ContractRawCudaTileMapKernel",
    "GQAFwdFP8Fa3ContractRawCudaQKFirstKeyDebugKernel",
    "GQAFwdFP8Fa3ContractRawCudaQKTileDebugKernel",
    "GQAFwdFP8Fa3ContractRawCudaQKWgmmaTileDebugKernel",
    "GQAFwdFP8Fa3ContractRawCudaQKWgmmaSoftmaxTileDebugKernel",
    "GQAFwdFP8Fa3ContractRawCudaQKWgmmaLocalPVTileDebugKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedPreRescaleKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedPreRescaleKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapLocalPKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceBarrierKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel",
    "GQAFwdFP8Fa3ContractPtxAccBN224WsKernel",
    "GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel",
    "GQAFwdFP8Fa3ContractPtxAccFa3EpilogueReuseVSmemKernel",
    "GQAFwdFP8Fa3ContractPtxAccKernel",
    "GQAFwdFP8WgmmaKernel",
    "GQAFwdFP8WsPersistentKernel",
]

NUM_SMS = int(os.environ.get("V2P_NUM_SMS", "132"))
TMA_DTYPE_UINT8 = 0
TMA_INTERLEAVE_NONE = 0
TMA_SWIZZLE_128B = 3
TMA_L2_PROMOTION_128B = 2
TMA_OOB_FILL_NONE = 0
_ANCHOR_HELPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_anchor_helper.h"))
_FP8_GQA_HELPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_fp8_gqa_helper.h"))


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_wgmma_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    use_register_p: bool = True,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("GQAFwdFP8WgmmaKernel currently requires dim == 128.")

    groups = heads // heads_kv
    accum_dtype = "float"
    fp8_dtype = "float8_e4m3fn"
    fp8_k = 32
    scale = make_log2e_scale(dim)
    scale_block = 128
    scale_blocks = (seq_len + scale_block - 1) // scale_block

    @tilelang.jit(
        out_idx=[6, 7],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16", "-include", _FP8_GQA_HELPER_PATH],
    )
    def func(block_m: int, block_n: int, num_stages: int, threads: int):
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        q_scale_shape = (batch, heads, scale_blocks)
        kv_scale_shape = (batch, heads_kv, scale_blocks)

        online_softmax = make_online_softmax(scale, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)

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
            with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, dim], fp8_dtype)
                k_shared = T.alloc_shared([block_n, dim], fp8_dtype)
                if not use_register_p:
                    p_shared = T.alloc_shared([block_m, block_n], fp8_dtype)
                v_tc_shared = T.alloc_shared([dim, fp8_k], fp8_dtype)
                o_shared = T.alloc_shared([block_m, dim], out_dtype)

                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                p_fp8 = T.alloc_fragment([block_m, fp8_k], fp8_dtype)
                acc_delta = T.alloc_fragment([block_m, dim], accum_dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                if use_register_p:
                    T.annotate_layout({
                        o_shared: tilelang.layout.make_swizzled_layout(o_shared),
                    })
                else:
                    T.annotate_layout({
                        p_shared: tilelang.layout.make_swizzled_layout(p_shared),
                        o_shared: tilelang.layout.make_swizzled_layout(o_shared),
                    })

                head_kv = by // groups
                row_base = bx * block_m
                q_scale_idx = row_base // scale_block

                T.copy(q[bz, row_base:row_base + block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for n_idx in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=num_stages):
                    T.copy(k[bz, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :], k_shared)
                    T.wgmma_gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=True,
                    )
                    T.wait_wgmma(0)

                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] *= q_scale[bz, by, q_scale_idx] * k_scale[bz, head_kv, n_idx]

                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum,
                                   logsum)

                    if not use_register_p:
                        T.copy(acc_s, p_shared)
                    T.clear(acc_delta)
                    for pv_k in range(block_n // fp8_k):
                        if use_register_p:
                            T.call_extern(
                                "handle",
                                "tl::fp8_acc_to_pv_a_frag_64x128",
                                acc_s.data,
                                p_fp8.data,
                                pv_k,
                            )
                        else:
                            T.copy(p_shared[:, pv_k * fp8_k:(pv_k + 1) * fp8_k], p_fp8)
                        for d, n in T.Parallel(dim, fp8_k):
                            v_tc_shared[d, n] = v[bz, n_idx * block_n + pv_k * fp8_k + n,
                                                  head_kv, d]

                        T.wgmma_gemm(
                            p_fp8,
                            v_tc_shared,
                            acc_delta,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=False,
                        )
                        T.wait_wgmma(0)

                    rescale(acc_o, scores_scale)
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] += acc_delta[i, j] * v_scale[bz, head_kv, n_idx]

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, output[bz, row_base:row_base + block_m, by, :])

                for i in T.Parallel(block_m):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, row_base:row_base + block_m])

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_ws_persistent_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    use_register_p: bool = False,
    use_full_pv_gemm: bool = False,
    use_ldsm_v_transpose: bool = False,
    use_tl_full_v_transpose: bool = False,
    use_tl_pack_ldsm_v_transpose: bool = False,
    use_fa3_pv_extern: bool = False,
    use_fa3_v_tma_direct: bool = False,
    use_ptx_pv: bool = False,
    use_ptx_pv_accumulate: bool = False,
    use_ptx_pv_direct_store: bool = False,
    use_ptx_pv_fa3_epilogue_store: bool = False,
    use_ptx_pv_fa3_epilogue_reuse_v_smem: bool = False,
    use_fa3_v_inplace_transform: bool = False,
    use_fa3_v_inplace_barrier_transform: bool = False,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("GQAFwdFP8WsPersistentKernel currently requires dim == 128.")

    block_m = 128
    half_m = block_m // 2
    groups = heads // heads_kv
    accum_dtype = "float"
    fp8_dtype = "float8_e4m3fn"
    fp8_k = 32
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
    def func(block_n: int):
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        q_scale_shape = (batch, heads, scale_blocks)
        kv_scale_shape = (batch, heads_kv, scale_blocks)

        fuse_score_scale_into_softmax = (
            use_fa3_pv_extern and use_fa3_v_tma_direct and
            use_ptx_pv_accumulate and block_n == 224
        )
        if fuse_score_scale_into_softmax:
            online_softmax_1 = make_online_softmax_with_score_scale(
                scale, accum_dtype, half_m, block_n)
            online_softmax_2 = make_online_softmax_with_score_scale(
                scale, accum_dtype, half_m, block_n)
        else:
            online_softmax_1 = make_online_softmax(scale, accum_dtype, half_m, block_n)
            online_softmax_2 = make_online_softmax(scale, accum_dtype, half_m, block_n)
        rescale_1 = make_rescale(half_m, dim)
        rescale_2 = make_rescale(half_m, dim)
        pv_accumulate_helper = (
            "tl::fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224"
            if block_n == 224 else
            "tl::fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x128"
        )
        v_inplace_transform_helper = (
            "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm_barrier_each_iter"
            if use_fa3_v_inplace_barrier_transform else
            "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm"
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
                k_smem_0 = T.alloc_shared([block_n, dim], fp8_dtype)
                k_smem_1 = T.alloc_shared([block_n, dim], fp8_dtype)
                if not (use_fa3_pv_extern and use_fa3_v_tma_direct):
                    v_smem_0 = T.alloc_shared([block_n, dim], fp8_dtype)
                    v_smem_1 = T.alloc_shared([block_n, dim], fp8_dtype)
                if use_tl_pack_ldsm_v_transpose:
                    v_perm_smem_0 = T.alloc_shared([block_n, dim], fp8_dtype)
                    v_perm_smem_1 = T.alloc_shared([block_n, dim], fp8_dtype)
                if use_fa3_pv_extern and use_fa3_v_tma_direct and use_fa3_v_inplace_transform:
                    v_vt_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                    v_vt_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                    v_tc_smem_0 = v_vt_smem_0
                    v_tc_smem_1 = v_vt_smem_1
                else:
                    v_tc_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                    v_tc_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                    if use_fa3_pv_extern:
                        v_vt_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                        v_vt_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                if not use_full_pv_gemm and not use_fa3_pv_extern:
                    v_tc_shared_1 = T.alloc_shared([dim, fp8_k], fp8_dtype)
                    v_tc_shared_2 = T.alloc_shared([dim, fp8_k], fp8_dtype)
                if use_fa3_pv_extern and not use_ptx_pv_accumulate:
                    acc_delta_shared_1 = T.alloc_shared([half_m, dim], accum_dtype)
                    acc_delta_shared_2 = T.alloc_shared([half_m, dim], accum_dtype)
                if use_ptx_pv_accumulate and not use_ptx_pv_direct_store:
                    if use_ptx_pv_fa3_epilogue_store:
                        if not use_ptx_pv_fa3_epilogue_reuse_v_smem:
                            o_shared_1 = T.alloc_shared([half_m, dim], out_dtype)
                            o_shared_2 = T.alloc_shared([half_m, dim], out_dtype)
                    else:
                        o_shared_1 = T.alloc_shared([half_m, dim], accum_dtype)
                        o_shared_2 = T.alloc_shared([half_m, dim], accum_dtype)
                if use_ptx_pv_accumulate:
                    ss_shared_1 = T.alloc_shared([half_m], accum_dtype)
                    ss_shared_2 = T.alloc_shared([half_m], accum_dtype)
                    ls_shared_1 = T.alloc_shared([half_m], accum_dtype)
                    ls_shared_2 = T.alloc_shared([half_m], accum_dtype)
                    if block_n == 224 and use_ptx_pv_fa3_epilogue_store:
                        acc_o_layout_seed_1 = T.alloc_shared([half_m, dim], out_dtype)
                        acc_o_layout_seed_2 = T.alloc_shared([half_m, dim], out_dtype)
                if not use_fa3_pv_extern:
                    o_shared_1 = T.alloc_shared([half_m, dim], out_dtype)
                    o_shared_2 = T.alloc_shared([half_m, dim], out_dtype)
                if not use_register_p and not use_fa3_pv_extern:
                    p_shared_1 = T.alloc_shared([half_m, block_n], fp8_dtype)
                    p_shared_2 = T.alloc_shared([half_m, block_n], fp8_dtype)

                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                if not use_fa3_pv_extern:
                    p_fp8_1 = T.alloc_fragment(
                        [half_m, block_n] if use_full_pv_gemm else [half_m, fp8_k],
                        fp8_dtype,
                    )
                if not use_ptx_pv_accumulate:
                    acc_delta_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
                if not use_fa3_pv_extern:
                    p_fp8_2 = T.alloc_fragment(
                        [half_m, block_n] if use_full_pv_gemm else [half_m, fp8_k],
                        fp8_dtype,
                    )
                if not use_ptx_pv_accumulate:
                    acc_delta_2 = T.alloc_fragment([half_m, dim], accum_dtype)
                acc_o_2 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_2 = T.alloc_fragment([half_m], accum_dtype)
                smp_2 = T.alloc_fragment([half_m], accum_dtype)
                ss_2 = T.alloc_fragment([half_m], accum_dtype)
                ssum_2 = T.alloc_fragment([half_m], accum_dtype)
                ls_2 = T.alloc_fragment([half_m], accum_dtype)

                k_full = T.alloc_barrier(arrive_count=128)
                k_empty = T.alloc_barrier(arrive_count=256)
                v_raw_full = T.alloc_barrier(arrive_count=128)
                if (use_fa3_pv_extern and not use_fa3_v_tma_direct) or use_tl_pack_ldsm_v_transpose:
                    v_pack_full = T.alloc_barrier(arrive_count=128)
                v_full = T.alloc_barrier(arrive_count=128)
                v_empty = T.alloc_barrier(arrive_count=256)
                if use_fa3_v_inplace_transform:
                    v_empty_0 = T.alloc_barrier(arrive_count=256)
                    v_empty_1 = T.alloc_barrier(arrive_count=256)
                q_full_1 = T.alloc_barrier(arrive_count=128)
                q_full_2 = T.alloc_barrier(arrive_count=128)
                if use_fa3_pv_extern:
                    if use_fa3_v_tma_direct:
                        if block_n == 224:
                            T.annotate_layout({
                                q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                                q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                            })
                        else:
                            T.annotate_layout({
                                q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                                q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                                v_vt_smem_0: tilelang.layout.make_full_bank_swizzled_layout(
                                    v_vt_smem_0
                                ),
                                v_vt_smem_1: tilelang.layout.make_full_bank_swizzled_layout(
                                    v_vt_smem_1
                                ),
                            })
                    else:
                        T.annotate_layout({
                            q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                            q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                        })
                else:
                    T.annotate_layout({
                        q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                        q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                        o_shared_1: tilelang.layout.make_swizzled_layout(o_shared_1),
                        o_shared_2: tilelang.layout.make_swizzled_layout(o_shared_2),
                    })
                if not use_register_p and not use_fa3_pv_extern:
                    T.annotate_layout({
                        p_shared_1: tilelang.layout.make_swizzled_layout(p_shared_1),
                        p_shared_2: tilelang.layout.make_swizzled_layout(p_shared_2),
                    })
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
                    for tile_b, tile_hkv, _tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len, block_m), groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_hkv
                        loop_range = T.ceildiv(seq_len, block_n)
                        if use_fa3_pv_extern and use_fa3_v_tma_direct and block_n == 224:
                            T.reads(v[tile_b, 0:seq_len, head_kv, :])
                            T.writes(
                                v_vt_smem_0[0:dim, 0:block_n],
                                v_vt_smem_1[0:dim, 0:block_n],
                            )

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            if n_idx > 0:
                                if use_fa3_v_inplace_transform:
                                    if gi_vp >= 2:
                                        if gi_vp % 2 == 0:
                                            T.barrier_wait(v_empty_0, (gi_vp // 2 - 1) % 2)
                                        else:
                                            T.barrier_wait(v_empty_1, (gi_vp // 2 - 1) % 2)
                                else:
                                    T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                                if gi_vp % 2 == 0:
                                    if use_fa3_pv_extern and use_fa3_v_tma_direct:
                                        if block_n == 224:
                                            if tx == 0:
                                                T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                                v_desc = T.create_tma_descriptor(
                                                    TMA_DTYPE_UINT8, 4, v.data,
                                                    dim, heads_kv, seq_len, batch,
                                                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                                    dim, 1, block_n, 1,
                                                    1, 1, 1, 1,
                                                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                                )
                                                tir.call_extern(
                                                    "handle",
                                                    "tl::fp8_tma_load_4d_ptx",
                                                    v_desc,
                                                    v_raw_full[0],
                                                    T.access_ptr(v_vt_smem_0, "w"),
                                                    0,
                                                    head_kv,
                                                    (n_idx - 1) * block_n,
                                                    tile_b,
                                                )
                                        else:
                                            T.tma_copy(
                                                v[tile_b, (n_idx - 1) * block_n:n_idx * block_n, head_kv, :],
                                                v_vt_smem_0,
                                                barrier=v_raw_full,
                                            )
                                        T.barrier_arrive(v_raw_full)
                                        T.barrier_wait(v_raw_full, gi_vp % 2)
                                        if block_n == 224:
                                            if use_fa3_v_inplace_transform:
                                                T.call_extern(
                                                    "handle",
                                                    v_inplace_transform_helper,
                                                    v_vt_smem_0.access_ptr("rw"),
                                                    v_vt_smem_0.access_ptr("rw"),
                                                )
                                            else:
                                                T.call_extern(
                                                    "handle",
                                                    "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                                    v_vt_smem_0.access_ptr("r"),
                                                    v_tc_smem_0.access_ptr("w"),
                                                )
                                        else:
                                            T.call_extern(
                                                "handle",
                                                "tl::fp8_transpose_v_128x128_fa3_src_ldsm_stsm",
                                                v_vt_smem_0.access_ptr("r"),
                                                v_tc_smem_0.access_ptr("w"),
                                            )
                                    else:
                                        T.tma_copy(
                                            v[tile_b, (n_idx - 1) * block_n:n_idx * block_n, head_kv, :],
                                            v_smem_0,
                                            barrier=v_raw_full,
                                        )
                                        T.barrier_arrive(v_raw_full)
                                        T.barrier_wait(v_raw_full, gi_vp % 2)
                                        if use_fa3_pv_extern:
                                            if block_n == 224:
                                                T.call_extern(
                                                    "handle",
                                                    "tl::fp8_transpose_v_128x224_ldsm_stsm",
                                                    v_smem_0.access_ptr("r"),
                                                    v_tc_smem_0.access_ptr("w"),
                                                )
                                            else:
                                                T.call_extern(
                                                    "handle",
                                                    "tl::fp8_pack_v_128x128_fa3_vt",
                                                    v_smem_0.access_ptr("r"),
                                                    v_vt_smem_0.access_ptr("w"),
                                                )
                                                T.barrier_arrive(v_pack_full)
                                                T.barrier_wait(v_pack_full, gi_vp % 2)
                                                T.call_extern(
                                                    "handle",
                                                    "tl::fp8_transpose_v_128x128_fa3_src_ldsm_stsm",
                                                    v_vt_smem_0.access_ptr("r"),
                                                    v_tc_smem_0.access_ptr("w"),
                                                )
                                        elif use_tl_full_v_transpose:
                                            T.call_extern(
                                                "handle",
                                                "tl::fp8_transpose_v_128x128_tl_full_swizzle",
                                                v_smem_0.access_ptr("r"),
                                                v_tc_smem_0.access_ptr("w"),
                                            )
                                        elif use_tl_pack_ldsm_v_transpose:
                                            T.call_extern(
                                                "handle",
                                                "tl::fp8_pack_v_128x128_tl_ldsm_src",
                                                v_smem_0.access_ptr("r"),
                                                v_perm_smem_0.access_ptr("w"),
                                            )
                                            T.barrier_arrive(v_pack_full)
                                            T.barrier_wait(v_pack_full, gi_vp % 2)
                                            T.call_extern(
                                                "handle",
                                                "tl::fp8_transpose_v_128x128_ldsm_stsm",
                                                v_perm_smem_0.access_ptr("r"),
                                                v_tc_smem_0.access_ptr("w"),
                                            )
                                        elif use_ldsm_v_transpose:
                                            T.call_extern(
                                                "handle",
                                                "tl::fp8_transpose_v_128x128_ldsm_stsm",
                                                v_smem_0.access_ptr("r"),
                                                v_tc_smem_0.access_ptr("w"),
                                            )
                                        else:
                                            for d, n in T.Parallel(dim, block_n):
                                                v_tc_smem_0[d, n] = v_smem_0[n, d]
                                else:
                                    if use_fa3_pv_extern and use_fa3_v_tma_direct:
                                        if block_n == 224:
                                            if tx == 0:
                                                T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                                v_desc = T.create_tma_descriptor(
                                                    TMA_DTYPE_UINT8, 4, v.data,
                                                    dim, heads_kv, seq_len, batch,
                                                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                                    dim, 1, block_n, 1,
                                                    1, 1, 1, 1,
                                                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                                )
                                                tir.call_extern(
                                                    "handle",
                                                    "tl::fp8_tma_load_4d_ptx",
                                                    v_desc,
                                                    v_raw_full[0],
                                                    T.access_ptr(v_vt_smem_1, "w"),
                                                    0,
                                                    head_kv,
                                                    (n_idx - 1) * block_n,
                                                    tile_b,
                                                )
                                        else:
                                            T.tma_copy(
                                                v[tile_b, (n_idx - 1) * block_n:n_idx * block_n, head_kv, :],
                                                v_vt_smem_1,
                                                barrier=v_raw_full,
                                            )
                                        T.barrier_arrive(v_raw_full)
                                        T.barrier_wait(v_raw_full, gi_vp % 2)
                                        if block_n == 224:
                                            if use_fa3_v_inplace_transform:
                                                T.call_extern(
                                                    "handle",
                                                    v_inplace_transform_helper,
                                                    v_vt_smem_1.access_ptr("rw"),
                                                    v_vt_smem_1.access_ptr("rw"),
                                                )
                                            else:
                                                T.call_extern(
                                                    "handle",
                                                    "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                                    v_vt_smem_1.access_ptr("r"),
                                                    v_tc_smem_1.access_ptr("w"),
                                                )
                                        else:
                                            T.call_extern(
                                                "handle",
                                                "tl::fp8_transpose_v_128x128_fa3_src_ldsm_stsm",
                                                v_vt_smem_1.access_ptr("r"),
                                                v_tc_smem_1.access_ptr("w"),
                                            )
                                    else:
                                        T.tma_copy(
                                            v[tile_b, (n_idx - 1) * block_n:n_idx * block_n, head_kv, :],
                                            v_smem_1,
                                            barrier=v_raw_full,
                                        )
                                        T.barrier_arrive(v_raw_full)
                                        T.barrier_wait(v_raw_full, gi_vp % 2)
                                        if use_fa3_pv_extern:
                                            if block_n == 224:
                                                T.call_extern(
                                                    "handle",
                                                    "tl::fp8_transpose_v_128x224_ldsm_stsm",
                                                    v_smem_1.access_ptr("r"),
                                                    v_tc_smem_1.access_ptr("w"),
                                                )
                                            else:
                                                T.call_extern(
                                                    "handle",
                                                    "tl::fp8_pack_v_128x128_fa3_vt",
                                                    v_smem_1.access_ptr("r"),
                                                    v_vt_smem_1.access_ptr("w"),
                                                )
                                                T.barrier_arrive(v_pack_full)
                                                T.barrier_wait(v_pack_full, gi_vp % 2)
                                                T.call_extern(
                                                    "handle",
                                                    "tl::fp8_transpose_v_128x128_fa3_src_ldsm_stsm",
                                                    v_vt_smem_1.access_ptr("r"),
                                                    v_tc_smem_1.access_ptr("w"),
                                                )
                                        elif use_tl_full_v_transpose:
                                            T.call_extern(
                                                "handle",
                                                "tl::fp8_transpose_v_128x128_tl_full_swizzle",
                                                v_smem_1.access_ptr("r"),
                                                v_tc_smem_1.access_ptr("w"),
                                            )
                                        elif use_tl_pack_ldsm_v_transpose:
                                            T.call_extern(
                                                "handle",
                                                "tl::fp8_pack_v_128x128_tl_ldsm_src",
                                                v_smem_1.access_ptr("r"),
                                                v_perm_smem_1.access_ptr("w"),
                                            )
                                            T.barrier_arrive(v_pack_full)
                                            T.barrier_wait(v_pack_full, gi_vp % 2)
                                            T.call_extern(
                                                "handle",
                                                "tl::fp8_transpose_v_128x128_ldsm_stsm",
                                                v_perm_smem_1.access_ptr("r"),
                                                v_tc_smem_1.access_ptr("w"),
                                            )
                                        elif use_ldsm_v_transpose:
                                            T.call_extern(
                                                "handle",
                                                "tl::fp8_transpose_v_128x128_ldsm_stsm",
                                                v_smem_1.access_ptr("r"),
                                                v_tc_smem_1.access_ptr("w"),
                                            )
                                        else:
                                            for d, n in T.Parallel(dim, block_n):
                                                v_tc_smem_1[d, n] = v_smem_1[n, d]
                                T.barrier_arrive(v_full)
                                gi_vp = gi_vp + 1
                            T.barrier_wait(k_empty, (gi_kp + 1) % 2)
                            if gi_kp % 2 == 0:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :],
                                    k_smem_0,
                                    barrier=k_full,
                                )
                            else:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :],
                                    k_smem_1,
                                    barrier=k_full,
                                )
                            T.barrier_arrive(k_full)
                            gi_kp = gi_kp + 1

                        if use_fa3_v_inplace_transform:
                            if gi_vp >= 2:
                                if gi_vp % 2 == 0:
                                    T.barrier_wait(v_empty_0, (gi_vp // 2 - 1) % 2)
                                else:
                                    T.barrier_wait(v_empty_1, (gi_vp // 2 - 1) % 2)
                        else:
                            T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                        if gi_vp % 2 == 0:
                            if use_fa3_pv_extern and use_fa3_v_tma_direct:
                                if block_n == 224:
                                    if tx == 0:
                                        T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                        v_desc_tail = T.create_tma_descriptor(
                                            TMA_DTYPE_UINT8, 4, v.data,
                                            dim, heads_kv, seq_len, batch,
                                            1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                            dim, 1, block_n, 1,
                                            1, 1, 1, 1,
                                            TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                            TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                        )
                                        tir.call_extern(
                                            "handle",
                                            "tl::fp8_tma_load_4d_ptx",
                                            v_desc_tail,
                                            v_raw_full[0],
                                            T.access_ptr(v_vt_smem_0, "w"),
                                            0,
                                            head_kv,
                                            (loop_range - 1) * block_n,
                                            tile_b,
                                        )
                                else:
                                    T.tma_copy(
                                        v[tile_b, (loop_range - 1) * block_n:loop_range * block_n, head_kv, :],
                                        v_vt_smem_0,
                                        barrier=v_raw_full,
                                    )
                                T.barrier_arrive(v_raw_full)
                                T.barrier_wait(v_raw_full, gi_vp % 2)
                                if block_n == 224:
                                    if use_fa3_v_inplace_transform:
                                        T.call_extern(
                                            "handle",
                                            v_inplace_transform_helper,
                                            v_vt_smem_0.access_ptr("rw"),
                                            v_vt_smem_0.access_ptr("rw"),
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                            v_vt_smem_0.access_ptr("r"),
                                            v_tc_smem_0.access_ptr("w"),
                                        )
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x128_fa3_src_ldsm_stsm",
                                        v_vt_smem_0.access_ptr("r"),
                                        v_tc_smem_0.access_ptr("w"),
                                    )
                            else:
                                T.tma_copy(
                                    v[tile_b, (loop_range - 1) * block_n:loop_range * block_n, head_kv, :],
                                    v_smem_0,
                                    barrier=v_raw_full,
                                )
                                T.barrier_arrive(v_raw_full)
                                T.barrier_wait(v_raw_full, gi_vp % 2)
                                if use_fa3_pv_extern:
                                    if block_n == 224:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_transpose_v_128x224_ldsm_stsm",
                                            v_smem_0.access_ptr("r"),
                                            v_tc_smem_0.access_ptr("w"),
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pack_v_128x128_fa3_vt",
                                            v_smem_0.access_ptr("r"),
                                            v_vt_smem_0.access_ptr("w"),
                                        )
                                        T.barrier_arrive(v_pack_full)
                                        T.barrier_wait(v_pack_full, gi_vp % 2)
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_transpose_v_128x128_fa3_src_ldsm_stsm",
                                            v_vt_smem_0.access_ptr("r"),
                                            v_tc_smem_0.access_ptr("w"),
                                        )
                                elif use_tl_full_v_transpose:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x128_tl_full_swizzle",
                                        v_smem_0.access_ptr("r"),
                                        v_tc_smem_0.access_ptr("w"),
                                    )
                                elif use_tl_pack_ldsm_v_transpose:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pack_v_128x128_tl_ldsm_src",
                                        v_smem_0.access_ptr("r"),
                                        v_perm_smem_0.access_ptr("w"),
                                    )
                                    T.barrier_arrive(v_pack_full)
                                    T.barrier_wait(v_pack_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x128_ldsm_stsm",
                                        v_perm_smem_0.access_ptr("r"),
                                        v_tc_smem_0.access_ptr("w"),
                                    )
                                elif use_ldsm_v_transpose:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x128_ldsm_stsm",
                                        v_smem_0.access_ptr("r"),
                                        v_tc_smem_0.access_ptr("w"),
                                    )
                                else:
                                    for d, n in T.Parallel(dim, block_n):
                                        v_tc_smem_0[d, n] = v_smem_0[n, d]
                        else:
                            if use_fa3_pv_extern and use_fa3_v_tma_direct:
                                if block_n == 224:
                                    if tx == 0:
                                        T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                        v_desc_tail = T.create_tma_descriptor(
                                            TMA_DTYPE_UINT8, 4, v.data,
                                            dim, heads_kv, seq_len, batch,
                                            1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                            dim, 1, block_n, 1,
                                            1, 1, 1, 1,
                                            TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                            TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                        )
                                        tir.call_extern(
                                            "handle",
                                            "tl::fp8_tma_load_4d_ptx",
                                            v_desc_tail,
                                            v_raw_full[0],
                                            T.access_ptr(v_vt_smem_1, "w"),
                                            0,
                                            head_kv,
                                            (loop_range - 1) * block_n,
                                            tile_b,
                                        )
                                else:
                                    T.tma_copy(
                                        v[tile_b, (loop_range - 1) * block_n:loop_range * block_n, head_kv, :],
                                        v_vt_smem_1,
                                        barrier=v_raw_full,
                                    )
                                T.barrier_arrive(v_raw_full)
                                T.barrier_wait(v_raw_full, gi_vp % 2)
                                if block_n == 224:
                                    if use_fa3_v_inplace_transform:
                                        T.call_extern(
                                            "handle",
                                            v_inplace_transform_helper,
                                            v_vt_smem_1.access_ptr("rw"),
                                            v_vt_smem_1.access_ptr("rw"),
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                            v_vt_smem_1.access_ptr("r"),
                                            v_tc_smem_1.access_ptr("w"),
                                        )
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x128_fa3_src_ldsm_stsm",
                                        v_vt_smem_1.access_ptr("r"),
                                        v_tc_smem_1.access_ptr("w"),
                                    )
                            else:
                                T.tma_copy(
                                    v[tile_b, (loop_range - 1) * block_n:loop_range * block_n, head_kv, :],
                                    v_smem_1,
                                    barrier=v_raw_full,
                                )
                                T.barrier_arrive(v_raw_full)
                                T.barrier_wait(v_raw_full, gi_vp % 2)
                                if use_fa3_pv_extern:
                                    if block_n == 224:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_transpose_v_128x224_ldsm_stsm",
                                            v_smem_1.access_ptr("r"),
                                            v_tc_smem_1.access_ptr("w"),
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pack_v_128x128_fa3_vt",
                                            v_smem_1.access_ptr("r"),
                                            v_vt_smem_1.access_ptr("w"),
                                        )
                                        T.barrier_arrive(v_pack_full)
                                        T.barrier_wait(v_pack_full, gi_vp % 2)
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_transpose_v_128x128_fa3_src_ldsm_stsm",
                                            v_vt_smem_1.access_ptr("r"),
                                            v_tc_smem_1.access_ptr("w"),
                                        )
                                elif use_tl_full_v_transpose:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x128_tl_full_swizzle",
                                        v_smem_1.access_ptr("r"),
                                        v_tc_smem_1.access_ptr("w"),
                                    )
                                elif use_tl_pack_ldsm_v_transpose:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pack_v_128x128_tl_ldsm_src",
                                        v_smem_1.access_ptr("r"),
                                        v_perm_smem_1.access_ptr("w"),
                                    )
                                    T.barrier_arrive(v_pack_full)
                                    T.barrier_wait(v_pack_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x128_ldsm_stsm",
                                        v_perm_smem_1.access_ptr("r"),
                                        v_tc_smem_1.access_ptr("w"),
                                    )
                                elif use_ldsm_v_transpose:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x128_ldsm_stsm",
                                        v_smem_1.access_ptr("r"),
                                        v_tc_smem_1.access_ptr("w"),
                                    )
                                else:
                                    for d, n in T.Parallel(dim, block_n):
                                        v_tc_smem_1[d, n] = v_smem_1[n, d]
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
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base:row_base + half_m, tile_h, :],
                                   q_shared_1, barrier=q_full_1)
                        T.barrier_arrive(q_full_1)
                        T.barrier_wait(q_full_1, gi_q1 % 2)
                        gi_q1 = gi_q1 + 1

                        if use_ptx_pv_accumulate:
                            if block_n != 224:
                                for i, j in T.Parallel(half_m, dim):
                                    acc_o_1[i, j] = 0.0
                            T.call_extern(
                                "handle",
                                "tl::fp8_zero_raw_acc_64",
                                acc_o_1.data,
                            )
                        else:
                            T.clear(acc_o_1)
                        T.clear(ls_1)
                        T.fill(sm_1, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc1 % 2)
                            if gi_kc1 % 2 == 0:
                                T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            else:
                                T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_s_1, num_regs=112 if block_n == 224 else 64)
                            T.barrier_arrive(k_empty)
                            if fuse_score_scale_into_softmax:
                                online_softmax_1(
                                    acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1,
                                    q_scale[tile_b, tile_h, q_scale_idx]
                                    * k_scale[tile_b, head_kv, n_idx],
                                )
                            else:
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_1[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                            if not use_register_p and not use_fa3_pv_extern:
                                T.copy(acc_s_1, p_shared_1)
                            if use_ptx_pv_accumulate:
                                T.copy(ss_1, ss_shared_1)

                            T.barrier_wait(v_full, gi_vc1 % 2)
                            if not use_ptx_pv_accumulate:
                                T.clear(acc_delta_1)
                            if use_fa3_pv_extern:
                                if gi_vc1 % 2 == 0:
                                    if use_ptx_pv_accumulate:
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
                                    elif use_ptx_pv:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_ptx_unit_from_acc_pretransposed_to_smem_64x128x128",
                                            acc_s_1.data,
                                            v_tc_smem_0.access_ptr("r"),
                                            4,
                                            acc_delta_shared_1.access_ptr("w"),
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_cute_unit_from_acc_pretransposed_64x128x128",
                                            acc_s_1.data,
                                            v_tc_smem_0.access_ptr("r"),
                                            4,
                                            acc_delta_shared_1.access_ptr("w"),
                                        )
                                else:
                                    if use_ptx_pv_accumulate:
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
                                    elif use_ptx_pv:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_ptx_unit_from_acc_pretransposed_to_smem_64x128x128",
                                            acc_s_1.data,
                                            v_tc_smem_1.access_ptr("r"),
                                            4,
                                            acc_delta_shared_1.access_ptr("w"),
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_cute_unit_from_acc_pretransposed_64x128x128",
                                            acc_s_1.data,
                                            v_tc_smem_1.access_ptr("r"),
                                            4,
                                            acc_delta_shared_1.access_ptr("w"),
                                        )
                                if not use_ptx_pv_accumulate:
                                    T.copy(acc_delta_shared_1, acc_delta_1)
                            elif use_full_pv_gemm:
                                T.copy(p_shared_1, p_fp8_1)
                                if gi_vc1 % 2 == 0:
                                    T.wgmma_gemm(p_fp8_1, v_tc_smem_0, acc_delta_1,
                                                 transpose_B=True,
                                                 policy=T.GemmWarpPolicy.FullRow,
                                                 clear_accum=False)
                                else:
                                    T.wgmma_gemm(p_fp8_1, v_tc_smem_1, acc_delta_1,
                                                 transpose_B=True,
                                                 policy=T.GemmWarpPolicy.FullRow,
                                                 clear_accum=False)
                                T.wait_wgmma(0)
                            else:
                                for pv_k in range(block_n // fp8_k):
                                    if use_register_p:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_acc_to_pv_a_frag_64x128",
                                            acc_s_1.data,
                                            p_fp8_1.data,
                                            pv_k,
                                        )
                                    else:
                                        T.copy(p_shared_1[:, pv_k * fp8_k:(pv_k + 1) * fp8_k],
                                               p_fp8_1)
                                    for d, n in T.Parallel(dim, fp8_k):
                                        if gi_vc1 % 2 == 0:
                                            v_tc_shared_1[d, n] = v_tc_smem_0[d, pv_k * fp8_k + n]
                                        else:
                                            v_tc_shared_1[d, n] = v_tc_smem_1[d, pv_k * fp8_k + n]
                                    T.wgmma_gemm(p_fp8_1, v_tc_shared_1, acc_delta_1,
                                                 transpose_B=True,
                                                 policy=T.GemmWarpPolicy.FullRow,
                                                 clear_accum=False)
                                    T.wait_wgmma(0)
                            if use_fa3_v_inplace_transform:
                                if gi_vc1 % 2 == 0:
                                    T.barrier_arrive(v_empty_0)
                                else:
                                    T.barrier_arrive(v_empty_1)
                            else:
                                T.barrier_arrive(v_empty)
                            gi_vc1 = gi_vc1 + 1
                            if not use_ptx_pv_accumulate:
                                rescale_1(acc_o_1, ss_1)
                                for i, j in T.Parallel(half_m, dim):
                                    acc_o_1[i, j] += (
                                        acc_delta_1[i, j] * v_scale[tile_b, head_kv, n_idx]
                                    )
                            gi_kc1 = gi_kc1 + 1

                        if use_ptx_pv_accumulate:
                            T.copy(ls_1, ls_shared_1)
                            if block_n == 224 and use_ptx_pv_fa3_epilogue_store:
                                T.copy(acc_o_1, acc_o_layout_seed_1)
                            if use_ptx_pv_direct_store:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_fa3_raw_acc_store_global_64x128",
                                    acc_o_1.data,
                                    ls_shared_1.access_ptr("r"),
                                    4,
                                    T.address_of(output[tile_b, row_base, tile_h, 0]),
                                    heads * dim,
                                )
                            elif use_ptx_pv_fa3_epilogue_store:
                                if use_ptx_pv_fa3_epilogue_reuse_v_smem:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                                        acc_o_1.data,
                                        ls_shared_1.access_ptr("r"),
                                        4,
                                        v_tc_smem_0.access_ptr("w"),
                                        T.address_of(output[tile_b, row_base, tile_h, 0]),
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(barrier_id=3, arrive_count=128)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                                        v_tc_smem_0.access_ptr("r"),
                                        T.address_of(output[tile_b, row_base, tile_h, 0]),
                                        heads * dim,
                                    )
                                else:
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
                            else:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_fa3_raw_acc_store_64x128",
                                    acc_o_1.data,
                                    ls_shared_1.access_ptr("r"),
                                    4,
                                    o_shared_1.access_ptr("w"),
                                )
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=3, arrive_count=128)
                                T.copy(o_shared_1,
                                       output[tile_b, row_base:row_base + half_m, tile_h, :])
                        else:
                            for i, j in T.Parallel(half_m, dim):
                                acc_o_1[i, j] /= ls_1[i]
                        if use_fa3_pv_extern and not use_ptx_pv_accumulate:
                            T.copy(acc_o_1, output[tile_b, row_base:row_base + half_m, tile_h, :])
                        elif not use_fa3_pv_extern:
                            T.copy(acc_o_1, o_shared_1)
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=3, arrive_count=128)
                            T.copy(o_shared_1,
                                   output[tile_b, row_base:row_base + half_m, tile_h, :])
                        for i in T.Parallel(half_m):
                            ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * scale
                        T.copy(ls_1, lse[tile_b, tile_h, row_base:row_base + half_m])

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
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                                   q_shared_2, barrier=q_full_2)
                        T.barrier_arrive(q_full_2)
                        T.barrier_wait(q_full_2, gi_q2 % 2)
                        gi_q2 = gi_q2 + 1

                        if use_ptx_pv_accumulate:
                            if block_n != 224:
                                for i, j in T.Parallel(half_m, dim):
                                    acc_o_2[i, j] = 0.0
                            T.call_extern(
                                "handle",
                                "tl::fp8_zero_raw_acc_64",
                                acc_o_2.data,
                            )
                        else:
                            T.clear(acc_o_2)
                        T.clear(ls_2)
                        T.fill(sm_2, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc2 % 2)
                            if gi_kc2 % 2 == 0:
                                T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            else:
                                T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_s_2, num_regs=112 if block_n == 224 else 64)
                            T.barrier_arrive(k_empty)
                            if fuse_score_scale_into_softmax:
                                online_softmax_2(
                                    acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2,
                                    q_scale[tile_b, tile_h, q_scale_idx]
                                    * k_scale[tile_b, head_kv, n_idx],
                                )
                            else:
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_2[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                            if not use_register_p and not use_fa3_pv_extern:
                                T.copy(acc_s_2, p_shared_2)
                            if use_ptx_pv_accumulate:
                                T.copy(ss_2, ss_shared_2)

                            T.barrier_wait(v_full, gi_vc2 % 2)
                            if not use_ptx_pv_accumulate:
                                T.clear(acc_delta_2)
                            if use_fa3_pv_extern:
                                if gi_vc2 % 2 == 0:
                                    if use_ptx_pv_accumulate:
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
                                    elif use_ptx_pv:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_ptx_unit_from_acc_pretransposed_to_smem_64x128x128",
                                            acc_s_2.data,
                                            v_tc_smem_0.access_ptr("r"),
                                            4,
                                            acc_delta_shared_2.access_ptr("w"),
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_cute_unit_from_acc_pretransposed_64x128x128",
                                            acc_s_2.data,
                                            v_tc_smem_0.access_ptr("r"),
                                            4,
                                            acc_delta_shared_2.access_ptr("w"),
                                        )
                                else:
                                    if use_ptx_pv_accumulate:
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
                                    elif use_ptx_pv:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_ptx_unit_from_acc_pretransposed_to_smem_64x128x128",
                                            acc_s_2.data,
                                            v_tc_smem_1.access_ptr("r"),
                                            4,
                                            acc_delta_shared_2.access_ptr("w"),
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_cute_unit_from_acc_pretransposed_64x128x128",
                                            acc_s_2.data,
                                            v_tc_smem_1.access_ptr("r"),
                                            4,
                                            acc_delta_shared_2.access_ptr("w"),
                                        )
                                if not use_ptx_pv_accumulate:
                                    T.copy(acc_delta_shared_2, acc_delta_2)
                            elif use_full_pv_gemm:
                                T.copy(p_shared_2, p_fp8_2)
                                if gi_vc2 % 2 == 0:
                                    T.wgmma_gemm(p_fp8_2, v_tc_smem_0, acc_delta_2,
                                                 transpose_B=True,
                                                 policy=T.GemmWarpPolicy.FullRow,
                                                 clear_accum=False)
                                else:
                                    T.wgmma_gemm(p_fp8_2, v_tc_smem_1, acc_delta_2,
                                                 transpose_B=True,
                                                 policy=T.GemmWarpPolicy.FullRow,
                                                 clear_accum=False)
                                T.wait_wgmma(0)
                            else:
                                for pv_k in range(block_n // fp8_k):
                                    if use_register_p:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_acc_to_pv_a_frag_64x128",
                                            acc_s_2.data,
                                            p_fp8_2.data,
                                            pv_k,
                                        )
                                    else:
                                        T.copy(p_shared_2[:, pv_k * fp8_k:(pv_k + 1) * fp8_k],
                                               p_fp8_2)
                                    for d, n in T.Parallel(dim, fp8_k):
                                        if gi_vc2 % 2 == 0:
                                            v_tc_shared_2[d, n] = v_tc_smem_0[d, pv_k * fp8_k + n]
                                        else:
                                            v_tc_shared_2[d, n] = v_tc_smem_1[d, pv_k * fp8_k + n]
                                    T.wgmma_gemm(p_fp8_2, v_tc_shared_2, acc_delta_2,
                                                 transpose_B=True,
                                                 policy=T.GemmWarpPolicy.FullRow,
                                                 clear_accum=False)
                                    T.wait_wgmma(0)
                            if use_fa3_v_inplace_transform:
                                if gi_vc2 % 2 == 0:
                                    T.barrier_arrive(v_empty_0)
                                else:
                                    T.barrier_arrive(v_empty_1)
                            else:
                                T.barrier_arrive(v_empty)
                            gi_vc2 = gi_vc2 + 1
                            if not use_ptx_pv_accumulate:
                                rescale_2(acc_o_2, ss_2)
                                for i, j in T.Parallel(half_m, dim):
                                    acc_o_2[i, j] += (
                                        acc_delta_2[i, j] * v_scale[tile_b, head_kv, n_idx]
                                    )
                            gi_kc2 = gi_kc2 + 1

                        if use_ptx_pv_accumulate:
                            T.copy(ls_2, ls_shared_2)
                            if block_n == 224 and use_ptx_pv_fa3_epilogue_store:
                                T.copy(acc_o_2, acc_o_layout_seed_2)
                            if use_ptx_pv_direct_store:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_fa3_raw_acc_store_global_64x128",
                                    acc_o_2.data,
                                    ls_shared_2.access_ptr("r"),
                                    4,
                                    T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                                    heads * dim,
                                )
                            elif use_ptx_pv_fa3_epilogue_store:
                                if use_ptx_pv_fa3_epilogue_reuse_v_smem:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                                        acc_o_2.data,
                                        ls_shared_2.access_ptr("r"),
                                        4,
                                        v_tc_smem_1.access_ptr("w"),
                                        T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                                    )
                                    T.fence_proxy_async()
                                    T.sync_threads(barrier_id=4, arrive_count=128)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                                        v_tc_smem_1.access_ptr("r"),
                                        T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                                        heads * dim,
                                    )
                                else:
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
                            else:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_fa3_raw_acc_store_64x128",
                                    acc_o_2.data,
                                    ls_shared_2.access_ptr("r"),
                                    4,
                                    o_shared_2.access_ptr("w"),
                                )
                                T.fence_proxy_async()
                                T.sync_threads(barrier_id=4, arrive_count=128)
                                T.copy(
                                    o_shared_2,
                                    output[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                                )
                        else:
                            for i, j in T.Parallel(half_m, dim):
                                acc_o_2[i, j] /= ls_2[i]
                        if use_fa3_pv_extern and not use_ptx_pv_accumulate:
                            T.copy(
                                acc_o_2,
                                output[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                            )
                        elif not use_fa3_pv_extern:
                            T.copy(acc_o_2, o_shared_2)
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.copy(
                                o_shared_2,
                                output[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                            )
                        for i in T.Parallel(half_m):
                            ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                        T.copy(ls_2, lse[tile_b, tile_h, row_base + half_m:row_base + block_m])

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_bn224_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("GQAFwdFP8Fa3ContractPtxAccBN224Kernel requires dim == 128.")
    if seq_len % 224 != 0:
        raise ValueError("GQAFwdFP8Fa3ContractPtxAccBN224Kernel requires seq_len % 224 == 0.")

    block_m = 64
    block_n = 224
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

        online_softmax = make_online_softmax(scale, accum_dtype, block_m, block_n)

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
            with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=128) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, dim], fp8_dtype)
                k_smem = T.alloc_shared([block_n, dim], fp8_dtype)
                v_smem = T.alloc_shared([block_n, dim], fp8_dtype)
                v_tc_smem = T.alloc_shared([dim, block_n], fp8_dtype)
                acc_o_layout_seed = T.alloc_shared([block_m, dim], out_dtype)
                ss_shared = T.alloc_shared([block_m], accum_dtype)
                ls_shared = T.alloc_shared([block_m], accum_dtype)

                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                sm = T.alloc_fragment([block_m], accum_dtype)
                smp = T.alloc_fragment([block_m], accum_dtype)
                ss = T.alloc_fragment([block_m], accum_dtype)
                ssum = T.alloc_fragment([block_m], accum_dtype)
                ls_frag = T.alloc_fragment([block_m], accum_dtype)

                T.annotate_layout({
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                })

                head_kv = by // groups
                row_base = bx * block_m
                q_scale_idx = row_base // scale_block

                T.copy(q[bz, row_base:row_base + block_m, by, :], q_shared)
                T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o.data)
                T.clear(ls_frag)
                T.fill(sm, -T.infinity(accum_dtype))

                for n_idx in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=0):
                    T.copy(k[bz, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :], k_smem)
                    T.copy(v[bz, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :], v_smem)
                    T.wgmma_gemm(
                        q_shared,
                        k_smem,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        clear_accum=True,
                    )
                    T.wait_wgmma(0)
                    T.warpgroup_fence_operand(acc_s, num_regs=112)

                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] *= (
                            q_scale[bz, by, q_scale_idx] * k_scale[bz, head_kv, n_idx]
                        )
                    online_softmax(acc_s, sm, smp, ss, ssum, ls_frag)
                    T.copy(ss, ss_shared)

                    T.call_extern(
                        "handle",
                        "tl::fp8_transpose_v_128x224_ldsm_stsm",
                        v_smem.access_ptr("r"),
                        v_tc_smem.access_ptr("w"),
                    )
                    T.call_extern(
                        "handle",
                        "tl::fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224",
                        acc_s.data,
                        v_tc_smem.access_ptr("r"),
                        4,
                        ss_shared.access_ptr("r"),
                        v_scale[bz, head_kv, n_idx],
                        acc_o.data,
                    )

                T.copy(ls_frag, ls_shared)
                # Seed TileLang's fragment layout map for acc_o before the FA3
                # raw epilogue helper consumes acc_o.data in this slim kernel.
                T.copy(acc_o, acc_o_layout_seed)
                T.call_extern(
                    "handle",
                    "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                    acc_o.data,
                    ls_shared.access_ptr("r"),
                    4,
                    v_tc_smem.access_ptr("w"),
                    T.address_of(output[bz, row_base, by, 0]),
                )
                T.fence_proxy_async()
                T.sync_threads(barrier_id=3, arrive_count=128)
                T.call_extern(
                    "handle",
                    "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                    v_tc_smem.access_ptr("r"),
                    T.address_of(output[bz, row_base, by, 0]),
                    heads * dim,
                )

                for i in T.Parallel(block_m):
                    ls_frag[i] = T.log2(ls_frag[i]) + sm[i] * scale
                T.copy(ls_frag, lse[bz, by, row_base:row_base + block_m])

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    streaming_p: bool = False,
    plain_wait: bool = False,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("BN224 overlap FP8 GQA requires dim == 128.")
    if seq_len % 224 != 0:
        raise ValueError("BN224 overlap FP8 GQA requires seq_len % 224 == 0.")

    block_m = 128
    half_m = block_m // 2
    block_n = 224
    fp8_k = 32
    groups = heads // heads_kv
    accum_dtype = "float"
    fp8_dtype = "float8_e4m3fn"
    scale = make_log2e_scale(dim)
    scale_block = 128
    scale_blocks = (seq_len + scale_block - 1) // scale_block
    pack_p_helper = (
        "tl::fp8_pack_p_fa3_raw_64x128x224_to_smem_streaming"
        if streaming_p else
        "tl::fp8_pack_p_fa3_raw_64x128x224_to_smem"
    )
    pv_begin_accumulate_helper = (
        "tl::fp8_pv_ptx_unit_begin_accumulate_from_p_smem_streaming_fa3_raw_64x128x224"
        if streaming_p else
        "tl::fp8_pv_ptx_unit_begin_accumulate_from_p_smem_fa3_raw_64x128x224"
    )
    wait_wgmma_helper = (
        "tl::wait_wgmma_plain_fence_noarg"
        if plain_wait else
        "tl::wait_wgmma_anchor"
    )

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

        online_softmax_1 = make_online_softmax(scale, accum_dtype, half_m, block_n)
        online_softmax_2 = make_online_softmax(scale, accum_dtype, half_m, block_n)

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
                k_smem_0 = T.alloc_shared([block_n, dim], fp8_dtype)
                k_smem_1 = T.alloc_shared([block_n, dim], fp8_dtype)
                v_vt_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_vt_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_tc_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_tc_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                ss_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ss_shared_2 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_2 = T.alloc_shared([half_m], accum_dtype)
                acc_o_layout_seed = T.alloc_shared([half_m, dim], fp8_dtype)
                p_smem_1 = T.alloc_shared([half_m, block_n], fp8_dtype)
                p_smem_2 = T.alloc_shared([half_m, block_n], fp8_dtype)
                anchor_sink = T.alloc_shared([2], "int32")

                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
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
                v_empty = T.alloc_barrier(arrive_count=256)
                q_full_1 = T.alloc_barrier(arrive_count=128)
                q_full_2 = T.alloc_barrier(arrive_count=128)

                T.annotate_layout({
                    q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                    q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                })
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
                    for tile_b, tile_hkv, _tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len, block_m), groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_hkv
                        loop_range = T.ceildiv(seq_len, block_n)
                        T.reads(v[tile_b, 0:seq_len, head_kv, :])
                        T.writes(
                            v_vt_smem_0[0:dim, 0:block_n],
                            v_vt_smem_1[0:dim, 0:block_n],
                        )

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_empty, (gi_kp + 1) % 2)
                            if gi_kp % 2 == 0:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :],
                                    k_smem_0,
                                    barrier=k_full,
                                )
                            else:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :],
                                    k_smem_1,
                                    barrier=k_full,
                                )
                            T.barrier_arrive(k_full)
                            if n_idx > 0:
                                T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                                if gi_vp % 2 == 0:
                                    if tx == 0:
                                        T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                        v_desc = T.create_tma_descriptor(
                                            TMA_DTYPE_UINT8, 4, v.data,
                                            dim, heads_kv, seq_len, batch,
                                            1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                            dim, 1, block_n, 1,
                                            1, 1, 1, 1,
                                            TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                            TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                        )
                                        tir.call_extern(
                                            "handle",
                                            "tl::fp8_tma_load_4d_ptx",
                                            v_desc,
                                            v_raw_full[0],
                                            T.access_ptr(v_vt_smem_0, "w"),
                                            0,
                                            head_kv,
                                            (n_idx - 1) * block_n,
                                            tile_b,
                                        )
                                    T.barrier_arrive(v_raw_full)
                                    T.barrier_wait(v_raw_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                        v_vt_smem_0.access_ptr("r"),
                                        v_tc_smem_0.access_ptr("w"),
                                    )
                                else:
                                    if tx == 0:
                                        T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                        v_desc = T.create_tma_descriptor(
                                            TMA_DTYPE_UINT8, 4, v.data,
                                            dim, heads_kv, seq_len, batch,
                                            1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                            dim, 1, block_n, 1,
                                            1, 1, 1, 1,
                                            TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                            TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                        )
                                        tir.call_extern(
                                            "handle",
                                            "tl::fp8_tma_load_4d_ptx",
                                            v_desc,
                                            v_raw_full[0],
                                            T.access_ptr(v_vt_smem_1, "w"),
                                            0,
                                            head_kv,
                                            (n_idx - 1) * block_n,
                                            tile_b,
                                        )
                                    T.barrier_arrive(v_raw_full)
                                    T.barrier_wait(v_raw_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                        v_vt_smem_1.access_ptr("r"),
                                        v_tc_smem_1.access_ptr("w"),
                                    )
                                T.barrier_arrive(v_full)
                                gi_vp = gi_vp + 1
                            gi_kp = gi_kp + 1

                        T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                        if gi_vp % 2 == 0:
                            if visible_pv_shared_v:
                                for d, n in T.Parallel(dim, block_n):
                                    v_vt_smem_0[d, n] = v[
                                        tile_b,
                                        (loop_range - 1) * block_n + n,
                                        head_kv,
                                        d,
                                    ]
                            if tx == 0:
                                if not visible_pv_shared_v:
                                    T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                    v_desc_tail = T.create_tma_descriptor(
                                        TMA_DTYPE_UINT8, 4, v.data,
                                        dim, heads_kv, seq_len, batch,
                                        1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                        dim, 1, block_n, 1,
                                        1, 1, 1, 1,
                                        TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                        TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                    )
                                    tir.call_extern(
                                        "handle",
                                        "tl::fp8_tma_load_4d_ptx",
                                        v_desc_tail,
                                        v_raw_full[0],
                                        T.access_ptr(v_vt_smem_0, "w"),
                                        0,
                                        head_kv,
                                        (loop_range - 1) * block_n,
                                        tile_b,
                                    )
                            if not visible_pv_shared_v:
                                T.barrier_arrive(v_raw_full)
                                T.barrier_wait(v_raw_full, gi_vp % 2)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                    v_vt_smem_0.access_ptr("r"),
                                    v_tc_smem_0.access_ptr("w"),
                                )
                        else:
                            if visible_pv_shared_v:
                                for d, n in T.Parallel(dim, block_n):
                                    v_vt_smem_1[d, n] = v[
                                        tile_b,
                                        (loop_range - 1) * block_n + n,
                                        head_kv,
                                        d,
                                    ]
                            if tx == 0:
                                if not visible_pv_shared_v:
                                    T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                    v_desc_tail = T.create_tma_descriptor(
                                        TMA_DTYPE_UINT8, 4, v.data,
                                        dim, heads_kv, seq_len, batch,
                                        1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                        dim, 1, block_n, 1,
                                        1, 1, 1, 1,
                                        TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                        TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                    )
                                    tir.call_extern(
                                        "handle",
                                        "tl::fp8_tma_load_4d_ptx",
                                        v_desc_tail,
                                        v_raw_full[0],
                                        T.access_ptr(v_vt_smem_1, "w"),
                                        0,
                                        head_kv,
                                        (loop_range - 1) * block_n,
                                        tile_b,
                                    )
                            if not visible_pv_shared_v:
                                T.barrier_arrive(v_raw_full)
                                T.barrier_wait(v_raw_full, gi_vp % 2)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                    v_vt_smem_1.access_ptr("r"),
                                    v_tc_smem_1.access_ptr("w"),
                                )
                        T.barrier_arrive(v_full)
                        gi_vp = gi_vp + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
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
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base:row_base + half_m, tile_h, :],
                                   q_shared_1, barrier=q_full_1)
                        T.barrier_arrive(q_full_1)
                        T.barrier_wait(q_full_1, gi_q1 % 2)
                        gi_q1 = gi_q1 + 1

                        T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o_1.data)
                        T.clear(ls_1)
                        T.fill(sm_1, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc1 % 2)
                            T.sync_threads(barrier_id=1, arrive_count=256)
                            if gi_kc1 % 2 == 0:
                                T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            else:
                                T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            if n_idx == 0:
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 2, 256)
                                if plain_wait:
                                    T.call_extern("handle", f"{wait_wgmma_helper}<1>")
                                else:
                                    T.call_extern(
                                        "handle",
                                        f"{wait_wgmma_helper}<1>",
                                        T.address_of(anchor_sink[0]),
                                        gi_kc1,
                                    )
                                T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_1[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                T.call_extern(
                                    "handle",
                                    pack_p_helper,
                                    acc_s_1.data,
                                    p_smem_1.access_ptr("w"),
                                )
                            else:
                                T.barrier_wait(v_full, gi_vc1 % 2)
                                if gi_vc1 % 2 == 0:
                                    T.call_extern(
                                        "handle",
                                        pv_begin_accumulate_helper,
                                        p_smem_1.access_ptr("r"),
                                        v_tc_smem_0.access_ptr("r"),
                                        4,
                                        acc_o_1.data,
                                    )
                                else:
                                    T.call_extern(
                                        "handle",
                                        pv_begin_accumulate_helper,
                                        p_smem_1.access_ptr("r"),
                                        v_tc_smem_1.access_ptr("r"),
                                        4,
                                        acc_o_1.data,
                                    )
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 2, 256)
                                if plain_wait:
                                    T.call_extern("handle", f"{wait_wgmma_helper}<1>")
                                else:
                                    T.call_extern(
                                        "handle",
                                        f"{wait_wgmma_helper}<1>",
                                        T.address_of(anchor_sink[0]),
                                        gi_kc1,
                                    )
                                T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_1[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                T.copy(ss_1, ss_shared_1)
                                if plain_wait:
                                    T.call_extern("handle", f"{wait_wgmma_helper}<0>")
                                else:
                                    T.call_extern(
                                        "handle",
                                        f"{wait_wgmma_helper}<0>",
                                        T.address_of(anchor_sink[0]),
                                        gi_kc1,
                                    )
                                T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                T.barrier_arrive(v_empty)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_fa3_raw_acc_rescale_keep_ptx_layout_64x128",
                                    acc_o_1.data,
                                    ss_shared_1.access_ptr("r"),
                                )
                                T.call_extern(
                                    "handle",
                                    pack_p_helper,
                                    acc_s_1.data,
                                    p_smem_1.access_ptr("w"),
                                )
                                gi_vc1 = gi_vc1 + 1
                            gi_kc1 = gi_kc1 + 1

                        T.barrier_wait(v_full, gi_vc1 % 2)
                        if gi_vc1 % 2 == 0:
                            T.call_extern(
                                "handle",
                                pv_begin_accumulate_helper,
                                p_smem_1.access_ptr("r"),
                                v_tc_smem_0.access_ptr("r"),
                                4,
                                acc_o_1.data,
                            )
                        else:
                            T.call_extern(
                                "handle",
                                pv_begin_accumulate_helper,
                                p_smem_1.access_ptr("r"),
                                v_tc_smem_1.access_ptr("r"),
                                4,
                                acc_o_1.data,
                            )
                        T.call_extern(
                            "handle",
                            "tl::fp8_wgmma_wait0_fence",
                        )
                        T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                        T.barrier_arrive(v_empty)
                        gi_vc1 = gi_vc1 + 1

                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_permute_to_canonical_64x128",
                            acc_o_1.data,
                        )
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_scale_64x128",
                            acc_o_1.data,
                            v_scale[tile_b, head_kv, 0],
                        )
                        T.copy(ls_1, ls_shared_1)
                        T.copy(acc_o_1, acc_o_layout_seed)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                            acc_o_1.data,
                            ls_shared_1.access_ptr("r"),
                            4,
                            v_tc_smem_0.access_ptr("w"),
                            T.address_of(output[tile_b, row_base, tile_h, 0]),
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=3, arrive_count=128)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                            v_tc_smem_0.access_ptr("r"),
                            T.address_of(output[tile_b, row_base, tile_h, 0]),
                            heads * dim,
                        )
                        for i in T.Parallel(half_m):
                            ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * scale
                        T.copy(ls_1, lse[tile_b, tile_h, row_base:row_base + half_m])

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
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                                   q_shared_2, barrier=q_full_2)
                        T.barrier_arrive(q_full_2)
                        T.barrier_wait(q_full_2, gi_q2 % 2)
                        gi_q2 = gi_q2 + 1

                        T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o_2.data)
                        T.clear(ls_2)
                        T.fill(sm_2, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc2 % 2)
                            T.sync_threads(barrier_id=2, arrive_count=256)
                            if gi_kc2 % 2 == 0:
                                T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            else:
                                T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            if n_idx == 0:
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
                                if plain_wait:
                                    T.call_extern("handle", f"{wait_wgmma_helper}<1>")
                                else:
                                    T.call_extern(
                                        "handle",
                                        f"{wait_wgmma_helper}<1>",
                                        T.address_of(anchor_sink[1]),
                                        gi_kc2,
                                    )
                                T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_2[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                T.call_extern(
                                    "handle",
                                    pack_p_helper,
                                    acc_s_2.data,
                                    p_smem_2.access_ptr("w"),
                                )
                            else:
                                T.barrier_wait(v_full, gi_vc2 % 2)
                                if gi_vc2 % 2 == 0:
                                    T.call_extern(
                                        "handle",
                                        pv_begin_accumulate_helper,
                                        p_smem_2.access_ptr("r"),
                                        v_tc_smem_0.access_ptr("r"),
                                        4,
                                        acc_o_2.data,
                                    )
                                else:
                                    T.call_extern(
                                        "handle",
                                        pv_begin_accumulate_helper,
                                        p_smem_2.access_ptr("r"),
                                        v_tc_smem_1.access_ptr("r"),
                                        4,
                                        acc_o_2.data,
                                    )
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
                                if plain_wait:
                                    T.call_extern("handle", f"{wait_wgmma_helper}<1>")
                                else:
                                    T.call_extern(
                                        "handle",
                                        f"{wait_wgmma_helper}<1>",
                                        T.address_of(anchor_sink[1]),
                                        gi_kc2,
                                    )
                                T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_2[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                T.copy(ss_2, ss_shared_2)
                                if plain_wait:
                                    T.call_extern("handle", f"{wait_wgmma_helper}<0>")
                                else:
                                    T.call_extern(
                                        "handle",
                                        f"{wait_wgmma_helper}<0>",
                                        T.address_of(anchor_sink[1]),
                                        gi_kc2,
                                    )
                                T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                T.barrier_arrive(v_empty)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_fa3_raw_acc_rescale_keep_ptx_layout_64x128",
                                    acc_o_2.data,
                                    ss_shared_2.access_ptr("r"),
                                )
                                T.call_extern(
                                    "handle",
                                    pack_p_helper,
                                    acc_s_2.data,
                                    p_smem_2.access_ptr("w"),
                                )
                                gi_vc2 = gi_vc2 + 1
                            gi_kc2 = gi_kc2 + 1

                        T.barrier_wait(v_full, gi_vc2 % 2)
                        if gi_vc2 % 2 == 0:
                            T.call_extern(
                                "handle",
                                pv_begin_accumulate_helper,
                                p_smem_2.access_ptr("r"),
                                v_tc_smem_0.access_ptr("r"),
                                4,
                                acc_o_2.data,
                            )
                        else:
                            T.call_extern(
                                "handle",
                                pv_begin_accumulate_helper,
                                p_smem_2.access_ptr("r"),
                                v_tc_smem_1.access_ptr("r"),
                                4,
                                acc_o_2.data,
                            )
                        T.call_extern(
                            "handle",
                            "tl::fp8_wgmma_wait0_fence",
                        )
                        T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                        T.barrier_arrive(v_empty)
                        gi_vc2 = gi_vc2 + 1

                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_permute_to_canonical_64x128",
                            acc_o_2.data,
                        )
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_scale_64x128",
                            acc_o_2.data,
                            v_scale[tile_b, head_kv, 0],
                        )
                        T.copy(ls_2, ls_shared_2)
                        T.copy(acc_o_2, acc_o_layout_seed)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                            acc_o_2.data,
                            ls_shared_2.access_ptr("r"),
                            4,
                            v_tc_smem_1.access_ptr("w"),
                            T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=4, arrive_count=128)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                            v_tc_smem_1.access_ptr("r"),
                            T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                            heads * dim,
                        )
                        for i in T.Parallel(half_m):
                            ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                        T.copy(ls_2, lse[tile_b, tile_h, row_base + half_m:row_base + block_m])

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_local_p_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("BN224 overlap local-P FP8 GQA requires dim == 128.")
    if seq_len % 224 != 0:
        raise ValueError("BN224 overlap local-P FP8 GQA requires seq_len % 224 == 0.")

    block_m = 128
    half_m = block_m // 2
    block_n = 224
    fp8_k = 32
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

        online_softmax_1 = make_online_softmax(scale, accum_dtype, half_m, block_n)
        online_softmax_2 = make_online_softmax(scale, accum_dtype, half_m, block_n)

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
                k_smem_0 = T.alloc_shared([block_n, dim], fp8_dtype)
                k_smem_1 = T.alloc_shared([block_n, dim], fp8_dtype)
                v_vt_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_vt_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_tc_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_tc_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                ss_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ss_shared_2 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_2 = T.alloc_shared([half_m], accum_dtype)
                p_smem_1 = T.alloc_shared([half_m, block_n], fp8_dtype)
                p_smem_2 = T.alloc_shared([half_m, block_n], fp8_dtype)
                acc_o_layout_seed = T.alloc_shared([half_m, dim], out_dtype)
                anchor_sink = T.alloc_shared([2], "int32")
                p_pv_local_1 = T.alloc_local([112], fp8_dtype)
                p_pv_local_2 = T.alloc_local([112], fp8_dtype)
                acc_delta_local_1 = T.alloc_local([64], accum_dtype)
                acc_delta_local_2 = T.alloc_local([64], accum_dtype)

                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
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
                v_empty = T.alloc_barrier(arrive_count=256)
                q_full_1 = T.alloc_barrier(arrive_count=128)
                q_full_2 = T.alloc_barrier(arrive_count=128)

                T.annotate_layout({
                    q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                    q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                })
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
                    for tile_b, tile_hkv, _tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len, block_m), groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_hkv
                        loop_range = T.ceildiv(seq_len, block_n)
                        T.reads(v[tile_b, 0:seq_len, head_kv, :])
                        T.writes(
                            v_vt_smem_0[0:dim, 0:block_n],
                            v_vt_smem_1[0:dim, 0:block_n],
                        )

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_empty, (gi_kp + 1) % 2)
                            if gi_kp % 2 == 0:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :],
                                    k_smem_0,
                                    barrier=k_full,
                                )
                            else:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :],
                                    k_smem_1,
                                    barrier=k_full,
                                )
                            T.barrier_arrive(k_full)
                            if n_idx > 0:
                                T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                                if gi_vp % 2 == 0:
                                    if tx == 0:
                                        T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                        v_desc = T.create_tma_descriptor(
                                            TMA_DTYPE_UINT8, 4, v.data,
                                            dim, heads_kv, seq_len, batch,
                                            1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                            dim, 1, block_n, 1,
                                            1, 1, 1, 1,
                                            TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                            TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                        )
                                        tir.call_extern(
                                            "handle",
                                            "tl::fp8_tma_load_4d_ptx",
                                            v_desc,
                                            v_raw_full[0],
                                            T.access_ptr(v_vt_smem_0, "w"),
                                            0,
                                            head_kv,
                                            (n_idx - 1) * block_n,
                                            tile_b,
                                        )
                                    T.barrier_arrive(v_raw_full)
                                    T.barrier_wait(v_raw_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                        v_vt_smem_0.access_ptr("r"),
                                        v_tc_smem_0.access_ptr("w"),
                                    )
                                else:
                                    if tx == 0:
                                        T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                        v_desc = T.create_tma_descriptor(
                                            TMA_DTYPE_UINT8, 4, v.data,
                                            dim, heads_kv, seq_len, batch,
                                            1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                            dim, 1, block_n, 1,
                                            1, 1, 1, 1,
                                            TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                            TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                        )
                                        tir.call_extern(
                                            "handle",
                                            "tl::fp8_tma_load_4d_ptx",
                                            v_desc,
                                            v_raw_full[0],
                                            T.access_ptr(v_vt_smem_1, "w"),
                                            0,
                                            head_kv,
                                            (n_idx - 1) * block_n,
                                            tile_b,
                                        )
                                    T.barrier_arrive(v_raw_full)
                                    T.barrier_wait(v_raw_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                        v_vt_smem_1.access_ptr("r"),
                                        v_tc_smem_1.access_ptr("w"),
                                    )
                                T.barrier_arrive(v_full)
                                gi_vp = gi_vp + 1
                            gi_kp = gi_kp + 1

                        T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                        if gi_vp % 2 == 0:
                            if tx == 0:
                                T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                v_desc_tail = T.create_tma_descriptor(
                                    TMA_DTYPE_UINT8, 4, v.data,
                                    dim, heads_kv, seq_len, batch,
                                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                    dim, 1, block_n, 1,
                                    1, 1, 1, 1,
                                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                )
                                tir.call_extern(
                                    "handle",
                                    "tl::fp8_tma_load_4d_ptx",
                                    v_desc_tail,
                                    v_raw_full[0],
                                    T.access_ptr(v_vt_smem_0, "w"),
                                    0,
                                    head_kv,
                                    (loop_range - 1) * block_n,
                                    tile_b,
                                )
                            T.barrier_arrive(v_raw_full)
                            T.barrier_wait(v_raw_full, gi_vp % 2)
                            T.call_extern(
                                "handle",
                                "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                v_vt_smem_0.access_ptr("r"),
                                v_tc_smem_0.access_ptr("w"),
                            )
                        else:
                            if tx == 0:
                                T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                v_desc_tail = T.create_tma_descriptor(
                                    TMA_DTYPE_UINT8, 4, v.data,
                                    dim, heads_kv, seq_len, batch,
                                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                    dim, 1, block_n, 1,
                                    1, 1, 1, 1,
                                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                )
                                tir.call_extern(
                                    "handle",
                                    "tl::fp8_tma_load_4d_ptx",
                                    v_desc_tail,
                                    v_raw_full[0],
                                    T.access_ptr(v_vt_smem_1, "w"),
                                    0,
                                    head_kv,
                                    (loop_range - 1) * block_n,
                                    tile_b,
                                )
                            T.barrier_arrive(v_raw_full)
                            T.barrier_wait(v_raw_full, gi_vp % 2)
                            T.call_extern(
                                "handle",
                                "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                v_vt_smem_1.access_ptr("r"),
                                v_tc_smem_1.access_ptr("w"),
                            )
                        T.barrier_arrive(v_full)
                        gi_vp = gi_vp + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
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
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base:row_base + half_m, tile_h, :],
                                   q_shared_1, barrier=q_full_1)
                        T.barrier_arrive(q_full_1)
                        T.barrier_wait(q_full_1, gi_q1 % 2)
                        gi_q1 = gi_q1 + 1

                        T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o_1.data)
                        T.clear(ls_1)
                        T.fill(sm_1, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc1 % 2)
                            T.sync_threads(barrier_id=1, arrive_count=256)
                            if gi_kc1 % 2 == 0:
                                T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            else:
                                T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            if n_idx == 0:
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 2, 256)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_1[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pack_p_fa3_raw_64x128x224_to_local_p_experimental",
                                    acc_s_1.data,
                                    p_pv_local_1.data,
                                )
                            else:
                                T.barrier_wait(v_full, gi_vc1 % 2)
                                if gi_vc1 % 2 == 0:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pv_ptx_unit_begin_from_local_p_experimental_64x128x224",
                                        p_pv_local_1.data,
                                        v_tc_smem_0.access_ptr("r"),
                                        4,
                                        acc_delta_local_1.data,
                                    )
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pv_ptx_unit_begin_from_local_p_experimental_64x128x224",
                                        p_pv_local_1.data,
                                        v_tc_smem_1.access_ptr("r"),
                                        4,
                                        acc_delta_local_1.data,
                                    )
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 2, 256)
                                T.call_extern(
                                    "handle",
                                    "tl::wait_wgmma_anchor<1>",
                                    T.address_of(anchor_sink[0]),
                                    gi_kc1,
                                )
                                T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_1[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                T.copy(ss_1, ss_shared_1)
                                T.call_extern(
                                    "handle",
                                    "tl::wait_wgmma_anchor<0>",
                                    T.address_of(anchor_sink[0]),
                                    gi_kc1,
                                )
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pv_ptx_unit_wait_update_fa3_raw_64x128x224",
                                    acc_delta_local_1.data,
                                    4,
                                    ss_shared_1.access_ptr("r"),
                                    v_scale[tile_b, head_kv, n_idx - 1],
                                    acc_o_1.data,
                                )
                                T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                T.barrier_arrive(v_empty)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pack_p_fa3_raw_64x128x224_to_local_p_experimental",
                                    acc_s_1.data,
                                    p_pv_local_1.data,
                                )
                                gi_vc1 = gi_vc1 + 1
                            gi_kc1 = gi_kc1 + 1

                        T.barrier_wait(v_full, gi_vc1 % 2)
                        if gi_vc1 % 2 == 0:
                            T.call_extern(
                                "handle",
                                "tl::fp8_pv_ptx_unit_begin_from_local_p_experimental_64x128x224",
                                p_pv_local_1.data,
                                v_tc_smem_0.access_ptr("r"),
                                4,
                                acc_delta_local_1.data,
                            )
                        else:
                            T.call_extern(
                                "handle",
                                "tl::fp8_pv_ptx_unit_begin_from_local_p_experimental_64x128x224",
                                p_pv_local_1.data,
                                v_tc_smem_1.access_ptr("r"),
                                4,
                                acc_delta_local_1.data,
                            )
                        T.call_extern(
                            "handle",
                            "tl::fp8_pv_ptx_unit_wait_update_tail_fa3_raw_64x128x224",
                            acc_delta_local_1.data,
                            4,
                            v_scale[tile_b, head_kv, loop_range - 1],
                            acc_o_1.data,
                        )
                        T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                        T.barrier_arrive(v_empty)
                        gi_vc1 = gi_vc1 + 1
                        T.copy(ls_1, ls_shared_1)
                        T.copy(acc_o_1, acc_o_layout_seed)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                            acc_o_1.data,
                            ls_shared_1.access_ptr("r"),
                            4,
                            v_tc_smem_0.access_ptr("w"),
                            T.address_of(output[tile_b, row_base, tile_h, 0]),
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=3, arrive_count=128)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                            v_tc_smem_0.access_ptr("r"),
                            T.address_of(output[tile_b, row_base, tile_h, 0]),
                            heads * dim,
                        )
                        for i in T.Parallel(half_m):
                            ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * scale
                        T.copy(ls_1, lse[tile_b, tile_h, row_base:row_base + half_m])

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
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                                   q_shared_2, barrier=q_full_2)
                        T.barrier_arrive(q_full_2)
                        T.barrier_wait(q_full_2, gi_q2 % 2)
                        gi_q2 = gi_q2 + 1

                        T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o_2.data)
                        T.clear(ls_2)
                        T.fill(sm_2, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc2 % 2)
                            T.sync_threads(barrier_id=2, arrive_count=256)
                            if gi_kc2 % 2 == 0:
                                T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            else:
                                T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            if n_idx == 0:
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_2[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pack_p_fa3_raw_64x128x224_to_local_p_experimental",
                                    acc_s_2.data,
                                    p_pv_local_2.data,
                                )
                            else:
                                T.barrier_wait(v_full, gi_vc2 % 2)
                                if gi_vc2 % 2 == 0:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pv_ptx_unit_begin_from_local_p_experimental_64x128x224",
                                        p_pv_local_2.data,
                                        v_tc_smem_0.access_ptr("r"),
                                        4,
                                        acc_delta_local_2.data,
                                    )
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pv_ptx_unit_begin_from_local_p_experimental_64x128x224",
                                        p_pv_local_2.data,
                                        v_tc_smem_1.access_ptr("r"),
                                        4,
                                        acc_delta_local_2.data,
                                    )
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
                                T.call_extern(
                                    "handle",
                                    "tl::wait_wgmma_anchor<1>",
                                    T.address_of(anchor_sink[1]),
                                    gi_kc2,
                                )
                                T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_2[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                T.copy(ss_2, ss_shared_2)
                                T.call_extern(
                                    "handle",
                                    "tl::wait_wgmma_anchor<0>",
                                    T.address_of(anchor_sink[1]),
                                    gi_kc2,
                                )
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pv_ptx_unit_wait_update_fa3_raw_64x128x224",
                                    acc_delta_local_2.data,
                                    4,
                                    ss_shared_2.access_ptr("r"),
                                    v_scale[tile_b, head_kv, n_idx - 1],
                                    acc_o_2.data,
                                )
                                T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                T.barrier_arrive(v_empty)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pack_p_fa3_raw_64x128x224_to_local_p_experimental",
                                    acc_s_2.data,
                                    p_pv_local_2.data,
                                )
                                gi_vc2 = gi_vc2 + 1
                            gi_kc2 = gi_kc2 + 1

                        T.barrier_wait(v_full, gi_vc2 % 2)
                        if gi_vc2 % 2 == 0:
                            T.call_extern(
                                "handle",
                                "tl::fp8_pv_ptx_unit_begin_from_local_p_experimental_64x128x224",
                                p_pv_local_2.data,
                                v_tc_smem_0.access_ptr("r"),
                                4,
                                acc_delta_local_2.data,
                            )
                        else:
                            T.call_extern(
                                "handle",
                                "tl::fp8_pv_ptx_unit_begin_from_local_p_experimental_64x128x224",
                                p_pv_local_2.data,
                                v_tc_smem_1.access_ptr("r"),
                                4,
                                acc_delta_local_2.data,
                            )
                        T.call_extern(
                            "handle",
                            "tl::fp8_pv_ptx_unit_wait_update_tail_fa3_raw_64x128x224",
                            acc_delta_local_2.data,
                            4,
                            v_scale[tile_b, head_kv, loop_range - 1],
                            acc_o_2.data,
                        )
                        T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                        T.barrier_arrive(v_empty)
                        gi_vc2 = gi_vc2 + 1
                        T.copy(ls_2, ls_shared_2)
                        T.copy(acc_o_2, acc_o_layout_seed)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                            acc_o_2.data,
                            ls_shared_2.access_ptr("r"),
                            4,
                            v_tc_smem_1.access_ptr("w"),
                            T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=4, arrive_count=128)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                            v_tc_smem_1.access_ptr("r"),
                            T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                            heads * dim,
                        )
                        for i in T.Parallel(half_m):
                            ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                        T.copy(ls_2, lse[tile_b, tile_h, row_base + half_m:row_base + block_m])

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_frag_p_local_delta_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    correct_delta_rescale: bool = False,
    pre_rescale_delta: bool = False,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("BN224 overlap frag-P local-delta FP8 GQA requires dim == 128.")
    if seq_len % 224 != 0:
        raise ValueError("BN224 overlap frag-P local-delta FP8 GQA requires seq_len % 224 == 0.")

    block_m = 128
    half_m = block_m // 2
    block_n = 224
    groups = heads // heads_kv
    accum_dtype = "float"
    fp8_dtype = "float8_e4m3fn"
    scale = make_log2e_scale(dim)
    scale_block = 128
    scale_blocks = (seq_len + scale_block - 1) // scale_block
    wait_update_helper = (
        "tl::fp8_pv_ptx_unit_wait_add_rescaled_delta_fa3_raw_64x128x224"
        if pre_rescale_delta else
        "tl::fp8_pv_ptx_unit_wait_update_rescale_fa3_raw_64x128x224"
        if correct_delta_rescale else
        "tl::fp8_pv_ptx_unit_wait_update_fa3_raw_64x128x224"
    )

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

        online_softmax_1 = make_online_softmax(scale, accum_dtype, half_m, block_n)
        online_softmax_2 = make_online_softmax(scale, accum_dtype, half_m, block_n)

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
                k_smem_0 = T.alloc_shared([block_n, dim], fp8_dtype)
                k_smem_1 = T.alloc_shared([block_n, dim], fp8_dtype)
                v_vt_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_vt_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_tc_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_tc_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                ss_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ss_shared_2 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_2 = T.alloc_shared([half_m], accum_dtype)
                acc_o_layout_seed = T.alloc_shared([half_m, dim], out_dtype)
                anchor_sink = T.alloc_shared([2], "int32")
                p_pv_frag_1 = T.alloc_fragment([half_m, block_n], fp8_dtype)
                p_pv_frag_2 = T.alloc_fragment([half_m, block_n], fp8_dtype)
                acc_delta_local_1 = T.alloc_local([64], accum_dtype)
                acc_delta_local_2 = T.alloc_local([64], accum_dtype)

                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
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
                v_empty = T.alloc_barrier(arrive_count=256)
                q_full_1 = T.alloc_barrier(arrive_count=128)
                q_full_2 = T.alloc_barrier(arrive_count=128)

                T.annotate_layout({
                    q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                    q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                })
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
                    for tile_b, tile_hkv, _tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len, block_m), groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_hkv
                        loop_range = T.ceildiv(seq_len, block_n)
                        T.reads(v[tile_b, 0:seq_len, head_kv, :])
                        T.writes(
                            v_vt_smem_0[0:dim, 0:block_n],
                            v_vt_smem_1[0:dim, 0:block_n],
                        )

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_empty, (gi_kp + 1) % 2)
                            if gi_kp % 2 == 0:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :],
                                    k_smem_0,
                                    barrier=k_full,
                                )
                            else:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :],
                                    k_smem_1,
                                    barrier=k_full,
                                )
                            T.barrier_arrive(k_full)
                            if n_idx > 0:
                                T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                                if gi_vp % 2 == 0:
                                    if tx == 0:
                                        T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                        v_desc = T.create_tma_descriptor(
                                            TMA_DTYPE_UINT8, 4, v.data,
                                            dim, heads_kv, seq_len, batch,
                                            1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                            dim, 1, block_n, 1,
                                            1, 1, 1, 1,
                                            TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                            TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                        )
                                        tir.call_extern(
                                            "handle",
                                            "tl::fp8_tma_load_4d_ptx",
                                            v_desc,
                                            v_raw_full[0],
                                            T.access_ptr(v_vt_smem_0, "w"),
                                            0,
                                            head_kv,
                                            (n_idx - 1) * block_n,
                                            tile_b,
                                        )
                                    T.barrier_arrive(v_raw_full)
                                    T.barrier_wait(v_raw_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                        v_vt_smem_0.access_ptr("r"),
                                        v_tc_smem_0.access_ptr("w"),
                                    )
                                else:
                                    if tx == 0:
                                        T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                        v_desc = T.create_tma_descriptor(
                                            TMA_DTYPE_UINT8, 4, v.data,
                                            dim, heads_kv, seq_len, batch,
                                            1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                            dim, 1, block_n, 1,
                                            1, 1, 1, 1,
                                            TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                            TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                        )
                                        tir.call_extern(
                                            "handle",
                                            "tl::fp8_tma_load_4d_ptx",
                                            v_desc,
                                            v_raw_full[0],
                                            T.access_ptr(v_vt_smem_1, "w"),
                                            0,
                                            head_kv,
                                            (n_idx - 1) * block_n,
                                            tile_b,
                                        )
                                    T.barrier_arrive(v_raw_full)
                                    T.barrier_wait(v_raw_full, gi_vp % 2)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                        v_vt_smem_1.access_ptr("r"),
                                        v_tc_smem_1.access_ptr("w"),
                                    )
                                T.barrier_arrive(v_full)
                                gi_vp = gi_vp + 1
                            gi_kp = gi_kp + 1

                        T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                        if gi_vp % 2 == 0:
                            if tx == 0:
                                T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                v_desc_tail = T.create_tma_descriptor(
                                    TMA_DTYPE_UINT8, 4, v.data,
                                    dim, heads_kv, seq_len, batch,
                                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                    dim, 1, block_n, 1,
                                    1, 1, 1, 1,
                                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                )
                                tir.call_extern(
                                    "handle",
                                    "tl::fp8_tma_load_4d_ptx",
                                    v_desc_tail,
                                    v_raw_full[0],
                                    T.access_ptr(v_vt_smem_0, "w"),
                                    0,
                                    head_kv,
                                    (loop_range - 1) * block_n,
                                    tile_b,
                                )
                            T.barrier_arrive(v_raw_full)
                            T.barrier_wait(v_raw_full, gi_vp % 2)
                            T.call_extern(
                                "handle",
                                "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                v_vt_smem_0.access_ptr("r"),
                                v_tc_smem_0.access_ptr("w"),
                            )
                        else:
                            if tx == 0:
                                T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                v_desc_tail = T.create_tma_descriptor(
                                    TMA_DTYPE_UINT8, 4, v.data,
                                    dim, heads_kv, seq_len, batch,
                                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                    dim, 1, block_n, 1,
                                    1, 1, 1, 1,
                                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                )
                                tir.call_extern(
                                    "handle",
                                    "tl::fp8_tma_load_4d_ptx",
                                    v_desc_tail,
                                    v_raw_full[0],
                                    T.access_ptr(v_vt_smem_1, "w"),
                                    0,
                                    head_kv,
                                    (loop_range - 1) * block_n,
                                    tile_b,
                                )
                            T.barrier_arrive(v_raw_full)
                            T.barrier_wait(v_raw_full, gi_vp % 2)
                            T.call_extern(
                                "handle",
                                "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                v_vt_smem_1.access_ptr("r"),
                                v_tc_smem_1.access_ptr("w"),
                            )
                        T.barrier_arrive(v_full)
                        gi_vp = gi_vp + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
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
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base:row_base + half_m, tile_h, :],
                                   q_shared_1, barrier=q_full_1)
                        T.barrier_arrive(q_full_1)
                        T.barrier_wait(q_full_1, gi_q1 % 2)
                        gi_q1 = gi_q1 + 1

                        T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o_1.data)
                        T.clear(ls_1)
                        T.fill(sm_1, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc1 % 2)
                            T.sync_threads(barrier_id=1, arrive_count=256)
                            if gi_kc1 % 2 == 0:
                                T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            else:
                                T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            if n_idx == 0:
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 2, 256)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_1[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                T.copy(acc_s_1, p_pv_frag_1)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pack_p_fa3_raw_64x128x224",
                                    acc_s_1.data,
                                    p_pv_frag_1.data,
                                )
                            else:
                                T.barrier_wait(v_full, gi_vc1 % 2)
                                if gi_vc1 % 2 == 0:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                        p_pv_frag_1.data,
                                        v_tc_smem_0.access_ptr("r"),
                                        4,
                                        acc_delta_local_1.data,
                                    )
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                        p_pv_frag_1.data,
                                        v_tc_smem_1.access_ptr("r"),
                                        4,
                                        acc_delta_local_1.data,
                                    )
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 2, 256)
                                T.call_extern(
                                    "handle",
                                    "tl::wait_wgmma_anchor<1>",
                                    T.address_of(anchor_sink[0]),
                                    gi_kc1,
                                )
                                T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_1[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                T.copy(ss_1, ss_shared_1)
                                T.call_extern(
                                    "handle",
                                    "tl::wait_wgmma_anchor<0>",
                                    T.address_of(anchor_sink[0]),
                                    gi_kc1,
                                )
                                T.call_extern(
                                    "handle",
                                    wait_update_helper,
                                    acc_delta_local_1.data,
                                    4,
                                    ss_shared_1.access_ptr("r"),
                                    v_scale[tile_b, head_kv, n_idx - 1],
                                    acc_o_1.data,
                                )
                                T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                T.barrier_arrive(v_empty)
                                T.copy(acc_s_1, p_pv_frag_1)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pack_p_fa3_raw_64x128x224",
                                    acc_s_1.data,
                                    p_pv_frag_1.data,
                                )
                                gi_vc1 = gi_vc1 + 1
                            gi_kc1 = gi_kc1 + 1

                        T.barrier_wait(v_full, gi_vc1 % 2)
                        if gi_vc1 % 2 == 0:
                            T.call_extern(
                                "handle",
                                "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                p_pv_frag_1.data,
                                v_tc_smem_0.access_ptr("r"),
                                4,
                                acc_delta_local_1.data,
                            )
                        else:
                            T.call_extern(
                                "handle",
                                "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                p_pv_frag_1.data,
                                v_tc_smem_1.access_ptr("r"),
                                4,
                                acc_delta_local_1.data,
                            )
                        T.call_extern(
                            "handle",
                            "tl::fp8_pv_ptx_unit_wait_update_tail_fa3_raw_64x128x224",
                            acc_delta_local_1.data,
                            4,
                            v_scale[tile_b, head_kv, loop_range - 1],
                            acc_o_1.data,
                        )
                        T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                        T.barrier_arrive(v_empty)
                        gi_vc1 = gi_vc1 + 1
                        T.copy(ls_1, ls_shared_1)
                        T.copy(acc_o_1, acc_o_layout_seed)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                            acc_o_1.data,
                            ls_shared_1.access_ptr("r"),
                            4,
                            v_tc_smem_0.access_ptr("w"),
                            T.address_of(output[tile_b, row_base, tile_h, 0]),
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=3, arrive_count=128)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                            v_tc_smem_0.access_ptr("r"),
                            T.address_of(output[tile_b, row_base, tile_h, 0]),
                            heads * dim,
                        )
                        for i in T.Parallel(half_m):
                            ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * scale
                        T.copy(ls_1, lse[tile_b, tile_h, row_base:row_base + half_m])

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
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                                   q_shared_2, barrier=q_full_2)
                        T.barrier_arrive(q_full_2)
                        T.barrier_wait(q_full_2, gi_q2 % 2)
                        gi_q2 = gi_q2 + 1

                        T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o_2.data)
                        T.clear(ls_2)
                        T.fill(sm_2, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc2 % 2)
                            T.sync_threads(barrier_id=2, arrive_count=256)
                            if gi_kc2 % 2 == 0:
                                T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            else:
                                T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            if n_idx == 0:
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_2[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                T.copy(acc_s_2, p_pv_frag_2)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pack_p_fa3_raw_64x128x224",
                                    acc_s_2.data,
                                    p_pv_frag_2.data,
                                )
                            else:
                                T.barrier_wait(v_full, gi_vc2 % 2)
                                if gi_vc2 % 2 == 0:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                        p_pv_frag_2.data,
                                        v_tc_smem_0.access_ptr("r"),
                                        4,
                                        acc_delta_local_2.data,
                                    )
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                        p_pv_frag_2.data,
                                        v_tc_smem_1.access_ptr("r"),
                                        4,
                                        acc_delta_local_2.data,
                                    )
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
                                T.call_extern(
                                    "handle",
                                    "tl::wait_wgmma_anchor<1>",
                                    T.address_of(anchor_sink[1]),
                                    gi_kc2,
                                )
                                T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_2[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                T.copy(ss_2, ss_shared_2)
                                T.call_extern(
                                    "handle",
                                    "tl::wait_wgmma_anchor<0>",
                                    T.address_of(anchor_sink[1]),
                                    gi_kc2,
                                )
                                T.call_extern(
                                    "handle",
                                    wait_update_helper,
                                    acc_delta_local_2.data,
                                    4,
                                    ss_shared_2.access_ptr("r"),
                                    v_scale[tile_b, head_kv, n_idx - 1],
                                    acc_o_2.data,
                                )
                                T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                T.barrier_arrive(v_empty)
                                T.copy(acc_s_2, p_pv_frag_2)
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pack_p_fa3_raw_64x128x224",
                                    acc_s_2.data,
                                    p_pv_frag_2.data,
                                )
                                gi_vc2 = gi_vc2 + 1
                            gi_kc2 = gi_kc2 + 1

                        T.barrier_wait(v_full, gi_vc2 % 2)
                        if gi_vc2 % 2 == 0:
                            T.call_extern(
                                "handle",
                                "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                p_pv_frag_2.data,
                                v_tc_smem_0.access_ptr("r"),
                                4,
                                acc_delta_local_2.data,
                            )
                        else:
                            T.call_extern(
                                "handle",
                                "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                p_pv_frag_2.data,
                                v_tc_smem_1.access_ptr("r"),
                                4,
                                acc_delta_local_2.data,
                            )
                        T.call_extern(
                            "handle",
                            "tl::fp8_pv_ptx_unit_wait_update_tail_fa3_raw_64x128x224",
                            acc_delta_local_2.data,
                            4,
                            v_scale[tile_b, head_kv, loop_range - 1],
                            acc_o_2.data,
                        )
                        T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                        T.barrier_arrive(v_empty)
                        gi_vc2 = gi_vc2 + 1
                        T.copy(ls_2, ls_shared_2)
                        T.copy(acc_o_2, acc_o_layout_seed)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                            acc_o_2.data,
                            ls_shared_2.access_ptr("r"),
                            4,
                            v_tc_smem_1.access_ptr("w"),
                            T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=4, arrive_count=128)
                        T.call_extern(
                            "handle",
                            "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                            v_tc_smem_1.access_ptr("r"),
                            T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                            heads * dim,
                        )
                        for i in T.Parallel(half_m):
                            ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                        T.copy(ls_2, lse[tile_b, tile_h, row_base + half_m:row_base + block_m])

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_fragment_delta_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    visible_pv: bool = False,
    visible_pv_shared_v: bool = False,
    visible_pv_emitter_k224: bool = False,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("BN224 overlap fragment-delta FP8 GQA requires dim == 128.")
    if seq_len % 224 != 0:
        raise ValueError("BN224 overlap fragment-delta FP8 GQA requires seq_len % 224 == 0.")

    block_m = 128
    half_m = block_m // 2
    block_n = 224
    fp8_k = 32
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

        online_softmax_1 = make_online_softmax(scale, accum_dtype, half_m, block_n)
        online_softmax_2 = make_online_softmax(scale, accum_dtype, half_m, block_n)
        pv_wgmma_emitter = None
        if visible_pv_emitter_k224:
            pv_wgmma_emitter = TensorCoreIntrinEmitter(
                a_dtype=T.float8_e4m3fn,
                b_dtype=T.float8_e4m3fn,
                accum_dtype=T.float32,
                a_transposed=False,
                b_transposed=True,
                block_row_warps=4,
                block_col_warps=1,
                warp_row_tiles=16,
                warp_col_tiles=128,
                chunk=block_n,
            )

        @T.macro
        def load_p_fragment_k224(a_local, p_shared):
            tx, _, warp_m = pv_wgmma_emitter.extract_thread_binding(
                pv_wgmma_emitter.get_thread_binding()
            )
            row_off, col_off = get_ldmatrix_offset(
                "A",
                tx,
                0,
                p_shared.shape[-1],
                T.float8_e4m3fn,
                False,
            )
            for ldm_k in T.serial(block_n // fp8_k):
                T.ptx_ldmatrix(
                    T.bool(False),
                    4,
                    T.access_ptr(
                        p_shared[
                            warp_m * 16 + row_off,
                            ldm_k * fp8_k + col_off,
                        ],
                        "r",
                        extent=8,
                    ),
                    T.access_ptr(a_local[ldm_k * 16], "w", extent=8),
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
                k_smem_0 = T.alloc_shared([block_n, dim], fp8_dtype)
                k_smem_1 = T.alloc_shared([block_n, dim], fp8_dtype)
                v_vt_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_vt_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_tc_smem_0 = T.alloc_shared([dim, block_n], fp8_dtype)
                v_tc_smem_1 = T.alloc_shared([dim, block_n], fp8_dtype)
                if visible_pv and not visible_pv_emitter_k224:
                    v_tc_shared_1 = T.alloc_shared([dim, fp8_k], fp8_dtype)
                    v_tc_shared_2 = T.alloc_shared([dim, fp8_k], fp8_dtype)
                ss_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ss_shared_2 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_2 = T.alloc_shared([half_m], accum_dtype)
                if not visible_pv:
                    acc_o_layout_seed = T.alloc_shared([half_m, dim], out_dtype)
                anchor_sink = T.alloc_shared([2], "int32")
                if visible_pv:
                    p_smem_1 = T.alloc_shared([half_m, block_n], fp8_dtype)
                    p_smem_2 = T.alloc_shared([half_m, block_n], fp8_dtype)
                    if visible_pv_emitter_k224:
                        p_pv_frag_1 = T.alloc_local([112], fp8_dtype)
                        p_pv_frag_2 = T.alloc_local([112], fp8_dtype)
                    else:
                        p_pv_frag_1 = T.alloc_fragment([half_m, fp8_k], fp8_dtype)
                        p_pv_frag_2 = T.alloc_fragment([half_m, fp8_k], fp8_dtype)
                else:
                    p_pv_frag_1 = T.alloc_fragment([half_m, block_n], fp8_dtype)
                    p_pv_frag_2 = T.alloc_fragment([half_m, block_n], fp8_dtype)
                acc_delta_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                acc_delta_2 = T.alloc_fragment([half_m, dim], accum_dtype)

                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
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
                v_empty = T.alloc_barrier(arrive_count=256)
                q_full_1 = T.alloc_barrier(arrive_count=128)
                q_full_2 = T.alloc_barrier(arrive_count=128)

                if visible_pv_emitter_k224:
                    T.annotate_layout({
                        q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                        q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                        v_tc_smem_0: tilelang.layout.make_quarter_bank_swizzled_layout(v_tc_smem_0),
                        v_tc_smem_1: tilelang.layout.make_quarter_bank_swizzled_layout(v_tc_smem_1),
                        acc_delta_1: pv_wgmma_emitter.make_mma_store_layout(acc_delta_1),
                        acc_delta_2: pv_wgmma_emitter.make_mma_store_layout(acc_delta_2),
                    })
                else:
                    T.annotate_layout({
                        q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                        q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                    })
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
                    for tile_b, tile_hkv, _tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len, block_m), groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_hkv
                        loop_range = T.ceildiv(seq_len, block_n)
                        T.reads(v[tile_b, 0:seq_len, head_kv, :])
                        T.writes(
                            v_vt_smem_0[0:dim, 0:block_n],
                            v_vt_smem_1[0:dim, 0:block_n],
                        )

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_empty, (gi_kp + 1) % 2)
                            if gi_kp % 2 == 0:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :],
                                    k_smem_0,
                                    barrier=k_full,
                                )
                            else:
                                T.tma_copy(
                                    k[tile_b, n_idx * block_n:(n_idx + 1) * block_n, head_kv, :],
                                    k_smem_1,
                                    barrier=k_full,
                                )
                            T.barrier_arrive(k_full)
                            if n_idx > 0:
                                T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                                if gi_vp % 2 == 0:
                                    if visible_pv_emitter_k224:
                                        for d, n in T.Parallel(dim, block_n):
                                            v_tc_smem_0[d, n] = v[
                                                tile_b,
                                                (n_idx - 1) * block_n + n,
                                                head_kv,
                                                d,
                                            ]
                                    elif visible_pv_shared_v:
                                        for d, n in T.Parallel(dim, block_n):
                                            v_vt_smem_0[d, n] = v[
                                                tile_b,
                                                (n_idx - 1) * block_n + n,
                                                head_kv,
                                                d,
                                            ]
                                    if tx == 0:
                                        if not visible_pv_shared_v and not visible_pv_emitter_k224:
                                            T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                            v_desc = T.create_tma_descriptor(
                                                TMA_DTYPE_UINT8, 4, v.data,
                                                dim, heads_kv, seq_len, batch,
                                                1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                                dim, 1, block_n, 1,
                                                1, 1, 1, 1,
                                                TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                                TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                            )
                                            tir.call_extern(
                                                "handle",
                                                "tl::fp8_tma_load_4d_ptx",
                                                v_desc,
                                                v_raw_full[0],
                                                T.access_ptr(v_vt_smem_0, "w"),
                                                0,
                                                head_kv,
                                                (n_idx - 1) * block_n,
                                                tile_b,
                                            )
                                    if not visible_pv_shared_v and not visible_pv_emitter_k224:
                                        T.barrier_arrive(v_raw_full)
                                        T.barrier_wait(v_raw_full, gi_vp % 2)
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                            v_vt_smem_0.access_ptr("r"),
                                            v_tc_smem_0.access_ptr("w"),
                                        )
                                else:
                                    if visible_pv_emitter_k224:
                                        for d, n in T.Parallel(dim, block_n):
                                            v_tc_smem_1[d, n] = v[
                                                tile_b,
                                                (n_idx - 1) * block_n + n,
                                                head_kv,
                                                d,
                                            ]
                                    elif visible_pv_shared_v:
                                        for d, n in T.Parallel(dim, block_n):
                                            v_vt_smem_1[d, n] = v[
                                                tile_b,
                                                (n_idx - 1) * block_n + n,
                                                head_kv,
                                                d,
                                            ]
                                    if tx == 0:
                                        if not visible_pv_shared_v and not visible_pv_emitter_k224:
                                            T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                            v_desc = T.create_tma_descriptor(
                                                TMA_DTYPE_UINT8, 4, v.data,
                                                dim, heads_kv, seq_len, batch,
                                                1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                                dim, 1, block_n, 1,
                                                1, 1, 1, 1,
                                                TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                                TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                            )
                                            tir.call_extern(
                                                "handle",
                                                "tl::fp8_tma_load_4d_ptx",
                                                v_desc,
                                                v_raw_full[0],
                                                T.access_ptr(v_vt_smem_1, "w"),
                                                0,
                                                head_kv,
                                                (n_idx - 1) * block_n,
                                                tile_b,
                                            )
                                    if not visible_pv_shared_v and not visible_pv_emitter_k224:
                                        T.barrier_arrive(v_raw_full)
                                        T.barrier_wait(v_raw_full, gi_vp % 2)
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                            v_vt_smem_1.access_ptr("r"),
                                            v_tc_smem_1.access_ptr("w"),
                                        )
                                T.barrier_arrive(v_full)
                                gi_vp = gi_vp + 1
                            gi_kp = gi_kp + 1

                        T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                        if gi_vp % 2 == 0:
                            if tx == 0:
                                T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                v_desc_tail = T.create_tma_descriptor(
                                    TMA_DTYPE_UINT8, 4, v.data,
                                    dim, heads_kv, seq_len, batch,
                                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                    dim, 1, block_n, 1,
                                    1, 1, 1, 1,
                                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                )
                                tir.call_extern(
                                    "handle",
                                    "tl::fp8_tma_load_4d_ptx",
                                    v_desc_tail,
                                    v_raw_full[0],
                                    T.access_ptr(v_vt_smem_0, "w"),
                                    0,
                                    head_kv,
                                    (loop_range - 1) * block_n,
                                    tile_b,
                                )
                            T.barrier_arrive(v_raw_full)
                            T.barrier_wait(v_raw_full, gi_vp % 2)
                            T.call_extern(
                                "handle",
                                "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                v_vt_smem_0.access_ptr("r"),
                                v_tc_smem_0.access_ptr("w"),
                            )
                        else:
                            if tx == 0:
                                T.mbarrier_expect_tx(v_raw_full, dim * block_n)
                                v_desc_tail = T.create_tma_descriptor(
                                    TMA_DTYPE_UINT8, 4, v.data,
                                    dim, heads_kv, seq_len, batch,
                                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                                    dim, 1, block_n, 1,
                                    1, 1, 1, 1,
                                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                                )
                                tir.call_extern(
                                    "handle",
                                    "tl::fp8_tma_load_4d_ptx",
                                    v_desc_tail,
                                    v_raw_full[0],
                                    T.access_ptr(v_vt_smem_1, "w"),
                                    0,
                                    head_kv,
                                    (loop_range - 1) * block_n,
                                    tile_b,
                                )
                            T.barrier_arrive(v_raw_full)
                            T.barrier_wait(v_raw_full, gi_vp % 2)
                            T.call_extern(
                                "handle",
                                "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                                v_vt_smem_1.access_ptr("r"),
                                v_tc_smem_1.access_ptr("w"),
                            )
                        T.barrier_arrive(v_full)
                        gi_vp = gi_vp + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
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
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base:row_base + half_m, tile_h, :],
                                   q_shared_1, barrier=q_full_1)
                        T.barrier_arrive(q_full_1)
                        T.barrier_wait(q_full_1, gi_q1 % 2)
                        gi_q1 = gi_q1 + 1

                        T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o_1.data)
                        T.copy(acc_o_1, acc_delta_1)
                        T.clear(ls_1)
                        T.fill(sm_1, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc1 % 2)
                            T.sync_threads(barrier_id=1, arrive_count=256)
                            if gi_kc1 % 2 == 0:
                                T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            else:
                                T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            if n_idx == 0:
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 2, 256)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_1[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                if visible_pv:
                                    T.copy(acc_s_1, p_smem_1)
                                else:
                                    T.copy(acc_s_1, p_pv_frag_1)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pack_p_fa3_raw_64x128x224",
                                        acc_s_1.data,
                                        p_pv_frag_1.data,
                                    )
                            else:
                                T.barrier_wait(v_full, gi_vc1 % 2)
                                if visible_pv:
                                    if visible_pv_emitter_k224:
                                        load_p_fragment_k224(p_pv_frag_1, p_smem_1)
                                        if gi_vc1 % 2 == 0:
                                            pv_wgmma_emitter._assign_b_shared_layout(
                                                tilelang.layout.make_quarter_bank_swizzled_layout(v_tc_smem_0)
                                            )
                                            pv_wgmma_emitter.wgmma_rs(
                                                p_pv_frag_1[:],
                                                v_tc_smem_0[:, :],
                                                acc_delta_1[:, :],
                                                clear_accum=True,
                                                wg_wait=-1,
                                            )
                                        else:
                                            pv_wgmma_emitter._assign_b_shared_layout(
                                                tilelang.layout.make_quarter_bank_swizzled_layout(v_tc_smem_1)
                                            )
                                            pv_wgmma_emitter.wgmma_rs(
                                                p_pv_frag_1[:],
                                                v_tc_smem_1[:, :],
                                                acc_delta_1[:, :],
                                                clear_accum=True,
                                                wg_wait=-1,
                                            )
                                    else:
                                        for pv_k in range(block_n // fp8_k):
                                            T.copy(
                                                p_smem_1[:, pv_k * fp8_k:(pv_k + 1) * fp8_k],
                                                p_pv_frag_1,
                                            )
                                            for d, n in T.Parallel(dim, fp8_k):
                                                if visible_pv_shared_v:
                                                    if gi_vc1 % 2 == 0:
                                                        v_tc_shared_1[d, n] = v_vt_smem_0[
                                                            d, pv_k * fp8_k + n]
                                                    else:
                                                        v_tc_shared_1[d, n] = v_vt_smem_1[
                                                            d, pv_k * fp8_k + n]
                                                else:
                                                    v_tc_shared_1[d, n] = v[
                                                        tile_b,
                                                        (n_idx - 1) * block_n + pv_k * fp8_k + n,
                                                        head_kv,
                                                        d,
                                                    ]
                                            T.wgmma_gemm(
                                                p_pv_frag_1,
                                                v_tc_shared_1,
                                                acc_delta_1,
                                                transpose_B=True,
                                                policy=T.GemmWarpPolicy.FullRow,
                                                clear_accum=(pv_k == 0),
                                            )
                                else:
                                    if gi_vc1 % 2 == 0:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                            p_pv_frag_1.data,
                                            v_tc_smem_0.access_ptr("r"),
                                            4,
                                            acc_delta_1.data,
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                            p_pv_frag_1.data,
                                            v_tc_smem_1.access_ptr("r"),
                                            4,
                                            acc_delta_1.data,
                                        )
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 2, 256)
                                if visible_pv:
                                    if visible_pv_emitter_k224:
                                        T.call_extern(
                                            "handle",
                                            "tl::wait_wgmma_anchor<1>",
                                            T.address_of(anchor_sink[0]),
                                            gi_kc1,
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::wait_wgmma_anchor<7>",
                                            T.address_of(anchor_sink[0]),
                                            gi_kc1,
                                        )
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::wait_wgmma_anchor<1>",
                                        T.address_of(anchor_sink[0]),
                                        gi_kc1,
                                    )
                                T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_1[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                T.copy(ss_1, ss_shared_1)
                                T.call_extern(
                                    "handle",
                                    "tl::wait_wgmma_anchor<0>",
                                    T.address_of(anchor_sink[0]),
                                    gi_kc1,
                                )
                                if visible_pv:
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(acc_delta_1, num_regs=64)
                                    for i, j in T.Parallel(half_m, dim):
                                        acc_o_1[i, j] = (
                                            acc_o_1[i, j]
                                            + acc_delta_1[i, j]
                                            * v_scale[tile_b, head_kv, n_idx - 1]
                                        ) * ss_1[i]
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pv_ptx_unit_wait_update_fa3_raw_64x128x224",
                                        acc_delta_1.data,
                                        4,
                                        ss_shared_1.access_ptr("r"),
                                        v_scale[tile_b, head_kv, n_idx - 1],
                                        acc_o_1.data,
                                    )
                                    T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                T.barrier_arrive(v_empty)
                                if visible_pv:
                                    T.copy(acc_s_1, p_smem_1)
                                else:
                                    T.copy(acc_s_1, p_pv_frag_1)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pack_p_fa3_raw_64x128x224",
                                        acc_s_1.data,
                                        p_pv_frag_1.data,
                                    )
                                gi_vc1 = gi_vc1 + 1
                            gi_kc1 = gi_kc1 + 1

                        T.barrier_wait(v_full, gi_vc1 % 2)
                        if visible_pv:
                            if visible_pv_emitter_k224:
                                load_p_fragment_k224(p_pv_frag_1, p_smem_1)
                                if gi_vc1 % 2 == 0:
                                    pv_wgmma_emitter._assign_b_shared_layout(
                                        tilelang.layout.make_quarter_bank_swizzled_layout(v_tc_smem_0)
                                    )
                                    pv_wgmma_emitter.wgmma_rs(
                                        p_pv_frag_1[:],
                                        v_tc_smem_0[:, :],
                                        acc_delta_1[:, :],
                                        clear_accum=True,
                                        wg_wait=-1,
                                    )
                                else:
                                    pv_wgmma_emitter._assign_b_shared_layout(
                                        tilelang.layout.make_quarter_bank_swizzled_layout(v_tc_smem_1)
                                    )
                                    pv_wgmma_emitter.wgmma_rs(
                                        p_pv_frag_1[:],
                                        v_tc_smem_1[:, :],
                                        acc_delta_1[:, :],
                                        clear_accum=True,
                                        wg_wait=-1,
                                    )
                            else:
                                for pv_k in range(block_n // fp8_k):
                                    T.copy(
                                        p_smem_1[:, pv_k * fp8_k:(pv_k + 1) * fp8_k],
                                        p_pv_frag_1,
                                    )
                                    for d, n in T.Parallel(dim, fp8_k):
                                        if visible_pv_shared_v:
                                            if gi_vc1 % 2 == 0:
                                                v_tc_shared_1[d, n] = v_vt_smem_0[
                                                    d, pv_k * fp8_k + n]
                                            else:
                                                v_tc_shared_1[d, n] = v_vt_smem_1[
                                                    d, pv_k * fp8_k + n]
                                        else:
                                            v_tc_shared_1[d, n] = v[
                                                tile_b,
                                                (loop_range - 1) * block_n + pv_k * fp8_k + n,
                                                head_kv,
                                                d,
                                            ]
                                    T.wgmma_gemm(
                                        p_pv_frag_1,
                                        v_tc_shared_1,
                                        acc_delta_1,
                                        transpose_B=True,
                                        policy=T.GemmWarpPolicy.FullRow,
                                        clear_accum=(pv_k == 0),
                                    )
                        else:
                            if gi_vc1 % 2 == 0:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                    p_pv_frag_1.data,
                                    v_tc_smem_0.access_ptr("r"),
                                    4,
                                    acc_delta_1.data,
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                    p_pv_frag_1.data,
                                    v_tc_smem_1.access_ptr("r"),
                                    4,
                                    acc_delta_1.data,
                                )
                        if visible_pv:
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_delta_1, num_regs=64)
                            for i, j in T.Parallel(half_m, dim):
                                acc_o_1[i, j] += (
                                    acc_delta_1[i, j]
                                    * v_scale[tile_b, head_kv, loop_range - 1]
                                )
                        else:
                            T.call_extern(
                                "handle",
                                "tl::fp8_pv_ptx_unit_wait_update_tail_fa3_raw_64x128x224",
                                acc_delta_1.data,
                                4,
                                v_scale[tile_b, head_kv, loop_range - 1],
                                acc_o_1.data,
                            )
                            T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                        T.barrier_arrive(v_empty)
                        gi_vc1 = gi_vc1 + 1
                        if visible_pv:
                            for i, j in T.Parallel(half_m, dim):
                                acc_o_1[i, j] /= ls_1[i]
                            T.copy(
                                acc_o_1,
                                output[tile_b, row_base:row_base + half_m, tile_h, :],
                            )
                        else:
                            T.copy(ls_1, ls_shared_1)
                            T.copy(acc_o_1, acc_o_layout_seed)
                            T.call_extern(
                                "handle",
                                "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                                acc_o_1.data,
                                ls_shared_1.access_ptr("r"),
                                4,
                                v_tc_smem_0.access_ptr("w"),
                                T.address_of(output[tile_b, row_base, tile_h, 0]),
                            )
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=3, arrive_count=128)
                            T.call_extern(
                                "handle",
                                "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                                v_tc_smem_0.access_ptr("r"),
                                T.address_of(output[tile_b, row_base, tile_h, 0]),
                                heads * dim,
                            )
                        for i in T.Parallel(half_m):
                            ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * scale
                        T.copy(ls_1, lse[tile_b, tile_h, row_base:row_base + half_m])

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
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                                   q_shared_2, barrier=q_full_2)
                        T.barrier_arrive(q_full_2)
                        T.barrier_wait(q_full_2, gi_q2 % 2)
                        gi_q2 = gi_q2 + 1

                        T.call_extern("handle", "tl::fp8_zero_raw_acc_64", acc_o_2.data)
                        T.copy(acc_o_2, acc_delta_2)
                        T.clear(ls_2)
                        T.fill(sm_2, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc2 % 2)
                            T.sync_threads(barrier_id=2, arrive_count=256)
                            if gi_kc2 % 2 == 0:
                                T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            else:
                                T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2,
                                             transpose_B=True,
                                             policy=T.GemmWarpPolicy.FullRow,
                                             clear_accum=True)
                            if n_idx == 0:
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_2[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                if visible_pv:
                                    T.copy(acc_s_2, p_smem_2)
                                else:
                                    T.copy(acc_s_2, p_pv_frag_2)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pack_p_fa3_raw_64x128x224",
                                        acc_s_2.data,
                                        p_pv_frag_2.data,
                                    )
                            else:
                                T.barrier_wait(v_full, gi_vc2 % 2)
                                if visible_pv:
                                    if visible_pv_emitter_k224:
                                        load_p_fragment_k224(p_pv_frag_2, p_smem_2)
                                        if gi_vc2 % 2 == 0:
                                            pv_wgmma_emitter._assign_b_shared_layout(
                                                tilelang.layout.make_quarter_bank_swizzled_layout(v_tc_smem_0)
                                            )
                                            pv_wgmma_emitter.wgmma_rs(
                                                p_pv_frag_2[:],
                                                v_tc_smem_0[:, :],
                                                acc_delta_2[:, :],
                                                clear_accum=True,
                                                wg_wait=-1,
                                            )
                                        else:
                                            pv_wgmma_emitter._assign_b_shared_layout(
                                                tilelang.layout.make_quarter_bank_swizzled_layout(v_tc_smem_1)
                                            )
                                            pv_wgmma_emitter.wgmma_rs(
                                                p_pv_frag_2[:],
                                                v_tc_smem_1[:, :],
                                                acc_delta_2[:, :],
                                                clear_accum=True,
                                                wg_wait=-1,
                                            )
                                    else:
                                        for pv_k in range(block_n // fp8_k):
                                            T.copy(
                                                p_smem_2[:, pv_k * fp8_k:(pv_k + 1) * fp8_k],
                                                p_pv_frag_2,
                                            )
                                            for d, n in T.Parallel(dim, fp8_k):
                                                if visible_pv_shared_v:
                                                    if gi_vc2 % 2 == 0:
                                                        v_tc_shared_2[d, n] = v_vt_smem_0[
                                                            d, pv_k * fp8_k + n]
                                                    else:
                                                        v_tc_shared_2[d, n] = v_vt_smem_1[
                                                            d, pv_k * fp8_k + n]
                                                else:
                                                    v_tc_shared_2[d, n] = v[
                                                        tile_b,
                                                        (n_idx - 1) * block_n + pv_k * fp8_k + n,
                                                        head_kv,
                                                        d,
                                                    ]
                                            T.wgmma_gemm(
                                                p_pv_frag_2,
                                                v_tc_shared_2,
                                                acc_delta_2,
                                                transpose_B=True,
                                                policy=T.GemmWarpPolicy.FullRow,
                                                clear_accum=(pv_k == 0),
                                            )
                                else:
                                    if gi_vc2 % 2 == 0:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                            p_pv_frag_2.data,
                                            v_tc_smem_0.access_ptr("r"),
                                            4,
                                            acc_delta_2.data,
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                            p_pv_frag_2.data,
                                            v_tc_smem_1.access_ptr("r"),
                                            4,
                                            acc_delta_2.data,
                                        )
                                T.call_extern("handle", "tl::tileops_barrier_arrive_named", 1, 256)
                                if visible_pv:
                                    if visible_pv_emitter_k224:
                                        T.call_extern(
                                            "handle",
                                            "tl::wait_wgmma_anchor<1>",
                                            T.address_of(anchor_sink[1]),
                                            gi_kc2,
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            "tl::wait_wgmma_anchor<7>",
                                            T.address_of(anchor_sink[1]),
                                            gi_kc2,
                                        )
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::wait_wgmma_anchor<1>",
                                        T.address_of(anchor_sink[1]),
                                        gi_kc2,
                                    )
                                T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                                T.barrier_arrive(k_empty)
                                for i, j in T.Parallel(half_m, block_n):
                                    acc_s_2[i, j] *= (
                                        q_scale[tile_b, tile_h, q_scale_idx]
                                        * k_scale[tile_b, head_kv, n_idx]
                                    )
                                online_softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                T.copy(ss_2, ss_shared_2)
                                T.call_extern(
                                    "handle",
                                    "tl::wait_wgmma_anchor<0>",
                                    T.address_of(anchor_sink[1]),
                                    gi_kc2,
                                )
                                if visible_pv:
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(acc_delta_2, num_regs=64)
                                    for i, j in T.Parallel(half_m, dim):
                                        acc_o_2[i, j] = (
                                            acc_o_2[i, j]
                                            + acc_delta_2[i, j]
                                            * v_scale[tile_b, head_kv, n_idx - 1]
                                        ) * ss_2[i]
                                else:
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pv_ptx_unit_wait_update_fa3_raw_64x128x224",
                                        acc_delta_2.data,
                                        4,
                                        ss_shared_2.access_ptr("r"),
                                        v_scale[tile_b, head_kv, n_idx - 1],
                                        acc_o_2.data,
                                    )
                                    T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                T.barrier_arrive(v_empty)
                                if visible_pv:
                                    T.copy(acc_s_2, p_smem_2)
                                else:
                                    T.copy(acc_s_2, p_pv_frag_2)
                                    T.call_extern(
                                        "handle",
                                        "tl::fp8_pack_p_fa3_raw_64x128x224",
                                        acc_s_2.data,
                                        p_pv_frag_2.data,
                                    )
                                gi_vc2 = gi_vc2 + 1
                            gi_kc2 = gi_kc2 + 1

                        T.barrier_wait(v_full, gi_vc2 % 2)
                        if visible_pv:
                            if visible_pv_emitter_k224:
                                load_p_fragment_k224(p_pv_frag_2, p_smem_2)
                                if gi_vc2 % 2 == 0:
                                    pv_wgmma_emitter._assign_b_shared_layout(
                                        tilelang.layout.make_quarter_bank_swizzled_layout(v_tc_smem_0)
                                    )
                                    pv_wgmma_emitter.wgmma_rs(
                                        p_pv_frag_2[:],
                                        v_tc_smem_0[:, :],
                                        acc_delta_2[:, :],
                                        clear_accum=True,
                                        wg_wait=-1,
                                    )
                                else:
                                    pv_wgmma_emitter._assign_b_shared_layout(
                                        tilelang.layout.make_quarter_bank_swizzled_layout(v_tc_smem_1)
                                    )
                                    pv_wgmma_emitter.wgmma_rs(
                                        p_pv_frag_2[:],
                                        v_tc_smem_1[:, :],
                                        acc_delta_2[:, :],
                                        clear_accum=True,
                                        wg_wait=-1,
                                    )
                            else:
                                for pv_k in range(block_n // fp8_k):
                                    T.copy(
                                        p_smem_2[:, pv_k * fp8_k:(pv_k + 1) * fp8_k],
                                        p_pv_frag_2,
                                    )
                                    for d, n in T.Parallel(dim, fp8_k):
                                        if visible_pv_shared_v:
                                            if gi_vc2 % 2 == 0:
                                                v_tc_shared_2[d, n] = v_vt_smem_0[
                                                    d, pv_k * fp8_k + n]
                                            else:
                                                v_tc_shared_2[d, n] = v_vt_smem_1[
                                                    d, pv_k * fp8_k + n]
                                        else:
                                            v_tc_shared_2[d, n] = v[
                                                tile_b,
                                                (loop_range - 1) * block_n + pv_k * fp8_k + n,
                                                head_kv,
                                                d,
                                            ]
                                    T.wgmma_gemm(
                                        p_pv_frag_2,
                                        v_tc_shared_2,
                                        acc_delta_2,
                                        transpose_B=True,
                                        policy=T.GemmWarpPolicy.FullRow,
                                        clear_accum=(pv_k == 0),
                                    )
                        else:
                            if gi_vc2 % 2 == 0:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                    p_pv_frag_2.data,
                                    v_tc_smem_0.access_ptr("r"),
                                    4,
                                    acc_delta_2.data,
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    "tl::fp8_pv_ptx_unit_begin_from_p_fa3_raw_64x128x224",
                                    p_pv_frag_2.data,
                                    v_tc_smem_1.access_ptr("r"),
                                    4,
                                    acc_delta_2.data,
                                )
                        if visible_pv:
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_delta_2, num_regs=64)
                            for i, j in T.Parallel(half_m, dim):
                                acc_o_2[i, j] += (
                                    acc_delta_2[i, j]
                                    * v_scale[tile_b, head_kv, loop_range - 1]
                                )
                        else:
                            T.call_extern(
                                "handle",
                                "tl::fp8_pv_ptx_unit_wait_update_tail_fa3_raw_64x128x224",
                                acc_delta_2.data,
                                4,
                                v_scale[tile_b, head_kv, loop_range - 1],
                                acc_o_2.data,
                            )
                            T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                        T.barrier_arrive(v_empty)
                        gi_vc2 = gi_vc2 + 1
                        if visible_pv:
                            for i, j in T.Parallel(half_m, dim):
                                acc_o_2[i, j] /= ls_2[i]
                            T.copy(
                                acc_o_2,
                                output[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                            )
                        else:
                            T.copy(ls_2, ls_shared_2)
                            T.copy(acc_o_2, acc_o_layout_seed)
                            T.call_extern(
                                "handle",
                                "tl::fp8_fa3_raw_acc_store_smem_cute_reuse_64x128",
                                acc_o_2.data,
                                ls_shared_2.access_ptr("r"),
                                4,
                                v_tc_smem_1.access_ptr("w"),
                                T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                            )
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.call_extern(
                                "handle",
                                "tl::fp8_fa3_o_smem_store_global_cute_reuse_64x128",
                                v_tc_smem_1.access_ptr("r"),
                                T.address_of(output[tile_b, row_base + half_m, tile_h, 0]),
                                heads * dim,
                            )
                        for i in T.Parallel(half_m):
                            ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                        T.copy(ls_2, lse[tile_b, tile_h, row_base + half_m:row_base + block_m])

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_raw_cuda_smoke_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("Raw CUDA FP8 GQA smoke kernel currently requires dim == 128.")
    if out_dtype not in ("float16", "bfloat16"):
        raise ValueError("Raw CUDA FP8 GQA smoke kernel outputs float16 or bfloat16.")

    output_type = "__nv_bfloat16" if out_dtype == "bfloat16" else "__half"
    source = f"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" __global__ void gqa_fp8_raw_cuda_smoke(
    void* k,
    void* k_scale,
    float* lse,
    void* output,
    void* q,
    void* q_scale,
    void* v,
    void* v_scale) {{
  {output_type}* out = reinterpret_cast<{output_type}*>(output);
  constexpr long long total_o = {batch}LL * {seq_len}LL * {heads}LL * {dim}LL;
  constexpr long long total_lse = {batch}LL * {heads}LL * {seq_len}LL;
  long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long i = tid; i < total_o; i += stride) {{
    out[i] = {output_type}(0.0f);
  }}
  for (long long i = tid; i < total_lse; i += stride) {{
    lse[i] = 0.0f;
  }}
}}
"""

    @tilelang.jit(
        out_idx=[6, 7],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def func():
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        scale_blocks = (seq_len + 127) // 128
        q_scale_shape = (batch, heads, scale_blocks)
        kv_scale_shape = (batch, heads_kv, scale_blocks)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, "float8_e4m3fn"),
            k: T.Tensor(kv_shape, "float8_e4m3fn"),
            v: T.Tensor(kv_shape, "float8_e4m3fn"),
            q_scale: T.Tensor(q_scale_shape, "float"),
            k_scale: T.Tensor(kv_scale_shape, "float"),
            v_scale: T.Tensor(kv_scale_shape, "float"),
            output: T.Tensor(q_shape, out_dtype),
            lse: T.Tensor([batch, heads, seq_len], "float"),
        ) -> None:
            T.CUDASourceCodeKernel(
                NUM_SMS,
                threads=256,
                source_code_or_path=source,
                entry_name="gqa_fp8_raw_cuda_smoke",
            )

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_raw_cuda_tilemap_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("Raw CUDA FP8 GQA tile-map kernel currently requires dim == 128.")
    if out_dtype not in ("float16", "bfloat16"):
        raise ValueError("Raw CUDA FP8 GQA tile-map kernel outputs float16 or bfloat16.")

    block_m = 128
    groups = heads // heads_kv
    m_tiles = (seq_len + block_m - 1) // block_m
    logical_tiles = batch * heads_kv * m_tiles * groups
    output_type = "__nv_bfloat16" if out_dtype == "bfloat16" else "__half"
    source = f"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" __global__ void gqa_fp8_raw_cuda_tilemap(
    void* k,
    void* k_scale,
    float* lse,
    void* output,
    void* q,
    void* q_scale,
    void* v,
    void* v_scale) {{
  {output_type}* out = reinterpret_cast<{output_type}*>(output);

  constexpr int kBatch = {batch};
  constexpr int kSeqLen = {seq_len};
  constexpr int kHeads = {heads};
  constexpr int kHeadsKv = {heads_kv};
  constexpr int kDim = {dim};
  constexpr int kBlockM = {block_m};
  constexpr int kGroups = {groups};
  constexpr int kMTiles = {m_tiles};
  constexpr int kLogicalTiles = {logical_tiles};

  for (int tile = blockIdx.x; tile < kLogicalTiles; tile += gridDim.x) {{
    int t = tile;
    int tile_g = t % kGroups;
    t /= kGroups;
    int tile_m = t % kMTiles;
    t /= kMTiles;
    int tile_hkv = t % kHeadsKv;
    int tile_b = t / kHeadsKv;
    int tile_h = tile_hkv * kGroups + tile_g;
    int row_base = tile_m * kBlockM;
    int row_limit = min(kSeqLen, row_base + kBlockM);
    float sentinel = static_cast<float>(
        tile_b * 1000000 + tile_hkv * 100000 + tile_g * 10000 + tile_m);

    int rows = row_limit - row_base;
    int tile_elems = rows * kDim;
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {{
      int r = idx / kDim;
      int d = idx - r * kDim;
      long long o_idx =
          (((static_cast<long long>(tile_b) * kSeqLen + row_base + r) * kHeads
            + tile_h) * kDim + d);
      out[o_idx] = {output_type}(sentinel);
    }}

    for (int r = threadIdx.x; r < rows; r += blockDim.x) {{
      long long lse_idx =
          (static_cast<long long>(tile_b) * kHeads + tile_h) * kSeqLen
          + row_base + r;
      lse[lse_idx] = sentinel;
    }}
  }}
}}
"""

    @tilelang.jit(
        out_idx=[6, 7],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def func():
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        scale_blocks = (seq_len + 127) // 128
        q_scale_shape = (batch, heads, scale_blocks)
        kv_scale_shape = (batch, heads_kv, scale_blocks)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, "float8_e4m3fn"),
            k: T.Tensor(kv_shape, "float8_e4m3fn"),
            v: T.Tensor(kv_shape, "float8_e4m3fn"),
            q_scale: T.Tensor(q_scale_shape, "float"),
            k_scale: T.Tensor(kv_scale_shape, "float"),
            v_scale: T.Tensor(kv_scale_shape, "float"),
            output: T.Tensor(q_shape, out_dtype),
            lse: T.Tensor([batch, heads, seq_len], "float"),
        ) -> None:
            T.CUDASourceCodeKernel(
                NUM_SMS,
                threads=256,
                source_code_or_path=source,
                entry_name="gqa_fp8_raw_cuda_tilemap",
            )

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_raw_cuda_qk_first_key_debug_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("Raw CUDA FP8 GQA QK debug kernel currently requires dim == 128.")
    if out_dtype not in ("float16", "bfloat16"):
        raise ValueError("Raw CUDA FP8 GQA QK debug kernel outputs float16 or bfloat16.")

    block_m = 128
    groups = heads // heads_kv
    m_tiles = (seq_len + block_m - 1) // block_m
    logical_tiles = batch * heads_kv * m_tiles * groups
    scale_blocks = (seq_len + 127) // 128
    output_type = "__nv_bfloat16" if out_dtype == "bfloat16" else "__half"
    source = f"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" __global__ void gqa_fp8_raw_cuda_qk_first_key_debug(
    void* k,
    void* k_scale,
    float* lse,
    void* output,
    void* q,
    void* q_scale,
    void* v,
    void* v_scale) {{
  const __nv_fp8_e4m3* q_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(q);
  const __nv_fp8_e4m3* k_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(k);
  const float* q_scale_ptr = reinterpret_cast<const float*>(q_scale);
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale);
  {output_type}* out = reinterpret_cast<{output_type}*>(output);

  constexpr int kBatch = {batch};
  constexpr int kSeqLen = {seq_len};
  constexpr int kHeads = {heads};
  constexpr int kHeadsKv = {heads_kv};
  constexpr int kDim = {dim};
  constexpr int kBlockM = {block_m};
  constexpr int kGroups = {groups};
  constexpr int kMTiles = {m_tiles};
  constexpr int kScaleBlocks = {scale_blocks};
  constexpr int kLogicalTiles = {logical_tiles};
  constexpr float kSoftmaxScale = 0.08838834764831845f;  // 1 / sqrt(128)

  for (int tile = blockIdx.x; tile < kLogicalTiles; tile += gridDim.x) {{
    int t = tile;
    int tile_g = t % kGroups;
    t /= kGroups;
    int tile_m = t % kMTiles;
    t /= kMTiles;
    int tile_hkv = t % kHeadsKv;
    int tile_b = t / kHeadsKv;
    int tile_h = tile_hkv * kGroups + tile_g;
    int row_base = tile_m * kBlockM;
    int row_limit = min(kSeqLen, row_base + kBlockM);

    for (int row = row_base + threadIdx.x; row < row_limit; row += blockDim.x) {{
      float acc = 0.0f;
      #pragma unroll
      for (int d = 0; d < kDim; ++d) {{
        long long q_idx =
            (((static_cast<long long>(tile_b) * kSeqLen + row) * kHeads + tile_h)
             * kDim + d);
        long long k_idx =
            (((static_cast<long long>(tile_b) * kSeqLen + 0) * kHeadsKv + tile_hkv)
             * kDim + d);
        acc += static_cast<float>(q_ptr[q_idx]) * static_cast<float>(k_ptr[k_idx]);
      }}
      int q_scale_idx = row / kBlockM;
      float qs = q_scale_ptr[
          (static_cast<long long>(tile_b) * kHeads + tile_h) * kScaleBlocks
          + q_scale_idx];
      float ks = k_scale_ptr[
          (static_cast<long long>(tile_b) * kHeadsKv + tile_hkv) * kScaleBlocks];
      long long lse_idx =
          (static_cast<long long>(tile_b) * kHeads + tile_h) * kSeqLen + row;
      lse[lse_idx] = acc * qs * ks * kSoftmaxScale;

      long long o_idx =
          (((static_cast<long long>(tile_b) * kSeqLen + row) * kHeads + tile_h)
           * kDim);
      out[o_idx] = {output_type}(lse[lse_idx]);
    }}
  }}
}}
"""

    @tilelang.jit(
        out_idx=[6, 7],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def func():
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        q_scale_shape = (batch, heads, scale_blocks)
        kv_scale_shape = (batch, heads_kv, scale_blocks)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, "float8_e4m3fn"),
            k: T.Tensor(kv_shape, "float8_e4m3fn"),
            v: T.Tensor(kv_shape, "float8_e4m3fn"),
            q_scale: T.Tensor(q_scale_shape, "float"),
            k_scale: T.Tensor(kv_scale_shape, "float"),
            v_scale: T.Tensor(kv_scale_shape, "float"),
            output: T.Tensor(q_shape, out_dtype),
            lse: T.Tensor([batch, heads, seq_len], "float"),
        ) -> None:
            T.CUDASourceCodeKernel(
                NUM_SMS,
                threads=256,
                source_code_or_path=source,
                entry_name="gqa_fp8_raw_cuda_qk_first_key_debug",
            )

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_raw_cuda_qk_tile_debug_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("Raw CUDA FP8 GQA QK tile debug kernel currently requires dim == 128.")
    if seq_len < 224:
        raise ValueError("Raw CUDA FP8 GQA QK tile debug kernel requires seq_len >= 224.")

    block_m = 128
    block_n = 224
    groups = heads // heads_kv
    m_tiles = (seq_len + block_m - 1) // block_m
    logical_tiles = batch * heads_kv * m_tiles * groups
    scale_blocks = (seq_len + 127) // 128
    source = f"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" __global__ void gqa_fp8_raw_cuda_qk_tile_debug(
    void* k,
    void* k_scale,
    float* lse,
    float* output,
    void* q,
    void* q_scale,
    void* v,
    void* v_scale) {{
  const __nv_fp8_e4m3* q_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(q);
  const __nv_fp8_e4m3* k_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(k);
  const float* q_scale_ptr = reinterpret_cast<const float*>(q_scale);
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale);

  constexpr int kBatch = {batch};
  constexpr int kSeqLen = {seq_len};
  constexpr int kHeads = {heads};
  constexpr int kHeadsKv = {heads_kv};
  constexpr int kDim = {dim};
  constexpr int kBlockM = {block_m};
  constexpr int kBlockN = {block_n};
  constexpr int kGroups = {groups};
  constexpr int kMTiles = {m_tiles};
  constexpr int kScaleBlocks = {scale_blocks};
  constexpr int kLogicalTiles = {logical_tiles};
  constexpr float kSoftmaxScale = 0.08838834764831845f;  // 1 / sqrt(128)

  for (int tile = blockIdx.x; tile < kLogicalTiles; tile += gridDim.x) {{
    int t = tile;
    int tile_g = t % kGroups;
    t /= kGroups;
    int tile_m = t % kMTiles;
    t /= kMTiles;
    int tile_hkv = t % kHeadsKv;
    int tile_b = t / kHeadsKv;
    int tile_h = tile_hkv * kGroups + tile_g;
    int row_base = tile_m * kBlockM;
    int row_limit = min(kSeqLen, row_base + kBlockM);
    int rows = row_limit - row_base;

    int elems = rows * kBlockN;
    for (int idx = threadIdx.x; idx < elems; idx += blockDim.x) {{
      int local_row = idx / kBlockN;
      int col = idx - local_row * kBlockN;
      int row = row_base + local_row;
      float acc = 0.0f;
      #pragma unroll
      for (int d = 0; d < kDim; ++d) {{
        long long q_idx =
            (((static_cast<long long>(tile_b) * kSeqLen + row) * kHeads + tile_h)
             * kDim + d);
        long long k_idx =
            (((static_cast<long long>(tile_b) * kSeqLen + col) * kHeadsKv + tile_hkv)
             * kDim + d);
        acc += static_cast<float>(q_ptr[q_idx]) * static_cast<float>(k_ptr[k_idx]);
      }}

      float qs = q_scale_ptr[
          (static_cast<long long>(tile_b) * kHeads + tile_h) * kScaleBlocks
          + row / kBlockM];
      float ks = k_scale_ptr[
          (static_cast<long long>(tile_b) * kHeadsKv + tile_hkv) * kScaleBlocks
          + col / 128];
      float score = acc * qs * ks * kSoftmaxScale;
      long long out_idx =
          ((((static_cast<long long>(tile_b) * kHeads + tile_h) * kMTiles + tile_m)
            * kBlockM + local_row) * kBlockN + col);
      output[out_idx] = score;
      if (col == 0) {{
        long long lse_idx =
            (static_cast<long long>(tile_b) * kHeads + tile_h) * kSeqLen + row;
        lse[lse_idx] = score;
      }}
    }}
  }}
}}
"""

    @tilelang.jit(
        out_idx=[6, 7],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def func():
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        q_scale_shape = (batch, heads, scale_blocks)
        kv_scale_shape = (batch, heads_kv, scale_blocks)
        score_shape = (batch, heads, m_tiles, block_m, block_n)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, "float8_e4m3fn"),
            k: T.Tensor(kv_shape, "float8_e4m3fn"),
            v: T.Tensor(kv_shape, "float8_e4m3fn"),
            q_scale: T.Tensor(q_scale_shape, "float"),
            k_scale: T.Tensor(kv_scale_shape, "float"),
            v_scale: T.Tensor(kv_scale_shape, "float"),
            output: T.Tensor(score_shape, "float"),
            lse: T.Tensor([batch, heads, seq_len], "float"),
        ) -> None:
            T.CUDASourceCodeKernel(
                NUM_SMS,
                threads=256,
                source_code_or_path=source,
                entry_name="gqa_fp8_raw_cuda_qk_tile_debug",
            )

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_raw_cuda_qk_wgmma_tile_debug_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("Raw CUDA FP8 GQA QK WGMMA debug kernel requires dim == 128.")
    if seq_len < 224:
        raise ValueError("Raw CUDA FP8 GQA QK WGMMA debug kernel requires seq_len >= 224.")
    if seq_len % 128 != 0:
        raise ValueError(
            "Raw CUDA FP8 GQA QK WGMMA debug kernel currently requires seq_len % 128 == 0."
        )
    if out_dtype != "float32":
        raise ValueError("Raw CUDA FP8 GQA QK WGMMA debug kernel outputs float32 scores.")

    block_m = 128
    half_m = 64
    block_n = 224
    groups = heads // heads_kv
    m_tiles = (seq_len + block_m - 1) // block_m
    logical_tiles = batch * heads_kv * m_tiles * groups
    scale_blocks = (seq_len + 127) // 128
    source = f"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "{_FP8_GQA_HELPER_PATH}"

__device__ __forceinline__ int tl_fp8_full_swizzle_128b_index(int row, int col) {{
  int const row_phase = row & 7;
  int const row_tile = row >> 3;
  int const vec_col = col >> 4;
  int const vec = col & 15;
  return row_tile * 1024 + ((row_phase * 8 + (vec_col ^ row_phase)) * 16) + vec;
}}

extern "C" __global__ void gqa_fp8_raw_cuda_qk_wgmma_tile_debug(
    void* k,
    void* k_scale,
    float* lse,
    float* output,
    void* q,
    void* q_scale,
    void* v,
    void* v_scale) {{
  const unsigned char* q_bytes_in = reinterpret_cast<const unsigned char*>(q);
  const unsigned char* k_bytes_in = reinterpret_cast<const unsigned char*>(k);
  const float* q_scale_ptr = reinterpret_cast<const float*>(q_scale);
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale);

  __shared__ __align__(1024) fp8_e4_t q_smem[{half_m} * {dim}];
  __shared__ __align__(1024) fp8_e4_t k_smem[{block_n} * {dim}];
  unsigned char* q_bytes = reinterpret_cast<unsigned char*>(q_smem);
  unsigned char* k_bytes = reinterpret_cast<unsigned char*>(k_smem);

  constexpr int kBatch = {batch};
  constexpr int kSeqLen = {seq_len};
  constexpr int kHeads = {heads};
  constexpr int kHeadsKv = {heads_kv};
  constexpr int kDim = {dim};
  constexpr int kHalfM = {half_m};
  constexpr int kBlockM = {block_m};
  constexpr int kBlockN = {block_n};
  constexpr int kGroups = {groups};
  constexpr int kMTiles = {m_tiles};
  constexpr int kScaleBlocks = {scale_blocks};
  constexpr int kLogicalTiles = {logical_tiles};
  constexpr float kSoftmaxScale = 0.08838834764831845f;  // 1 / sqrt(128)

  for (int tile = blockIdx.x; tile < kLogicalTiles; tile += gridDim.x) {{
    int t = tile;
    int tile_g = t % kGroups;
    t /= kGroups;
    int tile_m = t % kMTiles;
    t /= kMTiles;
    int tile_hkv = t % kHeadsKv;
    int tile_b = t / kHeadsKv;
    int tile_h = tile_hkv * kGroups + tile_g;
    int row_base = tile_m * kBlockM;

    for (int idx = threadIdx.x; idx < kBlockN * kDim; idx += blockDim.x) {{
      int col = idx / kDim;
      int d = idx - col * kDim;
      long long k_idx =
          (((static_cast<long long>(tile_b) * kSeqLen + col) * kHeadsKv + tile_hkv)
           * kDim + d);
      k_bytes[tl_fp8_full_swizzle_128b_index(col, d)] = k_bytes_in[k_idx];
    }}

    for (int half = 0; half < 2; ++half) {{
      int half_row_base = row_base + half * kHalfM;
      for (int idx = threadIdx.x; idx < kHalfM * kDim; idx += blockDim.x) {{
        int local_row = idx / kDim;
        int d = idx - local_row * kDim;
        int row = half_row_base + local_row;
        long long q_idx =
            (((static_cast<long long>(tile_b) * kSeqLen + row) * kHeads + tile_h)
             * kDim + d);
        q_bytes[tl_fp8_full_swizzle_128b_index(local_row, d)] = q_bytes_in[q_idx];
      }}
      __syncthreads();

      float acc_s[112];
      #pragma unroll
      for (int i = 0; i < 112; ++i) {{
        acc_s[i] = 0.0f;
      }}

      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, q_smem);
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, k_smem);

      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 112);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int warp_j = 0; warp_j < 7; ++warp_j) {{
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {{
          tl::wgmma_ss<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3,
                       tl::DataType::kFloat32, 64, 32, 32, false, false, 1, 1>(
              uint64_t(desc_a + ((ki * 32) >> 4)),
              uint64_t(desc_b + (((warp_j * 4096) + (ki * 32)) >> 4)),
              reinterpret_cast<uint32_t*>(acc_s + warp_j * 16),
              ((0 < ki) ? 1 : 0));
        }}
      }}
      tl::warpgroup_commit_batch();
      tl::wait_wgmma<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 112);

      float qs = q_scale_ptr[
          (static_cast<long long>(tile_b) * kHeads + tile_h) * kScaleBlocks
          + row_base / kBlockM];
      float ks0 = k_scale_ptr[(static_cast<long long>(tile_b) * kHeadsKv + tile_hkv)
                              * kScaleBlocks + 0];
      float ks1 = k_scale_ptr[(static_cast<long long>(tile_b) * kHeadsKv + tile_hkv)
                              * kScaleBlocks + 1];
      float* tile_out =
          output + ((((static_cast<long long>(tile_b) * kHeads + tile_h) * kMTiles + tile_m)
                     * kBlockM + half * kHalfM) * kBlockN);
      float* tile_lse =
          lse + (static_cast<long long>(tile_b) * kHeads + tile_h) * kSeqLen
              + half_row_base;
      #pragma unroll
      for (int warp_j = 0; warp_j < 7; ++warp_j) {{
        tl::fp8_qk_acc_store_global_64x32_chunk(
            acc_s + warp_j * 16, warp_j * 32, qs, ks0, ks1, kSoftmaxScale,
            tile_out, kBlockN, tile_lse);
      }}
      __syncthreads();
    }}
  }}
}}
"""

    @tilelang.jit(
        out_idx=[6, 7],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def func():
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        q_scale_shape = (batch, heads, scale_blocks)
        kv_scale_shape = (batch, heads_kv, scale_blocks)
        score_shape = (batch, heads, m_tiles, block_m, block_n)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, "float8_e4m3fn"),
            k: T.Tensor(kv_shape, "float8_e4m3fn"),
            v: T.Tensor(kv_shape, "float8_e4m3fn"),
            q_scale: T.Tensor(q_scale_shape, "float"),
            k_scale: T.Tensor(kv_scale_shape, "float"),
            v_scale: T.Tensor(kv_scale_shape, "float"),
            output: T.Tensor(score_shape, "float"),
            lse: T.Tensor([batch, heads, seq_len], "float"),
        ) -> None:
            T.CUDASourceCodeKernel(
                NUM_SMS,
                threads=128,
                source_code_or_path=source,
                entry_name="gqa_fp8_raw_cuda_qk_wgmma_tile_debug",
            )

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_fp8_fa3_contract_raw_cuda_qk_wgmma_softmax_tile_debug_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
) -> Callable:
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim != 128:
        raise ValueError("Raw CUDA FP8 GQA QK WGMMA softmax debug requires dim == 128.")
    if seq_len < 224:
        raise ValueError("Raw CUDA FP8 GQA QK WGMMA softmax debug requires seq_len >= 224.")
    if seq_len % 128 != 0:
        raise ValueError(
            "Raw CUDA FP8 GQA QK WGMMA softmax debug currently requires seq_len % 128 == 0."
        )
    if out_dtype != "float32":
        raise ValueError("Raw CUDA FP8 GQA QK WGMMA softmax debug outputs float32.")

    block_m = 128
    half_m = 64
    block_n = 224
    groups = heads // heads_kv
    m_tiles = (seq_len + block_m - 1) // block_m
    logical_tiles = batch * heads_kv * m_tiles * groups
    scale_blocks = (seq_len + 127) // 128
    source = f"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include "{_FP8_GQA_HELPER_PATH}"

__device__ __forceinline__ int tl_fp8_full_swizzle_128b_index(int row, int col) {{
  int const row_phase = row & 7;
  int const row_tile = row >> 3;
  int const vec_col = col >> 4;
  int const vec = col & 15;
  return row_tile * 1024 + ((row_phase * 8 + (vec_col ^ row_phase)) * 16) + vec;
}}

extern "C" __global__ void gqa_fp8_raw_cuda_qk_wgmma_softmax_tile_debug(
    void* k,
    void* k_scale,
    float* lse,
    float* output,
    void* q,
    void* q_scale,
    void* v,
    void* v_scale) {{
  const unsigned char* q_bytes_in = reinterpret_cast<const unsigned char*>(q);
  const unsigned char* k_bytes_in = reinterpret_cast<const unsigned char*>(k);
  const float* q_scale_ptr = reinterpret_cast<const float*>(q_scale);
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale);

  __shared__ __align__(1024) fp8_e4_t q_smem[{half_m} * {dim}];
  __shared__ __align__(1024) fp8_e4_t k_smem[{block_n} * {dim}];
  unsigned char* q_bytes = reinterpret_cast<unsigned char*>(q_smem);
  unsigned char* k_bytes = reinterpret_cast<unsigned char*>(k_smem);

  constexpr int kBatch = {batch};
  constexpr int kSeqLen = {seq_len};
  constexpr int kHeads = {heads};
  constexpr int kHeadsKv = {heads_kv};
  constexpr int kDim = {dim};
  constexpr int kHalfM = {half_m};
  constexpr int kBlockM = {block_m};
  constexpr int kBlockN = {block_n};
  constexpr int kGroups = {groups};
  constexpr int kMTiles = {m_tiles};
  constexpr int kScaleBlocks = {scale_blocks};
  constexpr int kLogicalTiles = {logical_tiles};
  constexpr float kSoftmaxScale = 0.08838834764831845f;  // 1 / sqrt(128)

  for (int tile = blockIdx.x; tile < kLogicalTiles; tile += gridDim.x) {{
    int t = tile;
    int tile_g = t % kGroups;
    t /= kGroups;
    int tile_m = t % kMTiles;
    t /= kMTiles;
    int tile_hkv = t % kHeadsKv;
    int tile_b = t / kHeadsKv;
    int tile_h = tile_hkv * kGroups + tile_g;
    int row_base = tile_m * kBlockM;

    for (int idx = threadIdx.x; idx < kBlockN * kDim; idx += blockDim.x) {{
      int col = idx / kDim;
      int d = idx - col * kDim;
      long long k_idx =
          (((static_cast<long long>(tile_b) * kSeqLen + col) * kHeadsKv + tile_hkv)
           * kDim + d);
      k_bytes[tl_fp8_full_swizzle_128b_index(col, d)] = k_bytes_in[k_idx];
    }}

    for (int half = 0; half < 2; ++half) {{
      int half_row_base = row_base + half * kHalfM;
      for (int idx = threadIdx.x; idx < kHalfM * kDim; idx += blockDim.x) {{
        int local_row = idx / kDim;
        int d = idx - local_row * kDim;
        int row = half_row_base + local_row;
        long long q_idx =
            (((static_cast<long long>(tile_b) * kSeqLen + row) * kHeads + tile_h)
             * kDim + d);
        q_bytes[tl_fp8_full_swizzle_128b_index(local_row, d)] = q_bytes_in[q_idx];
      }}
      __syncthreads();

      float acc_s[112];
      #pragma unroll
      for (int i = 0; i < 112; ++i) {{
        acc_s[i] = 0.0f;
      }}

      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, q_smem);
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, k_smem);

      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 112);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int warp_j = 0; warp_j < 7; ++warp_j) {{
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {{
          tl::wgmma_ss<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3,
                       tl::DataType::kFloat32, 64, 32, 32, false, false, 1, 1>(
              uint64_t(desc_a + ((ki * 32) >> 4)),
              uint64_t(desc_b + (((warp_j * 4096) + (ki * 32)) >> 4)),
              reinterpret_cast<uint32_t*>(acc_s + warp_j * 16),
              ((0 < ki) ? 1 : 0));
        }}
      }}
      tl::warpgroup_commit_batch();
      tl::wait_wgmma<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s + 0), 112);

      float qs = q_scale_ptr[
          (static_cast<long long>(tile_b) * kHeads + tile_h) * kScaleBlocks
          + row_base / kBlockM];
      float ks0 = k_scale_ptr[(static_cast<long long>(tile_b) * kHeadsKv + tile_hkv)
                              * kScaleBlocks + 0];
      float ks1 = k_scale_ptr[(static_cast<long long>(tile_b) * kHeadsKv + tile_hkv)
                              * kScaleBlocks + 1];
      float* half_out =
          output + ((((static_cast<long long>(tile_b) * kHeads + tile_h) * kMTiles + tile_m)
                     * kBlockM + half * kHalfM) * kBlockN);
      #pragma unroll
      for (int warp_j = 0; warp_j < 7; ++warp_j) {{
        tl::fp8_qk_acc_store_global_64x32_chunk(
            acc_s + warp_j * 16, warp_j * 32, qs, ks0, ks1, kSoftmaxScale,
            half_out, kBlockN, nullptr);
      }}
      __syncthreads();

      for (int row = threadIdx.x; row < kHalfM; row += blockDim.x) {{
        float* row_out = half_out + row * kBlockN;
        float row_max = -CUDART_INF_F;
        #pragma unroll
        for (int col = 0; col < kBlockN; ++col) {{
          row_max = fmaxf(row_max, row_out[col]);
        }}
        float row_sum = 0.0f;
        #pragma unroll
        for (int col = 0; col < kBlockN; ++col) {{
          float p = expf(row_out[col] - row_max);
          row_out[col] = p;
          row_sum += p;
        }}
        float inv_sum = 1.0f / row_sum;
        #pragma unroll
        for (int col = 0; col < kBlockN; ++col) {{
          row_out[col] = row_out[col] * inv_sum;
        }}
        lse[(static_cast<long long>(tile_b) * kHeads + tile_h) * kSeqLen
            + half_row_base + row] = logf(row_sum) + row_max;
      }}
      __syncthreads();
    }}
  }}
}}
"""

    @tilelang.jit(
        out_idx=[6, 7],
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def func():
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        q_scale_shape = (batch, heads, scale_blocks)
        kv_scale_shape = (batch, heads_kv, scale_blocks)
        score_shape = (batch, heads, m_tiles, block_m, block_n)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, "float8_e4m3fn"),
            k: T.Tensor(kv_shape, "float8_e4m3fn"),
            v: T.Tensor(kv_shape, "float8_e4m3fn"),
            q_scale: T.Tensor(q_scale_shape, "float"),
            k_scale: T.Tensor(kv_scale_shape, "float"),
            v_scale: T.Tensor(kv_scale_shape, "float"),
            output: T.Tensor(score_shape, "float"),
            lse: T.Tensor([batch, heads, seq_len], "float"),
        ) -> None:
            T.CUDASourceCodeKernel(
                NUM_SMS,
                threads=128,
                source_code_or_path=source,
                entry_name="gqa_fp8_raw_cuda_qk_wgmma_softmax_tile_debug",
            )

        return main

    return func


@torch.library.custom_op("top::gqa_fwd_fp8_wgmma_wrapped_kernel", mutates_args=())
def _gqa_fwd_fp8_wgmma_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    use_register_p: bool,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_fp8_wgmma_kernel(batch, heads, heads_kv, seq_len, dim, out_dtype,
                                     use_register_p)(
        block_m, block_n, num_stages, threads)(q, k, v, q_scale, k_scale, v_scale)


@_gqa_fwd_fp8_wgmma_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    use_register_p: bool,
    *inputs: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch_dtype = torch.float16 if out_dtype == "float16" else torch.bfloat16
    fake_o = torch.empty((batch, seq_len, heads, dim), dtype=torch_dtype, device=inputs[0].device)
    fake_lse = torch.empty((batch, heads, seq_len), dtype=torch.float32, device=inputs[0].device)
    return fake_o, fake_lse


@torch.library.custom_op("top::gqa_fwd_fp8_ws_persistent_wrapped_kernel", mutates_args=())
def _gqa_fwd_fp8_ws_persistent_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    block_n: int,
    use_register_p: bool,
    use_full_pv_gemm: bool,
    use_ldsm_v_transpose: bool,
    use_tl_full_v_transpose: bool,
    use_tl_pack_ldsm_v_transpose: bool,
    use_fa3_pv_extern: bool,
    use_fa3_v_tma_direct: bool,
    use_ptx_pv: bool,
    use_ptx_pv_accumulate: bool,
    use_ptx_pv_direct_store: bool,
    use_ptx_pv_fa3_epilogue_store: bool,
    use_ptx_pv_fa3_epilogue_reuse_v_smem: bool,
    use_fa3_v_inplace_transform: bool,
    use_fa3_v_inplace_barrier_transform: bool,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_fp8_ws_persistent_kernel(batch, heads, heads_kv, seq_len, dim,
                                             out_dtype, use_register_p, use_full_pv_gemm,
                                             use_ldsm_v_transpose, use_tl_full_v_transpose,
                                             use_tl_pack_ldsm_v_transpose, use_fa3_pv_extern,
                                             use_fa3_v_tma_direct, use_ptx_pv,
                                             use_ptx_pv_accumulate,
                                             use_ptx_pv_direct_store,
                                             use_ptx_pv_fa3_epilogue_store,
                                             use_ptx_pv_fa3_epilogue_reuse_v_smem,
                                             use_fa3_v_inplace_transform,
                                             use_fa3_v_inplace_barrier_transform)(block_n)(
                                                 q, k, v, q_scale, k_scale, v_scale)


@_gqa_fwd_fp8_ws_persistent_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    out_dtype: str,
    block_n: int,
    use_register_p: bool,
    use_full_pv_gemm: bool,
    use_ldsm_v_transpose: bool,
    use_tl_full_v_transpose: bool,
    use_tl_pack_ldsm_v_transpose: bool,
    use_fa3_pv_extern: bool,
    use_fa3_v_tma_direct: bool,
    use_ptx_pv: bool,
    use_ptx_pv_accumulate: bool,
    use_ptx_pv_direct_store: bool,
    use_ptx_pv_fa3_epilogue_store: bool,
    use_ptx_pv_fa3_epilogue_reuse_v_smem: bool,
    use_fa3_v_inplace_transform: bool,
    use_fa3_v_inplace_barrier_transform: bool,
    *inputs: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch_dtype = torch.float16 if out_dtype == "float16" else torch.bfloat16
    fake_o = torch.empty((batch, seq_len, heads, dim), dtype=torch_dtype, device=inputs[0].device)
    fake_lse = torch.empty((batch, heads, seq_len), dtype=torch.float32, device=inputs[0].device)
    return fake_o, fake_lse


def _expand_fa3_gqa_descales(
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept FA3 [batch, heads_kv] descales while preserving internal scale layout.

    The TileLang kernels below still index scale metadata as
    q: [batch, heads, scale_blocks] and k/v: [batch, heads_kv, scale_blocks].
    FA3 exposes q/k/v descales as [batch, heads_kv], with q grouped by KV head.
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
                    "q_descale/q_scale must have shape "
                    f"({batch}, {heads_kv}) for FA3 descales or "
                    f"({batch}, {heads}, {scale_blocks}) for internal scales, got {tuple(x.shape)}."
            )
            return _record_stream(_as_float_contiguous(x))
        if x.ndim != 2 or tuple(x.shape) != (batch, heads_kv):
            raise ValueError(
                "q_descale must have FA3 shape "
                f"({batch}, {heads_kv}) or internal shape "
                f"({batch}, {heads}, {scale_blocks}), got {tuple(x.shape)}."
            )
        x = _as_float_contiguous(x).repeat_interleave(group_size, dim=1)
        return _record_stream(x[:, :, None].expand(batch, heads, scale_blocks).contiguous())

    def _expand_kv(name: str, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if tuple(x.shape) != (batch, heads_kv, scale_blocks):
                raise ValueError(
                    f"{name} must have FA3 shape ({batch}, {heads_kv}) or internal shape "
                    f"({batch}, {heads_kv}, {scale_blocks}), got {tuple(x.shape)}."
                )
            return _record_stream(_as_float_contiguous(x))
        if x.ndim != 2 or tuple(x.shape) != (batch, heads_kv):
            raise ValueError(
                f"{name} must have FA3 shape ({batch}, {heads_kv}) or internal shape "
                f"({batch}, {heads_kv}, {scale_blocks}), got {tuple(x.shape)}."
            )
        x = _as_float_contiguous(x)
        return _record_stream(x[:, :, None].expand(batch, heads_kv, scale_blocks).contiguous())

    return _expand_q(q_descale), _expand_kv("k_descale", k_descale), _expand_kv(
        "v_descale", v_descale)


class GQAFwdFP8WgmmaKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if dim != 128:
            raise ValueError("GQAFwdFP8WgmmaKernel currently requires dim == 128.")
        if seq_len % 128 != 0:
            raise ValueError("GQAFwdFP8WgmmaKernel currently requires seq_len % 128 == 0.")
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("GQAFwdFP8WgmmaKernel currently outputs float16 or bfloat16.")

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_wgmma_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
            bool(self.config.get("use_register_p", False)),
        )

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 128,
            "num_stages": 0,
            "threads": 128,
            "use_register_p": False,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError("GQAFwdFP8WgmmaKernel expects q/k/v to be torch.float8_e4m3fn.")
        q_scale, k_scale, v_scale = _expand_fa3_gqa_descales(
            q_scale,
            k_scale,
            v_scale,
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
        )
        return _gqa_fwd_fp8_wgmma_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
            self.dim,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
            bool(self.config.get("use_register_p", False)),
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
        )


class GQAFwdFP8WsPersistentKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if dim != 128:
            raise ValueError("GQAFwdFP8WsPersistentKernel currently requires dim == 128.")
        if seq_len % 128 != 0:
            raise ValueError("GQAFwdFP8WsPersistentKernel currently requires seq_len % 128 == 0.")
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8WsPersistentKernel currently outputs float16 or bfloat16.")

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        if self.config.get("use_register_p", False) and self.config.get("use_full_pv_gemm", False):
            raise ValueError(
                "GQAFwdFP8WsPersistentKernel full-PV path currently requires shared P "
                "(use_register_p=False).")
        if self.config.get("use_tl_full_v_transpose", False) and not self.config.get(
                "use_full_pv_gemm", False):
            raise ValueError(
                "use_tl_full_v_transpose currently targets the full-PV SWIZZLE_128B "
                "contract and requires use_full_pv_gemm=True.")
        if self.config.get("use_tl_pack_ldsm_v_transpose", False) and not self.config.get(
                "use_full_pv_gemm", False):
            raise ValueError(
                "use_tl_pack_ldsm_v_transpose targets the full-PV SWIZZLE_128B "
                "contract and requires use_full_pv_gemm=True.")
        if self.config.get("use_fa3_pv_extern", False) and (
                self.config.get("use_register_p", False)
                or self.config.get("use_full_pv_gemm", False)
                or self.config.get("use_tl_full_v_transpose", False)
                or self.config.get("use_tl_pack_ldsm_v_transpose", False)
                or self.config.get("use_ldsm_v_transpose", False)):
            raise ValueError(
                "use_fa3_pv_extern is an experimental standalone PV path and cannot "
                "be combined with use_register_p/use_full_pv_gemm/use_tl_full_v_transpose/"
                "use_tl_pack_ldsm_v_transpose/use_ldsm_v_transpose.")
        if self.config.get("use_fa3_v_tma_direct", False) and not self.config.get(
                "use_fa3_pv_extern", False):
            raise ValueError("use_fa3_v_tma_direct requires use_fa3_pv_extern=True.")
        if self.config.get("use_fa3_v_inplace_transform", False) and not (
                self.config.get("use_fa3_v_tma_direct", False)
                and self.config.get("use_fa3_pv_extern", False)):
            raise ValueError(
                "use_fa3_v_inplace_transform requires use_fa3_v_tma_direct=True "
                "and use_fa3_pv_extern=True."
            )
        if self.config.get("use_fa3_v_inplace_barrier_transform", False) and not self.config.get(
                "use_fa3_v_inplace_transform", False):
            raise ValueError(
                "use_fa3_v_inplace_barrier_transform requires "
                "use_fa3_v_inplace_transform=True."
            )
        if self.config.get("use_ptx_pv", False) and not self.config.get(
                "use_fa3_pv_extern", False):
            raise ValueError("use_ptx_pv requires use_fa3_pv_extern=True.")
        if self.config.get("use_ptx_pv_accumulate", False):
            if not self.config.get("use_fa3_pv_extern", False):
                raise ValueError("use_ptx_pv_accumulate requires use_fa3_pv_extern=True.")
            self.config["use_ptx_pv"] = True
        if self.config.get("use_ptx_pv_direct_store", False):
            if not self.config.get("use_ptx_pv_accumulate", False):
                raise ValueError("use_ptx_pv_direct_store requires use_ptx_pv_accumulate=True.")
        if self.config.get("use_ptx_pv_fa3_epilogue_store", False):
            if not self.config.get("use_ptx_pv_accumulate", False):
                raise ValueError(
                    "use_ptx_pv_fa3_epilogue_store requires use_ptx_pv_accumulate=True."
                )
            if self.config.get("use_ptx_pv_direct_store", False):
                raise ValueError(
                    "use_ptx_pv_fa3_epilogue_store cannot be combined with "
                    "use_ptx_pv_direct_store."
                )
        if self.config.get("use_ptx_pv_fa3_epilogue_reuse_v_smem", False):
            if not self.config.get("use_ptx_pv_fa3_epilogue_store", False):
                raise ValueError(
                "use_ptx_pv_fa3_epilogue_reuse_v_smem requires "
                "use_ptx_pv_fa3_epilogue_store=True."
                )
        self.kernel = _gqa_fwd_fp8_ws_persistent_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
            bool(self.config.get("use_register_p", False)),
            bool(self.config.get("use_full_pv_gemm", False)),
            bool(self.config.get("use_ldsm_v_transpose", False)),
            bool(self.config.get("use_tl_full_v_transpose", False)),
            bool(self.config.get("use_tl_pack_ldsm_v_transpose", False)),
            bool(self.config.get("use_fa3_pv_extern", False)),
            bool(self.config.get("use_fa3_v_tma_direct", False)),
            bool(self.config.get("use_ptx_pv", False)),
            bool(self.config.get("use_ptx_pv_accumulate", False)),
            bool(self.config.get("use_ptx_pv_direct_store", False)),
            bool(self.config.get("use_ptx_pv_fa3_epilogue_store", False)),
            bool(self.config.get("use_ptx_pv_fa3_epilogue_reuse_v_smem", False)),
            bool(self.config.get("use_fa3_v_inplace_transform", False)),
            bool(self.config.get("use_fa3_v_inplace_barrier_transform", False)),
        )

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "threads": 384,
            "use_register_p": False,
            "use_full_pv_gemm": False,
            "use_ldsm_v_transpose": False,
            "use_tl_full_v_transpose": False,
            "use_tl_pack_ldsm_v_transpose": False,
            "use_fa3_pv_extern": False,
            "use_fa3_v_tma_direct": False,
            "use_ptx_pv": False,
            "use_ptx_pv_accumulate": False,
            "use_ptx_pv_direct_store": False,
            "use_ptx_pv_fa3_epilogue_store": False,
            "use_ptx_pv_fa3_epilogue_reuse_v_smem": False,
            "use_fa3_v_inplace_transform": False,
            "use_fa3_v_inplace_barrier_transform": False,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError("GQAFwdFP8WsPersistentKernel expects q/k/v to be torch.float8_e4m3fn.")
        q_scale, k_scale, v_scale = _expand_fa3_gqa_descales(
            q_scale,
            k_scale,
            v_scale,
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
        )
        return _gqa_fwd_fp8_ws_persistent_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
            self.dim,
            self.dtype_str,
            self.config["block_n"],
            bool(self.config.get("use_register_p", False)),
            bool(self.config.get("use_full_pv_gemm", False)),
            bool(self.config.get("use_ldsm_v_transpose", False)),
            bool(self.config.get("use_tl_full_v_transpose", False)),
            bool(self.config.get("use_tl_pack_ldsm_v_transpose", False)),
            bool(self.config.get("use_fa3_pv_extern", False)),
            bool(self.config.get("use_fa3_v_tma_direct", False)),
            bool(self.config.get("use_ptx_pv", False)),
            bool(self.config.get("use_ptx_pv_accumulate", False)),
            bool(self.config.get("use_ptx_pv_direct_store", False)),
            bool(self.config.get("use_ptx_pv_fa3_epilogue_store", False)),
            bool(self.config.get("use_ptx_pv_fa3_epilogue_reuse_v_smem", False)),
            bool(self.config.get("use_fa3_v_inplace_transform", False)),
            bool(self.config.get("use_fa3_v_inplace_barrier_transform", False)),
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
        )


class GQAFwdFP8Fa3ContractKernel(GQAFwdFP8WsPersistentKernel):
    """Experimental FP8 GQA path that keeps PV in the FA3 contract.

    This class is intentionally a separate entry point from
    GQAFwdFP8WsPersistentKernel so PTX work on the FA3 P/V/O contract can move
    forward without perturbing the default TileOps path.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        config = dict(config or {})
        if config.get("use_fa3_pv_extern") is False:
            raise ValueError("GQAFwdFP8Fa3ContractKernel requires use_fa3_pv_extern=True.")
        if config.get("use_fa3_v_tma_direct") is False:
            raise ValueError("GQAFwdFP8Fa3ContractKernel requires use_fa3_v_tma_direct=True.")
        config["use_fa3_pv_extern"] = True
        config["use_fa3_v_tma_direct"] = True
        super().__init__(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            out_dtype,
            config=config,
            tune=tune,
        )

    @property
    def default_config(self) -> dict:
        config = super().default_config
        config.update({
            "use_fa3_pv_extern": True,
            "use_fa3_v_tma_direct": True,
            # Reserved for the next step: PTX register pack / PV / O update inside
            # this experimental FA3-contract entry point.
            "use_ptx_pv": False,
        })
        return config


class GQAFwdFP8Fa3ContractPtxAccKernel(GQAFwdFP8Fa3ContractKernel):
    """FA3-contract FP8 GQA path with PTX PV and FA3-layout O accumulation."""

    @property
    def default_config(self) -> dict:
        config = super().default_config
        config.update({
            "use_ptx_pv": True,
            "use_ptx_pv_accumulate": True,
        })
        return config


class GQAFwdFP8Fa3ContractPtxAccBN224Kernel(GQAFwdFP8WsPersistentKernel):
    """BN224 FA3-contract experiment with a slim single-consumer TileLang kernel."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        if heads % heads_kv != 0:
            raise ValueError("GQAFwdFP8Fa3ContractPtxAccBN224Kernel requires heads % heads_kv == 0.")
        if dim != 128:
            raise ValueError("GQAFwdFP8Fa3ContractPtxAccBN224Kernel currently requires dim == 128.")
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224Kernel currently requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224Kernel currently outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
        )()

    @property
    def default_config(self) -> dict:
        return {
            "block_n": 224,
            "threads": 128,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224Kernel expects q/k/v to be torch.float8_e4m3fn."
            )
        q_scale, k_scale, v_scale = _expand_fa3_gqa_descales(
            q_scale,
            k_scale,
            v_scale,
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
        )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractPtxAccBN224WsKernel(GQAFwdFP8WsPersistentKernel):
    """WS persistent BN224 FA3-contract path with P/V swizzle aligned to FA3."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsKernel currently requires seq_len % 224 == 0."
            )
        super().__init__(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            out_dtype,
            config=config,
            tune=tune,
        )

    @property
    def default_config(self) -> dict:
        config = super().default_config
        config.update({
            "block_n": 224,
            "use_fa3_pv_extern": True,
            "use_fa3_v_tma_direct": False,
            "use_ptx_pv": True,
            "use_ptx_pv_accumulate": True,
            "use_ptx_pv_fa3_epilogue_store": True,
            "use_ptx_pv_fa3_epilogue_reuse_v_smem": True,
        })
        return config


class GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(GQAFwdFP8Fa3ContractPtxAccBN224WsKernel):
    """BN224 WS FA3-contract path with PTX TMA directly landing V into FA3 Vt."""

    @property
    def default_config(self) -> dict:
        config = super().default_config
        config.update({
            "use_fa3_v_tma_direct": True,
            "use_fa3_v_inplace_transform": True,
            "use_fa3_v_inplace_barrier_transform": True,
            "use_ptx_pv_fa3_epilogue_reuse_v_smem": False,
        })
        return config

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(q, k, v, q_descale, k_descale, v_descale)


class GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel):
    """BN224 WS FA3-contract path with in-place Vt -> Vtc transform per V stage."""

    @property
    def default_config(self) -> dict:
        config = super().default_config
        config.update({
            "use_fa3_v_inplace_transform": True,
            "use_ptx_pv_fa3_epilogue_reuse_v_smem": False,
        })
        return config


class GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceBarrierKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceKernel):
    """BN224 WS FA3-contract path with producer-barriered in-place V transform."""

    @property
    def default_config(self) -> dict:
        config = super().default_config
        config.update({
            "use_fa3_v_inplace_barrier_transform": True,
        })
        return config


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel(Kernel):
    """Separate BN224 WS FP8 kernel for QK(n) / PV(n-1) overlap experiments."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
        )()

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 224,
            "threads": 384,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel):
    """BN224 overlap kernel with streaming 4-reg P operands for each PV WGMMA."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPKernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPKernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
            True,
        )()


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel):
    """Streaming-P overlap kernel using plain wait_group plus wgmma fence."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
            True,
            True,
        )()


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel(Kernel):
    """BN224 overlap experiment using fragment P and fragment delta buffers."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_fragment_delta_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
        )()

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 224,
            "threads": 384,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel):
    """BN224 overlap experiment with K32 TileLang-visible PV into fragment delta."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_fragment_delta_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
            True,
        )()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel):
    """BN224 visible-PV delta experiment that reuses producer-loaded shared V."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_fragment_delta_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
            True,
            True,
        )()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel):
    """BN224 visible-PV delta experiment using a K224 WGMMA emitter bracket."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_fragment_delta_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
            True,
            False,
            True,
        )()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel(Kernel):
    """BN224 overlap experiment using fragment P and local delta buffers."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_frag_p_local_delta_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
        )()

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 224,
            "threads": 384,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel):
    """BN224 FP8 WS candidate with FP16-style QK[n] / PV[n-1] ping-pong.

    This keeps P[n-1] in a PV operand-A fragment and starts PV[n-1] while
    QK[n] is in flight. The PV result is held in a per-thread delta buffer
    until the online-softmax rescale for block n is known, then folded into
    acc_o with v_scale[n-1].
    """


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel):
    """Frag-P local-delta overlap kernel with corrected delayed-delta rescale."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedKernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedKernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_frag_p_local_delta_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
            True,
        )()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedKernel):
    """BN224 FP8 WS ping-pong candidate with corrected delayed-delta rescale."""


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedPreRescaleKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel):
    """Corrected local-delta kernel that rescales O before waiting for PV."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedPreRescaleKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedPreRescaleKernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedPreRescaleKernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedPreRescaleKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_frag_p_local_delta_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
            True,
            True,
        )()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedPreRescaleKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedPreRescaleKernel(
        GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaCorrectedPreRescaleKernel):
    """BN224 FP8 WS ping-pong candidate with pre-wait corrected delta merge."""


class GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapLocalPKernel(Kernel):
    """Copied BN224 overlap experiment using local raw P buffers across extern calls."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapLocalPKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapLocalPKernel currently requires dim == 128."
            )
        if seq_len % 224 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapLocalPKernel requires seq_len % 224 == 0."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapLocalPKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_bn224_ws_overlap_local_p_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
        )()

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 224,
            "threads": 384,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapLocalPKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractRawCudaSmokeKernel(Kernel):
    """Raw CUDA source-code launch smoke test for future FP8 GQA PTX kernels."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaSmokeKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaSmokeKernel currently requires dim == 128."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaSmokeKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_raw_cuda_smoke_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
        )()

    @property
    def default_config(self) -> dict:
        return {
            "threads": 256,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaSmokeKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractRawCudaTileMapKernel(Kernel):
    """Raw CUDA persistent tile-map smoke test for future FP8 GQA PTX kernels."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaTileMapKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaTileMapKernel currently requires dim == 128."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaTileMapKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_raw_cuda_tilemap_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
        )()

    @property
    def default_config(self) -> dict:
        return {
            "threads": 256,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaTileMapKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractRawCudaQKFirstKeyDebugKernel(Kernel):
    """Raw CUDA debug kernel that computes Q[row] dot K[0] into lse."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKFirstKeyDebugKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKFirstKeyDebugKernel currently requires dim == 128."
            )
        if out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKFirstKeyDebugKernel outputs float16 or bfloat16."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = out_dtype
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_raw_cuda_qk_first_key_debug_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            self.dtype_str,
        )()

    @property
    def default_config(self) -> dict:
        return {
            "threads": 256,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKFirstKeyDebugKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractRawCudaQKTileDebugKernel(Kernel):
    """Raw CUDA debug kernel that returns the first 128x224 QK score tile."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKTileDebugKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKTileDebugKernel currently requires dim == 128."
            )
        if seq_len < 224:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKTileDebugKernel requires seq_len >= 224."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = torch.float32
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_raw_cuda_qk_tile_debug_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            "float32",
        )()

    @property
    def default_config(self) -> dict:
        return {
            "threads": 256,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKTileDebugKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractRawCudaQKWgmmaTileDebugKernel(Kernel):
    """Raw CUDA debug kernel that computes the first 128x224 QK tile with WGMMA."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKWgmmaTileDebugKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKWgmmaTileDebugKernel currently requires dim == 128."
            )
        if seq_len < 224:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKWgmmaTileDebugKernel requires seq_len >= 224."
            )
        if seq_len % 128 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKWgmmaTileDebugKernel currently requires seq_len % 128 == 0."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = torch.float32
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_raw_cuda_qk_wgmma_tile_debug_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            "float32",
        )()

    @property
    def default_config(self) -> dict:
        return {
            "threads": 128,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKWgmmaTileDebugKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractRawCudaQKWgmmaSoftmaxTileDebugKernel(Kernel):
    """Raw CUDA debug kernel that computes local softmax over one 128x224 QK tile."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        out_dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if heads % heads_kv != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKWgmmaSoftmaxTileDebugKernel requires heads % heads_kv == 0."
            )
        if dim != 128:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKWgmmaSoftmaxTileDebugKernel currently requires dim == 128."
            )
        if seq_len < 224:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKWgmmaSoftmaxTileDebugKernel requires seq_len >= 224."
            )
        if seq_len % 128 != 0:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKWgmmaSoftmaxTileDebugKernel currently requires seq_len % 128 == 0."
            )

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = torch.float32
        self.init_config(config, tune)
        self.kernel = _gqa_fwd_fp8_fa3_contract_raw_cuda_qk_wgmma_softmax_tile_debug_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            "float32",
        )()

    @property
    def default_config(self) -> dict:
        return {
            "threads": 128,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "GQAFwdFP8Fa3ContractRawCudaQKWgmmaSoftmaxTileDebugKernel expects q/k/v to be torch.float8_e4m3fn."
            )
        return self.kernel(q, k, v, q_scale, k_scale, v_scale)


class GQAFwdFP8Fa3ContractPtxAccDirectStoreKernel(GQAFwdFP8Fa3ContractPtxAccKernel):
    """FA3-contract FP8 GQA path with PTX PV and direct global output store."""

    @property
    def default_config(self) -> dict:
        config = super().default_config
        config.update({
            "use_ptx_pv_direct_store": True,
        })
        return config


class GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel(GQAFwdFP8Fa3ContractPtxAccKernel):
    """FA3-contract FP8 GQA path with PTX PV and FA3-style output epilogue store."""

    @property
    def default_config(self) -> dict:
        config = super().default_config
        config.update({
            "use_ptx_pv_fa3_epilogue_store": True,
        })
        return config


class GQAFwdFP8Fa3ContractPtxAccFa3EpilogueReuseVSmemKernel(
        GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel):
    """FA3 epilogue-store experiment that reuses V operand smem as output scratch."""

    @property
    def default_config(self) -> dict:
        config = super().default_config
        config.update({
            "use_ptx_pv_fa3_epilogue_reuse_v_smem": True,
        })
        return config
