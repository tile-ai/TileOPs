import functools
import os
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.online_softmax import (
    make_log2e_scale,
    make_online_softmax_with_mask_guard,
    make_rescale,
)

__all__ = [
    "GQAFwdWsPersistentCausalKernel",
    "GQAFwdWsPersistentKernel",
]

NUM_SMS = int(os.environ.get("V2P_NUM_SMS", "132"))
_ANCHOR_HELPER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_anchor_helper.h"))


@functools.lru_cache(maxsize=32)
def _gqa_fwd_ws_persistent_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    dtype: str = "float16",
) -> Callable:
    assert heads % heads_kv == 0 and dim == 128
    block_m = 128
    block_n = 128
    half_m = block_m // 2
    groups = heads // heads_kv
    scale = make_log2e_scale(dim)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16", "-include", _ANCHOR_HELPER_PATH],
    )
    def func(block_m: int, block_n: int):
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        softmax_1 = make_online_softmax_with_mask_guard(scale, accum_dtype, half_m, block_n)
        softmax_2 = make_online_softmax_with_mask_guard(scale, accum_dtype, half_m, block_n)
        rescale_1 = make_rescale(half_m, dim)
        rescale_2 = make_rescale(half_m, dim)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(kv_shape, dtype),
            v: T.Tensor(kv_shape, dtype),
            output: T.Tensor(q_shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ) -> None:
            with T.Kernel(NUM_SMS, 1, 1, threads=384) as (bx, _by, _bz):
                q_shared_1 = T.alloc_shared([half_m, dim], dtype)
                q_shared_2 = T.alloc_shared([half_m, dim], dtype)
                k_smem_0 = T.alloc_shared([block_n, dim], dtype)
                k_smem_1 = T.alloc_shared([block_n, dim], dtype)
                v_smem_0 = T.alloc_shared([block_n, dim], dtype)
                v_smem_1 = T.alloc_shared([block_n, dim], dtype)

                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_1 = T.alloc_fragment([half_m, block_n], dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_2 = T.alloc_fragment([half_m, block_n], dtype)
                acc_o_2 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_2 = T.alloc_fragment([half_m], accum_dtype)
                smp_2 = T.alloc_fragment([half_m], accum_dtype)
                ss_2 = T.alloc_fragment([half_m], accum_dtype)
                ssum_2 = T.alloc_fragment([half_m], accum_dtype)
                ls_2 = T.alloc_fragment([half_m], accum_dtype)

                anchor_sink = T.alloc_shared([2], "int32")
                k_full = T.alloc_barrier(arrive_count=128)
                k_empty = T.alloc_barrier(arrive_count=256)
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
                    for tile_b, tile_hkv, tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len, block_m), groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        tile_h = tile_hkv * groups + tile_g
                        head_kv = tile_hkv
                        loop_range = T.ceildiv(seq_len, block_n)

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
                                    T.tma_copy(
                                        v[tile_b, (n_idx - 1) * block_n:n_idx * block_n, head_kv, :],
                                        v_smem_0,
                                        barrier=v_full,
                                    )
                                else:
                                    T.tma_copy(
                                        v[tile_b, (n_idx - 1) * block_n:n_idx * block_n, head_kv, :],
                                        v_smem_1,
                                        barrier=v_full,
                                    )
                                T.barrier_arrive(v_full)
                                gi_vp = gi_vp + 1
                            gi_kp = gi_kp + 1

                        T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                        if gi_vp % 2 == 0:
                            T.tma_copy(
                                v[tile_b, (loop_range - 1) * block_n:loop_range * block_n, head_kv, :],
                                v_smem_0,
                                barrier=v_full,
                            )
                        else:
                            T.tma_copy(
                                v[tile_b, (loop_range - 1) * block_n:loop_range * block_n, head_kv, :],
                                v_smem_1,
                                barrier=v_full,
                            )
                        T.barrier_arrive(v_full)
                        gi_vp = gi_vp + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    T.call_extern("handle", "tl::barrier_arrive_named", 1, 256)
                    for tile_b, tile_hkv, tile_m, tile_g in T.Persistent(
                        [batch, heads_kv, T.ceildiv(seq_len, block_m), groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        tile_h = tile_hkv * groups + tile_g
                        row_base = tile_m * block_m
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(q[tile_b, row_base:row_base + half_m, tile_h, :], q_shared_1, barrier=q_full_1)
                        T.barrier_arrive(q_full_1)
                        T.barrier_wait(q_full_1, gi_q1 % 2)
                        gi_q1 = gi_q1 + 1

                        T.clear(acc_o_1)
                        T.clear(ls_1)
                        T.fill(sm_1, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc1 % 2)
                            T.sync_threads(barrier_id=1, arrive_count=256)
                            if n_idx == 0:
                                if gi_kc1 % 2 == 0:
                                    T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                else:
                                    T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                T.call_extern("handle", "tl::barrier_arrive_named", 2, 256)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_s_1, num_regs=64)
                                T.barrier_arrive(k_empty)
                                softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                T.copy(acc_s_1, acc_s_cast_1)
                            else:
                                if gi_kc1 % 2 == 0:
                                    T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                else:
                                    T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                T.barrier_wait(v_full, gi_vc1 % 2)
                                if gi_vc1 % 2 == 0:
                                    T.wgmma_gemm(acc_s_cast_1, v_smem_0, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                else:
                                    T.wgmma_gemm(acc_s_cast_1, v_smem_1, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                T.call_extern("handle", "tl::barrier_arrive_named", 2, 256)
                                T.call_extern("handle", "tl::wait_wgmma_anchor<1>", T.address_of(anchor_sink[0]), gi_kc1)
                                T.warpgroup_fence_operand(acc_s_1, num_regs=64)
                                T.barrier_arrive(k_empty)
                                softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                T.call_extern("handle", "tl::wait_wgmma_anchor<0>", T.address_of(anchor_sink[0]), gi_kc1)
                                T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                T.barrier_arrive(v_empty)
                                rescale_1(acc_o_1, ss_1)
                                T.copy(acc_s_1, acc_s_cast_1)
                                gi_vc1 = gi_vc1 + 1
                            gi_kc1 = gi_kc1 + 1

                        T.barrier_wait(v_full, gi_vc1 % 2)
                        if gi_vc1 % 2 == 0:
                            T.wgmma_gemm(acc_s_cast_1, v_smem_0, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                        else:
                            T.wgmma_gemm(acc_s_cast_1, v_smem_1, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                        T.barrier_arrive(v_empty)
                        gi_vc1 = gi_vc1 + 1
                        for i, j in T.Parallel(half_m, dim):
                            acc_o_1[i, j] /= ls_1[i]
                        T.copy(acc_o_1, q_shared_1)
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=3, arrive_count=128)
                        T.copy(q_shared_1, output[tile_b, row_base:row_base + half_m, tile_h, :])
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
                        row_base = tile_m * block_m
                        loop_range = T.ceildiv(seq_len, block_n)

                        T.tma_copy(
                            q[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                            q_shared_2,
                            barrier=q_full_2,
                        )
                        T.barrier_arrive(q_full_2)
                        T.barrier_wait(q_full_2, gi_q2 % 2)
                        gi_q2 = gi_q2 + 1

                        T.clear(acc_o_2)
                        T.clear(ls_2)
                        T.fill(sm_2, -T.infinity(accum_dtype))

                        for n_idx in T.Pipelined(loop_range, num_stages=0):
                            T.barrier_wait(k_full, gi_kc2 % 2)
                            T.sync_threads(barrier_id=2, arrive_count=256)
                            if n_idx == 0:
                                if gi_kc2 % 2 == 0:
                                    T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                else:
                                    T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                T.call_extern("handle", "tl::barrier_arrive_named", 1, 256)
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc_s_2, num_regs=64)
                                T.barrier_arrive(k_empty)
                                softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                T.copy(acc_s_2, acc_s_cast_2)
                            else:
                                if gi_kc2 % 2 == 0:
                                    T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                else:
                                    T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                T.barrier_wait(v_full, gi_vc2 % 2)
                                if gi_vc2 % 2 == 0:
                                    T.wgmma_gemm(acc_s_cast_2, v_smem_0, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                else:
                                    T.wgmma_gemm(acc_s_cast_2, v_smem_1, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                T.call_extern("handle", "tl::barrier_arrive_named", 1, 256)
                                T.call_extern("handle", "tl::wait_wgmma_anchor<1>", T.address_of(anchor_sink[1]), gi_kc2)
                                T.warpgroup_fence_operand(acc_s_2, num_regs=64)
                                T.barrier_arrive(k_empty)
                                softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                T.call_extern("handle", "tl::wait_wgmma_anchor<0>", T.address_of(anchor_sink[1]), gi_kc2)
                                T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                T.barrier_arrive(v_empty)
                                rescale_2(acc_o_2, ss_2)
                                T.copy(acc_s_2, acc_s_cast_2)
                                gi_vc2 = gi_vc2 + 1
                            gi_kc2 = gi_kc2 + 1

                        T.barrier_wait(v_full, gi_vc2 % 2)
                        if gi_vc2 % 2 == 0:
                            T.wgmma_gemm(acc_s_cast_2, v_smem_0, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                        else:
                            T.wgmma_gemm(acc_s_cast_2, v_smem_1, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                        T.barrier_arrive(v_empty)
                        gi_vc2 = gi_vc2 + 1
                        for i, j in T.Parallel(half_m, dim):
                            acc_o_2[i, j] /= ls_2[i]
                        T.copy(acc_o_2, q_shared_2)
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=4, arrive_count=128)
                        T.copy(q_shared_2, output[tile_b, row_base + half_m:row_base + block_m, tile_h, :])
                        for i in T.Parallel(half_m):
                            ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                        T.copy(ls_2, lse[tile_b, tile_h, row_base + half_m:row_base + block_m])

        return main

    return func


@functools.lru_cache(maxsize=32)
def _gqa_fwd_ws_persistent_causal_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    dtype: str = "float16",
) -> Callable:
    assert heads % heads_kv == 0 and dim == 128
    block_m = 128
    block_n = 128
    half_m = block_m // 2
    groups = heads // heads_kv
    hkv_sections = heads_kv
    scale = make_log2e_scale(dim)
    accum_dtype = "float"
    m_blocks = (seq_len + block_m - 1) // block_m
    if m_blocks % 2 != 0:
        raise ValueError(f"M_blocks={m_blocks} is odd (seq_len={seq_len}, block_m={block_m}).")
    half_m_blocks = m_blocks // 2

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16", "-include", _ANCHOR_HELPER_PATH],
    )
    def func(block_m: int, block_n: int):
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        softmax_1 = make_online_softmax_with_mask_guard(scale, accum_dtype, half_m, block_n)
        softmax_2 = make_online_softmax_with_mask_guard(scale, accum_dtype, half_m, block_n)
        rescale_1 = make_rescale(half_m, dim)
        rescale_2 = make_rescale(half_m, dim)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(kv_shape, dtype),
            v: T.Tensor(kv_shape, dtype),
            output: T.Tensor(q_shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ) -> None:
            with T.Kernel(NUM_SMS, 1, 1, threads=384) as (bx, _by, _bz):
                q_shared_1 = T.alloc_shared([half_m, dim], dtype)
                q_shared_2 = T.alloc_shared([half_m, dim], dtype)
                k_smem_0 = T.alloc_shared([block_n, dim], dtype)
                k_smem_1 = T.alloc_shared([block_n, dim], dtype)
                v_smem_0 = T.alloc_shared([block_n, dim], dtype)
                v_smem_1 = T.alloc_shared([block_n, dim], dtype)

                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_1 = T.alloc_fragment([half_m, block_n], dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)

                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_s_cast_2 = T.alloc_fragment([half_m, block_n], dtype)
                acc_o_2 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_2 = T.alloc_fragment([half_m], accum_dtype)
                smp_2 = T.alloc_fragment([half_m], accum_dtype)
                ss_2 = T.alloc_fragment([half_m], accum_dtype)
                ssum_2 = T.alloc_fragment([half_m], accum_dtype)
                ls_2 = T.alloc_fragment([half_m], accum_dtype)

                anchor_sink = T.alloc_shared([2], "int32")
                k_full = T.alloc_barrier(arrive_count=128)
                k_empty = T.alloc_barrier(arrive_count=256)
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
                    for tile_b, tile_hkv_section, pair_idx, tile_local_hkv, tile_g in T.Persistent(
                        [batch, hkv_sections, half_m_blocks, 1, groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_hkv_section + tile_local_hkv
                        for sub_idx in range(2):
                            tile_m = pair_idx + sub_idx * (m_blocks - 1 - 2 * pair_idx)
                            loop_range = T.ceildiv((tile_m + 1) * block_m, block_n)

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
                                        T.tma_copy(
                                            v[tile_b, (n_idx - 1) * block_n:n_idx * block_n, head_kv, :],
                                            v_smem_0,
                                            barrier=v_full,
                                        )
                                    else:
                                        T.tma_copy(
                                            v[tile_b, (n_idx - 1) * block_n:n_idx * block_n, head_kv, :],
                                            v_smem_1,
                                            barrier=v_full,
                                        )
                                    T.barrier_arrive(v_full)
                                    gi_vp = gi_vp + 1
                                gi_kp = gi_kp + 1

                            T.barrier_wait(v_empty, (gi_vp + 1) % 2)
                            if gi_vp % 2 == 0:
                                T.tma_copy(
                                    v[tile_b, (loop_range - 1) * block_n:loop_range * block_n, head_kv, :],
                                    v_smem_0,
                                    barrier=v_full,
                                )
                            else:
                                T.tma_copy(
                                    v[tile_b, (loop_range - 1) * block_n:loop_range * block_n, head_kv, :],
                                    v_smem_1,
                                    barrier=v_full,
                                )
                            T.barrier_arrive(v_full)
                            gi_vp = gi_vp + 1

                elif tx < 256:
                    T.inc_max_nreg(240)
                    T.call_extern("handle", "tl::barrier_arrive_named", 1, 256)
                    for tile_b, tile_hkv_section, pair_idx, tile_local_hkv, tile_g in T.Persistent(
                        [batch, hkv_sections, half_m_blocks, 1, groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_hkv_section + tile_local_hkv
                        tile_h = head_kv * groups + tile_g
                        for sub_idx in range(2):
                            tile_m = pair_idx + sub_idx * (m_blocks - 1 - 2 * pair_idx)
                            row_base = tile_m * block_m
                            loop_range = T.ceildiv((tile_m + 1) * block_m, block_n)

                            T.tma_copy(q[tile_b, row_base:row_base + half_m, tile_h, :], q_shared_1, barrier=q_full_1)
                            T.barrier_arrive(q_full_1)
                            T.barrier_wait(q_full_1, gi_q1 % 2)
                            gi_q1 = gi_q1 + 1

                            T.clear(acc_o_1)
                            T.clear(ls_1)
                            T.fill(sm_1, -T.infinity(accum_dtype))

                            for n_idx in T.Pipelined(loop_range, num_stages=0):
                                T.barrier_wait(k_full, gi_kc1 % 2)
                                T.sync_threads(barrier_id=1, arrive_count=256)
                                if n_idx == 0:
                                    if gi_kc1 % 2 == 0:
                                        T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                    else:
                                        T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                    T.call_extern("handle", "tl::barrier_arrive_named", 2, 256)
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(acc_s_1, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    if n_idx == loop_range - 1:
                                        for i, j in T.Parallel(half_m, block_n):
                                            acc_s_1[i, j] = T.if_then_else(
                                                row_base + i >= n_idx * block_n + j,
                                                acc_s_1[i, j],
                                                -T.infinity(accum_dtype),
                                            )
                                    softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                    T.copy(acc_s_1, acc_s_cast_1)
                                else:
                                    if gi_kc1 % 2 == 0:
                                        T.wgmma_gemm(q_shared_1, k_smem_0, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                    else:
                                        T.wgmma_gemm(q_shared_1, k_smem_1, acc_s_1, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                    T.barrier_wait(v_full, gi_vc1 % 2)
                                    if gi_vc1 % 2 == 0:
                                        T.wgmma_gemm(acc_s_cast_1, v_smem_0, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(acc_s_cast_1, v_smem_1, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                                    T.call_extern("handle", "tl::barrier_arrive_named", 2, 256)
                                    T.call_extern("handle", "tl::wait_wgmma_anchor<1>", T.address_of(anchor_sink[0]), gi_kc1)
                                    T.warpgroup_fence_operand(acc_s_1, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    if n_idx == loop_range - 1:
                                        for i, j in T.Parallel(half_m, block_n):
                                            acc_s_1[i, j] = T.if_then_else(
                                                row_base + i >= n_idx * block_n + j,
                                                acc_s_1[i, j],
                                                -T.infinity(accum_dtype),
                                            )
                                    softmax_1(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1)
                                    T.call_extern("handle", "tl::wait_wgmma_anchor<0>", T.address_of(anchor_sink[0]), gi_kc1)
                                    T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                    T.barrier_arrive(v_empty)
                                    rescale_1(acc_o_1, ss_1)
                                    T.copy(acc_s_1, acc_s_cast_1)
                                    gi_vc1 = gi_vc1 + 1
                                gi_kc1 = gi_kc1 + 1

                            T.barrier_wait(v_full, gi_vc1 % 2)
                            if gi_vc1 % 2 == 0:
                                T.wgmma_gemm(acc_s_cast_1, v_smem_0, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(acc_s_cast_1, v_smem_1, acc_o_1, policy=T.GemmWarpPolicy.FullRow)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                            T.barrier_arrive(v_empty)
                            gi_vc1 = gi_vc1 + 1
                            for i, j in T.Parallel(half_m, dim):
                                acc_o_1[i, j] /= ls_1[i]
                            T.copy(acc_o_1, q_shared_1)
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=3, arrive_count=128)
                            T.copy(q_shared_1, output[tile_b, row_base:row_base + half_m, tile_h, :])
                            for i in T.Parallel(half_m):
                                ls_1[i] = T.log2(ls_1[i]) + sm_1[i] * scale
                            T.copy(ls_1, lse[tile_b, tile_h, row_base:row_base + half_m])

                else:
                    T.inc_max_nreg(240)
                    for tile_b, tile_hkv_section, pair_idx, tile_local_hkv, tile_g in T.Persistent(
                        [batch, hkv_sections, half_m_blocks, 1, groups],
                        wave_size=NUM_SMS,
                        index=bx,
                        group_size=8,
                    ):
                        head_kv = tile_hkv_section + tile_local_hkv
                        tile_h = head_kv * groups + tile_g
                        for sub_idx in range(2):
                            tile_m = pair_idx + sub_idx * (m_blocks - 1 - 2 * pair_idx)
                            row_base = tile_m * block_m
                            loop_range = T.ceildiv((tile_m + 1) * block_m, block_n)

                            T.tma_copy(
                                q[tile_b, row_base + half_m:row_base + block_m, tile_h, :],
                                q_shared_2,
                                barrier=q_full_2,
                            )
                            T.barrier_arrive(q_full_2)
                            T.barrier_wait(q_full_2, gi_q2 % 2)
                            gi_q2 = gi_q2 + 1

                            T.clear(acc_o_2)
                            T.clear(ls_2)
                            T.fill(sm_2, -T.infinity(accum_dtype))

                            for n_idx in T.Pipelined(loop_range, num_stages=0):
                                T.barrier_wait(k_full, gi_kc2 % 2)
                                T.sync_threads(barrier_id=2, arrive_count=256)
                                if n_idx == 0:
                                    if gi_kc2 % 2 == 0:
                                        T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                    else:
                                        T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                    T.call_extern("handle", "tl::barrier_arrive_named", 1, 256)
                                    T.wait_wgmma(0)
                                    T.warpgroup_fence_operand(acc_s_2, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    if n_idx == loop_range - 1:
                                        for i, j in T.Parallel(half_m, block_n):
                                            acc_s_2[i, j] = T.if_then_else(
                                                row_base + half_m + i >= n_idx * block_n + j,
                                                acc_s_2[i, j],
                                                -T.infinity(accum_dtype),
                                            )
                                    softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                    T.copy(acc_s_2, acc_s_cast_2)
                                else:
                                    if gi_kc2 % 2 == 0:
                                        T.wgmma_gemm(q_shared_2, k_smem_0, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                    else:
                                        T.wgmma_gemm(q_shared_2, k_smem_1, acc_s_2, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                                    T.barrier_wait(v_full, gi_vc2 % 2)
                                    if gi_vc2 % 2 == 0:
                                        T.wgmma_gemm(acc_s_cast_2, v_smem_0, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                    else:
                                        T.wgmma_gemm(acc_s_cast_2, v_smem_1, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                                    T.call_extern("handle", "tl::barrier_arrive_named", 1, 256)
                                    T.call_extern("handle", "tl::wait_wgmma_anchor<1>", T.address_of(anchor_sink[1]), gi_kc2)
                                    T.warpgroup_fence_operand(acc_s_2, num_regs=64)
                                    T.barrier_arrive(k_empty)
                                    if n_idx == loop_range - 1:
                                        for i, j in T.Parallel(half_m, block_n):
                                            acc_s_2[i, j] = T.if_then_else(
                                                row_base + half_m + i >= n_idx * block_n + j,
                                                acc_s_2[i, j],
                                                -T.infinity(accum_dtype),
                                            )
                                    softmax_2(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2)
                                    T.call_extern("handle", "tl::wait_wgmma_anchor<0>", T.address_of(anchor_sink[1]), gi_kc2)
                                    T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                    T.barrier_arrive(v_empty)
                                    rescale_2(acc_o_2, ss_2)
                                    T.copy(acc_s_2, acc_s_cast_2)
                                    gi_vc2 = gi_vc2 + 1
                                gi_kc2 = gi_kc2 + 1

                            T.barrier_wait(v_full, gi_vc2 % 2)
                            if gi_vc2 % 2 == 0:
                                T.wgmma_gemm(acc_s_cast_2, v_smem_0, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                            else:
                                T.wgmma_gemm(acc_s_cast_2, v_smem_1, acc_o_2, policy=T.GemmWarpPolicy.FullRow)
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                            T.barrier_arrive(v_empty)
                            gi_vc2 = gi_vc2 + 1
                            for i, j in T.Parallel(half_m, dim):
                                acc_o_2[i, j] /= ls_2[i]
                            T.copy(acc_o_2, q_shared_2)
                            T.fence_proxy_async()
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.copy(q_shared_2, output[tile_b, row_base + half_m:row_base + block_m, tile_h, :])
                            for i in T.Parallel(half_m):
                                ls_2[i] = T.log2(ls_2[i]) + sm_2[i] * scale
                            T.copy(ls_2, lse[tile_b, tile_h, row_base + half_m:row_base + block_m])

        return main

    return func


class GQAFwdWsPersistentKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if is_causal:
            raise ValueError("GQAFwdWsPersistentKernel only supports non-causal forward.")
        if dtype != torch.float16:
            raise ValueError("GQAFwdWsPersistentKernel currently supports float16 only.")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.kernel = _gqa_fwd_ws_persistent_kernel(batch, heads, heads_kv, seq_len, dim, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_m": 128, "block_n": 128}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kernel(self.config["block_m"], self.config["block_n"])(q, k, v)


class GQAFwdWsPersistentCausalKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if not is_causal:
            raise ValueError("GQAFwdWsPersistentCausalKernel only supports causal forward.")
        if dtype != torch.float16:
            raise ValueError("GQAFwdWsPersistentCausalKernel currently supports float16 only.")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.kernel = _gqa_fwd_ws_persistent_causal_kernel(batch, heads, heads_kv, seq_len, dim, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_m": 128, "block_n": 128}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kernel(self.config["block_m"], self.config["block_n"])(q, k, v)
