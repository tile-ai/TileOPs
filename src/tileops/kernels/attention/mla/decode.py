import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import MlaDecodeCall, MLADecodeFwdInterface
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_shared_memory_optin

__all__ = ["MLADecodeMmaKernel", "MLADecodeWsKernel"]


def _split_combine(batch, heads, num_split, dim, dtype, lse_dtype):
    """Merge the splits' partial outputs, each weighted by its share of the log-sum-exp."""
    accum_dtype = "float"

    @T.macro
    def combine(
        glse: T.Tensor([batch, heads, num_split], lse_dtype),
        Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (hid, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local_split = T.alloc_local([1], accum_dtype)
            lse_logsum_local = T.alloc_local([1], accum_dtype)
            lse_max_local = T.alloc_local([1], accum_dtype)
            scale_local = T.alloc_local([1], accum_dtype)

            T.annotate_layout(
                {
                    lse_logsum_local: T.Fragment(
                        lse_logsum_local.shape, forward_thread_fn=lambda i: i
                    ),
                }
            )

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local[0] = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local[0] = T.max(lse_max_local[0], glse[bz, hid, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split[0] = glse[bz, hid, k]
                lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
            lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, hid, k, i]
                lse_local_split[0] = glse[bz, hid, k]
                scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local[0]
            for i in T.Parallel(dim):
                Output[bz, hid, i] = o_accum_local[i]

    return combine


@functools.lru_cache(maxsize=32)
def _mla_decode_ws_kernel(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype="float16"):
    sm_scale = (1.0 / (dim + pe_dim)) ** 0.5 * LOG2E
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num

    @tilelang.jit(
        out_idx=[6],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=[
            "-O3",
            "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--ptxas-options=-v,--register-usage-level=10",
            "-DNDEBUG",
        ],
    )
    def _mla_decode_ws_func(block_H, block_N, num_split, num_stages, threads=384):
        # A head count block_H does not divide leaves the last block partly filled; its
        # loads past the heads read zero and its stores there are dropped.
        VALID_BLOCK_H = min(block_H, kv_group_num)
        # Two 128-thread consumer warpgroups and one producer warpgroup, which gathers
        # KV 8 threads to a row, 16 rows a pass.
        if threads != 384:
            raise ValueError(f"the warp-specialized schedule runs 384 threads, threads={threads}")
        if block_N % 16 != 0:
            raise ValueError(f"the KV gather copies 16 rows a pass, block_N={block_N}")
        # Each split owns kv_per_split keys and a loop step consumes two tiles. A
        # length that does not fill every step masks the keys past its split's end.
        kv_per_split = tilelang.cdiv(seqlen_kv, num_split)
        ragged = seqlen_kv % (num_split * 2 * block_N) != 0
        may_leave_a_split_empty = (num_split - 1) * kv_per_split >= seqlen_kv

        @T.macro
        def load_row(KV, K_pe, kv_l, kv_r, k_tail, bid, cur_kv_head, r, kv_index, tx):
            with T.attr("default", "async_scope", 1):
                for u in T.serial(dim // 128):
                    for v in T.vectorized(8):
                        kv_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                            bid, kv_index, cur_kv_head, 64 * u + (tx - 256) % 8 * 8 + v
                        ]
                        kv_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                            bid, kv_index, cur_kv_head, dim // 2 + 64 * u + (tx - 256) % 8 * 8 + v
                        ]
            with T.attr("default", "async_scope", 1):
                for v in T.vectorized(8):
                    k_tail[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v] = K_pe[
                        bid, kv_index, cur_kv_head, (tx - 256) % 8 * 8 + v
                    ]

        @T.macro
        def gather_tile(KV, K_pe, kv_l, kv_r, k_tail, bid, cur_kv_head, tile_start, kv_end, tx):
            for r in T.serial(block_N // 16):
                kv_index = tile_start + r * 16 + (tx - 256) // 8
                if ragged:
                    # Zero, not stale: the row's softmax weight is zero, and zero
                    # times a NaN an earlier kernel left in shared memory is NaN.
                    if kv_index < kv_end:
                        load_row(KV, K_pe, kv_l, kv_r, k_tail, bid, cur_kv_head, r, kv_index, tx)
                    else:
                        for u in T.serial(dim // 128):
                            for v in T.vectorized(8):
                                kv_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = 0
                                kv_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = 0
                        for v in T.vectorized(8):
                            k_tail[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v] = 0
                else:
                    load_row(KV, K_pe, kv_l, kv_r, k_tail, bid, cur_kv_head, r, kv_index, tx)

        @T.macro
        def init_scores(acc_s, tile_start, kv_end):
            if ragged:
                for h_i, bi_i in T.Parallel(block_H, block_N):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        tile_start + bi_i < kv_end, 0, -T.infinity(accum_dtype)
                    )
            else:
                T.clear(acc_s)

        @T.macro
        def flash_attn(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(T.ceildiv(heads, VALID_BLOCK_H), batch, threads=threads) as (
                hid,
                bid,
            ):
                Q_shared_l = T.alloc_shared([block_H, dim // 2], dtype)
                Q_shared_r = T.alloc_shared([block_H, dim // 2], dtype)
                Q_tail_shared = T.alloc_shared([block_H, pe_dim], dtype)
                KV_shared_0_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_0_r = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_r = T.alloc_shared([block_N, dim // 2], dtype)
                K_tail_shared_0 = T.alloc_shared([block_N, pe_dim], dtype)
                K_tail_shared_1 = T.alloc_shared([block_N, pe_dim], dtype)
                O_shared_l = Q_shared_l
                O_shared_r = Q_shared_r

                acc_o_l = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                sumexp = T.alloc_fragment([block_H], accum_dtype)
                sum_exp_shared = T.alloc_shared([block_H], accum_dtype)
                sumexp_i = T.alloc_fragment([block_H], accum_dtype)
                alpha_shared = T.alloc_shared([block_H], accum_dtype, scope="shared")
                alpha_local = T.alloc_fragment([block_H], accum_dtype)
                m_i = T.alloc_fragment([block_H], accum_dtype)
                m_i_prev = T.alloc_fragment([block_H], accum_dtype)

                # TODO: Multi buffer
                bar_k_0_ready = T.alloc_barrier(arrive_count=128)
                bar_k_1_ready = T.alloc_barrier(arrive_count=128)
                bar_k_0_free = T.alloc_barrier(arrive_count=256)
                bar_k_1_free = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

                cur_kv_head = hid * VALID_BLOCK_H // kv_group_num
                kv_start = 0
                kv_end = seqlen_kv
                NI = T.ceildiv(seqlen_kv, block_N)

                tx = T.get_thread_binding()

                # Q/Q_pe -> shared copies must stay inside tx < 128 so that the copy
                # and the subsequent T.wgmma_gemm share the same 128-thread bounds.
                # Mixing a 384-thread copy with a 128-thread WGMMA on the same shared
                # buffer causes TileLang 0.1.9 layout inference to fail with
                # "no available layout found".
                if tx < 128:
                    T.set_max_nreg(240, 1)
                    T.copy(
                        Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, 0 : dim // 2],
                        Q_shared_l,
                    )
                    T.copy(
                        Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, dim // 2 : dim],
                        Q_shared_r,
                    )
                    T.copy(
                        Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_tail_shared
                    )
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
                    T.fill(acc_o_l, 0)

                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                        init_scores(acc_s, kv_start + (i_i * 2) * block_N, kv_end)
                        T.wgmma_gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True)

                        T.wait_wgmma(0)

                        if i_i != 0:
                            T.barrier_arrive(bar_sScale_and_sS_free)
                            T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_0_free[0])

                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                        init_scores(acc_s, kv_start + (i_i * 2 + 1) * block_N, kv_end)
                        T.wgmma_gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True)

                        T.wait_wgmma(0)

                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_1_free[0])

                    for h_i in T.Parallel(block_H):
                        sum_exp_shared[h_i] = sumexp[h_i]
                    for h_i, d_i in T.Parallel(block_H, dim // 2):
                        acc_o_l[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(acc_o_l, O_shared_l)
                    T.copy(
                        O_shared_l,
                        Output[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, 0 : dim // 2],
                    )

                elif tx >= 128 and tx < 256:
                    T.set_max_nreg(168, 1)
                    T.fill(acc_o_r, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                        T.barrier_arrive(bar_k_0_free[0])
                        T.barrier_arrive(bar_sScale_and_sS_free)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                        T.barrier_arrive(bar_k_1_free[0])
                        if i_i != T.ceildiv(NI, 2) - 1:
                            T.barrier_arrive(bar_sScale_and_sS_free)

                    for h_i, d_i in T.Parallel(block_H, dim // 2):
                        acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                    T.copy(acc_o_r, O_shared_r)
                    T.copy(
                        O_shared_r,
                        Output[
                            bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, dim // 2 : dim
                        ],
                    )

                elif tx >= 256:
                    T.set_max_nreg(80, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                        gather_tile(
                            KV,
                            K_pe,
                            KV_shared_0_l,
                            KV_shared_0_r,
                            K_tail_shared_0,
                            bid,
                            cur_kv_head,
                            kv_start + (i_i * 2) * block_N,
                            kv_end,
                            tx,
                        )
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])

                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                        gather_tile(
                            KV,
                            K_pe,
                            KV_shared_1_l,
                            KV_shared_1_r,
                            K_tail_shared_1,
                            bid,
                            cur_kv_head,
                            kv_start + (i_i * 2 + 1) * block_N,
                            kv_end,
                            tx,
                        )
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])

        @T.macro
        def flash_attn_split(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        ):
            with T.Kernel(batch, T.ceildiv(heads, VALID_BLOCK_H), num_split, threads=threads) as (
                bid,
                hid,
                bz,
            ):
                Q_shared_l = T.alloc_shared([block_H, dim // 2], dtype)
                Q_shared_r = T.alloc_shared([block_H, dim // 2], dtype)
                Q_tail_shared = T.alloc_shared([block_H, pe_dim], dtype)
                KV_shared_0_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_0_r = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_r = T.alloc_shared([block_N, dim // 2], dtype)
                K_tail_shared_0 = T.alloc_shared([block_N, pe_dim], dtype)
                K_tail_shared_1 = T.alloc_shared([block_N, pe_dim], dtype)
                O_shared_l = Q_shared_l
                O_shared_r = Q_shared_r

                acc_o_l = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                sumexp = T.alloc_fragment([block_H], accum_dtype)
                sum_exp_shared = T.alloc_shared([block_H], accum_dtype)
                sumexp_i = T.alloc_fragment([block_H], accum_dtype)
                alpha_shared = T.alloc_shared([block_H], accum_dtype, scope="shared")
                alpha_local = T.alloc_fragment([block_H], accum_dtype)
                m_i = T.alloc_fragment([block_H], accum_dtype)
                m_i_prev = T.alloc_fragment([block_H], accum_dtype)

                # TODO: Multi buffer
                bar_k_0_ready = T.alloc_barrier(arrive_count=128)
                bar_k_1_ready = T.alloc_barrier(arrive_count=128)
                bar_k_0_free = T.alloc_barrier(arrive_count=256)
                bar_k_1_free = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

                cur_kv_head = hid * VALID_BLOCK_H // kv_group_num
                kv_start = kv_per_split * bz
                kv_end = T.min(kv_start + kv_per_split, seqlen_kv)
                NI = T.ceildiv(kv_per_split, block_N)

                tx = T.get_thread_binding()

                # Q/Q_pe -> shared copies must stay inside tx < 128 so that the copy
                # and the subsequent T.wgmma_gemm share the same 128-thread bounds.
                # Mixing a 384-thread copy with a 128-thread WGMMA on the same shared
                # buffer causes TileLang 0.1.9 layout inference to fail with
                # "no available layout found".
                if tx < 128:
                    T.set_max_nreg(240, 1)
                    T.copy(
                        Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, 0 : dim // 2],
                        Q_shared_l,
                    )
                    T.copy(
                        Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, dim // 2 : dim],
                        Q_shared_r,
                    )
                    T.copy(
                        Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_tail_shared
                    )
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
                    T.fill(acc_o_l, 0)

                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                        init_scores(acc_s, kv_start + (i_i * 2) * block_N, kv_end)
                        T.wgmma_gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True)

                        T.wait_wgmma(0)

                        if i_i != 0:
                            T.barrier_arrive(bar_sScale_and_sS_free)
                            T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_0_free[0])

                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                        init_scores(acc_s, kv_start + (i_i * 2 + 1) * block_N, kv_end)
                        T.wgmma_gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True)

                        T.wait_wgmma(0)

                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_1_free[0])

                    for h_i in T.Parallel(block_H):
                        sum_exp_shared[h_i] = sumexp[h_i]
                    if may_leave_a_split_empty:
                        # An empty split has no key to divide by; its zero weight in the combine
                        # needs a finite partial.
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] = T.if_then_else(
                                sumexp[h_i] > 0, acc_o_l[h_i, d_i] / sumexp[h_i], 0
                            )
                    else:
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(acc_o_l, O_shared_l)
                    T.copy(
                        O_shared_l,
                        Output_partial[
                            bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz, 0 : dim // 2
                        ],
                    )
                    T.copy(sumexp, glse[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz])

                elif tx >= 128 and tx < 256:
                    T.set_max_nreg(168, 1)
                    T.fill(acc_o_r, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                        T.barrier_arrive(bar_k_0_free[0])
                        T.barrier_arrive(bar_sScale_and_sS_free)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                        T.barrier_arrive(bar_k_1_free[0])
                        if i_i != T.ceildiv(NI, 2) - 1:
                            T.barrier_arrive(bar_sScale_and_sS_free)

                    if may_leave_a_split_empty:
                        # An empty split has no key to divide by; its zero weight in the combine
                        # needs a finite partial.
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] = T.if_then_else(
                                sum_exp_shared[h_i] > 0, acc_o_r[h_i, d_i] / sum_exp_shared[h_i], 0
                            )
                    else:
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                    T.copy(acc_o_r, O_shared_r)
                    T.copy(
                        O_shared_r,
                        Output_partial[
                            bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz, dim // 2 : dim
                        ],
                    )

                elif tx >= 256:
                    T.set_max_nreg(80, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                        gather_tile(
                            KV,
                            K_pe,
                            KV_shared_0_l,
                            KV_shared_0_r,
                            K_tail_shared_0,
                            bid,
                            cur_kv_head,
                            kv_start + (i_i * 2) * block_N,
                            kv_end,
                            tx,
                        )
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])

                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                        gather_tile(
                            KV,
                            K_pe,
                            KV_shared_1_l,
                            KV_shared_1_r,
                            K_tail_shared_1,
                            bid,
                            cur_kv_head,
                            kv_start + (i_i * 2 + 1) * block_N,
                            kv_end,
                            tx,
                        )
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])

        combine = _split_combine(batch, heads, num_split, dim, dtype, dtype)

        @T.prim_func
        def main_split(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            flash_attn_split(Q, Q_pe, KV, K_pe, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def main_no_split(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            flash_attn(Q, Q_pe, KV, K_pe, Output)

        if num_split > 1:
            return main_split
        else:
            return main_no_split

    return _mla_decode_ws_func


@functools.lru_cache(maxsize=32)
def _mla_decode_mma_kernel(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype="float16"):
    sm_scale = (1.0 / (dim + pe_dim)) ** 0.5 * LOG2E
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[6],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _mla_decode_mma_func(block_H, block_N, num_split, num_stages, threads=128):
        # A length that does not fill every tile masks the keys past its split's end; loads
        # past the cache read zero.
        kv_per_split = tilelang.cdiv(seqlen_kv, num_split)
        ragged = seqlen_kv % (num_split * block_N) != 0
        may_leave_a_split_empty = (num_split - 1) * kv_per_split >= seqlen_kv

        @T.macro
        def flash_attn(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            # Head blocks vary fastest, so the blocks reading one request's cache run together.
            with T.Kernel(T.ceildiv(heads, block_H), batch, num_split, threads=threads) as (
                hid,
                bid,
                bz,
            ):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
                KV_shared = T.alloc_shared([block_N, dim], dtype)
                K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                O_shared = Q_shared

                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                sumexp = T.alloc_fragment([block_H], accum_dtype)
                sumexp_i = T.alloc_fragment([block_H], accum_dtype)
                alpha = T.alloc_fragment([block_H], accum_dtype)
                m_i = T.alloc_fragment([block_H], accum_dtype)
                m_i_prev = T.alloc_fragment([block_H], accum_dtype)

                kv_start = kv_per_split * bz
                kv_end = T.min(kv_start + kv_per_split, seqlen_kv)

                T.copy(Q[bid, hid * block_H : (hid + 1) * block_H, :], Q_shared)
                T.copy(Q_pe[bid, hid * block_H : (hid + 1) * block_H, :], Q_pe_shared)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
                T.fill(acc_o, 0)

                for k in T.Pipelined(T.ceildiv(kv_per_split, block_N), num_stages=num_stages):
                    tile_start = kv_start + k * block_N
                    T.copy(KV[bid, tile_start : tile_start + block_N, 0, :], KV_shared)
                    T.copy(K_pe[bid, tile_start : tile_start + block_N, 0, :], K_pe_shared)
                    if ragged:
                        for h_i, n_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, n_i] = T.if_then_else(
                                tile_start + n_i < kv_end, 0, -T.infinity(accum_dtype)
                            )
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.gemm(
                        Q_pe_shared,
                        K_pe_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(block_H):
                        alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, n_i in T.Parallel(block_H, block_N):
                        acc_s[h_i, n_i] = T.exp2(acc_s[h_i, n_i] * sm_scale - m_i[h_i] * sm_scale)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(block_H, dim):
                        acc_o[h_i, d_i] *= alpha[h_i]

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

                if num_split == 1:
                    for h_i, d_i in T.Parallel(block_H, dim):
                        acc_o[h_i, d_i] /= sumexp[h_i]
                    T.copy(acc_o, O_shared)
                    T.copy(O_shared, Output[bid, hid * block_H : (hid + 1) * block_H, :])
                else:
                    if may_leave_a_split_empty:
                        # An empty split has no key to divide by; its zero weight in the combine
                        # needs a finite partial.
                        for h_i, d_i in T.Parallel(block_H, dim):
                            acc_o[h_i, d_i] = T.if_then_else(
                                sumexp[h_i] > 0, acc_o[h_i, d_i] / sumexp[h_i], 0
                            )
                    else:
                        for h_i, d_i in T.Parallel(block_H, dim):
                            acc_o[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(sumexp, glse[bid, hid * block_H : (hid + 1) * block_H, bz])
                    T.copy(acc_o, O_shared)
                    T.copy(
                        O_shared, Output_partial[bid, hid * block_H : (hid + 1) * block_H, bz, :]
                    )

        # The split log-sum-exps stay in float32; rounded to the operand type they would
        # shift each split's weight in the combine.
        combine = _split_combine(batch, heads, num_split, dim, dtype, accum_dtype)

        @T.prim_func
        def main(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            flash_attn(Q, Q_pe, KV, K_pe, glse, Output_partial, Output)
            if num_split > 1:
                combine(glse, Output_partial, Output)

        return main

    return _mla_decode_mma_func


class MLADecodeWsKernel(Kernel, MLADecodeFwdInterface):
    supported_archs: list[int] = [90]
    # Where both run, a caller's replacement of this key wins over the MMA kernel.
    preferred_over = frozenset({"mla_decode_mma_kernel"})
    _build = staticmethod(_mla_decode_ws_kernel)

    @classmethod
    def applies(cls, call: MlaDecodeCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: MlaDecodeCall) -> Optional[str]:
        """Why *call* is outside the shapes the warp-specialized schedule serves."""
        if call.heads_kv != 1:
            return f"serves one KV head, got {call.heads_kv}"
        if call.heads < 64:
            return f"query heads fill at least one 64-row WGMMA tile, got {call.heads}"
        if call.dim % 128 != 0:
            return f"the KV gather walks dim in 128-column steps, got {call.dim}"
        if call.pe_dim != 64:
            return f"the KV tail gather copies exactly 64 columns, got {call.pe_dim}"
        return None

    @classmethod
    def entry_for(cls, call: MlaDecodeCall) -> Entry:
        return call, lambda: cls(
            call.batch,
            call.heads,
            call.heads_kv,
            call.seqlen_kv,
            call.dim,
            call.pe_dim,
            call.dtype,
            device_index=call.device.index if call.device is not None else None,
        )

    def __init__(
        self,
        batch,
        heads,
        kv_head_num,
        seqlen_kv,
        dim,
        pe_dim,
        dtype,
        config: Optional[dict] = None,
        tune=False,
        *,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.kv_head_num = kv_head_num
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim
        self.dtype = dtype

        self.kernel = self._build(
            self.batch,
            self.heads,
            self.kv_head_num,
            self.seqlen_kv,
            self.dim,
            self.pe_dim,
            self.dtype_str,
        )

        self.init_config(config, tune)

    @property
    def lse_dtype(self) -> torch.dtype:
        return self.dtype

    @property
    def default_config(self) -> dict:
        return {
            "block_H": min(64, self.heads // self.kv_head_num),
            "block_N": 64,
            "num_split": 2,
            "num_stages": 1,
            "threads": 384,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_H = [64, 128]
        block_N = [64, 128]
        num_split = [1, 2, 4]
        num_stages = [1, 2, 3]
        threads = [384]
        _configs = list(itertools.product(block_H, block_N, num_split, num_stages, threads))

        configs = [
            {
                "block_H": c[0],
                "block_N": c[1],
                "num_split": c[2],
                "num_stages": c[3],
                "threads": c[4],
            }
            for c in _configs
        ]
        return configs

    def forward(self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor, k_pe: torch.Tensor):
        if self.seqlen_kv == 0:
            # No keys: the softmax normalizer is zero, so the program would divide 0 by 0.
            # Attention over no keys is the empty sum, as torch's reference returns.
            return torch.zeros(
                (self.batch, self.heads, self.dim), dtype=self.dtype, device=q.device
            )
        glse = torch.empty(
            (self.batch, self.heads, self.config["num_split"]),
            dtype=self.lse_dtype,
            device=q.device,
        )
        Output_partial = torch.empty(
            (self.batch, self.heads, self.config["num_split"], self.dim),
            dtype=self.dtype,
            device=q.device,
        )
        return self.kernel(
            self.config["block_H"],
            self.config["block_N"],
            self.config["num_split"],
            self.config["num_stages"],
            self.config["threads"],
        )(q, q_pe, k, k_pe, glse, Output_partial)


class MLADecodeMmaKernel(MLADecodeWsKernel):
    """The same decode on MMA, for GPUs without WGMMA."""

    supported_archs: list[int] = [80, 86, 89]
    preferred_over = frozenset()
    _build = staticmethod(_mla_decode_mma_kernel)
    # Head rows of the default block, widest first, over a 32-key tile and four warps.
    _BLOCK_HS = (32, 16)
    _BLOCK_N = 32
    _THREADS = 128
    _NUM_SPLIT = 2

    @classmethod
    def refusal(cls, call: MlaDecodeCall) -> Optional[str]:
        """Why *call* is outside the shapes this schedule serves. An empty cache builds no program;
        otherwise shared memory is read at the narrowest head block."""
        if call.heads_kv != 1:
            return f"serves one KV head, got {call.heads_kv}"
        if call.seqlen_kv == 0:
            return None
        if call.pe_dim % 16 != 0:
            return f"the score GEMM steps pe_dim by 16, got {call.pe_dim}"
        # The output GEMM splits dim over the four warps, 8 columns at a time; TileLang lays a
        # warp's share out only up to 32 columns, or in multiples of 32.
        if call.dim % 32 != 0 or (call.dim > 128 and call.dim % 128 != 0):
            return f"dim must be a multiple of 32 up to 128, or of 128 past it; got {call.dim}"
        need = cls._shared_bytes(
            cls._BLOCK_HS[-1], call.dim, call.pe_dim, call.dtype.itemsize, call.seqlen_kv
        )
        if call.smem_budget and need > call.smem_budget:
            return (
                f"needs {need} bytes of shared memory per block at dim {call.dim}, "
                f"pe_dim {call.pe_dim} in {call.dtype}; the device gives {call.smem_budget}"
            )
        return None

    @property
    def lse_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def default_config(self) -> dict:
        return self._default_config_for(
            get_shared_memory_optin(self.device_index),
            self.dim,
            self.pe_dim,
            self.dtype.itemsize,
            self.seqlen_kv,
        )

    @classmethod
    def _default_config_for(
        cls, budget: int, dim: int, pe_dim: int, itemsize: int, seqlen_kv: int
    ) -> dict:
        """The config this kernel builds at *budget* bytes of shared memory per block."""
        block_h = next(
            (
                h
                for h in cls._BLOCK_HS
                if cls._shared_bytes(h, dim, pe_dim, itemsize, seqlen_kv) <= budget
            ),
            cls._BLOCK_HS[-1],
        )
        return {
            "block_H": block_h,
            "block_N": cls._BLOCK_N,
            "num_split": cls._NUM_SPLIT,
            "num_stages": 1,
            "threads": cls._THREADS,
        }

    @classmethod
    def _shared_bytes(
        cls, block_h: int, dim: int, pe_dim: int, itemsize: int, seqlen_kv: int
    ) -> int:
        """Shared memory of the default program: the query and key rows with their rope parts and
        the scores, plus the two reductions' workspaces of 4 bytes a thread, which TileLang folds
        into freed space when a split runs one key tile (both) or two (one)."""
        n = cls._BLOCK_N
        tiles = tilelang.cdiv(tilelang.cdiv(seqlen_kv, cls._NUM_SPLIT), n)
        workspaces = max(min(tiles, 3) - 1, 0)
        return (
            (block_h + n) * (dim + pe_dim) * itemsize
            + block_h * n * itemsize
            + workspaces * 4 * cls._THREADS
        )

    @property
    def autotune_configs(self) -> list[dict]:
        # Four warps: with eight, or a 16-key tile, the score and output GEMMs split the
        # head rows differently and the row statistics have no common layout.
        return [
            {"block_H": h, "block_N": n, "num_split": s, "num_stages": st, "threads": 128}
            for h, n, s, st in itertools.product([16, 32], [32, 64], [1, 2, 4], [1, 2])
        ]
