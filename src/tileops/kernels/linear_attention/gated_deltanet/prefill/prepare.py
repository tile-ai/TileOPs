# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under the MIT License.
# Adapted and modified for TileOps GatedDeltaNet prefill integration.
"""Gated DeltaNet private recurrence preparation and state-correction stages."""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import LOG2E

from .common import _gemm_v1, prepare_chunk_offsets


@functools.lru_cache(maxsize=32)
def _prefill_chunk_local_cumsum_bthd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dtype: str,
):
    num_chunks = seq_len // chunk_size

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func():
        @T.prim_func
        def chunk_cumsum_bthd_kernel(
            g: T.Tensor([batch, seq_len, head], dtype),
            out: T.Tensor([batch, seq_len, head], dtype),
        ):
            with T.Kernel(batch, num_chunks, threads=128) as (bid, cid):
                base = cid * chunk_size
                acc_s = T.alloc_shared([head], "float32")

                for hid in T.Parallel(head):
                    acc_s[hid] = T.float32(0.0)
                for i in T.Serial(chunk_size):
                    for hid in T.Parallel(head):
                        acc_s[hid] = acc_s[hid] + T.cast(g[bid, base + i, hid], "float32")
                        out[bid, base + i, hid] = T.cast(acc_s[hid], dtype)

        return chunk_cumsum_bthd_kernel

    return _func()


@functools.lru_cache(maxsize=32)
def _prefill_blocksolve_A_bthd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dtype: str,
    use_gate: bool = True,
):
    if chunk_size != 64 or dim_k not in (64, 128):
        raise ValueError("TileLang blocksolve-A currently expects chunk64 and K in {64, 128}")

    block_t = 64
    block_c = 16
    block_k = 64
    accum_dtype = "float32"
    solve_dtype = dtype

    @tilelang.jit(
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=32):
        @T.prim_func
        def prefill_blocksolve_A_bthd_tl(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            g: T.Tensor([batch, seq_len, head], dtype),
            beta: T.Tensor([batch, seq_len, head], dtype),
            A: T.Tensor([batch, seq_len, head, chunk_size], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_t, threads=threads) as (
                bid,
                hid,
                cid,
            ):
                base = cid * block_t
                k0 = T.alloc_shared([block_c, block_k], dtype)
                k1 = T.alloc_shared([block_c, block_k], dtype)
                k2 = T.alloc_shared([block_c, block_k], dtype)
                k3 = T.alloc_shared([block_c, block_k], dtype)
                g_s = T.alloc_shared([block_t], dtype)
                beta_s = T.alloc_shared([block_t], dtype)
                gate0_s = T.alloc_shared([block_t], dtype)
                gate1_s = T.alloc_shared([block_t], dtype)
                gate2_s = T.alloc_shared([block_t], dtype)
                gate3_s = T.alloc_shared([block_t], dtype)
                beta_f_s = T.alloc_shared([block_t], dtype)
                a_s = T.alloc_shared([10, block_c, block_c], solve_dtype)
                i_s = T.alloc_shared([4, block_c, block_c], solve_dtype)
                work_s = T.alloc_shared([1, block_c, block_c], solve_dtype)

                G00 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G10 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G11 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G20 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G21 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G22 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G30 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G31 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G32 = T.alloc_fragment([block_c, block_c], accum_dtype)
                G33 = T.alloc_fragment([block_c, block_c], accum_dtype)
                tmp = T.alloc_fragment([block_c, block_c], accum_dtype)

                T.annotate_layout(
                    {
                        k0: tilelang.layout.make_swizzled_layout(k0),
                        k1: tilelang.layout.make_swizzled_layout(k1),
                        k2: tilelang.layout.make_swizzled_layout(k2),
                        k3: tilelang.layout.make_swizzled_layout(k3),
                        a_s: tilelang.layout.make_swizzled_layout(a_s),
                        i_s: tilelang.layout.make_swizzled_layout(i_s),
                        work_s: tilelang.layout.make_swizzled_layout(work_s),
                    }
                )

                if use_gate:
                    T.copy(g[bid, base : base + block_t, hid], g_s, disable_tma=True)
                T.copy(
                    beta[bid, base : base + block_t, hid],
                    beta_s,
                    disable_tma=True,
                )

                T.clear(G00)
                T.clear(G10)
                T.clear(G11)
                T.clear(G20)
                T.clear(G21)
                T.clear(G22)
                T.clear(G30)
                T.clear(G31)
                T.clear(G32)
                T.clear(G33)

                for kt in T.Serial(dim_k // block_k):
                    koff = kt * block_k
                    T.async_copy(
                        k[bid, base : base + block_c, hid, koff : koff + block_k],
                        k0,
                    )
                    T.async_copy(
                        k[
                            bid,
                            base + block_c : base + 2 * block_c,
                            hid,
                            koff : koff + block_k,
                        ],
                        k1,
                    )
                    T.async_copy(
                        k[
                            bid,
                            base + 2 * block_c : base + 3 * block_c,
                            hid,
                            koff : koff + block_k,
                        ],
                        k2,
                    )
                    T.async_copy(
                        k[
                            bid,
                            base + 3 * block_c : base + 4 * block_c,
                            hid,
                            koff : koff + block_k,
                        ],
                        k3,
                    )
                    T.ptx_wait_group(0)
                    T.sync_threads()

                    T.gemm(k0, k0, G00, transpose_B=True)
                    T.gemm(k1, k0, G10, transpose_B=True)
                    T.gemm(k1, k1, G11, transpose_B=True)
                    T.gemm(k2, k0, G20, transpose_B=True)
                    T.gemm(k2, k1, G21, transpose_B=True)
                    T.gemm(k2, k2, G22, transpose_B=True)
                    T.gemm(k3, k0, G30, transpose_B=True)
                    T.gemm(k3, k1, G31, transpose_B=True)
                    T.gemm(k3, k2, G32, transpose_B=True)
                    T.gemm(k3, k3, G33, transpose_B=True)

                for t in T.Parallel(block_t):
                    g_val = T.cast(g_s[t], accum_dtype) if use_gate else T.float32(0.0)
                    gate0_s[t] = T.exp2(
                        (g_val - (T.cast(g_s[0], accum_dtype) if use_gate else T.float32(0.0)))
                        * LOG2E
                    )
                    gate1_s[t] = T.exp2(
                        (
                            g_val
                            - (T.cast(g_s[block_c], accum_dtype) if use_gate else T.float32(0.0))
                        )
                        * LOG2E
                    )
                    gate2_s[t] = T.exp2(
                        (
                            g_val
                            - (
                                T.cast(g_s[2 * block_c], accum_dtype)
                                if use_gate
                                else T.float32(0.0)
                            )
                        )
                        * LOG2E
                    )
                    gate3_s[t] = T.exp2(
                        (
                            g_val
                            - (
                                T.cast(g_s[3 * block_c], accum_dtype)
                                if use_gate
                                else T.float32(0.0)
                            )
                        )
                        * LOG2E
                    )
                    beta_f_s[t] = beta_s[t]
                T.sync_threads()

                for i, j in T.Parallel(block_c, block_c):
                    s00 = (
                        T.cast(beta_f_s[i], accum_dtype)
                        * T.cast(gate0_s[i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s11 = (
                        T.cast(beta_f_s[block_c + i], accum_dtype)
                        * T.cast(gate1_s[block_c + i], accum_dtype)
                        / T.cast(gate1_s[block_c + j], accum_dtype)
                    )
                    s22 = (
                        T.cast(beta_f_s[2 * block_c + i], accum_dtype)
                        * T.cast(gate2_s[2 * block_c + i], accum_dtype)
                        / T.cast(gate2_s[2 * block_c + j], accum_dtype)
                    )
                    s33 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate3_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate3_s[3 * block_c + j], accum_dtype)
                    )
                    s10 = (
                        T.cast(beta_f_s[block_c + i], accum_dtype)
                        * T.cast(gate0_s[block_c + i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s20 = (
                        T.cast(beta_f_s[2 * block_c + i], accum_dtype)
                        * T.cast(gate0_s[2 * block_c + i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s21 = (
                        T.cast(beta_f_s[2 * block_c + i], accum_dtype)
                        * T.cast(gate1_s[2 * block_c + i], accum_dtype)
                        / T.cast(gate1_s[block_c + j], accum_dtype)
                    )
                    s30 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate0_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate0_s[j], accum_dtype)
                    )
                    s31 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate1_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate1_s[block_c + j], accum_dtype)
                    )
                    s32 = (
                        T.cast(beta_f_s[3 * block_c + i], accum_dtype)
                        * T.cast(gate2_s[3 * block_c + i], accum_dtype)
                        / T.cast(gate2_s[2 * block_c + j], accum_dtype)
                    )
                    a_s[0, i, j] = T.if_then_else(
                        i > j,
                        -G00[i, j] * s00,
                        T.float32(0.0),
                    )
                    a_s[2, i, j] = T.if_then_else(
                        i > j,
                        -G11[i, j] * s11,
                        T.float32(0.0),
                    )
                    a_s[5, i, j] = T.if_then_else(
                        i > j,
                        -G22[i, j] * s22,
                        T.float32(0.0),
                    )
                    a_s[9, i, j] = T.if_then_else(
                        i > j,
                        -G33[i, j] * s33,
                        T.float32(0.0),
                    )
                    a_s[1, i, j] = G10[i, j] * s10
                    a_s[3, i, j] = G20[i, j] * s20
                    a_s[4, i, j] = G21[i, j] * s21
                    a_s[6, i, j] = G30[i, j] * s30
                    a_s[7, i, j] = G31[i, j] * s31
                    a_s[8, i, j] = G32[i, j] * s32
                    i_s[0, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                    i_s[1, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                    i_s[2, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                    i_s[3, i, j] = T.if_then_else(i == j, T.float32(1.0), T.float32(0.0))
                T.sync_threads()

                for _r in T.Serial(1):
                    T.clear(tmp)
                    T.gemm(a_s[0, :, :], i_s[0, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[0, i, j] = i_s[0, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[0, :, :], a_s[0, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[0, i, j] = tmp[i, j]

                    T.clear(tmp)
                    T.gemm(a_s[2, :, :], i_s[1, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[1, i, j] = i_s[1, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[2, :, :], a_s[2, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[2, i, j] = tmp[i, j]

                    T.clear(tmp)
                    T.gemm(a_s[5, :, :], i_s[2, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[2, i, j] = i_s[2, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[5, :, :], a_s[5, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[5, i, j] = tmp[i, j]

                    T.clear(tmp)
                    T.gemm(a_s[9, :, :], i_s[3, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        i_s[3, i, j] = i_s[3, i, j] + tmp[i, j]
                    T.clear(tmp)
                    T.gemm(a_s[9, :, :], a_s[9, :, :], tmp)
                    for i, j in T.Parallel(block_c, block_c):
                        a_s[9, i, j] = tmp[i, j]
                T.sync_threads()

                T.clear(tmp)
                T.gemm(i_s[1, :, :], a_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(work_s[0, :, :], i_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[1, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(i_s[2, :, :], a_s[4, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(work_s[0, :, :], i_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[4, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(a_s[3, :, :], i_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[4, :, :], a_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(i_s[2, :, :], work_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[3, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(a_s[6, :, :], i_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[7, :, :], a_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[8, :, :], a_s[3, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(i_s[3, :, :], work_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[6, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(a_s[7, :, :], i_s[1, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.clear(tmp)
                T.gemm(a_s[8, :, :], a_s[4, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = work_s[0, i, j] + tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(i_s[3, :, :], work_s[0, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[7, i, j] = -tmp[i, j]

                T.clear(tmp)
                T.gemm(i_s[3, :, :], a_s[8, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    work_s[0, i, j] = tmp[i, j]
                T.sync_threads()
                T.clear(tmp)
                T.gemm(work_s[0, :, :], i_s[2, :, :], tmp)
                for i, j in T.Parallel(block_c, block_c):
                    a_s[8, i, j] = -tmp[i, j]
                T.sync_threads()

                for i, j in T.Parallel(block_c, block_c):
                    A[bid, base + i, hid, j] = T.cast(i_s[0, i, j], dtype)
                    A[bid, base + i, hid, block_c + j] = T.cast(0, dtype)
                    A[bid, base + i, hid, 2 * block_c + j] = T.cast(0, dtype)
                    A[bid, base + i, hid, 3 * block_c + j] = T.cast(0, dtype)
                    A[bid, base + block_c + i, hid, j] = T.cast(a_s[1, i, j], dtype)
                    A[bid, base + block_c + i, hid, block_c + j] = T.cast(i_s[1, i, j], dtype)
                    A[bid, base + block_c + i, hid, 2 * block_c + j] = T.cast(0, dtype)
                    A[bid, base + block_c + i, hid, 3 * block_c + j] = T.cast(0, dtype)
                    A[bid, base + 2 * block_c + i, hid, j] = T.cast(a_s[3, i, j], dtype)
                    A[bid, base + 2 * block_c + i, hid, block_c + j] = T.cast(a_s[4, i, j], dtype)
                    A[bid, base + 2 * block_c + i, hid, 2 * block_c + j] = T.cast(
                        i_s[2, i, j], dtype
                    )
                    A[bid, base + 2 * block_c + i, hid, 3 * block_c + j] = T.cast(0, dtype)
                    A[bid, base + 3 * block_c + i, hid, j] = T.cast(a_s[6, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, block_c + j] = T.cast(a_s[7, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, 2 * block_c + j] = T.cast(
                        a_s[8, i, j], dtype
                    )
                    A[bid, base + 3 * block_c + i, hid, 3 * block_c + j] = T.cast(
                        i_s[3, i, j], dtype
                    )

        return prefill_blocksolve_A_bthd_tl

    return _func(32)


def _prefill_blocksolve_A_bthd(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    use_gate: bool = True,
) -> torch.Tensor:
    batch, seq_len, head, dim_k = k.shape
    A = torch.empty(batch, seq_len, head, chunk_size, dtype=k.dtype, device=k.device)
    kernel = _prefill_blocksolve_A_bthd_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        str(k.dtype).split(".")[-1],
        use_gate,
    )
    kernel(k, g, beta, A)
    return A


@functools.lru_cache(maxsize=32)
@tilelang.jit()
def _build_warmup_chunks_kernel(
    num_heads,
    chunk_size,
    threshold,
    accum_dtype,
    g_dtype,
    mask_dtype,
    seqlen_dtype,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_threads = tilelang.cdiv(num_heads, 32) * 32

    @T.prim_func
    def warmup_chunks_kernel(
        g: T.Tensor([1, num_tokens, num_heads], dtype=g_dtype),
        ht_mask: T.Tensor([batch_size], dtype=mask_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        num_warmup_chunks: T.Tensor([batch_size, num_heads], dtype=seqlen_dtype),
        fallback_mask: T.Tensor([batch_size, num_heads], dtype=mask_dtype),
    ):
        with T.Kernel(batch_size, threads=num_threads) as (bb,):
            if ht_mask[bb]:
                for i_h in T.Parallel(num_heads):
                    num_warmup_chunks[bb, i_h] = 0
            else:
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")
                num_iters = T.alloc_var("int32")
                seq_start_idx = cu_seqlens[bb]
                seq_end_idx = cu_seqlens[bb + 1]
                num_iters = (seq_end_idx - seq_start_idx) // chunk_size

                g_fragment = T.alloc_fragment((num_heads), dtype=accum_dtype)
                g_cumsum = T.alloc_fragment((num_heads), dtype=accum_dtype)
                n_fragment = T.alloc_fragment((num_heads), dtype=seqlen_dtype)
                f_fragment = T.alloc_fragment((num_heads), dtype=mask_dtype)
                T.clear(g_cumsum)
                T.fill(n_fragment, num_iters)
                T.fill(f_fragment, True)

                for i_s in T.serial(num_iters):
                    for i_h in T.Parallel(num_heads):
                        g_fragment[i_h] = g[0, seq_end_idx - i_s * chunk_size - 1, i_h]
                    for i_h in T.Parallel(num_heads):
                        g_cumsum[i_h] += g_fragment[i_h]
                    for i_h in T.Parallel(num_heads):
                        if g_cumsum[i_h] < threshold and n_fragment[i_h] == num_iters:
                            n_fragment[i_h] = i_s + 1
                            f_fragment[i_h] = False

                for i_h in T.Parallel(num_heads):
                    num_warmup_chunks[bb, i_h] = n_fragment[i_h]
                for i_h in T.Parallel(num_heads):
                    fallback_mask[bb, i_h] = f_fragment[i_h]

    return warmup_chunks_kernel


def get_warmup_chunks(
    g: torch.Tensor,  # [1, num_total_tokens, num_v_heads]
    cu_seqlens: torch.Tensor,  # [cp_real_batch_size + 1]
    ht_mask: torch.Tensor,  # [cp_real_batch_size]
    chunk_size: int = 64,
    threshold: float = -10.0,
):
    batch_size, num_tokens, num_heads = g.shape
    real_batch_size = ht_mask.shape[0]
    assert cu_seqlens.shape[0] == real_batch_size + 1
    assert batch_size == 1
    assert chunk_size == 64

    warmup_chunks_kernel = _build_warmup_chunks_kernel(
        num_heads=num_heads,
        chunk_size=chunk_size,
        threshold=threshold,
        accum_dtype="float32",
        g_dtype=g.dtype,
        mask_dtype=ht_mask.dtype,
        seqlen_dtype=cu_seqlens.dtype,
    )
    num_warmup_chunks = torch.empty(
        [real_batch_size, num_heads], dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    fallback_mask = torch.empty(
        [real_batch_size, num_heads], dtype=ht_mask.dtype, device=cu_seqlens.device
    )
    warmup_chunks_kernel(g, ht_mask, cu_seqlens, num_warmup_chunks, fallback_mask)

    return num_warmup_chunks, fallback_mask


@functools.lru_cache(maxsize=32)
# TileLang's warp specialization has every producer thread wait on the empty
# mbarrier while one thread issues the TMA; a producer warp that falls two
# phases behind waits on an aliased parity forever.
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _build_correct_h0_kernel(
    H,
    DK,
    DV,
    res_dtype,
    accum_dtype,
    buffer_dtype,
    seqlen_dtype,
    mask_dtype,
    use_raw_h0,
    block_DV: int = 32,
):
    cp_batch_size = T.dynamic("cp_batch_size")
    raw_batch_size = T.dynamic("raw_batch_size")

    @T.macro
    def kernel_body(
        bb,
        bh,
        bv,
        seq_start_idx,
        seq_end_idx,
        num_iters,
        ht_buffer,
        mt_buffer,
        fallback_mask,
        seq_map_r2c,
        cp_h0,
        h_fragment,
    ):
        h_shared = T.alloc_shared((DK, block_DV), dtype=buffer_dtype)
        hd_shared = T.alloc_shared((DK, block_DV), dtype=buffer_dtype)
        m_shared = T.alloc_shared((DK, DK), dtype=buffer_dtype)

        for i_s in T.Pipelined(num_iters - 1, num_stages=2):
            if fallback_mask[seq_start_idx + i_s, bh]:
                T.copy(h_fragment, hd_shared)
            T.copy(
                ht_buffer[seq_start_idx + i_s, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                h_shared,
            )
            T.copy(h_shared, h_fragment)
            if fallback_mask[seq_start_idx + i_s, bh]:
                T.copy(mt_buffer[seq_start_idx + i_s, bh, 0:DK, 0:DK], m_shared)
                T.gemm(m_shared, hd_shared, h_fragment, clear_accum=False)
            T.copy(
                h_fragment,
                cp_h0[
                    seq_start_idx + i_s + 1,
                    bh,
                    0:DK,
                    bv * block_DV : (bv + 1) * block_DV,
                ],
            )

    if use_raw_h0:

        @T.prim_func
        def correct_h0_kernel(
            raw_h0: T.Tensor([raw_batch_size, H, DK, DV], dtype=res_dtype),
            ht_buffer: T.Tensor([cp_batch_size, H, DK, DV], dtype=buffer_dtype),
            mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=buffer_dtype),
            fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
            seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
            cp_h0: T.Tensor([cp_batch_size, H, DK, DV], dtype=res_dtype),
        ):
            with T.Kernel(T.ceildiv(DV, block_DV) * H * raw_batch_size, threads=128) as (bbhv,):
                bbh, bv = (
                    bbhv // T.ceildiv(DV, block_DV),
                    bbhv % T.ceildiv(DV, block_DV),
                )
                bb, bh = bbh // H, bbh % H

                seq_start_idx = seq_map_r2c[bb]
                seq_end_idx = seq_map_r2c[bb + 1]
                num_iters = seq_end_idx - seq_start_idx

                h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
                T.copy(
                    raw_h0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                    h_fragment,
                )
                # The loop never writes the partition a sequence starts on, and a
                # fragment-to-global T.copy before it is dropped — the fragment's
                # layout comes from how the loop consumes it. Hence a plain store.
                for ik, iv in T.Parallel(DK, block_DV):
                    cp_h0[seq_start_idx, bh, ik, bv * block_DV + iv] = raw_h0[
                        bb, bh, ik, bv * block_DV + iv
                    ]

                kernel_body(
                    bb,
                    bh,
                    bv,
                    seq_start_idx,
                    seq_end_idx,
                    num_iters,
                    ht_buffer,
                    mt_buffer,
                    fallback_mask,
                    seq_map_r2c,
                    cp_h0,
                    h_fragment,
                )

    else:

        @T.prim_func
        def correct_h0_kernel(
            ht_buffer: T.Tensor([cp_batch_size, H, DK, DV], dtype=buffer_dtype),
            mt_buffer: T.Tensor([cp_batch_size, H, DK, DK], dtype=buffer_dtype),
            fallback_mask: T.Tensor([cp_batch_size, H], dtype=mask_dtype),
            seq_map_r2c: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
            cp_h0: T.Tensor([cp_batch_size, H, DK, DV], dtype=res_dtype),
        ):
            with T.Kernel(T.ceildiv(DV, block_DV) * H * raw_batch_size, threads=128) as (bbhv,):
                bbh, bv = (
                    bbhv // T.ceildiv(DV, block_DV),
                    bbhv % T.ceildiv(DV, block_DV),
                )
                bb, bh = bbh // H, bbh % H

                seq_start_idx = seq_map_r2c[bb]
                seq_end_idx = seq_map_r2c[bb + 1]
                num_iters = seq_end_idx - seq_start_idx

                h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
                T.clear(h_fragment)
                # See the raw_h0 branch.
                for ik, iv in T.Parallel(DK, block_DV):
                    cp_h0[seq_start_idx, bh, ik, bv * block_DV + iv] = 0.0

                kernel_body(
                    bb,
                    bh,
                    bv,
                    seq_start_idx,
                    seq_end_idx,
                    num_iters,
                    ht_buffer,
                    mt_buffer,
                    fallback_mask,
                    seq_map_r2c,
                    cp_h0,
                    h_fragment,
                )

    return correct_h0_kernel


def correct_initial_states(
    raw_h0: torch.Tensor | None,  # [raw_batch_size, num_v_heads, k_head_dim, v_head_dim]
    ht_buffer: torch.Tensor,  # [cp_batch_size, num_v_heads, k_head_dim, v_head_dim]
    mt_buffer: torch.Tensor,  # [cp_batch_size, num_v_heads, k_head_dim, k_head_dim]
    fallback_mask: torch.Tensor,  # [cp_batch_size, num_v_heads]
    seq_map_r2c: torch.Tensor,  # [raw_batch_size + 1]
):
    cp_batch_size = fallback_mask.shape[0]
    _, num_heads, k_head_dim, v_head_dim = ht_buffer.shape
    assert k_head_dim == v_head_dim and k_head_dim in (64, 128)

    if raw_h0 is None:
        res_dtype = torch.float32
        use_raw_h0 = False
    else:
        res_dtype = raw_h0.dtype
        use_raw_h0 = True

    correct_h0_kernel = _build_correct_h0_kernel(
        H=num_heads,
        DK=k_head_dim,
        DV=v_head_dim,
        res_dtype=res_dtype,
        accum_dtype="float32",
        buffer_dtype=ht_buffer.dtype,
        seqlen_dtype=seq_map_r2c.dtype,
        mask_dtype=fallback_mask.dtype,
        use_raw_h0=use_raw_h0,
    )
    cp_h0 = torch.empty(
        (cp_batch_size, num_heads, k_head_dim, v_head_dim),
        dtype=res_dtype,
        device=ht_buffer.device,
    )
    if use_raw_h0:
        correct_h0_kernel(
            raw_h0,
            ht_buffer,
            mt_buffer,
            fallback_mask,
            seq_map_r2c,
            cp_h0,
        )
    else:
        correct_h0_kernel(
            ht_buffer,
            mt_buffer,
            fallback_mask,
            seq_map_r2c,
            cp_h0,
        )

    return cp_h0


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
    compile_flags=["-O3", "-DENABLE_BF16", "-include", "tl_templates/cuda/gemm.h"],
)
def _build_prepare_h_kernel(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    h_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    store_h,
    is_varlen,
    is_cp,
    num_stages=2,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    if is_varlen:
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
        h_shape = (1, num_chunks, H, DK, DV)
    else:
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
        h_shape = (batch_size, num_chunks, H, DK, DV)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)
    m_shape = (batch_size, H, DK, DK)

    @T.prim_func
    def prepare_h_kernel(
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        num_warmup_chunks: T.Tensor([batch_size, H], dtype=seqlen_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
        mt: T.Tensor(m_shape, dtype=ht_dtype),
    ):
        with T.Kernel(batch_size * H, threads=512) as (bbh,):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            _seq_split_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")
            _chunk_split_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0

            num_iters = T.alloc_var("int32")
            num_iters = (
                num_warmup_chunks[bb, bh]
                if is_cp
                else T.ceildiv(seq_end_idx - seq_start_idx, block_S)
            )

            calc_mt = T.alloc_var("bool")
            calc_mt = is_cp and num_iters >= T.ceildiv(seq_end_idx - seq_start_idx, block_S)
            seq_start_idx = seq_end_idx - num_iters * block_S if is_cp else seq_start_idx

            k_shared = T.alloc_shared((num_stages, block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((num_stages, block_S, DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((num_stages, block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((num_stages, block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((num_stages, block_S), dtype=accum_dtype, scope="shared")
            h_shared = T.alloc_shared((DK, DV), dtype=qkva_dtype)
            x_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            y_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            m_shared_L = T.alloc_shared((DK, DK // 2), dtype=qkva_dtype)
            m_shared_R = T.alloc_shared((DK, DK // 2), dtype=qkva_dtype)
            z_shared_L = T.alloc_shared((block_S, DK // 2), dtype=qkva_dtype)
            z_shared_R = T.alloc_shared((block_S, DK // 2), dtype=qkva_dtype)
            g_rev_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            h_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            x_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            y_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            m_fragment_L = T.alloc_fragment((DK, DK // 2), dtype=accum_dtype)
            m_fragment_R = T.alloc_fragment((DK, DK // 2), dtype=accum_dtype)
            z_fragment_L = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
            z_fragment_R = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
            g_last_local_S = T.alloc_local((1), dtype=accum_dtype)
            g_last_local_X = T.alloc_local((1), dtype=accum_dtype)
            g_last_local_Y = T.alloc_local((1), dtype=accum_dtype)
            g_prod_X = T.alloc_fragment((1), dtype=accum_dtype)
            g_prod_Y = T.alloc_fragment((1), dtype=accum_dtype)

            data_is_ready = T.alloc_barrier(arrive_count=[96] * num_stages)
            data_is_free = T.alloc_barrier(arrive_count=[384] * num_stages)

            bar_0 = T.alloc_barrier(arrive_count=416)
            bar_1 = T.alloc_barrier(arrive_count=256)
            bar_2 = T.alloc_barrier(arrive_count=384)
            bar_3 = T.alloc_barrier(arrive_count=128)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 24
            CONSUMER_S_NREG = 168
            CONSUMER_X_NREG = 160
            CONSUMER_Y_NREG = 160

            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                if use_initial_state:
                    T.copy(h0[bb, bh, 0:DK, 0:DV], h_fragment)
                else:
                    T.clear(h_fragment)

                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % num_stages], (i_s // num_stages + 0) % 2)
                    T.barrier_arrive(bar_0)

                    T.barrier_wait(bar_0, i_s % 2)
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)

                    T.barrier_wait(bar_1, i_s % 2)
                    # S = g_last * S
                    g_last_local_S[0] = T.exp2(g_shared[i_s % num_stages, block_S - 1] * LOG2E)
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_fragment[j_k, j_v] *= g_last_local_S[0]
                    T.barrier_arrive(bar_2)

                    T.barrier_wait(bar_2, i_s % 2)
                    # S += X^T @ Y
                    _gemm_v1(
                        x_shared,
                        y_shared,
                        h_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    T.barrier_arrive(bar_3)

                    T.barrier_arrive(data_is_free[i_s % num_stages])

                if store_final_state:
                    T.copy(h_fragment, ht[bb, bh, 0:DK, 0:DV])

            elif tx < 256:
                T.set_max_nreg(CONSUMER_X_NREG, 1)

                if calc_mt:
                    for j_k, j_v in T.Parallel(DK, DK // 2):
                        if j_k == j_v + DK // 2:
                            m_fragment_R[j_k, j_v] = 1
                        else:
                            m_fragment_R[j_k, j_v] = 0
                    g_prod_X[0] = 0

                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % num_stages], (i_s // num_stages + 0) % 2)
                    T.barrier_arrive(bar_0)

                    T.barrier_wait(bar_0, i_s % 2)
                    # X = A^T @ K
                    _gemm_v1(
                        a_shared[i_s % num_stages, :, :],
                        k_shared[i_s % num_stages, :, :],
                        x_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )

                    # [STAGE = i_s % num_stages] 1
                    # X = - b * X
                    for j_s, j_k in T.Parallel(block_S, DK):
                        x_fragment[j_s, j_k] *= -b_shared[i_s % num_stages, j_s]
                    T.copy(x_fragment, x_shared)
                    T.barrier_arrive(bar_2)

                    if calc_mt:
                        g_prod_X[0] += g_shared[i_s % num_stages, block_S - 1]
                        T.copy(m_fragment_R, m_shared_R)

                        T.barrier_wait(bar_3, i_s % 2)
                        # Z = K @ M
                        _gemm_v1(
                            k_shared[i_s % num_stages, :, :],
                            m_shared_R,
                            z_fragment_R,
                            clear_accum=True,
                        )
                        T.copy(z_fragment_R, z_shared_R)
                        # M += X^T @ Z
                        _gemm_v1(
                            x_shared,
                            z_shared_R,
                            m_fragment_R,
                            transpose_A=True,
                            clear_accum=False,
                        )

                    T.barrier_arrive(data_is_free[i_s % num_stages])

                if calc_mt:
                    g_last_local_X[0] = T.exp2(g_prod_X[0] * LOG2E)
                    for j_k, j_v in T.Parallel(DK, DK // 2):
                        m_fragment_R[j_k, j_v] *= g_last_local_X[0]
                    T.copy(m_fragment_R, mt[bb, bh, 0:DK, DK // 2 :])

            elif tx < 384:
                T.set_max_nreg(CONSUMER_Y_NREG, 1)

                if calc_mt:
                    for j_k, j_v in T.Parallel(DK, DK // 2):
                        if j_k == j_v:
                            m_fragment_L[j_k, j_v] = 1
                        else:
                            m_fragment_L[j_k, j_v] = 0
                    g_prod_Y[0] = 0

                for i_s in T.serial(num_iters):
                    T.barrier_wait(data_is_ready[i_s % num_stages], (i_s // num_stages + 0) % 2)
                    T.barrier_arrive(bar_0)

                    T.barrier_wait(bar_0, i_s % 2)
                    # Precompute g_last/g
                    g_last_local_Y[0] = g_shared[i_s % num_stages, block_S - 1]
                    for j_s in T.Parallel(block_S):
                        g_rev_exp_shared[j_s] = T.exp2(
                            (g_last_local_Y[0] - g_shared[i_s % num_stages, j_s]) * LOG2E
                        )
                    g_last_local_Y[0] = T.exp2(g_last_local_Y[0] * LOG2E)
                    T.barrier_arrive(bar_1)

                    T.barrier_wait(bar_1, i_s % 2)
                    # U = K @ S
                    _gemm_v1(
                        k_shared[i_s % num_stages, :, :],
                        h_shared,
                        y_fragment,
                        clear_accum=True,
                    )
                    # Y = g_last * U - g_last/g * V
                    for j_s, j_v in T.Parallel(block_S, DV):
                        y_fragment[j_s, j_v] *= g_last_local_Y[0]
                    for j_s, j_v in T.Parallel(block_S, DV):
                        y_fragment[j_s, j_v] -= (
                            v_shared[i_s % num_stages, j_s, j_v] * g_rev_exp_shared[j_s]
                        )
                    T.copy(y_fragment, y_shared)
                    T.barrier_arrive(bar_2)

                    if calc_mt:
                        g_prod_Y[0] += g_shared[i_s % num_stages, block_S - 1]
                        T.copy(m_fragment_L, m_shared_L)

                        T.barrier_wait(bar_3, i_s % 2)
                        # Z = K @ M
                        _gemm_v1(
                            k_shared[i_s % num_stages, :, :],
                            m_shared_L,
                            z_fragment_L,
                            clear_accum=True,
                        )
                        T.copy(z_fragment_L, z_shared_L)
                        # M += X^T @ Z
                        _gemm_v1(
                            x_shared,
                            z_shared_L,
                            m_fragment_L,
                            transpose_A=True,
                            clear_accum=False,
                        )

                    T.barrier_arrive(data_is_free[i_s % num_stages])

                if calc_mt:
                    g_last_local_Y[0] = T.exp2(g_prod_Y[0] * LOG2E)
                    for j_k, j_v in T.Parallel(DK, DK // 2):
                        m_fragment_L[j_k, j_v] *= g_last_local_Y[0]
                    T.copy(m_fragment_L, mt[bb, bh, 0:DK, : DK // 2])

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % num_stages], (i_s // num_stages + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        T.tma_copy(
                            k[batch_idx, left:right, bhg, 0:DK],
                            k_shared[i_s % num_stages, :, :],
                            barrier=data_is_ready[i_s % num_stages],
                        )

                        T.barrier_arrive(data_is_ready[i_s % num_stages])

                elif tx < 384 + 64:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % num_stages], (i_s // num_stages + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        T.tma_copy(
                            v[batch_idx, left:right, bh, 0:DV],
                            v_shared[i_s % num_stages, :, :],
                            barrier=data_is_ready[i_s % num_stages],
                        )
                        # TODO: mask A for the last chunk
                        T.tma_copy(
                            a[batch_idx, left:right, bh, 0:block_S],
                            a_shared[i_s % num_stages, :, :],
                            barrier=data_is_ready[i_s % num_stages],
                        )

                        T.barrier_arrive(data_is_ready[i_s % num_stages])

                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % num_stages], (i_s // num_stages + 1) % 2)
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                g_shared[i_s % num_stages, j_s] = g[batch_idx, left + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    g_shared[i_s % num_stages, j_s] = g[batch_idx, left + j_s, bh]
                                else:
                                    g_shared[i_s % num_stages, j_s] = g[
                                        batch_idx, seq_end_idx - 1, bh
                                    ]
                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % num_stages, j_s] = b[batch_idx, left + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    b_shared[i_s % num_stages, j_s] = b[batch_idx, left + j_s, bh]
                                else:
                                    b_shared[i_s % num_stages, j_s] = 0

                        T.barrier_arrive(data_is_ready[i_s % num_stages])

                else:
                    for i_s in T.serial(num_iters):
                        T.barrier_arrive(bar_0)

                        T.barrier_wait(bar_0, i_s % 2)
                        T.barrier_wait(bar_1, i_s % 2)
                        if store_h:
                            T.copy(
                                h_shared,
                                h[batch_idx, chunk_start_idx + i_s, bh, 0:DK, 0:DV],
                            )

    return prepare_h_kernel


def fused_gdr_h(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = True,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    num_warmup_chunks: torch.LongTensor | None = None,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    assert K == V and K in (64, 128)
    assert chunk_size == 64

    if cu_seqlens is None:
        assert num_warmup_chunks is None
        real_batch_size = batch_size
        num_chunks = tilelang.cdiv(num_tokens, chunk_size) if output_h else 0
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        chunk_offsets = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        is_varlen = False
        is_cp = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        chunk_offsets, num_chunks = prepare_chunk_offsets(cu_seqlens, chunk_size)
        chunk_offsets = chunk_offsets.to(cu_seqlens.dtype)
        num_chunks = num_chunks if output_h else 0
        is_varlen = True
        if num_warmup_chunks is None:
            num_warmup_chunks = torch.empty(
                (real_batch_size, H), dtype=cu_seqlens.dtype, device=k.device
            )
            is_cp = False
        else:
            is_cp = True

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
        )
    h = torch.empty((batch_size, num_chunks, H, K, V), dtype=k.dtype, device=k.device)
    ht_dtype = k.dtype if is_cp else torch.float32
    final_state = torch.empty((real_batch_size, H, K, V), dtype=ht_dtype, device=k.device)
    final_correction = torch.empty((real_batch_size, H, K, K), dtype=ht_dtype, device=k.device)

    prepare_h_kernel = _build_prepare_h_kernel(
        H,
        Hg,
        K,
        V,
        chunk_size,
        qkva_dtype=k.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        h_dtype=h.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        store_h=output_h,
        is_varlen=is_varlen,
        is_cp=is_cp,
    )
    prepare_h_kernel(
        k,
        v,
        a,
        g,
        b,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        num_warmup_chunks,
        h,
        final_state,
        final_correction,
    )

    if not output_final_state:
        final_state = None
        final_correction = None
    if not output_h:
        h = None

    return h, final_state, final_correction
