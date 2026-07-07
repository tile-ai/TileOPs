"""Gated DeltaNet inference prefill kernel.

The prefill path shares the chunkwise state/output kernels with training
forward, but uses an inference-only prepare kernel that produces only ``w`` and
``u``. Training-only ``Aw``/``Au`` matrices are kept in shared memory and are
not written to global memory.
"""

import functools
import math
import os
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from .gated_deltanet_fwd import (
    _LOG2E,
    _chunk_local_cumsum,
    _h_recurrence_tl,
    _output_o_tl,
)

__all__ = ["GatedDeltaNetPrefillFwdKernel"]


def _normalize_prefill_layout(layout: str) -> str:
    layout = layout.lower()
    if layout == "bhsd":
        return "bhtd"
    if layout in ("bhtd", "bthd"):
        return layout
    raise ValueError(f"Unsupported layout: {layout}")


@functools.lru_cache(maxsize=32)
def _prefill_single_batch_cu_seqlens(
    num_tokens: int,
    device_idx: int,
) -> torch.Tensor:
    return torch.tensor([0, num_tokens], dtype=torch.int32, device=f"cuda:{device_idx}")


def _prefill_auto_cp_local_chunks(num_chunks: int, num_heads: int) -> int:
    env_max_local_chunks = os.environ.get(
        "TILEOPS_GDN_PREFILL_MAX_LOCAL_CHUNKS",
        os.environ.get("TILEOPS_GDN_PREFILL_CP_MAX_LOCAL_CHUNKS"),
    )
    if env_max_local_chunks:
        max_local_chunks = int(env_max_local_chunks)
    else:
        sm_count = torch.cuda.get_device_properties().multi_processor_count
        max_local_chunks = 2 ** round(
            math.log2(math.sqrt(num_heads * num_chunks / sm_count) * 3)
        )
        if num_heads >= 64 and num_chunks >= 512:
            max_local_chunks = max(max_local_chunks, 256)
    return max(max_local_chunks, 4)


def _prefill_partitioned_initial_state_bthd(
    k: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    from tileops.kernels.gated_deltanet.gdn_prefill import (
        correct_initial_states,
        fused_gdr_h,
        get_warmup_chunks,
    )

    batch, num_tokens, num_heads, _ = k.shape
    assert batch == 1
    raw_cu_seqlens = _prefill_single_batch_cu_seqlens(num_tokens, k.device.index)
    num_chunks = tilelang.cdiv(num_tokens, chunk_size)
    max_local_chunks = _prefill_auto_cp_local_chunks(num_chunks, num_heads)

    has_partition = num_chunks > max_local_chunks
    force_partition = (
        os.environ.get(
            "TILEOPS_GDN_PREFILL_FORCE_PARTITION",
            os.environ.get("TILEOPS_GDN_PREFILL_CP_FORCE", "0"),
        )
        == "1"
    )
    use_partition = has_partition and (
        force_partition or num_heads <= 40 or (num_heads <= 64 and num_chunks >= 128)
    )
    if not use_partition:
        return None, raw_cu_seqlens, None, None

    cp_cu_seqlens = []
    ht_mask = []
    seq_map_c2r = []
    seq_map_r2c = [0]
    split_tokens = max_local_chunks * chunk_size
    start = 0
    while start < num_tokens:
        cp_cu_seqlens.append(start)
        ht_mask.append(False)
        seq_map_c2r.append(0)
        start += split_tokens
    ht_mask[-1] = True
    seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(num_tokens)

    cp_cu_seqlens_t = torch.tensor(
        cp_cu_seqlens, dtype=torch.int32, device=k.device, requires_grad=False
    )
    seq_map_c2r_t = torch.tensor(seq_map_c2r, dtype=torch.int32, device=k.device)
    seq_map_r2c_t = torch.tensor(
        seq_map_r2c, dtype=torch.int32, device=k.device, requires_grad=False
    )
    ht_mask_t = torch.tensor(
        ht_mask, dtype=torch.bool, device=k.device, requires_grad=False
    )

    num_warmup_chunks, fallback_mask = get_warmup_chunks(
        g=g,
        cu_seqlens=cp_cu_seqlens_t,
        ht_mask=ht_mask_t,
        chunk_size=chunk_size,
        threshold=-10.0,
    )
    _, ht, mt = fused_gdr_h(
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        cu_seqlens=cp_cu_seqlens_t,
        num_warmup_chunks=num_warmup_chunks,
    )
    cp_h0 = correct_initial_states(
        raw_h0=None,
        ht_buffer=ht,
        mt_buffer=mt,
        fallback_mask=fallback_mask,
        seq_map_r2c=seq_map_r2c_t,
    )
    return cp_h0, cp_cu_seqlens_t, seq_map_c2r_t, raw_cu_seqlens


@functools.lru_cache(maxsize=32)
def _prefill_chunk_local_cumsum_bhtd_tl(
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
        def prefill_chunk_cumsum_bhtd(
            g: T.Tensor([batch, head, seq_len], dtype),
            out: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(batch * head * num_chunks, threads=128) as bx:
                bid = bx // (head * num_chunks)
                hid = (bx // num_chunks) % head
                cid = bx % num_chunks
                base = cid * chunk_size

                g_s = T.alloc_shared([chunk_size], dtype)
                out_s = T.alloc_shared([chunk_size], "float32")

                T.copy(g[bid, hid, base : base + chunk_size], g_s, disable_tma=True)

                out_s[0] = T.cast(g_s[0], "float32")
                for i in T.Serial(1, chunk_size):
                    out_s[i] = out_s[i - 1] + T.cast(g_s[i], "float32")
                for i in T.Parallel(chunk_size):
                    out[bid, hid, base + i] = T.cast(out_s[i], dtype)

        return prefill_chunk_cumsum_bhtd

    return _func()


@functools.lru_cache(maxsize=32)
def _prefill_prepare_w_u_bhtd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Compute Gated DeltaNet prefill ``w`` and ``u`` without global Aw/Au."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _fused_func(num_stages, threads=128):
        @T.prim_func
        def prefill_prepare_w_u_bhtd(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                k_shared = T.alloc_shared([block_C, dim_k], dtype)
                g_shared = T.alloc_shared([block_C], dtype)
                beta_shared = T.alloc_shared([block_C], dtype)
                k_beta_shared = T.alloc_shared([block_C, dim_k], dtype)
                v_beta_shared = T.alloc_shared([block_C, dim_v], dtype)
                S_shared = T.alloc_shared([block_C, block_C], dtype)
                P_shared = T.alloc_shared([block_C, block_C], dtype)
                g_pos_shared = T.alloc_shared([block_C], accum_dtype)
                g_neg_shared = T.alloc_shared([block_C], accum_dtype)

                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                w_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                u_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)

                T.copy(
                    k[bid, hid, by * block_C : (by + 1) * block_C, :],
                    k_shared,
                    disable_tma=True,
                )
                T.copy(
                    g[bid, hid, by * block_C : (by + 1) * block_C],
                    g_shared,
                    disable_tma=True,
                )
                T.copy(
                    beta[bid, hid, by * block_C : (by + 1) * block_C],
                    beta_shared,
                    disable_tma=True,
                )

                for i_s, i_k in T.Parallel(block_C, dim_k):
                    k_beta_shared[i_s, i_k] = k_shared[i_s, i_k] * beta_shared[i_s]
                for i in T.Parallel(block_C):
                    g_pos_shared[i] = T.exp2(g_shared[i] * _LOG2E)
                    g_neg_shared[i] = T.exp2(-g_shared[i] * _LOG2E)
                T.clear(gram_frag)
                T.gemm(k_beta_shared, k_shared, gram_frag, transpose_B=True)

                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(
                        i > j,
                        -gram_frag[i, j]
                        * g_pos_shared[i]
                        * g_neg_shared[j],
                        T.float32(0.0),
                    )
                T.clear(S_shared)
                for i in T.Parallel(block_C):
                    S_shared[i, i] = T.float32(1.0)

                for _r in T.Serial(num_rounds):
                    T.clear(temp_frag)
                    T.gemm(P_shared, S_shared, temp_frag)
                    for i, j in T.Parallel(block_C, block_C):
                        S_shared[i, j] = S_shared[i, j] + temp_frag[i, j]
                    T.clear(temp_frag)
                    T.gemm(P_shared, P_shared, temp_frag)
                    T.copy(temp_frag, P_shared)

                T.clear(w_frag)
                T.gemm(S_shared, k_beta_shared, w_frag)
                T.copy(
                    w_frag,
                    w[bid, hid, by * block_C : (by + 1) * block_C, :],
                    disable_tma=True,
                )

                for i, j in T.Parallel(block_C, dim_v):
                    v_beta_shared[i, j] = (
                        v[bid, hid, by * block_C + i, j] * beta_shared[i]
                    )
                T.clear(u_frag)
                T.gemm(S_shared, v_beta_shared, u_frag)
                T.copy(
                    u_frag,
                    u[bid, hid, by * block_C : (by + 1) * block_C, :],
                    disable_tma=True,
                )

        return prefill_prepare_w_u_bhtd

    return _fused_func


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
                        acc_s[hid] = acc_s[hid] + T.cast(
                            g[bid, base + i, hid], "float32"
                        )
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
):
    if chunk_size != 64 or dim_k != 128:
        raise ValueError("TileLang blocksolve-A currently expects chunk64 and K=128")

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

                T.annotate_layout({
                    k0: tilelang.layout.make_swizzled_layout(k0),
                    k1: tilelang.layout.make_swizzled_layout(k1),
                    k2: tilelang.layout.make_swizzled_layout(k2),
                    k3: tilelang.layout.make_swizzled_layout(k3),
                    a_s: tilelang.layout.make_swizzled_layout(a_s),
                    i_s: tilelang.layout.make_swizzled_layout(i_s),
                    work_s: tilelang.layout.make_swizzled_layout(work_s),
                })

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
                    g_val = T.cast(g_s[t], accum_dtype)
                    gate0_s[t] = T.exp2(
                        (g_val - T.cast(g_s[0], accum_dtype)) * _LOG2E
                    )
                    gate1_s[t] = T.exp2(
                        (g_val - T.cast(g_s[block_c], accum_dtype)) * _LOG2E
                    )
                    gate2_s[t] = T.exp2(
                        (g_val - T.cast(g_s[2 * block_c], accum_dtype)) * _LOG2E
                    )
                    gate3_s[t] = T.exp2(
                        (g_val - T.cast(g_s[3 * block_c], accum_dtype)) * _LOG2E
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
                    A[bid, base + block_c + i, hid, j] = T.cast(a_s[1, i, j], dtype)
                    A[bid, base + block_c + i, hid, block_c + j] = T.cast(i_s[1, i, j], dtype)
                    A[bid, base + 2 * block_c + i, hid, j] = T.cast(a_s[3, i, j], dtype)
                    A[bid, base + 2 * block_c + i, hid, block_c + j] = T.cast(a_s[4, i, j], dtype)
                    A[bid, base + 2 * block_c + i, hid, 2 * block_c + j] = T.cast(i_s[2, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, j] = T.cast(a_s[6, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, block_c + j] = T.cast(a_s[7, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, 2 * block_c + j] = T.cast(a_s[8, i, j], dtype)
                    A[bid, base + 3 * block_c + i, hid, 3 * block_c + j] = T.cast(i_s[3, i, j], dtype)

        return prefill_blocksolve_A_bthd_tl

    return _func(32)


def _prefill_blocksolve_A_bthd(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch, seq_len, head, dim_k = k.shape
    A = torch.zeros(batch, seq_len, head, chunk_size, dtype=k.dtype, device=k.device)
    kernel = _prefill_blocksolve_A_bthd_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        str(k.dtype).split(".")[-1],
    )
    kernel(k, g, beta, A)
    return A

@functools.lru_cache(maxsize=32)
def _prefill_recompute_w_u_from_A_bthd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
):
    block_c = chunk_size
    block_d = 64
    num_k_tiles = dim_k // block_d
    num_v_tiles = dim_v // block_d
    accum_dtype = "float32"

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def recompute_w_u_from_A_bthd(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            v: T.Tensor([batch, seq_len, head, dim_v], dtype),
            beta: T.Tensor([batch, seq_len, head], dtype),
            A: T.Tensor([batch, seq_len, head, chunk_size], dtype),
            w: T.Tensor([batch, seq_len, head, dim_k], dtype),
            u: T.Tensor([batch, seq_len, head, dim_v], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_c, threads=threads) as (
                bid,
                hid,
                by,
            ):
                base = by * block_c
                A_s = T.alloc_shared([block_c, block_c], dtype)
                beta_s = T.alloc_shared([block_c], dtype)
                x_s = T.alloc_shared([block_c, block_d], dtype)
                out_s = T.alloc_shared([block_c, block_d], dtype)
                out_frag = T.alloc_fragment([block_c, block_d], accum_dtype)

                T.annotate_layout({
                    A_s: tilelang.layout.make_swizzled_layout(A_s),
                    x_s: tilelang.layout.make_swizzled_layout(x_s),
                    out_s: tilelang.layout.make_swizzled_layout(out_s),
                })

                T.async_copy(A[bid, base : base + block_c, hid, 0:block_c], A_s)
                T.ptx_wait_group(0)
                T.copy(
                    beta[bid, base : base + block_c, hid],
                    beta_s,
                    disable_tma=True,
                )
                T.sync_threads()

                for vt in T.Serial(num_v_tiles):
                    v_offset = vt * block_d
                    T.async_copy(
                        v[
                            bid,
                            base : base + block_c,
                            hid,
                            v_offset : v_offset + block_d,
                        ],
                        x_s,
                    )
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    for i, d in T.Parallel(block_c, block_d):
                        x_s[i, d] = x_s[i, d] * beta_s[i]
                    T.clear(out_frag)
                    T.gemm(A_s, x_s, out_frag)
                    T.copy(out_frag, out_s)
                    T.copy(
                        out_s,
                        u[
                            bid,
                            base : base + block_c,
                            hid,
                            v_offset : v_offset + block_d,
                        ],
                        disable_tma=True,
                    )

                for kt in T.Serial(num_k_tiles):
                    k_offset = kt * block_d
                    T.async_copy(
                        k[
                            bid,
                            base : base + block_c,
                            hid,
                            k_offset : k_offset + block_d,
                        ],
                        x_s,
                    )
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    for i, d in T.Parallel(block_c, block_d):
                        x_s[i, d] = x_s[i, d] * beta_s[i]
                    T.clear(out_frag)
                    T.gemm(A_s, x_s, out_frag)
                    T.copy(out_frag, out_s)
                    T.copy(
                        out_s,
                        w[
                            bid,
                            base : base + block_c,
                            hid,
                            k_offset : k_offset + block_d,
                        ],
                        disable_tma=True,
                    )

        return recompute_w_u_from_A_bthd

    return _func


@functools.lru_cache(maxsize=32)
def _prefill_prepare_w_u_bthd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    accum_dtype = "float32"
    block_C = chunk_size
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _fused_func(num_stages, threads=128):
        @T.prim_func
        def prefill_prepare_w_u_bthd(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            v: T.Tensor([batch, seq_len, head, dim_v], dtype),
            g: T.Tensor([batch, seq_len, head], dtype),
            beta: T.Tensor([batch, seq_len, head], dtype),
            w: T.Tensor([batch, seq_len, head, dim_k], dtype),
            u: T.Tensor([batch, seq_len, head, dim_v], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (
                bid,
                hid,
                by,
            ):
                base = by * block_C
                k_shared = T.alloc_shared([block_C, dim_k], dtype)
                g_shared = T.alloc_shared([block_C], dtype)
                beta_shared = T.alloc_shared([block_C], dtype)
                k_beta_shared = T.alloc_shared([block_C, dim_k], dtype)
                v_beta_shared = T.alloc_shared([block_C, dim_v], dtype)
                S_shared = T.alloc_shared([block_C, block_C], dtype)
                P_shared = T.alloc_shared([block_C, block_C], dtype)
                g_pos_shared = T.alloc_shared([block_C], accum_dtype)
                g_neg_shared = T.alloc_shared([block_C], accum_dtype)

                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                w_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                u_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)

                for i, d in T.Parallel(block_C, dim_k):
                    k_shared[i, d] = k[bid, base + i, hid, d]
                for i in T.Parallel(block_C):
                    g_shared[i] = g[bid, base + i, hid]
                    beta_shared[i] = beta[bid, base + i, hid]

                for i, d in T.Parallel(block_C, dim_k):
                    k_beta_shared[i, d] = k_shared[i, d] * beta_shared[i]
                for i in T.Parallel(block_C):
                    g_pos_shared[i] = T.exp2(g_shared[i] * _LOG2E)
                    g_neg_shared[i] = T.exp2(-g_shared[i] * _LOG2E)
                T.clear(gram_frag)
                T.gemm(k_beta_shared, k_shared, gram_frag, transpose_B=True)

                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(
                        i > j,
                        -gram_frag[i, j]
                        * g_pos_shared[i]
                        * g_neg_shared[j],
                        T.float32(0.0),
                    )
                T.clear(S_shared)
                for i in T.Parallel(block_C):
                    S_shared[i, i] = T.float32(1.0)

                for _r in T.Serial(num_rounds):
                    T.clear(temp_frag)
                    T.gemm(P_shared, S_shared, temp_frag)
                    for i, j in T.Parallel(block_C, block_C):
                        S_shared[i, j] = S_shared[i, j] + temp_frag[i, j]
                    T.clear(temp_frag)
                    T.gemm(P_shared, P_shared, temp_frag)
                    T.copy(temp_frag, P_shared)

                T.clear(w_frag)
                T.gemm(S_shared, k_beta_shared, w_frag)
                for i, d in T.Parallel(block_C, dim_k):
                    w[bid, base + i, hid, d] = w_frag[i, d]

                for i, d in T.Parallel(block_C, dim_v):
                    v_beta_shared[i, d] = (
                        v[bid, base + i, hid, d] * beta_shared[i]
                    )
                T.clear(u_frag)
                T.gemm(S_shared, v_beta_shared, u_frag)
                for i, d in T.Parallel(block_C, dim_v):
                    u[bid, base + i, hid, d] = u_frag[i, d]

        return prefill_prepare_w_u_bthd

    return _fused_func


@functools.lru_cache(maxsize=32)
def _prefill_h_recurrence_bthd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
    block_v: int = 0,
):
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = dim_v if block_v <= 0 else block_v
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128):
        @T.prim_func
        def h_recurrence_bthd(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            g: T.Tensor([batch, seq_len, head], dtype),
            w: T.Tensor([batch, seq_len, head, dim_k], dtype),
            u: T.Tensor([batch, seq_len, head, dim_v], dtype),
            S_0: T.Tensor([batch, head, dim_k, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, seq_len, head, dim_v], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (
                vid,
                bid,
                hid,
            ):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)

                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                h_next_frag = T.alloc_fragment([dim_k, BV], accum_dtype)

                v_offset = vid * BV

                T.copy(S_0[bid, hid, :, v_offset : v_offset + BV], h_c, disable_tma=True)
                for i, j in T.Parallel(dim_k, BV):
                    S[bid, hid, 0, i, v_offset + j] = h_c[i, j]

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    base = t * block_C
                    for i, d in T.Parallel(block_C, dim_k):
                        k_c[i, d] = k[bid, base + i, hid, d]
                        w_c[i, d] = w[bid, base + i, hid, d]
                    for i in T.Parallel(block_C):
                        g_c[i] = g[bid, base + i, hid]
                    for i, d in T.Parallel(block_C, BV):
                        u_c[i, d] = u[bid, base + i, hid, v_offset + d]

                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_c[block_C - 1]) * _LOG2E
                        )

                    for i, j in T.Parallel(block_C, BV):
                        v_new[bid, base + i, hid, v_offset + j] = v_new_c[i, j]

                    for n, j in T.Parallel(block_C, BV):
                        v_new_c[n, j] = v_new_c[n, j] * T.exp2(
                            (g_c[block_C - 1] - g_c[n]) * _LOG2E
                        )
                    for i, j in T.Parallel(dim_k, BV):
                        h_next_frag[i, j] = h_c[i, j] * T.exp2(
                            g_c[block_C - 1] * _LOG2E
                        )
                    T.gemm(
                        k_c,
                        v_new_c,
                        h_next_frag,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(h_next_frag, h_c)
                    for i, j in T.Parallel(dim_k, BV):
                        S[bid, hid, t + 1, i, v_offset + j] = h_c[i, j]

        return h_recurrence_bthd

    return _func


@functools.lru_cache(maxsize=32)
def _prefill_output_o_bthd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def output_o_bthd(
            q: T.Tensor([batch, seq_len, head, dim_k], dtype),
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            g: T.Tensor([batch, seq_len, head], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, seq_len, head, dim_v], dtype),
            o: T.Tensor([batch, seq_len, head, dim_v], dtype),
        ):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                base = tid * block_C
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)

                o_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)

                for i, d in T.Parallel(block_C, dim_k):
                    q_c[i, d] = q[bid, base + i, hid, d]
                    k_c[i, d] = k[bid, base + i, hid, d]
                for i in T.Parallel(block_C):
                    g_c[i] = g[bid, base + i, hid]
                T.copy(S[bid, hid, tid, :, :], h_c, disable_tma=True)
                for i, d in T.Parallel(block_C, dim_v):
                    v_new_c[i, d] = v_new[bid, base + i, hid, d]

                T.clear(o_frag)
                T.gemm(q_c, h_c, o_frag)
                for i, j in T.Parallel(block_C, dim_v):
                    o_frag[i, j] = o_frag[i, j] * T.exp2(g_c[i] * _LOG2E)

                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(
                        i >= j,
                        attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                        T.float32(0.0),
                    )

                T.gemm(attn, v_new_c, o_frag)
                for i, d in T.Parallel(block_C, dim_v):
                    o[bid, base + i, hid, d] = o_frag[i, d]

        return output_o_bthd

    return _func


@functools.lru_cache(maxsize=32)
def _prefill_group_transition_summary_bthd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    group_chunks: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float16",
    block_v: int = 64,
):
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    num_groups = num_chunks // group_chunks
    dim_aug = dim_v + dim_k
    BV = block_v
    num_v_tiles = dim_aug // BV

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def group_transition_summary_bthd(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            g: T.Tensor([batch, seq_len, head], dtype),
            w: T.Tensor([batch, seq_len, head, dim_k], dtype),
            u: T.Tensor([batch, seq_len, head, dim_v], dtype),
            summary: T.Tensor([batch, head, num_groups, dim_k, dim_aug], dtype),
        ):
            with T.Kernel(num_v_tiles * num_groups, batch, head, threads=threads) as (
                vgid,
                bid,
                hid,
            ):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)

                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                h_next_frag = T.alloc_fragment([dim_k, BV], accum_dtype)

                vid = vgid % num_v_tiles
                gid = vgid // num_v_tiles
                v_offset = vid * BV
                start_chunk = gid * group_chunks

                for i, j in T.Parallel(dim_k, BV):
                    aug_col = v_offset + j
                    if aug_col < dim_v:
                        h_c[i, j] = 0.0
                    else:
                        a_col = aug_col - dim_v
                        if i == a_col:
                            h_c[i, j] = 1.0
                        else:
                            h_c[i, j] = 0.0

                for local_t in T.Pipelined(group_chunks, num_stages=num_stages):
                    t = start_chunk + local_t
                    base = t * block_C
                    for i, d in T.Parallel(block_C, dim_k):
                        k_c[i, d] = k[bid, base + i, hid, d]
                        w_c[i, d] = w[bid, base + i, hid, d]
                    for i in T.Parallel(block_C):
                        g_c[i] = g[bid, base + i, hid]
                    for i, j in T.Parallel(block_C, BV):
                        aug_col = v_offset + j
                        if aug_col < dim_v:
                            u_c[i, j] = u[bid, base + i, hid, aug_col]
                        else:
                            u_c[i, j] = 0.0

                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_c[block_C - 1]) * _LOG2E
                        )

                    for n, j in T.Parallel(block_C, BV):
                        v_new_c[n, j] = v_new_c[n, j] * T.exp2(
                            (g_c[block_C - 1] - g_c[n]) * _LOG2E
                        )
                    for i, j in T.Parallel(dim_k, BV):
                        h_next_frag[i, j] = h_c[i, j] * T.exp2(
                            g_c[block_C - 1] * _LOG2E
                        )
                    T.gemm(
                        k_c,
                        v_new_c,
                        h_next_frag,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(h_next_frag, h_c)

                for i, j in T.Parallel(dim_k, BV):
                    summary[bid, hid, gid, i, v_offset + j] = h_c[i, j]

        return group_transition_summary_bthd

    return _func


@functools.lru_cache(maxsize=32)
def _prefill_dense_group_start_scan_bthd_tl(
    batch: int,
    head: int,
    num_groups: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float16",
    block_v: int = 64,
):
    accum_dtype = "float32"
    dim_aug = dim_v + dim_k
    BV = block_v
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def dense_group_start_scan_bthd(
            summary: T.Tensor([batch, head, num_groups, dim_k, dim_aug], dtype),
            group_start: T.Tensor([batch, head, num_groups, dim_k, dim_v], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (
                vid,
                bid,
                hid,
            ):
                a_c = T.alloc_shared([dim_k, dim_k], dtype)
                b_c = T.alloc_shared([dim_k, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                h_next_frag = T.alloc_fragment([dim_k, BV], accum_dtype)

                v_offset = vid * BV

                for i, j in T.Parallel(dim_k, BV):
                    h_c[i, j] = 0.0

                for gid in T.Pipelined(num_groups, num_stages=num_stages):
                    for i, j in T.Parallel(dim_k, BV):
                        group_start[bid, hid, gid, i, v_offset + j] = h_c[i, j]
                        b_c[i, j] = summary[bid, hid, gid, i, v_offset + j]
                    for i, j in T.Parallel(dim_k, dim_k):
                        a_c[i, j] = summary[bid, hid, gid, i, dim_v + j]

                    T.clear(h_next_frag)
                    T.gemm(a_c, h_c, h_next_frag)
                    for i, j in T.Parallel(dim_k, BV):
                        h_c[i, j] = h_next_frag[i, j] + b_c[i, j]

        return dense_group_start_scan_bthd

    return _func


@functools.lru_cache(maxsize=32)
def _prefill_grouped_replay_bthd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    group_chunks: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float16",
    block_v: int = 64,
):
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    num_groups = num_chunks // group_chunks
    BV = block_v
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=256):
        @T.prim_func
        def h_grouped_replay_bthd(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            g: T.Tensor([batch, seq_len, head], dtype),
            w: T.Tensor([batch, seq_len, head, dim_k], dtype),
            u: T.Tensor([batch, seq_len, head, dim_v], dtype),
            group_start: T.Tensor([batch, head, num_groups, dim_k, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, seq_len, head, dim_v], dtype),
        ):
            with T.Kernel(num_v_tiles * num_groups, batch, head, threads=threads) as (
                vgid,
                bid,
                hid,
            ):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)

                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                h_next_frag = T.alloc_fragment([dim_k, BV], accum_dtype)

                vid = vgid % num_v_tiles
                gid = vgid // num_v_tiles
                v_offset = vid * BV
                start_chunk = gid * group_chunks

                T.copy(
                    group_start[bid, hid, gid, :, v_offset : v_offset + BV],
                    h_c,
                    disable_tma=True,
                )
                if gid == 0:
                    for i, j in T.Parallel(dim_k, BV):
                        S[bid, hid, 0, i, v_offset + j] = h_c[i, j]

                for local_t in T.Pipelined(group_chunks, num_stages=num_stages):
                    t = start_chunk + local_t
                    base = t * block_C
                    for i, d in T.Parallel(block_C, dim_k):
                        k_c[i, d] = k[bid, base + i, hid, d]
                        w_c[i, d] = w[bid, base + i, hid, d]
                    for i in T.Parallel(block_C):
                        g_c[i] = g[bid, base + i, hid]
                    for i, d in T.Parallel(block_C, BV):
                        u_c[i, d] = u[bid, base + i, hid, v_offset + d]

                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_c[block_C - 1]) * _LOG2E
                        )

                    for i, j in T.Parallel(block_C, BV):
                        v_new[bid, base + i, hid, v_offset + j] = v_new_c[i, j]

                    for n, j in T.Parallel(block_C, BV):
                        v_new_c[n, j] = v_new_c[n, j] * T.exp2(
                            (g_c[block_C - 1] - g_c[n]) * _LOG2E
                        )
                    for i, j in T.Parallel(dim_k, BV):
                        h_next_frag[i, j] = h_c[i, j] * T.exp2(
                            g_c[block_C - 1] * _LOG2E
                        )
                    T.gemm(
                        k_c,
                        v_new_c,
                        h_next_frag,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(h_next_frag, h_c)
                    for i, j in T.Parallel(dim_k, BV):
                        S[bid, hid, t + 1, i, v_offset + j] = h_c[i, j]

        return h_grouped_replay_bthd

    return _func


@torch.library.custom_op("tileops::gated_deltanet_prefill_fwd_kernel", mutates_args=())
def _gated_deltanet_prefill_wrapped_kernel(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    fused_num_stages: int,
    fused_threads: int,
    h_num_stages: int,
    h_threads: int,
    h_block_v: int,
    o_threads: int,
    layout: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    layout = _normalize_prefill_layout(layout)
    if layout == "bthd":
        g_cum = _prefill_chunk_local_cumsum_bthd_tl(
            batch, head, seq_len, chunk_size, dtype
        )(g)
        use_blocksolve_prepare = (
            chunk_size == 64
            and dim_k == 128
            and dim_v == 128
            and dtype == "float16"
        )
        use_partitioned_prefill = (
            os.environ.get(
                "TILEOPS_GDN_PREFILL_PARTITIONED",
                os.environ.get("TILEOPS_GDN_PREFILL_CP_SPLIT", "1"),
            )
            != "0"
            and batch == 1
            and use_blocksolve_prepare
        )
        if use_partitioned_prefill:
            from tileops.kernels.gated_deltanet.gdn_prefill import fused_gdr_fwd

            g_zero = torch.zeros_like(g_cum)
            A = _prefill_blocksolve_A_bthd(k, g_zero, beta, chunk_size)
            initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
                _prefill_partitioned_initial_state_bthd(
                    k=k,
                    v=v,
                    A=A,
                    g=g_cum,
                    beta=beta,
                    chunk_size=chunk_size,
                )
            )
            o, _h, final_state = fused_gdr_fwd(
                q=q,
                k=k,
                v=v,
                a=A,
                g=g_cum,
                b=beta,
                scale=1.0,
                initial_state=initial_state,
                output_final_state=True,
                output_h=False,
                output_o=True,
                cu_seqlens=cu_seqlens,
                cp_seq_map=cp_seq_map,
                raw_cu_seqlens=raw_cu_seqlens,
            )
            return o, final_state.to(q.dtype)

        o_fn = _prefill_output_o_bthd_tl(
            batch, head, seq_len, chunk_size, dim_k, dim_v, dtype
        )(o_threads)
        if use_blocksolve_prepare:
            A = _prefill_blocksolve_A_bthd(k, g_cum, beta, chunk_size)
            recompute_fn = _prefill_recompute_w_u_from_A_bthd_tl(
                batch, head, seq_len, chunk_size, dim_k, dim_v, dtype
            )(128)
            w, u = recompute_fn(k, v, beta, A)
        else:
            prepare_fn = _prefill_prepare_w_u_bthd_tl(
                batch, head, seq_len, chunk_size, dim_k, dim_v, dtype
            )(fused_num_stages, fused_threads)
            w, u = prepare_fn(k, v, g_cum, beta)
        group_chunks = 64
        use_scan_h = (
            batch == 1
            and head == 16
            and seq_len >= 65536
            and chunk_size == 64
            and dim_k == 128
            and dim_v == 128
            and dtype == "float16"
            and seq_len % (chunk_size * group_chunks) == 0
        )
        if use_scan_h:
            scan_block_v = 64
            scan_stages = 2
            scan_threads = 256
            summary_fn = _prefill_group_transition_summary_bthd_tl(
                batch,
                head,
                seq_len,
                chunk_size,
                group_chunks,
                dim_k,
                dim_v,
                dtype,
                block_v=scan_block_v,
            )(scan_stages, scan_threads)
            scan_fn = _prefill_dense_group_start_scan_bthd_tl(
                batch,
                head,
                seq_len // chunk_size // group_chunks,
                dim_k,
                dim_v,
                dtype,
                block_v=scan_block_v,
            )(scan_stages, scan_threads)
            grouped_h_fn = _prefill_grouped_replay_bthd_tl(
                batch,
                head,
                seq_len,
                chunk_size,
                group_chunks,
                dim_k,
                dim_v,
                dtype,
                block_v=scan_block_v,
            )(scan_stages, scan_threads)
            summary = summary_fn(k, g_cum, w, u)
            group_start = scan_fn(summary)
            states, v_new = grouped_h_fn(k, g_cum, w, u, group_start)
        else:
            h_fn = _prefill_h_recurrence_bthd_tl(
                batch,
                head,
                seq_len,
                chunk_size,
                dim_k,
                dim_v,
                dtype,
                block_v=h_block_v,
            )(h_num_stages, h_threads)
            S_0 = torch.zeros(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
            states, v_new = h_fn(k, g_cum, w, u, S_0)
        o = o_fn(q, k, g_cum, states, v_new)
        final_state = states[:, :, -1, :, :].contiguous()
        return o, final_state

    if chunk_size == 64 and batch * head > 64:
        g_cum = _prefill_chunk_local_cumsum_bhtd_tl(
            batch, head, seq_len, chunk_size, dtype
        )(g)
    else:
        g_cum = _chunk_local_cumsum(g.float(), chunk_size).to(g.dtype)
    prepare_fn = _prefill_prepare_w_u_bhtd_tl(
        batch, head, seq_len, chunk_size, dim_k, dim_v, dtype
    )(fused_num_stages, fused_threads)
    h_fn = _h_recurrence_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        block_v=h_block_v,
    )(h_num_stages, h_threads)
    o_fn = _output_o_tl(batch, head, seq_len, chunk_size, dim_k, dim_v, dtype)(
        o_threads
    )
    S_0 = torch.zeros(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    w, u = prepare_fn(k, v, g_cum, beta)
    states, v_new = h_fn(k, g_cum, w, u, S_0)
    o = o_fn(q, k, g_cum, states, v_new)
    final_state = states[:, :, -1, :, :].contiguous()
    return o, final_state


@_gated_deltanet_prefill_wrapped_kernel.register_fake
def _gated_deltanet_prefill_wrapped_kernel_fake(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    fused_num_stages: int,
    fused_threads: int,
    h_num_stages: int,
    h_threads: int,
    h_block_v: int,
    o_threads: int,
    layout: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del dtype, fused_num_stages, fused_threads, h_num_stages, h_threads, h_block_v, o_threads
    del k, v, g, beta
    layout = _normalize_prefill_layout(layout)
    if layout == "bthd":
        o = torch.empty(batch, seq_len, head, dim_v, dtype=q.dtype, device=q.device)
    else:
        o = torch.empty(batch, head, seq_len, dim_v, dtype=q.dtype, device=q.device)
    final_state = torch.empty(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    return o, final_state


class GatedDeltaNetPrefillFwdKernel(Kernel):
    """Gated DeltaNet zero-state prefill."""

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        head: int,
        seq_len: int,
        chunk_size: int,
        dim_k: int,
        dim_v: int,
        dtype: str = "float32",
        config: Optional[dict] = None,
        layout: str = "bhtd",
        tune: bool = False,
    ):
        super().__init__()
        layout = _normalize_prefill_layout(layout)
        self.batch = batch
        self.head = head
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.layout = layout
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        streams = self.batch * self.head
        if self.layout == "bthd":
            if self.chunk_size == 64 and streams >= 64:
                h_block_v = 32
                h_num_stages = 2
                h_threads = 256
            else:
                h_block_v = 16
                h_num_stages = 2
                h_threads = 128
            return {
                "fused_num_stages": 2,
                "fused_threads": 128,
                "h_num_stages": h_num_stages,
                "h_threads": h_threads,
                "h_block_v": h_block_v,
                "o_threads": 256,
            }

        if self.chunk_size >= 128 and 1 < streams <= 8:
            h_block_v = 8
            h_num_stages = 2
            h_threads = 64
        elif self.chunk_size >= 64 and streams <= 4:
            h_block_v = 8
            h_num_stages = 2
            h_threads = 128
        elif self.chunk_size >= 64 and streams <= 48:
            h_block_v = 16
            h_num_stages = 2
            h_threads = 128
        elif self.chunk_size == 64 and streams <= 66:
            h_block_v = 64
            h_num_stages = 3
            h_threads = 256
        elif self.chunk_size >= 64:
            h_block_v = 16
            h_num_stages = 3
            h_threads = 128
        else:
            h_block_v = 0
            h_num_stages = 2
            h_threads = 256
        fused_threads = 128 if self.chunk_size >= 64 else 256
        o_threads = 128 if self.chunk_size == 64 and streams > 64 else 256
        return {
            "fused_num_stages": 2,
            "fused_threads": fused_threads,
            "h_num_stages": h_num_stages,
            "h_threads": h_threads,
            "h_block_v": h_block_v,
            "o_threads": o_threads,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gated_deltanet_prefill_wrapped_kernel(
            self.batch,
            self.head,
            self.seq_len,
            self.chunk_size,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            self.config["fused_num_stages"],
            self.config["fused_threads"],
            self.config["h_num_stages"],
            self.config["h_threads"],
            self.config.get("h_block_v", 0),
            self.config["o_threads"],
            self.layout,
            q,
            k,
            v,
            g,
            beta,
        )
