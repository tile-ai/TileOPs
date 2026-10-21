"""SM90 warp-specialized MLA prefill backward for ``dim_qk = 192``, ``dim_v = 128``.

Fixed-length and packed calls share one program over packed ``(T, H, D)`` tensors; a
fixed-length call is the packed one with every request ``seqlen`` long, and only the
layout of ``lse`` and ``delta`` differs (batch-major ``(B, H, S)`` against head-major
``(H, T)``).

One persistent CTA per SM claims tiles of 96 keys of one head of one request. A producer
warp loads the tile's keys and values once and streams the query blocks its mask admits,
64 rows of Q, dO, lse and delta per step, through a two-stage TMA ring. Two consumer
warpgroups split the tile's keys, 48 each, and per step:

- compute ``S = Q K^T`` and ``dP = dO V^T`` for their keys (64 x 48, M = queries);
- write ``P`` and ``dS = P * (dP - delta)`` to shared memory;
- accumulate ``dV^T += dO^T P`` and ``dK^T += Q^T dS`` (M = head dim): dK and dV live
  transposed in registers, 120 f32 per thread;
- compute their columns of ``dQ = dS K`` over all 96 keys (warpgroup 0: columns 0..127,
  warpgroup 1: 128..191) and stage them in register order, which two more producer warps
  add into an f32 accumulator with bulk reductions.

The 48-wide products issue as ``wgmma m64n48k16`` from ``mla_bwd_wgmma.h``; TileLang
0.1.12 would split each into three n16 instructions. A third launch lands the dQ
accumulator in the layout and dtype of ``q``.

Tiles run longest request first, and within a request in key-block order, the heads of a
group interleaved so the tiles resident together share their query rows in L2.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import (
    TensorCoreIntrinEmitter as WgmmaEmitter,
)
from tilelang.layout import (
    make_quarter_bank_swizzled_layout,
    make_swizzled_layout,
)

from tileops._csrc import csrc_include
from tileops.kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    MLABwdCall,
    MLADenseBwdInterface,
    MLAVarlenBwdInterface,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_sm_count

__all__ = ["MLADenseBwdWSKernel", "MLAVarlenBwdWSKernel"]

# Query rows per step, keys per tile, keys per consumer warpgroup.
_BM, _BN, _HN = 64, 96, 48
_DIM_QK, _DIM_V = 192, 128
# dQ columns of consumer warpgroups 0 and 1: 64-column atoms of K's shared layout.
_DQ_COLS = (128, 64)
# f32 per 64 x 192 dQ tile, and per warpgroup part.
_DQ_TILE = _BM * _DIM_QK
# The race check does not model which thread owns a fragment element, nor that a
# store index such as ``_dq_slot`` or ``row // len`` is injective; it reports the
# register-order dQ stores and the delta and dQ output stores as races.
_NO_RACE_CHECK = {tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True}
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    **_NO_RACE_CHECK,
}


def _dq_slot(i, d):
    """``(reg, thread)`` of element ``(i, d)`` of a 64 x 192 dQ tile in the consumers' WGMMA
    accumulators, for part ``d // 128``: thread ``t`` of the part's warpgroup stores register
    ``r`` at ``[r, t]``, so each store instruction writes 128 consecutive f32."""
    c = d % _DQ_COLS[0]
    t = (i // 16) * 32 + (i % 8) * 4 + (c % 8) // 2
    r = (c // 8) * 4 + ((i % 16) // 8) * 2 + c % 2
    return r, t


def _dq_element(e):
    """``(i, d)`` of offset ``e`` of a dQ accumulator tile, the inverse of :func:`_dq_slot`:
    part 0's ``[64, 128]`` register-by-thread block, then part 1's ``[32, 128]``."""
    r = e // 128 - T.if_then_else(e < _BM * _DQ_COLS[0], 0, _DQ_COLS[0] // 2)
    t = e % 128
    lane = t % 32
    i = (t // 32) * 16 + lane // 4 + ((r % 4) // 2) * 8
    c = (r // 4) * 8 + (lane % 4) * 2 + r % 2
    return i, c + T.if_then_else(e < _BM * _DQ_COLS[0], 0, _DQ_COLS[0])


@functools.lru_cache(maxsize=32)
def _prep_kernel(batch: int, heads: int, dense_seqlen: int, dtype: str) -> Callable:
    """``delta[.., h, row] = sum_d o[row, h, d] * do[row, h, d]``, laid out like ``lse``."""
    block = 32

    @tilelang.jit(out_idx=[2], pass_configs=_NO_RACE_CHECK, compile_flags=["-O3"])
    def _prep_func() -> Callable:
        total = T.dynamic("total_q")
        delta_shape = (batch, heads, dense_seqlen) if dense_seqlen else (heads, total)

        @T.prim_func
        def _prep_main(
            o: T.Tensor((total, heads, _DIM_V), dtype),  # type: ignore
            do: T.Tensor((total, heads, _DIM_V), dtype),  # type: ignore
            delta: T.Tensor(delta_shape, "float32"),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(total, block), heads, threads=128) as (bx, h):
                o_f = T.alloc_fragment([block, _DIM_V], "float32")
                do_f = T.alloc_fragment([block, _DIM_V], "float32")
                rows = T.alloc_fragment([block], "float32")
                T.copy(o[bx * block : (bx + 1) * block, h, :], o_f)
                T.copy(do[bx * block : (bx + 1) * block, h, :], do_f)
                for i, d in T.Parallel(block, _DIM_V):
                    o_f[i, d] = o_f[i, d] * do_f[i, d]
                T.reduce_sum(o_f, rows, 1)
                for i in T.Parallel(block):
                    row = bx * block + i
                    if row < total:
                        if dense_seqlen:
                            delta[row // dense_seqlen, h, row % dense_seqlen] = rows[i]
                        else:
                            delta[h, row] = rows[i]

        return _prep_main

    return _prep_func()


@functools.lru_cache(maxsize=32)
def _dq_land_kernel(batch: int, heads: int, scale: float, dtype: str) -> Callable:
    """``dq = scale * dq_acc``, read from the accumulator's per-request register-order tiles."""

    @tilelang.jit(pass_configs=_NO_RACE_CHECK, compile_flags=["-O3"])
    def _dq_land_func() -> Callable:
        total = T.dynamic("total_q")
        q_tiles = T.dynamic("q_tiles")

        @T.prim_func
        def _dq_land_main(
            dq_acc: T.Tensor((heads, q_tiles, _DQ_TILE), "float32"),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            dq: T.Tensor((total, heads, _DIM_QK), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(q_tiles, heads, threads=256) as (qt, h):
                out_s = T.alloc_shared([_BM, _DIM_QK], dtype)
                # The request owning query tile qt, and the tiles of the requests before it.
                b = T.alloc_var("int32", init=batch)
                before = T.alloc_var("int32", init=0)
                cum = T.alloc_var("int32", init=0)
                for r in T.serial(batch):
                    step = T.ceildiv(cu_seqlens_q[r + 1] - cu_seqlens_q[r], _BM)
                    if (b == batch) & (qt < cum + step):
                        b = r
                        before = cum
                    cum = cum + step
                if b < batch:
                    row0 = (qt - before) * _BM
                    q_start = cu_seqlens_q[b]
                    len_q = cu_seqlens_q[b + 1] - q_start
                    # Read the tile in storage order and turn it into rows in shared memory.
                    for e in T.Parallel(_DQ_TILE):
                        i, d = _dq_element(e)
                        out_s[i, d] = T.cast(dq_acc[h, qt, e] * scale, dtype)
                    for i, d in T.Parallel(_BM, _DIM_QK):
                        if row0 + i < len_q:
                            dq[q_start + row0 + i, h, d] = out_s[i, d]

        return _dq_land_main

    return _dq_land_func()


def _acc_layout(buf, rows: int, first_thread: int, dtype: str):
    """The accumulator layout of a ``rows x 48`` WGMMA over the warpgroup starting at thread
    ``first_thread``: the registers the n48 helpers write, as TileLang lays out its own."""
    base = WgmmaEmitter(
        a_dtype=dtype,
        b_dtype=dtype,
        accum_dtype="float32",
        a_transposed=False,
        b_transposed=True,
        block_row_warps=4,
        block_col_warps=1,
        warp_row_tiles=rows // 4,
        warp_col_tiles=_HN,
        chunk=64,
    ).make_mma_store_layout(buf)
    return T.Fragment(
        [rows, _HN],
        forward_thread_fn=lambda i, j: base.map_forward_thread([i, j])[0] + first_thread,
        forward_index_fn=lambda i, j: base.map_forward_index([i, j]),
    )


@functools.lru_cache(maxsize=32)
def _bwd_ws_kernel(
    batch: int,
    heads: int,
    dense_seqlen: int,
    is_causal: bool,
    scale: float,
    dtype: str,
    num_ctas: int,
    group: int,
) -> Callable:
    """The main backward; ``dense_seqlen`` is the query length of a fixed-length call (its
    key length rides in ``cu_seqlens_kv``), ``0`` for a packed call."""
    accum = "float32"
    scale_log2 = scale * LOG2E
    tag = "bf16" if dtype == "bfloat16" else "fp16"
    c0, c1 = _DQ_COLS
    dense = dense_seqlen > 0
    # Requests are visited longest first: a CTA ranks every request in one pass.
    assert batch <= 256, "requests are ranked one per consumer thread"

    @tilelang.jit(
        pass_configs=_PASS_CONFIGS,
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            *csrc_include("tile_claim.h", "warp_uniform.h", "mla_bwd_wgmma.h"),
        ],
    )
    def _bwd_ws_func() -> Callable:
        total_q = T.dynamic("total_q")
        total_kv = T.dynamic("total_kv")
        q_tiles = T.dynamic("q_tiles")
        stat_shape = (batch, heads, dense_seqlen) if dense else (heads, total_q)

        @T.macro
        def decode(t, kv_cum, order, dec):
            """Tile ``t`` -> ``dec = (head, request, key block)``. Groups of ``group`` heads
            interleave over the (request, key block) pairs, requests longest first."""
            rt = (t // group) % kv_cum[batch]
            dec[0] = (t // (group * kv_cum[batch])) * group + t % group
            dec[1] = 0
            for r in T.serial(batch):
                if kv_cum[r + 1] <= rt:
                    dec[1] = r + 1
            dec[2] = rt - kv_cum[dec[1]]
            dec[1] = order[dec[1]]

        def first_q_block(kb, shift):
            if is_causal:
                return T.max(0, T.floordiv(kb * _BN - shift, _BM))
            return 0

        @T.macro
        def schedule_prologue(cu_q, cu_kv, order, kv_cum, q_cum, tx):
            """Rank the requests by key length, longest first, and prefix-sum their key blocks
            in that order and their query blocks in packed order."""
            if tx < batch:
                mine = cu_kv[tx + 1] - cu_kv[tx]
                rank = T.alloc_var("int32", init=0)
                for r in T.serial(batch):
                    other = cu_kv[r + 1] - cu_kv[r]
                    if (other > mine) | ((other == mine) & (r < tx)):
                        rank = rank + 1
                order[rank] = tx
            T.sync_threads()
            if tx == 0:
                kv_cum[0] = 0
                q_cum[0] = 0
                for r in T.serial(batch):
                    b = order[r]
                    kv_cum[r + 1] = kv_cum[r] + T.ceildiv(cu_kv[b + 1] - cu_kv[b], _BN)
                    q_cum[r + 1] = q_cum[r] + T.ceildiv(cu_q[r + 1] - cu_q[r], _BM)
            T.sync_threads()

        @T.macro
        def stat_rows(src, dst, slot, b, h, n, rows, end, lane):
            """One lane's two of the 64 per-row statistics of a step; rows past the request
            read as zero."""
            for e in T.unroll(2):
                i = lane + 32 * e
                if dense:
                    dst[slot, i] = T.if_then_else(rows + i < end, src[b, h, n * _BM + i], 0.0)
                else:
                    dst[slot, i] = T.if_then_else(rows + i < end, src[h, rows + i], 0.0)

        @T.macro
        def dq_writer(w, dq_s0, dq_s1, dq_acc, cu_q, cu_kv, order, kv_cum, q_cum, tile_s,
                      tile_full, tile_empty, dq_full, dq_empty, tx):  # fmt: skip
            ti = T.alloc_var("int32", init=0)
            g = T.alloc_var("int32", init=0)
            run = T.alloc_var("int32", init=1)
            dec = T.alloc_local([3], "int32")
            while run == 1:
                T.barrier_wait(tile_full[ti & 1], (ti >> 1) & 1)
                t = T.alloc_var("int32")
                t = tile_s[ti & 1]
                T.barrier_arrive(tile_empty[ti & 1])
                if t < 0:
                    run = 0
                else:
                    decode(t, kv_cum, order, dec)
                    h, b, kb = dec[0], dec[1], dec[2]
                    len_q = cu_q[b + 1] - cu_q[b]
                    shift = (cu_kv[b + 1] - cu_kv[b]) - len_q
                    for n in T.serial(first_q_block(kb, shift), T.ceildiv(len_q, _BM)):
                        T.barrier_wait(dq_full[w], g & 1)
                        if tx == 32 * (w + 1):
                            if w == 0:
                                T.call_extern(
                                    "handle",
                                    "tl::tma_store_add",
                                    T.access_ptr(dq_s0[0, 0], "r"),
                                    T.access_ptr(dq_acc[h, q_cum[b] + n, 0], "w"),
                                    _BM * c0 * 4,
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    "tl::tma_store_add",
                                    T.access_ptr(dq_s1[0, 0], "r"),
                                    T.access_ptr(dq_acc[h, q_cum[b] + n, _BM * c0], "w"),
                                    _BM * c1 * 4,
                                )
                            T.tma_store_arrive()
                            T.tma_store_wait(0)
                            T.barrier_arrive(dq_empty[w])
                        g = g + 1
                    ti = ti + 1

        @T.macro
        def ep_chunk(w, ep_s, acc, col0, mul, out, row0, len_kv, key0, h):
            """Stage 64 columns of a transposed accumulator as rows and store them."""
            # Over the whole fragment: a T.Parallel over a window at a nonzero row offset
            # of a fragment is dropped by TileLang 0.1.12 without an error.
            for d, j in T.Parallel(acc.shape[0], _HN):
                if (d >= col0) & (d < col0 + 64):
                    ep_s[w, j, d - col0] = T.cast(acc[d, j] * mul, dtype)
            T.fence_proxy_async()
            T.sync_threads(barrier_id=3 + w, arrive_count=128)
            if key0 + _HN <= len_kv:
                T.copy(ep_s[w, :, :], out[row0 : row0 + _HN, h, col0 : col0 + 64])
            else:
                # The tile ends past its request: a box store would write the next one.
                for j, d in T.Parallel(_HN, 64):
                    if key0 + j < len_kv:
                        out[row0 + j, h, col0 + d] = ep_s[w, j, d]
            T.sync_threads(barrier_id=3 + w, arrive_count=128)

        @T.macro
        def consumer(
            w, k_s, v_s, q_s, do_s, lse_s, dl_s, p_s, ds_s, dq_s0, dq_s1, ep_s, cu_q, cu_kv,
            order, kv_cum, tile_s, tile_full, tile_empty, kv_full, kv_empty, q_full, do_full,
            q_empty, do_empty, ds_ready, ds_free, dq_full, dq_empty, dk, dv,
        ):  # fmt: skip
            s = T.alloc_fragment([_BM, _HN], accum)
            dp = T.alloc_fragment([_BM, _HN], accum)
            dvt = T.alloc_fragment([_DIM_V, _HN], accum)
            dkt = T.alloc_fragment([_DIM_QK, _HN], accum)
            dq_f = T.alloc_fragment([_BM, 64], accum)
            dq_g = T.alloc_fragment([_BM, 64], accum)
            first = 128 * (w + 1)
            T.annotate_layout(
                {
                    s: _acc_layout(s, _BM, first, dtype),
                    dp: _acc_layout(dp, _BM, first, dtype),
                    dvt: _acc_layout(dvt, _DIM_V, first, dtype),
                    dkt: _acc_layout(dkt, _DIM_QK, first, dtype),
                }
            )
            ti = T.alloc_var("int32", init=0)
            g = T.alloc_var("int32", init=0)
            run = T.alloc_var("int32", init=1)
            dec = T.alloc_local([3], "int32")
            while run == 1:
                T.barrier_wait(tile_full[ti & 1], (ti >> 1) & 1)
                t = T.alloc_var("int32")
                t = tile_s[ti & 1]
                T.barrier_arrive(tile_empty[ti & 1])
                if t < 0:
                    run = 0
                else:
                    decode(t, kv_cum, order, dec)
                    h, b, kb = dec[0], dec[1], dec[2]
                    kv_start = cu_kv[b]
                    len_q = cu_q[b + 1] - cu_q[b]
                    len_kv = cu_kv[b + 1] - kv_start
                    shift = len_kv - len_q
                    key0 = kb * _BN + w * _HN
                    T.clear(dvt)
                    T.clear(dkt)
                    T.barrier_wait(kv_full, ti & 1)
                    for n in T.serial(first_q_block(kb, shift), T.ceildiv(len_q, _BM)):
                        slot = g % 2
                        phase = (g >> 1) & 1
                        q0 = n * _BM
                        T.barrier_wait(q_full[slot], phase)
                        T.call_extern(
                            "handle",
                            f"tileops::mla_bwd_s_{tag}",
                            s.data,
                            T.address_of(q_s[slot, 0, 0]),
                            T.address_of(k_s[w, 0, 0]),
                            0,
                        )
                        T.barrier_wait(do_full[slot], phase)
                        T.call_extern(
                            "handle",
                            f"tileops::mla_bwd_dp_{tag}",
                            dp.data,
                            T.address_of(do_s[slot, 0, 0]),
                            T.address_of(v_s[w, 0, 0]),
                            0,
                        )
                        T.wait_wgmma(1)
                        T.warpgroup_fence_operand(s, num_regs=24)
                        for i, j in T.Parallel(_BM, _HN):
                            key = key0 + j
                            row = q0 + i
                            in_range = (key < len_kv) & (row < len_q)
                            visible = in_range & (key <= row + shift) if is_causal else in_range
                            s[i, j] = T.if_then_else(
                                visible,
                                T.exp2(s[i, j] * scale_log2 - lse_s[slot, i] * LOG2E),
                                0,
                            )
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(dp, num_regs=24)
                        # Both warpgroups' previous dQ has read this warpgroup's dS.
                        T.barrier_wait(ds_free[w], (g + 1) & 1)
                        for i, j in T.Parallel(_BM, _HN):
                            dp[i, j] = s[i, j] * (dp[i, j] - dl_s[slot, i])
                        T.call_extern(
                            "handle",
                            f"tileops::mla_bwd_stsm_{tag}",
                            s.data,
                            T.address_of(p_s[w, 0, 0]),
                            T.get_thread_binding() - first,
                        )
                        T.call_extern(
                            "handle",
                            f"tileops::mla_bwd_stsm_{tag}",
                            dp.data,
                            T.address_of(ds_s[w, 0, 0]),
                            T.get_thread_binding() - first,
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=3 + w, arrive_count=128)
                        T.barrier_arrive(ds_ready[w])
                        T.call_extern(
                            "handle",
                            f"tileops::mla_bwd_dvt_{tag}",
                            dvt.data,
                            T.address_of(do_s[slot, 0, 0]),
                            T.address_of(p_s[w, 0, 0]),
                        )
                        T.barrier_wait(ds_ready[1 - w], g & 1)
                        if w == 0:
                            # Warpgroup 0's 128 columns as two 64-column atoms of K.
                            T.wgmma_gemm(
                                ds_s[0, :, :],
                                k_s[0, :, 0:64],
                                dq_f,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=True,
                            )
                            T.wgmma_gemm(
                                ds_s[1, :, :],
                                k_s[1, :, 0:64],
                                dq_f,
                                policy=T.GemmWarpPolicy.FullRow,
                            )
                            T.wgmma_gemm(
                                ds_s[0, :, :],
                                k_s[0, :, 64:128],
                                dq_g,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=True,
                            )
                            T.wgmma_gemm(
                                ds_s[1, :, :],
                                k_s[1, :, 64:128],
                                dq_g,
                                policy=T.GemmWarpPolicy.FullRow,
                            )
                        else:
                            T.wgmma_gemm(
                                ds_s[0, :, :],
                                k_s[0, :, c0:_DIM_QK],
                                dq_f,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=True,
                            )
                            T.wgmma_gemm(
                                ds_s[1, :, :],
                                k_s[1, :, c0:_DIM_QK],
                                dq_f,
                                policy=T.GemmWarpPolicy.FullRow,
                            )
                        # dK runs while dQ is staged.
                        T.call_extern(
                            "handle",
                            f"tileops::mla_bwd_dkt_{tag}",
                            dkt.data,
                            T.address_of(q_s[slot, 0, 0]),
                            T.address_of(ds_s[w, 0, 0]),
                        )
                        T.wait_wgmma(1)
                        T.warpgroup_fence_operand(dvt, num_regs=48)
                        T.warpgroup_fence_operand(dq_f, num_regs=32)
                        if w == 0:
                            T.warpgroup_fence_operand(dq_g, num_regs=32)
                        T.barrier_wait(dq_empty[w], (g + 1) & 1)
                        if w == 0:
                            for i, d in T.Parallel(_BM, 64):
                                r, tt = _dq_slot(i, d)
                                dq_s0[r, tt] = dq_f[i, d]
                            for i, d in T.Parallel(_BM, 64):
                                r, tt = _dq_slot(i, d + 64)
                                dq_s0[r, tt] = dq_g[i, d]
                        else:
                            for i, d in T.Parallel(_BM, c1):
                                r, tt = _dq_slot(i, d)
                                dq_s1[r, tt] = dq_f[i, d]
                        T.fence_proxy_async()
                        T.barrier_arrive(dq_full[w])
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(dkt, num_regs=72)
                        T.barrier_arrive(ds_free[0])
                        T.barrier_arrive(ds_free[1])
                        T.barrier_arrive(q_empty[slot])
                        T.barrier_arrive(do_empty[slot])
                        g = g + 1
                    # K and V are free: the next tile's load overlaps this epilogue.
                    T.barrier_arrive(kv_empty)
                    row0 = kv_start + key0
                    for c in T.unroll(5):
                        if c == 0:
                            ep_chunk(w, ep_s, dkt, 0, scale, dk, row0, len_kv, key0, h)
                        elif c == 1:
                            ep_chunk(w, ep_s, dkt, 64, scale, dk, row0, len_kv, key0, h)
                        elif c == 2:
                            ep_chunk(w, ep_s, dkt, 128, scale, dk, row0, len_kv, key0, h)
                        elif c == 3:
                            ep_chunk(w, ep_s, dvt, 0, 1.0, dv, row0, len_kv, key0, h)
                        else:
                            ep_chunk(w, ep_s, dvt, 64, 1.0, dv, row0, len_kv, key0, h)
                    ti = ti + 1

        @T.prim_func
        def _bwd_ws_main(
            q: T.Tensor((total_q, heads, _DIM_QK), dtype),  # type: ignore
            k: T.Tensor((total_kv, heads, _DIM_QK), dtype),  # type: ignore
            v: T.Tensor((total_kv, heads, _DIM_V), dtype),  # type: ignore
            do: T.Tensor((total_q, heads, _DIM_V), dtype),  # type: ignore
            lse: T.Tensor(stat_shape, accum),  # type: ignore
            delta: T.Tensor(stat_shape, accum),  # type: ignore
            cu_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
            dq_acc: T.Tensor((heads, q_tiles, _DQ_TILE), accum),  # type: ignore
            dk: T.Tensor((total_kv, heads, _DIM_QK), dtype),  # type: ignore
            dv: T.Tensor((total_kv, heads, _DIM_V), dtype),  # type: ignore
            sched: T.Tensor([2], "int32"),  # type: ignore
        ) -> None:
            with T.Kernel(num_ctas, threads=384) as bid:
                k_s = T.alloc_shared([2, _HN, _DIM_QK], dtype)
                v_s = T.alloc_shared([2, _HN, _DIM_V], dtype)
                q_s = T.alloc_shared([2, _BM, _DIM_QK], dtype)
                do_s = T.alloc_shared([2, _BM, _DIM_V], dtype)
                lse_s = T.alloc_shared([2, _BM], accum)
                dl_s = T.alloc_shared([2, _BM], accum)
                p_s = T.alloc_shared([2, _BM, _HN], dtype)
                ds_s = T.alloc_shared([2, _BM, _HN], dtype)
                dq_s0 = T.alloc_shared([c0 // 2, 128], accum)
                dq_s1 = T.alloc_shared([c1 // 2, 128], accum)
                ep_s = T.alloc_shared([2, _HN, 64], dtype)
                tile_s = T.alloc_shared([2], "int32")
                order = T.alloc_shared([batch], "int32")
                kv_cum = T.alloc_shared([batch + 1], "int32")
                q_cum = T.alloc_shared([batch + 1], "int32")
                T.annotate_layout(
                    {
                        k_s: make_swizzled_layout(k_s),
                        v_s: make_swizzled_layout(v_s),
                        q_s: make_swizzled_layout(q_s),
                        do_s: make_swizzled_layout(do_s),
                        p_s: make_quarter_bank_swizzled_layout(p_s),
                        ds_s: make_quarter_bank_swizzled_layout(ds_s),
                        ep_s: make_swizzled_layout(ep_s),
                    }
                )

                tile_full = T.alloc_barrier(arrive_count=[32, 32])
                tile_empty = T.alloc_barrier(arrive_count=[320, 320])
                kv_full = T.alloc_barrier(arrive_count=32)
                kv_empty = T.alloc_barrier(arrive_count=256)
                q_full = T.alloc_barrier(arrive_count=[32, 32])
                do_full = T.alloc_barrier(arrive_count=[32, 32])
                q_empty = T.alloc_barrier(arrive_count=[256, 256])
                do_empty = T.alloc_barrier(arrive_count=[256, 256])
                ds_ready = T.alloc_barrier(arrive_count=[128, 128])
                ds_free = T.alloc_barrier(arrive_count=[256, 256])
                dq_full = T.alloc_barrier(arrive_count=[128, 128])
                dq_empty = T.alloc_barrier(arrive_count=[1, 1])
                tx = T.get_thread_binding()
                schedule_prologue(cu_q, cu_kv, order, kv_cum, q_cum, tx)
                n_tiles = heads * kv_cum[batch]

                if tx < 128:
                    T.dec_max_nreg(24)
                    if tx < 32:
                        tcur = T.alloc_var("int32", init=bid)
                        ti = T.alloc_var("int32", init=0)
                        g = T.alloc_var("int32", init=0)
                        dec = T.alloc_local([3], "int32")
                        while tcur < n_tiles:
                            T.barrier_wait(tile_empty[ti & 1], ((ti >> 1) + 1) & 1)
                            if tx == 0:
                                tile_s[ti & 1] = tcur
                            T.barrier_arrive(tile_full[ti & 1])
                            decode(tcur, kv_cum, order, dec)
                            h, b, kb = dec[0], dec[1], dec[2]
                            q_start = cu_q[b]
                            kv_start = cu_kv[b]
                            len_q = cu_q[b + 1] - q_start
                            shift = (cu_kv[b + 1] - kv_start) - len_q
                            T.barrier_wait(kv_empty, (ti + 1) & 1)
                            for hh in T.unroll(2):
                                rows = kv_start + kb * _BN + hh * _HN
                                T.tma_copy(
                                    k[rows : rows + _HN, h, :], k_s[hh, :, :], barrier=kv_full
                                )
                                T.tma_copy(
                                    v[rows : rows + _HN, h, :], v_s[hh, :, :], barrier=kv_full
                                )
                            T.barrier_arrive(kv_full)
                            for n in T.serial(first_q_block(kb, shift), T.ceildiv(len_q, _BM)):
                                slot = g % 2
                                rows = q_start + n * _BM
                                T.barrier_wait(q_empty[slot], ((g >> 1) + 1) & 1)
                                T.tma_copy(
                                    q[rows : rows + _BM, h, :],
                                    q_s[slot, :, :],
                                    barrier=q_full[slot],
                                )
                                # lse and delta rows are 4 bytes apart and a request starts
                                # anywhere, so TMA's 16-byte stride rule does not hold: each
                                # lane loads two, released to the consumers by its arrival.
                                stat_rows(lse, lse_s, slot, b, h, n, rows, q_start + len_q, tx)
                                T.barrier_arrive(q_full[slot])
                                T.barrier_wait(do_empty[slot], ((g >> 1) + 1) & 1)
                                T.tma_copy(
                                    do[rows : rows + _BM, h, :],
                                    do_s[slot, :, :],
                                    barrier=do_full[slot],
                                )
                                stat_rows(delta, dl_s, slot, b, h, n, rows, q_start + len_q, tx)
                                T.barrier_arrive(do_full[slot])
                                g = g + 1
                            tcur = T.call_extern(
                                "int32", "tileops::claim_tile", T.access_ptr(sched[0], "rw")
                            )
                            ti = ti + 1
                        # A negative tile tells every role the list is done.
                        T.barrier_wait(tile_empty[ti & 1], ((ti >> 1) + 1) & 1)
                        if tx == 0:
                            tile_s[ti & 1] = -1
                        T.barrier_arrive(tile_full[ti & 1])
                    elif tx < 64:
                        dq_writer(0, dq_s0, dq_s1, dq_acc, cu_q, cu_kv, order, kv_cum, q_cum,
                                  tile_s, tile_full, tile_empty, dq_full, dq_empty, tx)  # fmt: skip
                    elif tx < 96:
                        dq_writer(1, dq_s0, dq_s1, dq_acc, cu_q, cu_kv, order, kv_cum, q_cum,
                                  tile_s, tile_full, tile_empty, dq_full, dq_empty, tx)  # fmt: skip
                elif tx < 256:
                    T.inc_max_nreg(240)
                    consumer(
                        0, k_s, v_s, q_s, do_s, lse_s, dl_s, p_s, ds_s, dq_s0, dq_s1, ep_s, cu_q,
                        cu_kv, order, kv_cum, tile_s, tile_full, tile_empty, kv_full, kv_empty,
                        q_full, do_full, q_empty, do_empty, ds_ready, ds_free, dq_full, dq_empty,
                        dk, dv,
                    )  # fmt: skip
                else:
                    T.inc_max_nreg(240)
                    consumer(
                        1, k_s, v_s, q_s, do_s, lse_s, dl_s, p_s, ds_s, dq_s0, dq_s1, ep_s, cu_q,
                        cu_kv, order, kv_cum, tile_s, tile_full, tile_empty, kv_full, kv_empty,
                        q_full, do_full, q_empty, do_empty, ds_ready, ds_free, dq_full, dq_empty,
                        dk, dv,
                    )  # fmt: skip
                T.sync_threads()
                if tx == 0:
                    T.call_extern("handle", "tileops::retire", T.access_ptr(sched[0], "rw"))

        return _bwd_ws_main

    return _bwd_ws_func()


class _MLABwdWSKernelBase(Kernel):
    """The three launches of one backward call over packed tensors."""

    supported_archs: list[int] = [90]

    @classmethod
    def refusal(cls, call: MLABwdCall) -> Optional[str]:
        if (call.dim_qk, call.dim_v) != (_DIM_QK, _DIM_V):
            return (
                f"serves dim_qk {_DIM_QK} and dim_v {_DIM_V}, the decompressed MLA head, got "
                f"{call.dim_qk} and {call.dim_v}"
            )
        if call.dtype not in ATTENTION_DTYPES:
            return f"runs float16 and bfloat16, got {call.dtype}"
        if call.batch > 256:
            return f"ranks at most 256 requests per call, got {call.batch}"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: MLABwdCall) -> Entry:
        args = dict(
            batch=call.batch,
            heads=call.heads,
            seqlen_q=call.seqlen_q,
            seqlen_kv=call.seqlen_kv,
            is_causal=call.is_causal,
            sm_scale=call.sm_scale,
            dtype=call.dtype,
            device_index=call.device.index if call.device is not None else None,
        )
        return tuple(args.items()), lambda: cls(**args)

    def __init__(
        self,
        batch: int,
        heads: int,
        seqlen_q: int,
        seqlen_kv: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        config: Optional[dict] = None,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch, self.heads = batch, heads
        self.seqlen_q, self.seqlen_kv = seqlen_q, seqlen_kv
        self.is_causal = is_causal
        self.dtype = dtype
        self.scale = _DIM_QK**-0.5 if sm_scale is None else sm_scale
        self.num_sms = get_sm_count(device_index)
        self.device = torch.device("cuda", device_index) if device_index is not None else None
        # [next tile, CTAs retired]; the kernel returns both to zero.
        self.sched = torch.zeros(2, dtype=torch.int32, device=self.device or "cuda")
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {}

    def _group(self, kv_blocks: int) -> int:
        """Heads interleaved per group: the largest divisor of ``heads`` whose key blocks of
        one request fill at most one wave of CTAs."""
        fit = max(1, self.num_sms // max(1, kv_blocks))
        return max(g for g in range(1, min(fit, self.heads) + 1) if self.heads % g == 0)

    def _run(self, q, k, v, o, do, lse, cu_q, cu_kv, dense_seqlen, kv_blocks):
        total_q = q.shape[0]
        delta = _prep_kernel(self.batch, self.heads, dense_seqlen, self.dtype_str)(o, do)
        q_tiles = total_q // _BM + self.batch
        dq_acc = torch.zeros(self.heads, q_tiles, _DQ_TILE, dtype=torch.float32, device=q.device)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        group = self._group(kv_blocks)
        tiles_bound = self.heads * (k.shape[0] // _BN + self.batch)
        main = _bwd_ws_kernel(
            self.batch,
            self.heads,
            dense_seqlen,
            self.is_causal,
            self.scale,
            self.dtype_str,
            min(self.num_sms, tiles_bound),
            group,
        )
        main(q, k, v, do, lse, delta, cu_q, cu_kv, dq_acc, dk, dv, self.sched)
        dq = torch.empty_like(q)
        _dq_land_kernel(self.batch, self.heads, self.scale, self.dtype_str)(dq_acc, cu_q, dq)
        return dq, dk, dv


class MLADenseBwdWSKernel(_MLABwdWSKernelBase, MLADenseBwdInterface):
    """Warp-specialized MLA prefill backward over fixed-length requests (BSHD)."""

    def forward(self, q, k, v, o, do, lse):
        self._require_devices(q=q, k=k, v=v, o=o, do=do, lse=lse)
        batch, seqlen_q, heads, _ = q.shape
        seqlen_kv = k.shape[1]
        cu_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
        cu_kv = torch.arange(
            0, (batch + 1) * seqlen_kv, seqlen_kv, dtype=torch.int32, device=q.device
        )
        dq, dk, dv = self._run(
            q.view(batch * seqlen_q, heads, -1),
            k.view(batch * seqlen_kv, heads, -1),
            v.view(batch * seqlen_kv, heads, -1),
            o.view(batch * seqlen_q, heads, -1),
            do.view(batch * seqlen_q, heads, -1),
            lse,
            cu_q,
            cu_kv,
            seqlen_q,
            -(-seqlen_kv // _BN),
        )
        return dq.view(q.shape), dk.view(k.shape), dv.view(v.shape)


class MLAVarlenBwdWSKernel(_MLABwdWSKernelBase, MLAVarlenBwdInterface):
    """Warp-specialized MLA prefill backward over packed requests (THD)."""

    def forward(self, q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv):
        self._require_devices(q=q, k=k, v=v, o=o, do=do, lse=lse)
        kv_blocks = -(-k.shape[0] // (_BN * self.batch))
        return self._run(q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv, 0, kv_blocks)
