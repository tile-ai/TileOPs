import functools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops.kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    AttentionCall,
    GQABwdInterface,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_sm_count

__all__ = ["GQABwdWSKernel"]


@functools.lru_cache(maxsize=32)
def _gqa_bwd_ws_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    heads_per_tile: int,
    fill: int,
    num_ctas: int,
    block_m: int,
    block_n: int,
    dq_row: int,
    dtype: str,
) -> Callable:
    sm_scale = dim**-0.5
    scale = sm_scale * LOG2E
    accum_dtype = "float"
    half_m, half_d = block_m // 2, dim // 2
    kv_blocks = seq_len // block_m
    dq_rows = block_n * half_d // dq_row
    group = heads // heads_kv
    # Query-head chunks per KV head: one chunk keeps a key block's dK and dV in registers to
    # the end; several add f32 partials that the last one to land writes out.
    chunks = group // heads_per_tile
    units = batch * heads_kv * chunks
    n_tiles = units * kv_blocks
    n_ed = seq_len // block_n
    # Bytes of one f32 half tile a warpgroup stages.
    half_bytes = dq_rows * dq_row * 4
    # Every tile runs an even number of steps per head, so the global step count starts
    # each head's steps even and the ring slot can be read from the query block index.
    assert seq_len % block_m == 0

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16", *csrc_include("tile_claim.h", "wgmma_rows.h")],
    )
    def _gqa_bwd_ws_func() -> Callable:
        def _dq_slot(i, j):
            """Where element ``(i, j)`` of a warpgroup's ``[block_n, dim // 2]`` dQ half sits in
            its f32 accumulator tile.

            Each warp stores four consecutive f32 of the WGMMA accumulator per thread, 512
            contiguous bytes per instruction, so the store to shared memory is free of bank
            conflicts. Returns ``(row, col)`` of a ``[block_n * dim // 2 // dq_row, dq_row]``
            tile.
            """
            lane = (i % 8) * 4 + (j % 8) // 2
            return (i // 16) * 8 + j // 8, lane * 4 + ((i % 16) // 8) * 2 + j % 2

        def _acc_layout(w):
            """Consumer warpgroup ``w``'s m64n128 WGMMA accumulator as the PTX arranges it: row
            i, column c at register 4 (c / 8) + 2 ((i % 16) / 8) + c % 2 of the warpgroup's
            thread 32 (i / 16) + 4 (i % 8) + (c % 8) / 2."""

            def forward_fn(i, c):
                thread = 128 * (w + 1) + (i // 16) * 32 + (i % 8) * 4 + (c % 8) // 2
                return thread, (c // 8) * 4 + ((i % 16) // 8) * 2 + c % 2

            return tilelang.layout.Fragment([half_m, dim], forward_fn=forward_fn)

        def _tile(t):
            """The key block, batch, KV head and first query head of tile ``t``."""
            kb = (t % (fill * kv_blocks)) // fill
            u = t // (fill * kv_blocks) * fill + t % fill
            hk = (u // chunks) % heads_kv
            return kb, u // (heads_kv * chunks), hk, hk * group + (u % chunks) * heads_per_tile

        shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        # dV and dK partials by (KV head key block * 2 + dV / dK, warpgroup, column half).
        dkv_shape = (
            (batch * heads_kv * kv_blocks * 2, 2, 2, dq_rows, dq_row) if chunks > 1 else (1,)
        )

        @T.macro
        def dq_writer(w, dq_s, dq, tile_s, tile_full, tile_empty, dq_full, dq_empty, tx):
            ti = T.alloc_var("int32", init=0)
            g = T.alloc_var("int32", init=0)
            run = T.alloc_var("int32", init=1)
            while run == 1:
                T.barrier_wait(tile_full[ti & 1], (ti >> 1) & 1)
                t = T.alloc_var("int32")
                t = tile_s[ti & 1]
                T.barrier_arrive(tile_empty[ti & 1])
                if t >= n_tiles:
                    run = 0
                else:
                    kb, b, _, h0 = _tile(t)
                    n_st = kb * (block_m // block_n) if is_causal else 0
                    for hj in T.serial(heads_per_tile):
                        for n in T.serial(n_st, n_ed):
                            slot = n % 2
                            T.barrier_wait(dq_full[w * 2 + slot], (g >> 1) & 1)
                            if tx == 32 * (w + 1):
                                T.call_extern(
                                    "handle",
                                    "tl::tma_store_add",
                                    T.access_ptr(dq_s[slot, w, 0, 0], "r"),
                                    T.access_ptr(dq[b * heads + h0 + hj, n, w, 0, 0], "w"),
                                    half_bytes,
                                )
                                T.tma_store_arrive()
                                T.tma_store_wait(0)
                                T.barrier_arrive(dq_empty[w * 2 + slot])
                            g = g + 1
                    ti = ti + 1

        @T.macro
        def add_dkv_partial(w, kind, acc, dq_s, dkv_g, kvt, sender):
            """Stage warpgroup ``w``'s f32 ``acc`` (dV for kind 0, dK for kind 1) in its two dQ
            stages and add it into the key block's accumulator."""
            for i, jj in T.Parallel(half_m, dim):
                r, c = _dq_slot(i, jj % half_d)
                dq_s[jj // half_d, w, r, c] = acc[i, jj]
            T.fence_proxy_async()
            T.sync_threads(barrier_id=3 + w, arrive_count=128)
            if sender:
                # The dQ writer reads dq_s through a raw pointer; T.atomic_add(use_tma=True)
                # would give dq_s a swizzled layout, and fails to lower on a linear one.
                for ch in T.unroll(2):
                    T.call_extern(
                        "handle",
                        "tl::tma_store_add",
                        T.access_ptr(dq_s[ch, w, 0, 0], "r"),
                        T.access_ptr(dkv_g[kvt * 2 + kind, w, ch, 0, 0], "w"),
                        half_bytes,
                    )
                T.tma_store_arrive()
                T.tma_store_wait(0)
            T.sync_threads(barrier_id=3 + w, arrive_count=128)

        @T.macro
        def write_dkv(dv_acc, dk_acc, dk, dv, b, rows, hk):
            """Write a consumer warpgroup's 64 rows of dV and dK straight from registers."""
            # T.copy from this fragment stores 4 bytes per lane and row; the helper exchanges
            # values within each lane quad so every lane stores 16 contiguous bytes.
            T.call_extern(
                "handle",
                "tileops::store_acc_rows",
                T.address_of(dv_acc[0, 0]),
                T.access_ptr(dv[b, rows, hk, 0], "w"),
                heads_kv * dim,
            )
            T.call_extern(
                "handle",
                "tileops::store_acc_rows",
                T.address_of(dk_acc[0, 0]),
                T.access_ptr(dk[b, rows, hk, 0], "w"),
                heads_kv * dim,
            )

        @T.macro
        def reload_dkv(w, kind, acc, dkv_g, kvt):
            """Load the summed dV (kind 0) or dK (kind 1) of warpgroup ``w`` into ``acc`` and
            zero the global sum; each thread reads and clears the floats it added."""
            for i, jj in T.Parallel(half_m, dim):
                r, c = _dq_slot(i, jj % half_d)
                acc[i, jj] = dkv_g[kvt * 2 + kind, w, jj // half_d, r, c]
                dkv_g[kvt * 2 + kind, w, jj // half_d, r, c] = T.float32(0)

        @T.macro
        def consumer(
            w, k_s, v_s, q_s, do_s, lse_s, dl_s, ds_s, dq_s, tile_s, tile_full, tile_empty,
            kv_full, kv_empty, q_full, do_full, q_empty, do_empty, ds_ready, dq_full, dq_empty, dk, dv,
            dkv_g, dkv_cnt, last_s, tx,
        ):  # fmt: skip
            s = T.alloc_fragment([half_m, block_n], accum_dtype)
            dp = T.alloc_fragment([half_m, block_n], accum_dtype)
            p16 = T.alloc_fragment([half_m, block_n], dtype)
            ds16 = T.alloc_fragment([half_m, block_n], dtype)
            dv_acc = T.alloc_fragment([half_m, dim], accum_dtype)
            dk_acc = T.alloc_fragment([half_m, dim], accum_dtype)
            # store_acc_rows reads dV and dK in the WGMMA accumulator's own arrangement.
            T.annotate_layout({dv_acc: _acc_layout(w), dk_acc: _acc_layout(w)})
            dq_acc = T.alloc_fragment([block_n, half_d], accum_dtype)
            lse_r = T.alloc_fragment([half_m, block_n], accum_dtype)
            dl_r = T.alloc_fragment([half_m, block_n], accum_dtype)
            ti = T.alloc_var("int32", init=0)
            g = T.alloc_var("int32", init=0)
            run = T.alloc_var("int32", init=1)
            while run == 1:
                T.barrier_wait(tile_full[ti & 1], (ti >> 1) & 1)
                t = T.alloc_var("int32")
                t = tile_s[ti & 1]
                T.barrier_arrive(tile_empty[ti & 1])
                if t >= n_tiles:
                    run = 0
                else:
                    kb, b, hk, _ = _tile(t)
                    row0 = kb * block_m
                    n_st = kb * (block_m // block_n) if is_causal else 0
                    T.clear(dv_acc)
                    T.clear(dk_acc)
                    T.barrier_wait(kv_full, ti & 1)
                    for _ in T.serial(heads_per_tile):
                        for n in T.serial(n_st, n_ed):
                            slot = n % 2
                            phase = (g >> 1) & 1
                            # The transposed scores and dP of this warpgroup's 64 key rows.
                            T.barrier_wait(q_full[slot], phase)
                            T.wgmma_gemm(
                                k_s[w, :, :],
                                q_s[slot, :, :],
                                s,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=True,
                            )
                            T.barrier_wait(do_full[slot], phase)
                            T.wgmma_gemm(
                                v_s[w, :, :],
                                do_s[slot, :, :],
                                dp,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=True,
                            )
                            # Read while the scores are in flight: an LDS behind the WGMMA
                            # operand reads would stall the exponentials.
                            for i, j in T.Parallel(half_m, block_n):
                                lse_r[i, j] = lse_s[slot, j]
                                dl_r[i, j] = dl_s[slot, j]
                            # An accumulator is read only after a wait that covers its WGMMA;
                            # a read of an in-flight one makes ptxas serialize every WGMMA in
                            # the kernel.
                            T.wait_wgmma(1)
                            T.warpgroup_fence_operand(s, num_regs=32)
                            for i, j in T.Parallel(half_m, block_n):
                                s[i, j] = s[i, j] * scale - lse_r[i, j]
                            if is_causal and n * block_n < row0 + (w + 1) * half_m:
                                for i, j in T.Parallel(half_m, block_n):
                                    s[i, j] = T.if_then_else(
                                        row0 + w * half_m + i <= n * block_n + j,
                                        s[i, j],
                                        -T.infinity(accum_dtype),
                                    )
                            # Half the exponentials run while dP is in flight; the other half
                            # interleave with the dS math, so the MUFU and FMA pipes overlap.
                            for i, j in T.Parallel(half_m, block_n):
                                if j < block_n // 2:
                                    s[i, j] = T.exp2(s[i, j])
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(dp, num_regs=32)
                            # dS is rounded to the input dtype after the softmax scale, which
                            # delta arrives multiplied by.
                            for i, j in T.Parallel(half_m, block_n):
                                if j >= block_n // 2:
                                    s[i, j] = T.exp2(s[i, j])
                                ds16[i, j] = s[i, j] * (dp[i, j] * sm_scale - dl_r[i, j])
                            T.copy(s, p16)
                            T.copy(ds16, ds_s[slot * 2 + w, :, :])
                            T.fence_proxy_async()
                            T.barrier_arrive(ds_ready[w])
                            T.wgmma_gemm(
                                p16, do_s[slot, :, :], dv_acc, policy=T.GemmWarpPolicy.FullRow
                            )
                            # dQ contracts over all 128 key rows, so each warpgroup reads both
                            # halves of dS and computes half of dQ's columns.
                            T.barrier_wait(ds_ready[1 - w], g & 1)
                            T.wgmma_gemm(
                                ds_s[slot * 2, :, :],
                                k_s[0, :, w * half_d : (w + 1) * half_d],
                                dq_acc,
                                transpose_A=True,
                                policy=T.GemmWarpPolicy.FullRow,
                                clear_accum=True,
                            )
                            T.wgmma_gemm(
                                ds_s[slot * 2 + 1, :, :],
                                k_s[1, :, w * half_d : (w + 1) * half_d],
                                dq_acc,
                                transpose_A=True,
                                policy=T.GemmWarpPolicy.FullRow,
                            )
                            T.wait_wgmma(2)
                            T.warpgroup_fence_operand(dv_acc, num_regs=64)
                            # dO's last reader, dV, is done: the producer can refill its stage
                            # while dK and dQ run.
                            T.barrier_arrive(do_empty[slot])
                            T.wgmma_gemm(
                                ds16, q_s[slot, :, :], dk_acc, policy=T.GemmWarpPolicy.FullRow
                            )
                            T.wait_wgmma(1)
                            T.warpgroup_fence_operand(dq_acc, num_regs=32)
                            # dK runs while dQ is written out.
                            T.barrier_wait(dq_empty[w * 2 + slot], ((g >> 1) + 1) & 1)
                            for i, j in T.Parallel(block_n, half_d):
                                r, c = _dq_slot(i, j)
                                dq_s[slot, w, r, c] = dq_acc[i, j]
                            T.fence_proxy_async()
                            T.barrier_arrive(dq_full[w * 2 + slot])
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(dk_acc, num_regs=64)
                            T.barrier_arrive(q_empty[slot])
                            g = g + 1
                    # K and V are free: the next tile's load overlaps this epilogue.
                    T.barrier_arrive(kv_empty)
                    rows = row0 + w * half_m
                    if chunks == 1:
                        write_dkv(dv_acc, dk_acc, dk, dv, b, rows, hk)
                    else:
                        # The two dQ stages are free once their writer has read them; they
                        # stage this warpgroup's f32 partials of dV, then dK.
                        T.barrier_wait(dq_empty[w * 2 + g % 2], ((g >> 1) + 1) & 1)
                        T.barrier_wait(dq_empty[w * 2 + (g + 1) % 2], (((g + 1) >> 1) + 1) & 1)
                        kvt = (b * heads_kv + hk) * kv_blocks + kb
                        sender = tx == 128 * (w + 1)
                        add_dkv_partial(w, 0, dv_acc, dq_s, dkv_g, kvt, sender)
                        add_dkv_partial(w, 1, dk_acc, dq_s, dkv_g, kvt, sender)
                        # Both warpgroups' partials are complete in global memory before the
                        # chunk counts itself once; the last chunk returns the count to zero.
                        if sender:
                            T.tma_store_wait(0, read=False)
                            T.fence_proxy_async()
                        T.sync_threads(barrier_id=2, arrive_count=256)
                        if tx == 128:
                            prev = T.atomic_add(
                                dkv_cnt[kvt], 1, memory_order="acq_rel", return_prev=True
                            )
                            last_s[0] = T.if_then_else(prev == chunks - 1, 1, 0)
                            if prev == chunks - 1:
                                dkv_cnt[kvt] = 0
                        T.sync_threads(barrier_id=2, arrive_count=256)
                        if last_s[0] != 0:
                            # The last chunk reads the sums back into its accumulators, leaves
                            # the global ones zeroed and writes dV and dK out.
                            reload_dkv(w, 0, dv_acc, dkv_g, kvt)
                            reload_dkv(w, 1, dk_acc, dkv_g, kvt)
                            write_dkv(dv_acc, dk_acc, dk, dv, b, rows, hk)
                    ti = ti + 1

        @T.prim_func
        def _gqa_bwd_ws_main(
            q: T.Tensor(shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            do: T.Tensor(shape, dtype),  # type: ignore
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            dq: T.Tensor([batch * heads, seq_len // block_n, 2, dq_rows, dq_row], accum_dtype),  # type: ignore
            dk: T.Tensor(kv_shape, dtype),  # type: ignore
            dv: T.Tensor(kv_shape, dtype),  # type: ignore
            dkv_g: T.Tensor(dkv_shape, accum_dtype),  # type: ignore
            sched: T.Tensor([2], "int32"),  # type: ignore
            dkv_cnt: T.Tensor([batch * heads_kv * kv_blocks], "int32"),  # type: ignore
        ) -> None:
            with T.Kernel(min(num_ctas, n_tiles), threads=384) as bid:
                k_s = T.alloc_shared([2, half_m, dim], dtype)
                v_s = T.alloc_shared([2, half_m, dim], dtype)
                q_s = T.alloc_shared([2, block_n, dim], dtype)
                do_s = T.alloc_shared([2, block_n, dim], dtype)
                lse_s = T.alloc_shared([2, block_n], accum_dtype)
                dl_s = T.alloc_shared([2, block_n], accum_dtype)
                # dS by slot * 2 + warpgroup.
                ds_s = T.alloc_shared([4, half_m, block_n], dtype)
                dq_s = T.alloc_shared([2, 2, dq_rows, dq_row], accum_dtype)
                tile_s = T.alloc_shared([2], "int32")
                # Whether this tile's chunk was the last to land its dK/dV partials.
                last_s = T.alloc_shared([1], "int32")

                # The claimed tile goes to the two consumer warpgroups and the two dQ
                # writer warps through a two-slot ring.
                tile_full = T.alloc_barrier(arrive_count=[32, 32])
                tile_empty = T.alloc_barrier(arrive_count=[320, 320])
                kv_full = T.alloc_barrier(arrive_count=32)
                kv_empty = T.alloc_barrier(arrive_count=256)
                # Q and dO move through the ring on their own barriers, so dO's stage is
                # released mid-step and its next load has most of a step to land.
                q_full = T.alloc_barrier(arrive_count=[32, 32])
                do_full = T.alloc_barrier(arrive_count=[32, 32])
                q_empty = T.alloc_barrier(arrive_count=[256, 256])
                do_empty = T.alloc_barrier(arrive_count=[256, 256])
                ds_ready = T.alloc_barrier(arrive_count=[128, 128])
                # dQ barriers by warpgroup * 2 + slot.
                dq_full = T.alloc_barrier(arrive_count=[128, 128, 128, 128])
                dq_empty = T.alloc_barrier(arrive_count=[1, 1, 1, 1])
                T.sync_threads()

                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    if tx < 32:
                        # One CTA per SM walks the tile list in launch order: query-head
                        # chunks in groups whose key blocks fill one wave, longest block
                        # first, so the tiles resident together share query tiles in L2.
                        tcur = T.alloc_var("int32", init=bid)
                        ti = T.alloc_var("int32", init=0)
                        g = T.alloc_var("int32", init=0)
                        while tcur < n_tiles:
                            T.barrier_wait(tile_empty[ti & 1], ((ti >> 1) + 1) & 1)
                            if tx == 0:
                                tile_s[ti & 1] = tcur
                            T.barrier_arrive(tile_full[ti & 1])
                            kb, b, hk, h0 = _tile(tcur)
                            row0 = kb * block_m
                            n_st = kb * (block_m // block_n) if is_causal else 0
                            T.barrier_wait(kv_empty, (ti + 1) & 1)
                            for hh in T.unroll(2):
                                rows = row0 + hh * half_m
                                T.tma_copy(
                                    k[b, rows : rows + half_m, hk, :],
                                    k_s[hh, :, :],
                                    barrier=kv_full,
                                )
                                T.tma_copy(
                                    v[b, rows : rows + half_m, hk, :],
                                    v_s[hh, :, :],
                                    barrier=kv_full,
                                )
                            T.barrier_arrive(kv_full)
                            for hj in T.serial(heads_per_tile):
                                for n in T.serial(n_st, n_ed):
                                    slot = n % 2
                                    cols = n * block_n
                                    T.barrier_wait(q_empty[slot], ((g >> 1) + 1) & 1)
                                    T.tma_copy(
                                        q[b, cols : cols + block_n, h0 + hj, :],
                                        q_s[slot, :, :],
                                        barrier=q_full[slot],
                                    )
                                    T.tma_copy(
                                        lse[b, h0 + hj, cols : cols + block_n],
                                        lse_s[slot, :],
                                        barrier=q_full[slot],
                                    )
                                    T.barrier_arrive(q_full[slot])
                                    T.barrier_wait(do_empty[slot], ((g >> 1) + 1) & 1)
                                    T.tma_copy(
                                        do[b, cols : cols + block_n, h0 + hj, :],
                                        do_s[slot, :, :],
                                        barrier=do_full[slot],
                                    )
                                    T.tma_copy(
                                        delta[b, h0 + hj, cols : cols + block_n],
                                        dl_s[slot, :],
                                        barrier=do_full[slot],
                                    )
                                    T.barrier_arrive(do_full[slot])
                                    g = g + 1
                            tcur = T.call_extern(
                                "int32", "tileops::claim_tile", T.access_ptr(sched[0], "rw")
                            )
                            ti = ti + 1
                        # An out-of-range tile tells every role the list is done.
                        T.barrier_wait(tile_empty[ti & 1], ((ti >> 1) + 1) & 1)
                        if tx == 0:
                            tile_s[ti & 1] = n_tiles
                        T.barrier_arrive(tile_full[ti & 1])
                    elif tx < 64:
                        # Warp 1 + w adds warpgroup w's half of each dQ tile into global
                        # memory: one contiguous 16 KB bulk reduction.
                        dq_writer(0, dq_s, dq, tile_s, tile_full, tile_empty, dq_full, dq_empty, tx)
                    elif tx < 96:
                        dq_writer(1, dq_s, dq, tile_s, tile_full, tile_empty, dq_full, dq_empty, tx)
                elif tx < 256:
                    T.inc_max_nreg(240)
                    consumer(
                        0, k_s, v_s, q_s, do_s, lse_s, dl_s, ds_s, dq_s, tile_s, tile_full, tile_empty,
                        kv_full, kv_empty, q_full, do_full, q_empty, do_empty, ds_ready, dq_full, dq_empty, dk, dv,
                        dkv_g, dkv_cnt, last_s, tx,
                    )  # fmt: skip
                else:
                    T.inc_max_nreg(240)
                    consumer(
                        1, k_s, v_s, q_s, do_s, lse_s, dl_s, ds_s, dq_s, tile_s, tile_full, tile_empty,
                        kv_full, kv_empty, q_full, do_full, q_empty, do_empty, ds_ready, dq_full, dq_empty, dk, dv,
                        dkv_g, dkv_cnt, last_s, tx,
                    )  # fmt: skip
                T.sync_threads()
                if tx == 0:
                    T.call_extern("handle", "tileops::retire", T.access_ptr(sched[0], "rw"))

        return _gqa_bwd_ws_main

    return _gqa_bwd_ws_func


@functools.lru_cache(maxsize=32)
def _gqa_bwd_ws_post_kernel(
    batch: int, heads: int, seq_len: int, dim: int, block_n: int, dq_row: int, dtype: str
) -> Callable:
    half_d = dim // 2
    dq_rows = block_n * half_d // dq_row

    @tilelang.jit
    def _gqa_bwd_ws_post_func() -> Callable:
        @T.prim_func
        def _gqa_bwd_ws_post(
            dq_accum: T.Tensor([batch * heads, seq_len // block_n, 2, dq_rows, dq_row], "float"),  # type: ignore
            dq: T.Tensor([batch, seq_len, heads, dim], dtype),  # type: ignore
        ) -> None:
            with T.Kernel(heads, seq_len // block_n, batch, threads=256) as (h, m, b):
                out_s = T.alloc_shared([block_n, dim], dtype)
                # Read the accumulator in storage order, four f32 per thread, and invert
                # _dq_slot on the shared-memory side.
                for w, r, lane, e in T.Parallel(2, dq_rows, 32, 4):
                    i = (r // 8) * 16 + (e // 2) * 8 + lane // 4
                    j = w * half_d + (r % 8) * 8 + (lane % 4) * 2 + e % 2
                    out_s[i, j] = dq_accum[b * heads + h, m, w, r, lane * 4 + e]
                T.copy(out_s, dq[b, m * block_n : (m + 1) * block_n, h, :])

        return _gqa_bwd_ws_post

    return _gqa_bwd_ws_post_func


class GQABwdWSKernel(Kernel, GQABwdInterface):
    """Warp-specialized causal or full GQA/MHA backward for head dim 128 on SM90.

    Persistent: one CTA per SM claims tiles in launch order, and the next tile's K and V
    load while the current tile's dK and dV are written out. A tile is one 128-row key block
    of one KV head and a chunk of the query heads that share it, walked head by head. One
    producer warp streams Q, dO, LSE and delta through a two-stage TMA ring; two consumer
    warpgroups each own 64 of the tile's key rows and keep dK and dV in registers across the
    chunk; one more producer warp per consumer warpgroup adds that warpgroup's half of each
    dQ tile into an f32 accumulator with a bulk reduction. The accumulator is laid out in
    WGMMA register order, and the second launch of :meth:`forward` lands it in the layout
    and dtype of ``q``. When a KV head's query heads span several chunks, each chunk adds
    f32 partials of dK and dV the same way, and the last chunk to land writes them out.

    The tile counter, the dK/dV accumulator and its landing counts live in the instance, and
    every launch leaves them zeroed, so one instance serves one stream at a time.
    """

    supported_archs: list[int] = [90]
    # The head dim the WGMMA tiles and the accumulator layout are written for.
    _HEAD_DIM = 128
    # Key rows one CTA owns, 64 per consumer warpgroup.
    _BLOCK_M = 128
    # Query rows per step.
    _BLOCK_N = 64
    # f32 per row of the dQ half tile a consumer warpgroup owns.
    _DQ_ROW = 128

    @classmethod
    def refusal(cls, call: AttentionCall) -> "str | None":
        """16-bit inputs with the default softmax scale at head dim 128, and a sequence of
        whole ``_BLOCK_M``-row key blocks."""
        if call.dim != cls._HEAD_DIM:
            return f"requires head dim {cls._HEAD_DIM}, got {call.dim}"
        if call.max_seqlen_q <= 0 or call.max_seqlen_q % cls._BLOCK_M != 0:
            return f"requires a positive multiple of {cls._BLOCK_M} rows, got {call.max_seqlen_q}"
        if call.dtype not in ATTENTION_DTYPES or call.is_fp8:
            return f"requires float16 or bfloat16, got {call.dtype}"
        if call.softcap != 0.0:
            return "does not serve softcap"
        if call.sm_scale is not None:
            return "requires the default softmax scale"
        if call.uses_sliding_window:
            return "does not serve a sliding window"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        index = call.device.index if call.device is not None else None
        args = (
            call.batch,
            call.heads,
            call.heads_kv,
            call.max_seqlen_q,
            call.dim,
            call.is_causal,
            call.dtype,
        )
        return (*args, index), lambda: cls(*args, device_index=index)

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
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        num_sms = get_sm_count(device_index)
        kv_blocks = seq_len // self._BLOCK_M
        self.heads_per_tile = self._heads_per_tile(
            batch, heads, heads_kv, kv_blocks, is_causal, num_sms
        )
        units = batch * heads // self.heads_per_tile
        # Chunks launched together: the largest divisor of the chunk count whose key blocks
        # fit in one wave of ``num_sms`` CTAs.
        fit = max(1, num_sms // kv_blocks)
        fill = max(f for f in range(1, min(fit, units) + 1) if units % f == 0)
        self.kernel = _gqa_bwd_ws_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            is_causal,
            self.heads_per_tile,
            fill,
            num_sms,
            self._BLOCK_M,
            self._BLOCK_N,
            self._DQ_ROW,
            self.dtype_str,
        )
        device = (
            torch.device("cuda", device_index) if device_index is not None else torch.device("cuda")
        )
        # [next tile, CTAs retired]; the kernel returns both to zero.
        self.sched = torch.zeros(2, dtype=torch.int32, device=device)
        # The f32 dK/dV partials of each key block and the count of chunks landed in each;
        # the last chunk to land returns both to zero.
        kv_tiles = batch * heads_kv * kv_blocks
        dq_rows = self._BLOCK_N * dim // 2 // self._DQ_ROW
        grouped = self.heads_per_tile < heads // heads_kv
        self.dkv_acc = torch.zeros(
            (kv_tiles * 2, 2, 2, dq_rows, self._DQ_ROW) if grouped else (1,),
            dtype=torch.float32,
            device=device,
        )
        self.dkv_cnt = torch.zeros(kv_tiles, dtype=torch.int32, device=device)
        self.init_config(config)
        self.main_kernel = self.kernel()
        self.post_kernel = _gqa_bwd_ws_post_kernel(
            batch, heads, seq_len, dim, self._BLOCK_N, self._DQ_ROW, self.dtype_str
        )()

    @staticmethod
    def _heads_per_tile(
        batch: int, heads: int, heads_kv: int, kv_blocks: int, is_causal: bool, num_sms: int
    ) -> int:
        """Query heads one tile walks: the most, among divisors of the group, whose longest
        tile is no longer than one SM's share of all the steps, so the longest tiles still
        balance across SMs."""
        group = heads // heads_kv
        # Steps of one query head at key block 0, and of every key block of every head.
        longest = 2 * kv_blocks
        total = batch * heads * (kv_blocks * (kv_blocks + 1) if is_causal else longest * kv_blocks)
        fits = [d for d in range(1, group + 1) if group % d == 0 and d * longest * num_sms <= total]
        return max(fits, default=1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
        delta: torch.Tensor,
        dq_accum: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(dq, dk, dv)`` in the input dtype; ``dq_accum`` arrives zero-filled."""
        dq_accum = dq_accum.view(
            self.batch * self.heads,
            self.seq_len // self._BLOCK_N,
            2,
            self._BLOCK_N * self.dim // 2 // self._DQ_ROW,
            self._DQ_ROW,
        )
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        self.main_kernel(
            q, k, v, do, lse, delta, dq_accum, dk, dv, self.dkv_acc, self.sched, self.dkv_cnt
        )
        dq = torch.empty_like(q)
        self.post_kernel(dq_accum, dq)
        return dq, dk, dv
