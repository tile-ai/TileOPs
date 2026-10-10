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

__all__ = ["MHABwdWSKernel"]

# Key rows one CTA owns, 64 per consumer warpgroup.
_BLOCK_M = 128
# Query rows per step.
_BLOCK_N = 64
# f32 per row of the dQ half tile a consumer warpgroup owns.
_DQ_ROW = 128


@functools.lru_cache(maxsize=32)
def _mha_bwd_ws_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    group: int,
    num_ctas: int,
    dtype: str,
) -> Callable:
    sm_scale = dim**-0.5
    scale = sm_scale * LOG2E
    accum_dtype = "float"
    block_m, block_n = _BLOCK_M, _BLOCK_N
    half_m, half_d = block_m // 2, dim // 2
    kv_blocks = seq_len // block_m
    dq_rows = block_n * half_d // _DQ_ROW
    n_tiles = batch * heads * kv_blocks
    n_ed = seq_len // block_n
    # Every tile runs an even number of steps, so the global step count starts each
    # tile even and the ring slot can be read from the tile's own step index.
    assert seq_len % block_m == 0

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16", *csrc_include("tile_claim.h")],
    )
    def _mha_bwd_ws_func() -> Callable:
        def _dq_slot(i, j):
            """Where element ``(i, j)`` of a warpgroup's ``[_BLOCK_N, dim // 2]`` dQ half sits in
            its f32 accumulator tile.

            Each warp stores four consecutive f32 of the WGMMA accumulator per thread, 512
            contiguous bytes per instruction, so the store to shared memory is free of bank
            conflicts. Returns ``(row, col)`` of a ``[_BLOCK_N * dim // 2 // _DQ_ROW, _DQ_ROW]``
            tile.
            """
            lane = (i % 8) * 4 + (j % 8) // 2
            return (i // 16) * 8 + j // 8, lane * 4 + ((i % 16) // 8) * 2 + j % 2

        shape = (batch, seq_len, heads, dim)

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
                    kb = (t % (group * kv_blocks)) // group
                    bh = t // (group * kv_blocks) * group + t % group
                    n_st = kb * (block_m // block_n) if is_causal else 0
                    for n in T.serial(n_st, n_ed):
                        slot = n % 2
                        T.barrier_wait(dq_full[w * 2 + slot], (g >> 1) & 1)
                        if tx == 32 * (w + 1):
                            T.call_extern(
                                "handle",
                                "tl::tma_store_add",
                                T.access_ptr(dq_s[slot, w, 0, 0], "r"),
                                T.access_ptr(dq[bh, n, w, 0, 0], "w"),
                                dq_rows * _DQ_ROW * 4,
                            )
                            T.call_extern("handle", "tl::tma_store_arrive")
                            T.call_extern("handle", "tl::tma_store_wait<0, true>")
                            T.barrier_arrive(dq_empty[w * 2 + slot])
                        g = g + 1
                    ti = ti + 1

        @T.macro
        def consumer(
            w, k_s, v_s, q_s, do_s, lse_s, dl_s, ds_s, dq_s, tile_s, tile_full, tile_empty,
            kv_full, kv_empty, q_full, do_full, q_empty, do_empty, ds_ready, dq_full, dq_empty, dk, dv,
        ):  # fmt: skip
            s = T.alloc_fragment([half_m, block_n], accum_dtype)
            dp = T.alloc_fragment([half_m, block_n], accum_dtype)
            p16 = T.alloc_fragment([half_m, block_n], dtype)
            ds16 = T.alloc_fragment([half_m, block_n], dtype)
            dv_acc = T.alloc_fragment([half_m, dim], accum_dtype)
            dk_acc = T.alloc_fragment([half_m, dim], accum_dtype)
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
                    kb = (t % (group * kv_blocks)) // group
                    bh = t // (group * kv_blocks) * group + t % group
                    h = bh % heads
                    b = bh // heads
                    row0 = kb * block_m
                    n_st = kb * (block_m // block_n) if is_causal else 0
                    T.clear(dv_acc)
                    T.clear(dk_acc)
                    T.barrier_wait(kv_full, ti & 1)
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
                        # The softmax scale is applied to dK and dQ once, at the end.
                        for i, j in T.Parallel(half_m, block_n):
                            if j >= block_n // 2:
                                s[i, j] = T.exp2(s[i, j])
                            ds16[i, j] = s[i, j] * (dp[i, j] - dl_r[i, j])
                        T.copy(s, p16)
                        T.copy(ds16, ds_s[slot * 2 + w, :, :])
                        T.fence_proxy_async()
                        T.barrier_arrive(ds_ready[w])
                        T.wgmma_gemm(p16, do_s[slot, :, :], dv_acc, policy=T.GemmWarpPolicy.FullRow)
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
                        # while dK and dQ run. The tile's last two stages stay until the
                        # epilogue has staged dK and dV through them.
                        if n < n_ed - 2:
                            T.barrier_arrive(do_empty[slot])
                        T.wgmma_gemm(ds16, q_s[slot, :, :], dk_acc, policy=T.GemmWarpPolicy.FullRow)
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
                        if n < n_ed - 2:
                            T.barrier_arrive(q_empty[slot])
                        g = g + 1
                    # K and V are free: the next tile's load overlaps this epilogue.
                    T.barrier_arrive(kv_empty)
                    for i, j in T.Parallel(half_m, dim):
                        dk_acc[i, j] = dk_acc[i, j] * sm_scale
                    # Both warpgroups are past their last read of Q and dO before either
                    # reuses that storage for its output tile.
                    T.sync_threads(barrier_id=2, arrive_count=256)
                    T.copy(dv_acc, q_s[w, :, :])
                    T.copy(dk_acc, do_s[w, :, :])
                    T.fence_proxy_async()
                    T.sync_threads(barrier_id=3 + w, arrive_count=128)
                    rows = row0 + w * half_m
                    T.copy(q_s[w, :, :], dv[b, rows : rows + half_m, h, :])
                    T.copy(do_s[w, :, :], dk[b, rows : rows + half_m, h, :])
                    # The stores have read both stages: hand them back to the producer.
                    T.sync_threads(barrier_id=3 + w, arrive_count=128)
                    for sl in T.unroll(2):
                        T.barrier_arrive(q_empty[sl])
                        T.barrier_arrive(do_empty[sl])
                    ti = ti + 1

        @T.prim_func
        def _mha_bwd_ws_main(
            q: T.Tensor(shape, dtype),  # type: ignore
            k: T.Tensor(shape, dtype),  # type: ignore
            v: T.Tensor(shape, dtype),  # type: ignore
            do: T.Tensor(shape, dtype),  # type: ignore
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            dq: T.Tensor([batch * heads, seq_len // block_n, 2, dq_rows, _DQ_ROW], accum_dtype),  # type: ignore
            dk: T.Tensor(shape, dtype),  # type: ignore
            dv: T.Tensor(shape, dtype),  # type: ignore
            sched: T.Tensor([2], "int32"),  # type: ignore
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
                dq_s = T.alloc_shared([2, 2, dq_rows, _DQ_ROW], accum_dtype)
                tile_s = T.alloc_shared([2], "int32")

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
                        # One CTA per SM walks the tile list in launch order: heads in
                        # groups whose key blocks fill one wave, longest block first, so
                        # the tiles resident together share query tiles in L2.
                        tcur = T.alloc_var("int32", init=bid)
                        ti = T.alloc_var("int32", init=0)
                        g = T.alloc_var("int32", init=0)
                        while tcur < n_tiles:
                            T.barrier_wait(tile_empty[ti & 1], ((ti >> 1) + 1) & 1)
                            if tx == 0:
                                tile_s[ti & 1] = tcur
                            T.barrier_arrive(tile_full[ti & 1])
                            kb = (tcur % (group * kv_blocks)) // group
                            bh = tcur // (group * kv_blocks) * group + tcur % group
                            h = bh % heads
                            b = bh // heads
                            row0 = kb * block_m
                            n_st = kb * (block_m // block_n) if is_causal else 0
                            T.barrier_wait(kv_empty, (ti + 1) & 1)
                            for hh in T.unroll(2):
                                rows = row0 + hh * half_m
                                T.tma_copy(
                                    k[b, rows : rows + half_m, h, :], k_s[hh, :, :], barrier=kv_full
                                )
                                T.tma_copy(
                                    v[b, rows : rows + half_m, h, :], v_s[hh, :, :], barrier=kv_full
                                )
                            T.barrier_arrive(kv_full)
                            for n in T.serial(n_st, n_ed):
                                slot = n % 2
                                cols = n * block_n
                                T.barrier_wait(q_empty[slot], ((g >> 1) + 1) & 1)
                                T.tma_copy(
                                    q[b, cols : cols + block_n, h, :],
                                    q_s[slot, :, :],
                                    barrier=q_full[slot],
                                )
                                T.tma_copy(
                                    lse[b, h, cols : cols + block_n],
                                    lse_s[slot, :],
                                    barrier=q_full[slot],
                                )
                                T.barrier_arrive(q_full[slot])
                                T.barrier_wait(do_empty[slot], ((g >> 1) + 1) & 1)
                                T.tma_copy(
                                    do[b, cols : cols + block_n, h, :],
                                    do_s[slot, :, :],
                                    barrier=do_full[slot],
                                )
                                T.tma_copy(
                                    delta[b, h, cols : cols + block_n],
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
                    )  # fmt: skip
                else:
                    T.inc_max_nreg(240)
                    consumer(
                        1, k_s, v_s, q_s, do_s, lse_s, dl_s, ds_s, dq_s, tile_s, tile_full, tile_empty,
                        kv_full, kv_empty, q_full, do_full, q_empty, do_empty, ds_ready, dq_full, dq_empty, dk, dv,
                    )  # fmt: skip
                T.sync_threads()
                if tx == 0:
                    T.call_extern("handle", "tileops::retire", T.access_ptr(sched[0], "rw"))

        return _mha_bwd_ws_main

    return _mha_bwd_ws_func


@functools.lru_cache(maxsize=32)
def _mha_bwd_ws_post_kernel(batch: int, heads: int, seq_len: int, dim: int, dtype: str) -> Callable:
    block_n = _BLOCK_N
    half_d = dim // 2
    dq_rows = block_n * half_d // _DQ_ROW
    sm_scale = dim**-0.5

    @tilelang.jit
    def _mha_bwd_ws_post_func() -> Callable:
        @T.prim_func
        def _mha_bwd_ws_post(
            dq_accum: T.Tensor([batch * heads, seq_len // block_n, 2, dq_rows, _DQ_ROW], "float"),  # type: ignore
            dq: T.Tensor([batch, seq_len, heads, dim], dtype),  # type: ignore
        ) -> None:
            with T.Kernel(heads, seq_len // block_n, batch, threads=256) as (h, m, b):
                out_s = T.alloc_shared([block_n, dim], dtype)
                # Read the accumulator in storage order, four f32 per thread, and invert
                # _dq_slot on the shared-memory side.
                for w, r, lane, e in T.Parallel(2, dq_rows, 32, 4):
                    i = (r // 8) * 16 + (e // 2) * 8 + lane // 4
                    j = w * half_d + (r % 8) * 8 + (lane % 4) * 2 + e % 2
                    out_s[i, j] = dq_accum[b * heads + h, m, w, r, lane * 4 + e] * sm_scale
                T.copy(out_s, dq[b, m * block_n : (m + 1) * block_n, h, :])

        return _mha_bwd_ws_post

    return _mha_bwd_ws_post_func


class MHABwdWSKernel(Kernel, GQABwdInterface):
    """Warp-specialized causal or full MHA backward for head dim 128 on SM90.

    Persistent: one CTA per SM claims key-block tiles in launch order, and the next
    tile's K and V load while the current tile's dK and dV are written out. One producer
    warp streams Q, dO, LSE and delta through a two-stage TMA ring; two consumer
    warpgroups each own 64 of a tile's 128 key rows and keep dK and dV in registers; one
    more producer warp per consumer warpgroup adds that warpgroup's half of each dQ tile
    into an f32 accumulator with a bulk reduction. The accumulator is laid out in WGMMA
    register order, and the second launch of :meth:`forward` lands it in the layout and
    dtype of ``q``.

    The tile counter lives in the instance and every launch leaves it zeroed, so one
    instance serves one stream at a time.
    """

    supported_archs: list[int] = [90]
    # The head dim the WGMMA tiles and the dQ layout are written for.
    _HEAD_DIM = 128

    @classmethod
    def applies(cls, call: AttentionCall) -> bool:
        """One query head per KV head, 16-bit inputs with the default softmax scale, and a
        sequence of whole ``_BLOCK_M``-row key blocks."""
        return (
            call.heads == call.heads_kv
            and call.dim == cls._HEAD_DIM
            and call.max_seqlen_q > 0
            and call.max_seqlen_q % _BLOCK_M == 0
            and call.dtype in ATTENTION_DTYPES
            and not call.is_fp8
            and call.softcap == 0.0
            and call.sm_scale is None
            and not call.uses_sliding_window
        )

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        index = call.device.index if call.device is not None else None
        args = (call.batch, call.heads, call.max_seqlen_q, call.dim, call.is_causal, call.dtype)
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        num_sms = get_sm_count(device_index)
        # Heads launched together: the largest divisor of ``batch * heads`` whose key blocks
        # fit in one wave of ``num_sms`` CTAs.
        fit = max(1, num_sms // (seq_len // _BLOCK_M))
        group = max(g for g in range(1, min(fit, batch * heads) + 1) if batch * heads % g == 0)
        self.kernel = _mha_bwd_ws_kernel(
            batch, heads, seq_len, dim, is_causal, group, num_sms, self.dtype_str
        )
        # [next tile, CTAs retired]; the kernel returns both to zero.
        self.sched = torch.zeros(
            2,
            dtype=torch.int32,
            device=torch.device("cuda", device_index)
            if device_index is not None
            else torch.device("cuda"),
        )
        self.init_config(config, tune)
        self.main_kernel = self.kernel()
        self.post_kernel = _mha_bwd_ws_post_kernel(batch, heads, seq_len, dim, self.dtype_str)()

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
            self.seq_len // _BLOCK_N,
            2,
            _BLOCK_N * self.dim // 2 // _DQ_ROW,
            _DQ_ROW,
        )
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        self.main_kernel(q, k, v, do, lse, delta, dq_accum, dk, dv, self.sched)
        dq = torch.empty_like(q)
        self.post_kernel(dq_accum, dq)
        return dq, dk, dv
