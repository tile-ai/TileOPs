import functools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Entry, Kernel

from .call_spec import AttentionCall, mha_bwd_ws_region
from .online_softmax import LOG2E

__all__ = ["MHABwdWsKernel"]

# Key rows one CTA owns, 64 per consumer warpgroup.
_BLOCK_M = 128
# Query rows per step.
_BLOCK_N = 64
# The dQ tile leaves shared memory as one TMA box of this many f32 per row.
_DQ_ROW = 256


def _dq_slot(i, j, half_d: int):
    """Where element ``(i, j)`` of a query tile's dQ sits in the f32 accumulator tile.

    Thread ``t`` of the two consumer warpgroups owns 32 consecutive elements, in the
    register order of the WGMMA accumulator, so its store to shared memory is 128-bit.
    Returns ``(row, col)`` of a ``[_BLOCK_N * dim // _DQ_ROW, _DQ_ROW]`` tile.
    """
    w, jj = j // half_d, j % half_d
    lane = (i % 8) * 4 + (jj % 8) // 2
    reg = (jj // 8) * 4 + ((i % 16) // 8) * 2 + jj % 2
    return lane, (w * 4 + i // 16) * 32 + reg


@functools.lru_cache(maxsize=32)
def _mha_bwd_ws_kernel(
    batch: int, heads: int, seq_len: int, dim: int, is_causal: bool, group: int, dtype: str
) -> Callable:
    sm_scale = dim**-0.5
    scale = sm_scale * LOG2E
    accum_dtype = "float"
    block_m, block_n = _BLOCK_M, _BLOCK_N
    half_m, half_d = block_m // 2, dim // 2
    kv_blocks = seq_len // block_m
    dq_rows = block_n * dim // _DQ_ROW

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _mha_bwd_ws_func() -> Callable:
        shape = (batch, seq_len, heads, dim)

        @T.macro
        def consumer(
            w, k_s, v_s, q_s, do_s, lse_s, dl_s, ds_s, dq_s,
            kv_full, in_full, in_empty, dq_full, dq_empty, dk, dv, b, h, row0, n_st, n_ed,
        ):  # fmt: skip
            s = T.alloc_fragment([half_m, block_n], accum_dtype)
            dp = T.alloc_fragment([half_m, block_n], accum_dtype)
            p16 = T.alloc_fragment([half_m, block_n], dtype)
            ds16 = T.alloc_fragment([half_m, block_n], dtype)
            dv_acc = T.alloc_fragment([half_m, dim], accum_dtype)
            dk_acc = T.alloc_fragment([half_m, dim], accum_dtype)
            dq_acc = T.alloc_fragment([block_n, half_d], accum_dtype)
            T.clear(dv_acc)
            T.clear(dk_acc)
            T.barrier_wait(kv_full, 0)
            for n in T.serial(n_st, n_ed):
                it = n - n_st
                slot = it % 2
                T.barrier_wait(in_full[slot], (it // 2) % 2)
                # The transposed scores and dP of this warpgroup's 64 key rows.
                T.wgmma_gemm(
                    k_s[w, :, :],
                    q_s[slot, :, :],
                    s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )
                T.wgmma_gemm(
                    v_s[w, :, :],
                    do_s[slot, :, :],
                    dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )
                # An accumulator is read only after a wait that covers its WGMMA; a read
                # of an in-flight one makes ptxas serialize every WGMMA in the kernel.
                T.wait_wgmma(1)
                T.warpgroup_fence_operand(s, num_regs=32)
                for i, j in T.Parallel(half_m, block_n):
                    s[i, j] = T.exp2(s[i, j] * scale - lse_s[slot, j])
                if is_causal and n * block_n < row0 + (w + 1) * half_m:
                    for i, j in T.Parallel(half_m, block_n):
                        s[i, j] = T.if_then_else(
                            row0 + w * half_m + i <= n * block_n + j, s[i, j], 0
                        )
                T.wait_wgmma(0)
                T.warpgroup_fence_operand(dp, num_regs=32)
                # The softmax scale is applied to dK and dQ once, at the end.
                for i, j in T.Parallel(half_m, block_n):
                    ds16[i, j] = s[i, j] * (dp[i, j] - dl_s[slot, j])
                T.copy(s, p16)
                T.copy(ds16, ds_s[slot, w, :, :])
                T.wgmma_gemm(p16, do_s[slot, :, :], dv_acc, policy=T.GemmWarpPolicy.FullRow)
                T.fence_proxy_async()
                T.sync_threads(barrier_id=1, arrive_count=256)
                # dQ contracts over all 128 key rows, so each warpgroup reads both halves
                # of dS and computes half of dQ's columns.
                T.wgmma_gemm(
                    ds_s[slot, 0, :, :],
                    k_s[0, :, w * half_d : (w + 1) * half_d],
                    dq_acc,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullRow,
                    clear_accum=True,
                )
                T.wgmma_gemm(
                    ds_s[slot, 1, :, :],
                    k_s[1, :, w * half_d : (w + 1) * half_d],
                    dq_acc,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.wait_wgmma(2)
                T.wgmma_gemm(ds16, q_s[slot, :, :], dk_acc, policy=T.GemmWarpPolicy.FullRow)
                T.wait_wgmma(1)
                T.warpgroup_fence_operand(dq_acc, num_regs=32)
                # dK runs while dQ is written out.
                T.barrier_wait(dq_empty[slot], (it // 2 + 1) % 2)
                for i, j in T.Parallel(block_n, half_d):
                    r, c = _dq_slot(i, w * half_d + j, half_d)
                    dq_s[slot, r, c] = dq_acc[i, j]
                T.fence_proxy_async()
                T.barrier_arrive(dq_full[slot])
                T.wait_wgmma(0)
                T.warpgroup_fence_operand(dk_acc, num_regs=64)
                T.warpgroup_fence_operand(dv_acc, num_regs=64)
                T.barrier_arrive(in_empty[slot])
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

        @T.prim_func
        def _mha_bwd_ws_main(
            q: T.Tensor(shape, dtype),  # type: ignore
            k: T.Tensor(shape, dtype),  # type: ignore
            v: T.Tensor(shape, dtype),  # type: ignore
            do: T.Tensor(shape, dtype),  # type: ignore
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            dq: T.Tensor([batch, heads, seq_len // block_n, dq_rows, _DQ_ROW], accum_dtype),  # type: ignore
            dk: T.Tensor(shape, dtype),  # type: ignore
            dv: T.Tensor(shape, dtype),  # type: ignore
        ) -> None:
            with T.Kernel(batch * heads * kv_blocks, threads=384) as bid:
                # Heads launch in groups whose key blocks fill one wave, longest block
                # first, so the CTAs resident together share query tiles in L2.
                kb = (bid % (group * kv_blocks)) // group
                bh = bid // (group * kv_blocks) * group + bid % group
                h = bh % heads
                b = bh // heads
                row0 = kb * block_m
                n_st = T.floordiv(row0, block_n) if is_causal else 0
                n_ed = T.ceildiv(seq_len, block_n)

                k_s = T.alloc_shared([2, half_m, dim], dtype)
                v_s = T.alloc_shared([2, half_m, dim], dtype)
                q_s = T.alloc_shared([2, block_n, dim], dtype)
                do_s = T.alloc_shared([2, block_n, dim], dtype)
                lse_s = T.alloc_shared([2, block_n], accum_dtype)
                dl_s = T.alloc_shared([2, block_n], accum_dtype)
                ds_s = T.alloc_shared([2, 2, half_m, block_n], dtype)
                dq_s = T.alloc_shared([2, dq_rows, _DQ_ROW], accum_dtype)

                kv_full = T.alloc_barrier(arrive_count=32)
                in_full = T.alloc_barrier(arrive_count=[32, 32])
                in_empty = T.alloc_barrier(arrive_count=[256, 256])
                dq_full = T.alloc_barrier(arrive_count=[256, 256])
                dq_empty = T.alloc_barrier(arrive_count=[32, 32])
                T.sync_threads()

                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    if tx < 32:
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
                            it = n - n_st
                            slot = it % 2
                            cols = n * block_n
                            T.barrier_wait(in_empty[slot], (it // 2 + 1) % 2)
                            T.tma_copy(
                                q[b, cols : cols + block_n, h, :],
                                q_s[slot, :, :],
                                barrier=in_full[slot],
                            )
                            T.tma_copy(
                                do[b, cols : cols + block_n, h, :],
                                do_s[slot, :, :],
                                barrier=in_full[slot],
                            )
                            T.tma_copy(
                                lse[b, h, cols : cols + block_n],
                                lse_s[slot, :],
                                barrier=in_full[slot],
                            )
                            T.tma_copy(
                                delta[b, h, cols : cols + block_n],
                                dl_s[slot, :],
                                barrier=in_full[slot],
                            )
                            T.barrier_arrive(in_full[slot])
                    elif tx < 64:
                        # One warp adds each finished dQ tile into global memory.
                        for n in T.serial(n_st, n_ed):
                            it = n - n_st
                            slot = it % 2
                            T.barrier_wait(dq_full[slot], (it // 2) % 2)
                            T.atomic_add(dq[b, h, n, :, :], dq_s[slot, :, :], use_tma=True)
                            T.barrier_arrive(dq_empty[slot])
                elif tx < 256:
                    T.inc_max_nreg(240)
                    consumer(
                        0, k_s, v_s, q_s, do_s, lse_s, dl_s, ds_s, dq_s,
                        kv_full, in_full, in_empty, dq_full, dq_empty, dk, dv, b, h, row0, n_st, n_ed,
                    )  # fmt: skip
                else:
                    T.inc_max_nreg(240)
                    consumer(
                        1, k_s, v_s, q_s, do_s, lse_s, dl_s, ds_s, dq_s,
                        kv_full, in_full, in_empty, dq_full, dq_empty, dk, dv, b, h, row0, n_st, n_ed,
                    )  # fmt: skip

        return _mha_bwd_ws_main

    return _mha_bwd_ws_func


@functools.lru_cache(maxsize=32)
def _mha_bwd_ws_post_kernel(batch: int, heads: int, seq_len: int, dim: int, dtype: str) -> Callable:
    block_n = _BLOCK_N
    dq_rows = block_n * dim // _DQ_ROW
    sm_scale = dim**-0.5
    half_d = dim // 2

    @tilelang.jit
    def _mha_bwd_ws_post_func() -> Callable:
        @T.prim_func
        def _mha_bwd_ws_post(
            dq_accum: T.Tensor([batch, heads, seq_len // block_n, dq_rows, _DQ_ROW], "float"),  # type: ignore
            dq: T.Tensor([batch, seq_len, heads, dim], dtype),  # type: ignore
        ) -> None:
            with T.Kernel(heads, seq_len // block_n, batch, threads=256) as (h, m, b):
                out = T.alloc_fragment([block_n, dim], dtype)
                for i, j in T.Parallel(block_n, dim):
                    r, c = _dq_slot(i, j, half_d)
                    out[i, j] = dq_accum[b, h, m, r, c] * sm_scale
                T.copy(out, dq[b, m * block_n : (m + 1) * block_n, h, :])

        return _mha_bwd_ws_post

    return _mha_bwd_ws_post_func


def _launch_group(batch_heads: int, kv_blocks: int, num_sms: int) -> int:
    """How many heads launch together: the largest divisor of ``batch * heads`` whose key
    blocks fit in one wave of ``num_sms`` CTAs."""
    fit = max(1, num_sms // kv_blocks)
    return max(g for g in range(1, min(fit, batch_heads) + 1) if batch_heads % g == 0)


class MHABwdWsKernel(Kernel):
    """Warp-specialized causal or full MHA backward for head dim 128 on SM90.

    One producer warp streams Q, dO, LSE and delta through a two-stage TMA ring; two
    consumer warpgroups each own 64 of the CTA's 128 key rows and keep dK and dV in
    registers; a second producer warp adds each dQ tile into an f32 accumulator with a
    TMA reduction. The accumulator is laid out in WGMMA register order, and the second
    launch of :meth:`forward` lands it in the layout and dtype of ``q``.
    """

    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call: AttentionCall) -> bool:
        return mha_bwd_ws_region(call)

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        index = call.device.index if call.device is not None else None
        args = (call.batch, call.heads, call.max_seqlen_q, call.dim, call.is_causal, call.dtype)
        return (*args, index), lambda: cls(*args, tune=call.tune, device_index=index)

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

        num_sms = torch.cuda.get_device_properties(
            device_index if device_index is not None else torch.cuda.current_device()
        ).multi_processor_count
        group = _launch_group(batch * heads, seq_len // _BLOCK_M, num_sms)
        self.kernel = _mha_bwd_ws_kernel(
            batch, heads, seq_len, dim, is_causal, group, self.dtype_str
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
            self.batch,
            self.heads,
            self.seq_len // _BLOCK_N,
            _BLOCK_N * self.dim // _DQ_ROW,
            _DQ_ROW,
        )
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        self.main_kernel(q, k, v, do, lse, delta, dq_accum, dk, dv)
        dq = torch.empty_like(q)
        self.post_kernel(dq_accum, dq)
        return dq, dk, dv
