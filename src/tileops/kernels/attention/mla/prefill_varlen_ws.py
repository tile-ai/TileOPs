"""SM90 warp-specialized packed variable-length MLA prefill kernel.

The key of head ``h`` is ``k_nope[:, h]`` followed by ``k_pe``, and ``k_pe`` is one
row per token shared by every head. Nothing here builds that concatenation: the
score is a ``dim_nope``-wide WGMMA against ``k_nope`` plus a ``dim_pe``-wide one
against ``k_pe``, issued back to back into the same accumulator, so the rope half
crosses memory once per token rather than once per token per head.

Queries and keys are the same tokens, so one ``cu_seqlens`` describes both and the
causal mask sits on the diagonal.

The structure is explicit rather than left to a pipeliner: one persistent CTA per
SM claims work items from a global counter; a 32-thread producer warp issues every
TMA load and hands each buffer over through an mbarrier; two consumer warpgroups
own 64 query rows each and overlap the QK of tile ``k`` with the PV of tile
``k - 1``.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops.kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    MlaVarlenCall,
    MlaVarlenFwdInterface,
)
from tileops.kernels.constants import LOG2E
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_sm_count

__all__ = ["MLAVarlenPrefillWSFwdKernel"]


@functools.lru_cache(maxsize=32)
# No out_idx: resolving a symbolic output shape costs host time on every call.
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
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
        "-DNDEBUG",
    ],
)
def _mla_varlen_ws_kernel(
    batch,
    heads,
    dim_nope,
    dim_pe,
    dim_v,
    is_causal,
    sm_scale,
    dtype,
    block_n,
    stages,
    q_slots,
    num_ctas,
):
    """A persistent CTA per SM: a TMA producer warp claims work, two consumer warpgroups run it."""
    score_scale = (dim_nope + dim_pe) ** -0.5 if sm_scale is None else sm_scale
    scale = score_scale * LOG2E
    accum = "float"
    half = 64  # rows per consumer warpgroup: one m64 WGMMA
    block_m = 2 * half
    consumers = 256  # threads of the two consumer warpgroups
    policy = T.GemmWarpPolicy.FullRow
    total_q = T.dynamic("total_q")
    q_tiling = GroupTiling(batch, block_m)

    @T.macro
    def row_max(acc_s, red, sm):
        """sm = max(sm, row max): per-thread partials over the column groups, then one
        cross-thread reduce."""
        T.reduce_max(T.reshape(acc_s, [half, block_n // 8, 8]), red, dim=1, clear=True)
        T.reduce_max(red, sm, dim=1, clear=False, batch=2)

    @T.macro
    def softmax_step(acc_s, sm, smp, alpha, ss, red):
        """Fold one score tile into the running max, rescale factor, and row sums."""
        T.copy(sm, smp)
        row_max(acc_s, red, sm)
        for i in T.Parallel(half):
            sm[i] = T.if_then_else((sm[i] - smp[i]) * scale > 8.0, sm[i], smp[i])
            alpha[i] = 1.0
            if sm[i] != smp[i]:
                alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
        for i, j in T.Parallel(half, block_n):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
        T.reduce_sum(acc_s, ss, dim=1, batch=2)

    @T.macro
    def score(qn_tile, qp_tile, KNs, KPs, sk, acc_s):
        """The two halves of the key, issued back to back into one accumulator."""
        T.wgmma_gemm(
            qn_tile, KNs[sk, :, :], acc_s, transpose_B=True, policy=policy, clear_accum=True
        )
        T.wgmma_gemm(
            qp_tile, KPs[sk, :, :], acc_s, transpose_B=True, policy=policy, clear_accum=False
        )

    @T.macro
    def kv_step(
        qn_tile, qp_tile, KNs, KPs, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o,
        sm, smp, alpha, ss, red, logsum, my_bar, nxt_bar, k, n, row, seq_len, tail: bool,
    ):  # fmt: skip
        """QK of KV tile k overlapped with PV of tile k - 1; n counts tiles across work items."""
        sk = n % stages
        svp = (n - 1) % stages
        T.sync_threads(my_bar, consumers)
        T.mbarrier_wait_parity(kready[sk], (n // stages) % 2)
        score(qn_tile, qp_tile, KNs, KPs, sk, acc_s)
        rescale = T.alloc_var("int32", init=0)
        for i in T.Parallel(half):
            if alpha[i] != 1.0:
                rescale = 1
        if rescale == 1:
            for i, j in T.Parallel(half, dim_v):
                acc_o[i, j] *= alpha[i]
        T.mbarrier_wait_parity(vready[svp], ((n - 1) // stages) % 2)
        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
        T.named_barrier_arrive(nxt_bar, consumers)
        T.wait_wgmma(1)
        T.mbarrier_arrive(kfree[sk])
        if tail:
            if is_causal:
                limit = row - k * block_n
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(limit + i >= j, acc_s[i, j], -T.infinity(accum))
            else:
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(
                        k * block_n + j < seq_len, acc_s[i, j], -T.infinity(accum)
                    )
        softmax_step(acc_s, sm, smp, alpha, ss, red)
        T.wait_wgmma(0)
        T.mbarrier_arrive(vfree[svp])
        for i in T.Parallel(half):
            logsum[i] = logsum[i] * alpha[i] + ss[i]
        T.copy(acc_s, pcast)

    @T.macro
    def locate(work, CuQ, tile_cum, lo, hi, request, q_row, meta):
        """Fill meta (head, seq_start, seq_len, q0, eff) for work item *work*."""
        q_tiling.decode(tile_cum[batch] - 1 - work // heads, tile_cum, lo, hi, request, q_row)
        meta[0] = work % heads
        meta[1] = CuQ[request[0]]
        meta[2] = CuQ[request[0] + 1] - meta[1]
        meta[3] = q_row[0]
        if is_causal:
            # Self-attention on the diagonal: the scan stops at this tile's last row.
            meta[4] = T.max(
                1, T.min(T.ceildiv(meta[2], block_n), T.ceildiv(meta[3] + block_m, block_n))
            )
        else:
            meta[4] = T.max(1, T.ceildiv(meta[2], block_n))

    @T.macro
    def fetch(Sched, lane, work):
        """Claim the next work item into work[0]: lane 0 claims, the warp shares it."""
        if lane == 0:
            work[0] = T.atomic_add(Sched[0], 1, return_prev=True)
        work[0] = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), work[0], 0, 32, 32)

    @T.macro
    def producer(
        Q, KN, KP, V, CuQ, Sched, QNs, QPs, KNs, KPs, Vs, item_slot, q_bar, qfree, kready,
        kfree, vready, vfree, total, tile_cum, lo, hi, request, q_row, meta, lane,
    ):  # fmt: skip
        """Claim work items until none remain; load each one's Q, then its KV tiles."""
        issued = T.alloc_var("int32", init=0)
        loaded = T.alloc_var("int32", init=0)
        work = T.alloc_local([1], "int32")
        fetch(Sched, lane, work)
        while work[0] < total:
            locate(work[0], CuQ, tile_cum, lo, hi, request, q_row, meta)
            head = meta[0]
            q_start = meta[1] + meta[3]
            seq_start = meta[1]
            slot = loaded % q_slots
            T.mbarrier_wait_parity(qfree[slot], ((loaded // q_slots) % 2) ^ 1)
            # The consumers read the item's geometry here rather than decode it again.
            for i in T.unroll(5):
                item_slot[slot, i] = meta[i]
            item_slot[slot, 5] = work[0]
            # q is one packed row of dim_nope + dim_pe; the halves are staged apart
            # because they contract against different keys.
            T.tma_copy(
                Q[q_start : q_start + half, head, 0:dim_nope],
                QNs[slot, 0, :, :],
                barrier=q_bar[slot],
            )
            T.tma_copy(
                Q[q_start + half : q_start + block_m, head, 0:dim_nope],
                QNs[slot, 1, :, :],
                barrier=q_bar[slot],
            )
            T.tma_copy(
                Q[q_start : q_start + half, head, dim_nope : dim_nope + dim_pe],
                QPs[slot, 0, :, :],
                barrier=q_bar[slot],
            )
            T.tma_copy(
                Q[q_start + half : q_start + block_m, head, dim_nope : dim_nope + dim_pe],
                QPs[slot, 1, :, :],
                barrier=q_bar[slot],
            )
            T.mbarrier_arrive(q_bar[slot])
            for k in T.serial(meta[4]):
                n = issued + k
                s = n % stages
                T.mbarrier_wait_parity(kfree[s], ((n // stages) % 2) ^ 1)
                T.tma_copy(
                    KN[seq_start + k * block_n : seq_start + (k + 1) * block_n, head, :],
                    KNs[s, :, :],
                    barrier=kready[s],
                )
                # one row per token, read without a head axis
                T.tma_copy(
                    KP[seq_start + k * block_n : seq_start + (k + 1) * block_n, :],
                    KPs[s, :, :],
                    barrier=kready[s],
                )
                T.mbarrier_arrive(kready[s])
                T.mbarrier_wait_parity(vfree[s], ((n // stages) % 2) ^ 1)
                T.tma_copy(
                    V[seq_start + k * block_n : seq_start + (k + 1) * block_n, head, :],
                    Vs[s, :, :],
                    barrier=vready[s],
                )
                T.mbarrier_arrive(vready[s])
            issued += meta[4]
            loaded += 1
            fetch(Sched, lane, work)
        # Stop the consumers, then leave the counters zeroed for the next launch.
        T.mbarrier_wait_parity(qfree[loaded % q_slots], ((loaded // q_slots) % 2) ^ 1)
        item_slot[loaded % q_slots, 5] = -1
        T.mbarrier_arrive(q_bar[loaded % q_slots])
        if lane == 0:
            finished = T.atomic_add(Sched[1], 1, return_prev=True)
            if finished == num_ctas - 1:
                Sched[0] = 0
                Sched[1] = 0

    @T.macro
    def consumer(
        wg: int, QNs, QPs, KNs, KPs, Vs, O, LSE, item_slot, q_bar, qfree, kready, kfree,
        vready, vfree, meta,
    ):  # fmt: skip
        """Consumer warpgroup *wg*: rows ``q0 + wg*64`` onward of each claimed query tile."""
        T.set_max_nreg(240, 1)
        my_bar = 1 + wg
        nxt_bar = 2 - wg
        acc_s = T.alloc_fragment([half, block_n], accum)
        pcast = T.alloc_fragment([half, block_n], dtype)
        acc_o = T.alloc_fragment([half, dim_v], accum)
        sm = T.alloc_fragment([half], accum)
        smp = T.alloc_fragment([half], accum)
        alpha = T.alloc_fragment([half], accum)
        ss = T.alloc_fragment([half], accum)
        red = T.alloc_fragment([half, 8], accum)
        logsum = T.alloc_fragment([half], accum)
        full = T.alloc_var("int32", init=0)
        # Finished KV tiles and items: they carry the barrier phases.
        done = T.alloc_var("int32", init=0)
        served = T.alloc_var("int32", init=0)
        work = T.alloc_var("int32", init=0)
        T.mbarrier_wait_parity(q_bar[0], 0)
        work = item_slot[0, 5]
        while work >= 0:
            for i in T.unroll(5):
                meta[i] = item_slot[served % q_slots, i]
            head = meta[0]
            seq_start = meta[1]
            seq_len = meta[2]
            row = meta[3] + wg * half
            eff = meta[4]
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(alpha, 1.0)
            T.fill(sm, -T.infinity(accum))
            if wg == 1 and served == 0:
                T.named_barrier_arrive(1, consumers)  # let warpgroup 0 go first

            # KV tile 0: QK and softmax only.
            s0 = done % stages
            T.sync_threads(my_bar, consumers)
            T.mbarrier_wait_parity(kready[s0], (done // stages) % 2)
            score(
                QNs[served % q_slots, wg, :, :],
                QPs[served % q_slots, wg, :, :],
                KNs,
                KPs,
                s0,
                acc_s,
            )
            T.named_barrier_arrive(nxt_bar, consumers)
            T.wait_wgmma(0)
            T.mbarrier_arrive(kfree[s0])
            if is_causal and row < block_n - 1:
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(row + i >= j, acc_s[i, j], -T.infinity(accum))
            elif not is_causal and seq_len < block_n:
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(j < seq_len, acc_s[i, j], -T.infinity(accum))
            T.reduce_max(acc_s, sm, dim=1, clear=False)
            for i, j in T.Parallel(half, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
            T.reduce_sum(acc_s, ss, dim=1, batch=2)
            for i in T.Parallel(half):
                logsum[i] = ss[i]
            T.copy(acc_s, pcast)

            # Tiles 1 .. full - 1 need no mask; full .. eff - 1 do.
            if is_causal:
                full = T.max(1, T.min(eff, T.floordiv(row + 1, block_n)))
            else:
                full = T.max(1, T.floordiv(seq_len, block_n))
            step = (KNs, KPs, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp)
            for k in T.serial(1, full):
                kv_step(
                    QNs[served % q_slots, wg, :, :], QPs[served % q_slots, wg, :, :], *step, alpha, ss, red,
                    logsum, my_bar, nxt_bar, k, done + k, row, seq_len, False,
                )  # fmt: skip
            for k in T.serial(full, eff):
                kv_step(
                    QNs[served % q_slots, wg, :, :], QPs[served % q_slots, wg, :, :], *step, alpha, ss, red,
                    logsum, my_bar, nxt_bar, k, done + k, row, seq_len, True,
                )  # fmt: skip
            # Every QK is done, so the producer may reload Q.
            T.mbarrier_arrive(qfree[served % q_slots])

            # PV of the last tile, then normalize and store.
            svp = (done + eff - 1) % stages
            for i, j in T.Parallel(half, dim_v):
                acc_o[i, j] *= alpha[i]
            T.mbarrier_wait_parity(vready[svp], ((done + eff - 1) // stages) % 2)
            T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
            T.wait_wgmma(0)
            T.mbarrier_arrive(vfree[svp])
            for i, j in T.Parallel(half, dim_v):
                if row + i < seq_len:
                    O[seq_start + row + i, head, j] = T.if_then_else(
                        logsum[i] > 0,
                        T.cast(acc_o[i, j] / logsum[i], dtype),
                        T.cast(0, dtype),
                    )
            # log-sum-exp in the natural base, so a caller merging chunked-context
            # partials needs nothing of this kernel's exp2 scaling.
            for i in T.Parallel(half):
                if row + i < seq_len:
                    LSE[seq_start + row + i, head] = T.if_then_else(
                        logsum[i] > 0,
                        T.log(logsum[i]) + sm[i] * score_scale,
                        -T.infinity(accum),
                    )
            done += eff
            served += 1
            T.mbarrier_wait_parity(q_bar[served % q_slots], (served // q_slots) % 2)
            work = item_slot[served % q_slots, 5]

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, heads, dim_nope + dim_pe], dtype),
        KN: T.Tensor([total_q, heads, dim_nope], dtype),
        KP: T.Tensor([total_q, dim_pe], dtype),
        V: T.Tensor([total_q, heads, dim_v], dtype),
        CuQ: T.Tensor([batch + 1], "int32"),
        O: T.Tensor([total_q, heads, dim_v], dtype),
        LSE: T.Tensor([total_q, heads], accum),
        Sched: T.Tensor([2], "int32"),
    ):
        with T.Kernel(num_ctas, threads=384):
            QNs = T.alloc_shared([q_slots, 2, half, dim_nope], dtype)
            QPs = T.alloc_shared([q_slots, 2, half, dim_pe], dtype)
            KNs = T.alloc_shared([stages, block_n, dim_nope], dtype)
            KPs = T.alloc_shared([stages, block_n, dim_pe], dtype)
            Vs = T.alloc_shared([stages, block_n, dim_v], dtype)
            tile_cum = T.alloc_shared([batch + 1], "int32")
            item_slot = T.alloc_shared([q_slots, 6], "int32")
            T.annotate_layout(
                {
                    QNs: make_swizzled_layout(QNs),
                    QPs: make_swizzled_layout(QPs),
                    KNs: make_swizzled_layout(KNs),
                    KPs: make_swizzled_layout(KPs),
                    Vs: make_swizzled_layout(Vs),
                }
            )
            q_bar = T.alloc_barrier([32] * q_slots)
            qfree = T.alloc_barrier([consumers] * q_slots)
            kready = T.alloc_barrier([32] * stages)
            kfree = T.alloc_barrier([consumers] * stages)
            vready = T.alloc_barrier([32] * stages)
            vfree = T.alloc_barrier([consumers] * stages)
            lo = T.alloc_local([1], "int32")
            hi = T.alloc_local([1], "int32")
            q_row = T.alloc_local([1], "int32")
            request = T.alloc_local([1], "int32")
            meta = T.alloc_local([5], "int32")

            q_tiling.cumsum_offsets(CuQ, tile_cum)
            T.sync_threads()
            tx = T.get_thread_binding()
            if tx >= consumers:
                T.set_max_nreg(24, 0)  # the producer only issues TMA
            if tx >= consumers and tx < consumers + 32:
                producer(
                    Q, KN, KP, V, CuQ, Sched, QNs, QPs, KNs, KPs, Vs, item_slot, q_bar, qfree,
                    kready, kfree, vready, vfree, tile_cum[batch] * heads, tile_cum, lo, hi,
                    request, q_row, meta, tx - consumers,
                )  # fmt: skip
            args = (QNs, QPs, KNs, KPs, Vs, O, LSE, item_slot, q_bar, qfree, kready, kfree,
                    vready, vfree, meta)  # fmt: skip
            with T.ws(0):
                consumer(0, *args)
            with T.ws(1):
                consumer(1, *args)

    return main


class MLAVarlenPrefillWSFwdKernel(Kernel, MlaVarlenFwdInterface):
    """SM90 warp-specialized packed-varlen MLA prefill.

    One persistent CTA per SM, a TMA producer warp, and two consumer warpgroups of
    64 query rows each.
    """

    supported_archs: list[int] = [90]
    _BLOCK_N: int = 128
    _STAGES: int = 2
    _Q_SLOTS: int = 1
    # The shared buffers hold the Q slot, both key halves and V: at block_n 128 and
    # two stages that is 208 KB of the 227 KB an SM90 block may use, leaving the
    # per-request prefix room. A second Q slot would need 256 KB and does not fit,
    # and measured against it a wider key tile is worth more than prefetching the
    # next item's Q: 10.91 ms against 12.10 ms on ds-v3-8x4k. Re-measure if any
    # buffer grows.
    _MAX_BATCH: int = 256

    @classmethod
    def applies(cls, call: MlaVarlenCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: MlaVarlenCall) -> Optional[str]:
        """Why *call* is outside the shapes this schedule serves."""
        if call.dtype not in ATTENTION_DTYPES:
            return f"serves float16 and bfloat16, got {call.dtype}"
        if call.dim_nope % 64 != 0 or call.dim_pe % 64 != 0 or call.dim_v % 64 != 0:
            return (
                "every head dimension is a whole number of 64-wide WGMMA operands, got "
                f"{call.dim_nope}, {call.dim_pe}, {call.dim_v}"
            )
        if call.batch > cls._MAX_BATCH:
            return f"the per-request prefix is held in shared memory, got batch {call.batch}"
        return None

    @classmethod
    def entry_for(cls, call: MlaVarlenCall) -> Entry:
        args = dict(
            batch=call.batch,
            heads=call.heads,
            dim_nope=call.dim_nope,
            dim_pe=call.dim_pe,
            dim_v=call.dim_v,
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
        dim_nope: int,
        dim_pe: int,
        dim_v: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        """Build the program for one call shape.

        Args:
            batch: Requests the packing carries, so the length of ``cu_seqlens`` less one.
            heads: Query heads.
            dim_nope: Width of the key's per-head half.
            dim_pe: Width of the key's shared rope half.
            dim_v: Width of a value row.
            is_causal: Whether a query row reads only keys at or before its position.
            dtype: Input and output dtype.
            sm_scale: Score scale, or ``None`` for ``(dim_nope + dim_pe) ** -0.5``.
            config: Unused; this kernel's geometry is fixed by its class constants.
            tune: Unused; there is nothing to search.
            device_index: CUDA device the program is built for.
        """
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.dim_nope = dim_nope
        self.dim_pe = dim_pe
        self.dim_v = dim_v
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = sm_scale
        # One counter per stream: concurrent launches on one counter steal each other's items.
        self._counters: dict = {}
        self.kernel = _mla_varlen_ws_kernel(
            batch,
            heads,
            dim_nope,
            dim_pe,
            dim_v,
            is_causal,
            sm_scale,
            self.dtype_str,
            self._BLOCK_N,
            self._STAGES,
            self._Q_SLOTS,
            get_sm_count(device_index),
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {}

    def forward(
        self,
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend each query row to the keys its own request's mask admits.

        Args:
            q: ``(total_tokens, heads, dim_nope + dim_pe)``.
            k_nope: ``(total_tokens, heads, dim_nope)``.
            k_pe: ``(total_tokens, dim_pe)``, one row per token for every head.
            v: ``(total_tokens, heads, dim_v)``.
            cu_seqlens: ``(batch + 1)`` packed request offsets.

        Returns:
            The ``(total_tokens, heads, dim_v)`` output and its float32
            ``(total_tokens, heads)`` log-sum-exp.
        """
        self._require_cuda(q=q, k_nope=k_nope, k_pe=k_pe, v=v)
        if torch.cuda.is_current_stream_capturing():
            # Each captured graph owns a counter from its private pool.
            counter = torch.zeros(2, dtype=torch.int32, device=q.device)
        else:
            stream = torch.cuda.current_stream(q.device).cuda_stream
            counter = self._counters.get(stream)
            if counter is None:
                counter = torch.zeros(2, dtype=torch.int32, device=q.device)
                self._counters[stream] = counter
        out = torch.empty((q.shape[0], self.heads, self.dim_v), dtype=q.dtype, device=q.device)
        lse = torch.empty((q.shape[0], self.heads), dtype=torch.float32, device=q.device)
        self.kernel(q, k_nope, k_pe, v, cu_seqlens, out, lse, counter)
        return out, lse
