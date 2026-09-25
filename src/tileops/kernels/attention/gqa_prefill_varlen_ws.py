"""SM90 warp-specialized packed variable-length GQA prefill kernel."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops.kernels.constants import LOG2E
from tileops.utils import get_sm_count

from ..grouped_tiling import GroupTiling
from .call_spec import ATTENTION_DTYPES, uses_sliding_window
from .varlen import VarlenKernel, varlen_entry

__all__ = ["GQAPrefillVarlenWSFwdKernel"]


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    out_idx=[5],
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
def _gqa_prefill_varlen_ws_kernel(
    batch, heads, heads_kv, dim, is_causal, sm_scale, softcap, dtype, block_n, stages, num_ctas
):
    """A persistent CTA per SM: a TMA producer warp claims work, two consumer warpgroups run it."""
    score_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = heads // heads_kv
    accum = "float"
    half = 64  # rows per consumer warpgroup: one m64 WGMMA
    block_m = 2 * half
    consumers = 256  # threads of the two consumer warpgroups
    policy = T.GemmWarpPolicy.FullRow
    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    q_tiling = GroupTiling(batch, block_m)

    @T.macro
    def apply_softcap(acc_s):
        for i, j in T.Parallel(half, block_n):
            capped = T.cast(softcap, accum) * T.tanh(
                acc_s[i, j] * T.cast(score_scale / softcap, accum)
            )
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum), -T.infinity(accum), capped
            )

    @T.macro
    def softmax_step(acc_s, sm, smp, alpha, ss):
        """Fold one score tile into the running max, rescale factor, and row sums."""
        if use_softcap:
            apply_softcap(acc_s)
        T.copy(sm, smp)
        T.reduce_max(acc_s, sm, dim=1, clear=False)
        for i in T.Parallel(half):
            alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
        for i, j in T.Parallel(half, block_n):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
        T.reduce_sum(acc_s, ss, dim=1)

    @T.macro
    def kv_step(
        q_tile,
        Ks,
        Vs,
        kready,
        kfree,
        vready,
        vfree,
        acc_s,
        pcast,
        acc_o,
        sm,
        smp,
        alpha,
        ss,
        logsum,
        my_bar,
        nxt_bar,
        k,
        n,
        row,
        causal_offset,
        kv_len,
        tail: bool,
    ):
        """QK of KV tile k overlapped with PV of tile k - 1; n counts tiles across work items."""
        sk = n % stages
        svp = (n - 1) % stages
        T.sync_threads(my_bar, consumers)
        T.mbarrier_wait_parity(kready[sk], (n // stages) % 2)
        T.wgmma_gemm(q_tile, Ks[sk, :, :], acc_s, transpose_B=True, policy=policy, clear_accum=True)
        for i, j in T.Parallel(half, dim):
            acc_o[i, j] *= alpha[i]
        T.mbarrier_wait_parity(vready[svp], ((n - 1) // stages) % 2)
        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
        T.named_barrier_arrive(nxt_bar, consumers)
        T.wait_wgmma(1)
        T.mbarrier_arrive(kfree[sk])
        if tail:
            if is_causal:
                limit = row + causal_offset - k * block_n
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(limit + i >= j, acc_s[i, j], -T.infinity(accum))
            else:
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(
                        k * block_n + j < kv_len, acc_s[i, j], -T.infinity(accum)
                    )
        softmax_step(acc_s, sm, smp, alpha, ss)
        T.wait_wgmma(0)
        T.mbarrier_arrive(vfree[svp])
        for i in T.Parallel(half):
            logsum[i] = logsum[i] * alpha[i] + ss[i]
        T.copy(acc_s, pcast)

    @T.macro
    def locate(work, CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta):
        """Fill meta (head, q_start, kv_start, q_len, kv_len, q0, eff) for work item *work*."""
        q_tiling.decode(tile_cum[batch] - 1 - work // heads, tile_cum, lo, hi, request, q_row)
        meta[0] = work % heads
        meta[1] = CuQ[request[0]]
        meta[2] = CuKV[request[0]]
        meta[3] = CuQ[request[0] + 1] - meta[1]
        meta[4] = CuKV[request[0] + 1] - meta[2]
        meta[5] = q_row[0]
        if is_causal:
            meta[6] = T.max(
                1,
                T.min(
                    T.ceildiv(meta[4], block_n),
                    T.ceildiv(meta[5] + block_m + meta[4] - meta[3], block_n),
                ),
            )
        else:
            # At least one tile: the consumers always wait on tile 0.
            meta[6] = T.max(1, T.ceildiv(meta[4], block_n))

    @T.macro
    def fetch(Sched, lane, work):
        """Claim the next work item into work[0]: lane 0 claims, the warp shares it."""
        if lane == 0:
            work[0] = T.atomic_add(Sched[0], 1, return_prev=True)
        work[0] = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), work[0], 0, 32, 32)

    @T.macro
    def producer(
        Q, K, V, CuQ, CuKV, Sched, Qs, Ks, Vs, item_slot, q_bar, qfree, kready, kfree, vready,
        vfree, total, tile_cum, lo, hi, request, q_row, meta, lane,
    ):  # fmt: skip
        """Claim work items until none remain; load each one's Q, then its KV tiles."""
        issued = T.alloc_var("int32", init=0)
        loaded = T.alloc_var("int32", init=0)
        work = T.alloc_local([1], "int32")
        fetch(Sched, lane, work)
        while work[0] < total:
            locate(work[0], CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta)
            head = meta[0]
            q_start = meta[1] + meta[5]
            kv_start = meta[2]
            cv = head // groups
            T.mbarrier_wait_parity(qfree, (loaded % 2) ^ 1)
            item_slot[0] = work[0]
            T.tma_copy(Q[q_start : q_start + half, head, :], Qs[0, :, :], barrier=q_bar)
            T.tma_copy(Q[q_start + half : q_start + block_m, head, :], Qs[1, :, :], barrier=q_bar)
            T.mbarrier_arrive(q_bar)
            for k in T.serial(meta[6]):
                n = issued + k
                s = n % stages
                T.mbarrier_wait_parity(kfree[s], ((n // stages) % 2) ^ 1)
                T.tma_copy(
                    K[kv_start + k * block_n : kv_start + (k + 1) * block_n, cv, :],
                    Ks[s, :, :],
                    barrier=kready[s],
                )
                T.mbarrier_arrive(kready[s])
                T.mbarrier_wait_parity(vfree[s], ((n // stages) % 2) ^ 1)
                T.tma_copy(
                    V[kv_start + k * block_n : kv_start + (k + 1) * block_n, cv, :],
                    Vs[s, :, :],
                    barrier=vready[s],
                )
                T.mbarrier_arrive(vready[s])
            issued += meta[6]
            loaded += 1
            fetch(Sched, lane, work)
        # Stop the consumers, then leave the counters zeroed for the next launch.
        T.mbarrier_wait_parity(qfree, (loaded % 2) ^ 1)
        item_slot[0] = -1
        T.mbarrier_arrive(q_bar)
        if lane == 0:
            finished = T.atomic_add(Sched[1], 1, return_prev=True)
            if finished == num_ctas - 1:
                Sched[0] = 0
                Sched[1] = 0

    @T.macro
    def consumer(
        wg: int, CuQ, CuKV, Qs, Ks, Vs, Os, O, item_slot, q_bar, qfree, kready, kfree, vready,
        vfree, tile_cum, lo, hi, request, q_row, meta,
    ):  # fmt: skip
        """Consumer warpgroup *wg*: rows ``q0 + wg*64`` onward of each claimed query tile."""
        T.set_max_nreg(240, 1)
        my_bar = 1 + wg
        nxt_bar = 2 - wg
        acc_s = T.alloc_fragment([half, block_n], accum)
        pcast = T.alloc_fragment([half, block_n], dtype)
        acc_o = T.alloc_fragment([half, dim], accum)
        sm = T.alloc_fragment([half], accum)
        smp = T.alloc_fragment([half], accum)
        alpha = T.alloc_fragment([half], accum)
        ss = T.alloc_fragment([half], accum)
        logsum = T.alloc_fragment([half], accum)
        full = T.alloc_var("int32", init=0)
        # Finished KV tiles and items: they carry the barrier phases.
        done = T.alloc_var("int32", init=0)
        served = T.alloc_var("int32", init=0)
        work = T.alloc_var("int32", init=0)
        T.mbarrier_wait_parity(q_bar, 0)
        work = item_slot[0]
        while work >= 0:
            locate(work, CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta)
            head = meta[0]
            q_start = meta[1]
            q_len = meta[3]
            kv_len = meta[4]
            causal_offset = meta[4] - meta[3]
            row = meta[5] + wg * half
            eff = meta[6]
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
            T.wgmma_gemm(
                Qs[wg, :, :], Ks[s0, :, :], acc_s, transpose_B=True, policy=policy, clear_accum=True
            )
            T.named_barrier_arrive(nxt_bar, consumers)
            T.wait_wgmma(0)
            T.mbarrier_arrive(kfree[s0])
            if is_causal and row + causal_offset < block_n - 1:
                limit = row + causal_offset
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(limit + i >= j, acc_s[i, j], -T.infinity(accum))
            elif not is_causal and kv_len < block_n:
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(j < kv_len, acc_s[i, j], -T.infinity(accum))
            if use_softcap:
                apply_softcap(acc_s)
            T.reduce_max(acc_s, sm, dim=1, clear=False)
            for i, j in T.Parallel(half, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
            T.reduce_sum(acc_s, ss, dim=1)
            for i in T.Parallel(half):
                logsum[i] = ss[i]
            T.copy(acc_s, pcast)

            # Tiles 1 .. full - 1 need no mask; full .. eff - 1 do.
            if is_causal:
                full = T.max(1, T.min(eff, T.floordiv(row + causal_offset + 1, block_n)))
            else:
                full = T.max(1, T.floordiv(kv_len, block_n))
            step = (Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp)
            for k in T.serial(1, full):
                kv_step(
                    Qs[wg, :, :], *step, alpha, ss, logsum, my_bar, nxt_bar, k, done + k, row,
                    causal_offset, kv_len, False,
                )  # fmt: skip
            for k in T.serial(full, eff):
                kv_step(
                    Qs[wg, :, :], *step, alpha, ss, logsum, my_bar, nxt_bar, k, done + k, row,
                    causal_offset, kv_len, True,
                )  # fmt: skip
            # Every QK is done, so the producer may reload Q.
            T.mbarrier_arrive(qfree)

            # PV of the last tile, then normalize and store.
            svp = (done + eff - 1) % stages
            for i, j in T.Parallel(half, dim):
                acc_o[i, j] *= alpha[i]
            T.mbarrier_wait_parity(vready[svp], ((done + eff - 1) // stages) % 2)
            T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
            T.wait_wgmma(0)
            T.mbarrier_arrive(vfree[svp])
            # Every row sees a key: causal needs kv_len >= q_len, non-causal kv_len > 0.
            if row + half <= q_len and kv_len >= (q_len if is_causal else 1):
                for i in T.Parallel(half):
                    alpha[i] = 1.0 / logsum[i]
                for i, j in T.Parallel(half, dim):
                    acc_o[i, j] *= alpha[i]
                # The warpgroup has stored the previous item before Os is reused.
                T.sync_threads(3 + wg, 128)
                T.copy(acc_o, Os[wg, :, :])
                T.sync_threads(3 + wg, 128)
                T.copy(Os[wg, :, :], O[q_start + row : q_start + row + half, head, :])
            else:
                # A partial tile, or rows with no visible key (written as zero).
                for i, j in T.Parallel(half, dim):
                    if row + i < q_len:
                        O[q_start + row + i, head, j] = T.if_then_else(
                            logsum[i] > 0,
                            T.cast(acc_o[i, j] / logsum[i], dtype),
                            T.cast(0, dtype),
                        )
            done += eff
            served += 1
            T.mbarrier_wait_parity(q_bar, served % 2)
            work = item_slot[0]

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, heads, dim], dtype),
        K: T.Tensor([total_kv, heads_kv, dim], dtype),
        V: T.Tensor([total_kv, heads_kv, dim], dtype),
        CuQ: T.Tensor([batch + 1], "int32"),
        CuKV: T.Tensor([batch + 1], "int32"),
        O: T.Tensor([total_q, heads, dim], dtype),
        Sched: T.Tensor([2], "int32"),
    ):
        with T.Kernel(num_ctas, threads=384):
            Qs = T.alloc_shared([2, half, dim], dtype)
            Ks = T.alloc_shared([stages, block_n, dim], dtype)
            Vs = T.alloc_shared([stages, block_n, dim], dtype)
            Os = T.alloc_shared([2, half, dim], dtype)
            tile_cum = T.alloc_shared([batch + 1], "int32")
            item_slot = T.alloc_shared([1], "int32")
            T.annotate_layout(
                {
                    Qs: make_swizzled_layout(Qs),
                    Ks: make_swizzled_layout(Ks),
                    Vs: make_swizzled_layout(Vs),
                    Os: make_swizzled_layout(Os),
                }
            )
            q_bar = T.alloc_barrier([32])
            qfree = T.alloc_barrier([consumers])
            kready = T.alloc_barrier([32] * stages)
            kfree = T.alloc_barrier([consumers] * stages)
            vready = T.alloc_barrier([32] * stages)
            vfree = T.alloc_barrier([consumers] * stages)
            lo = T.alloc_local([1], "int32")
            hi = T.alloc_local([1], "int32")
            q_row = T.alloc_local([1], "int32")
            request = T.alloc_local([1], "int32")
            meta = T.alloc_local([7], "int32")

            q_tiling.cumsum_offsets(CuQ, tile_cum)
            T.sync_threads()
            tx = T.get_thread_binding()
            if tx >= consumers:
                T.set_max_nreg(24, 0)  # the producer only issues TMA
            if tx >= consumers and tx < consumers + 32:
                producer(
                    Q, K, V, CuQ, CuKV, Sched, Qs, Ks, Vs, item_slot, q_bar, qfree, kready, kfree,
                    vready, vfree, tile_cum[batch] * heads, tile_cum, lo, hi, request, q_row, meta,
                    tx - consumers,
                )  # fmt: skip
            args = (CuQ, CuKV, Qs, Ks, Vs, Os, O, item_slot, q_bar, qfree, kready, kfree, vready)
            with T.ws(0):
                consumer(0, *args, vfree, tile_cum, lo, hi, request, q_row, meta)
            with T.ws(1):
                consumer(1, *args, vfree, tile_cum, lo, hi, request, q_row, meta)

    return main


class GQAPrefillVarlenWSFwdKernel(VarlenKernel):
    """SM90 warp-specialized packed prefill for 128-wide heads."""

    supported_archs: list[int] = [90]
    # Fitted on H200 at dim 128; re-measure against the general kernel to change.
    _BLOCK_N: int = 128
    _STAGES: int = 2

    @classmethod
    def applies(cls, call) -> bool:
        return (
            call.dtype in ATTENTION_DTYPES
            and call.dim == 128
            and not call.is_fp8
            and not call.fuse_rope
            and not uses_sliding_window(call)
            and not call.empty_kv
            and (call.backend == "varlen" or (call.backend == "auto" and not call.is_uniform))
        )

    @classmethod
    def entry_for(cls, call):
        return varlen_entry(cls, call)

    def _make_kernel(self):
        return _gqa_prefill_varlen_ws_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            self._BLOCK_N,
            self._STAGES,
            get_sm_count(self.device_index),
        )

    @property
    def default_config(self) -> dict:
        return {}

    def __init__(self, *args, **kwargs) -> None:
        # One counter per stream: concurrent launches on one counter steal each other's items.
        self._counters: dict = {}
        super().__init__(*args, **kwargs)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._require_cuda(q=q, k=k, v=v)
        if torch.cuda.is_current_stream_capturing():
            # Each captured graph owns a counter from its private pool.
            counter = torch.zeros(2, dtype=torch.int32, device=q.device)
        else:
            stream = torch.cuda.current_stream(q.device).cuda_stream
            counter = self._counters.get(stream)
            if counter is None:
                counter = torch.zeros(2, dtype=torch.int32, device=q.device)
                self._counters[stream] = counter
        return self.kernel(q, k, v, cu_seqlens_q, cu_seqlens_kv, counter)
