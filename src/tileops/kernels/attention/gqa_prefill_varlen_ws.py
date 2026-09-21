"""SM90 warp-specialized packed variable-length GQA prefill kernel."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops.kernels.constants import LOG2E

from ..grouped_tiling import GroupTiling
from .call_spec import ATTENTION_DTYPES, uses_sliding_window
from .varlen import VarlenKernel, varlen_entry

__all__ = ["GQAPrefillVarlenWsKernel"]


# SM90 warp-specialized packed variable-length attention.
BLOCK_M = 128
BLOCK_N = 128
NSK = 2
NSV = 2
THREADS = 384
NMMA = 256

_pc = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}
_cf = [
    "-O3",
    "--use_fast_math",
    "-Wno-deprecated-declarations",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-DNDEBUG",
]


@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[5], pass_configs=_pc, compile_flags=_cf)
def _gqa_prefill_varlen_ws_kernel(
    batch,
    H,
    Hkv,
    D,
    is_causal,
    sm_scale,
    softcap,
    dtype,
    block_M=BLOCK_M,
    block_N=BLOCK_N,
    nsK=NSK,
    nsV=NSV,
    threads=THREADS,
):
    """Build the Dense WS program; its online softmax carries the previous tile's alpha."""
    score_scale = (1.0 / D) ** 0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = H // Hkv
    accum = "float"
    half = block_M // 2
    Pol = T.GemmWarpPolicy.FullRow
    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    q_tiling = GroupTiling(batch, block_M)
    num_q_tiles = q_tiling.tile_upper_bound(total_q)

    @T.macro
    def apply_softcap(acc_s, rows, cols):
        for i, j in T.Parallel(rows, cols):
            capped = T.cast(softcap, accum) * T.tanh(
                acc_s[i, j] * T.cast(score_scale / softcap, accum)
            )
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum), -T.infinity(accum), capped
            )

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, H, D], dtype),
        K: T.Tensor([total_kv, Hkv, D], dtype),
        V: T.Tensor([total_kv, Hkv, D], dtype),
        CuQ: T.Tensor([batch + 1], "int32"),
        CuKV: T.Tensor([batch + 1], "int32"),
        O: T.Tensor([total_q, H, D], dtype),
    ):
        with T.Kernel(num_q_tiles, H, threads=threads) as (bx, by):
            Qs = T.alloc_shared([2, half, D], dtype)
            Ks = T.alloc_shared([nsK, block_N, D], dtype)
            Vs = T.alloc_shared([nsV, block_N, D], dtype)
            T.annotate_layout(
                {
                    Qs: make_swizzled_layout(Qs),
                    Ks: make_swizzled_layout(Ks),
                    Vs: make_swizzled_layout(Vs),
                }
            )

            q_bar = T.alloc_barrier([32])  # 1-warp producer (FlashInfer NUM_PRODUCER_THREADS=32)
            kready = T.alloc_barrier([32] * nsK)
            kfree = T.alloc_barrier([NMMA] * nsK)
            vready = T.alloc_barrier([32] * nsV)
            vfree = T.alloc_barrier([NMMA] * nsV)

            tile_cum = T.alloc_shared([batch + 1], "int32")
            lo = T.alloc_local([1], "int32")
            hi = T.alloc_local([1], "int32")
            q_row = T.alloc_local([1], "int32")
            request = T.alloc_local([1], "int32")

            q_tiling.cumsum_offsets(CuQ, tile_cum)
            T.sync_threads()
            if bx < tile_cum[batch]:
                q_tiling.decode(bx, tile_cum, lo, hi, request, q_row)

                cv = by // groups
                q_start = CuQ[request[0]]
                kv_start = CuKV[request[0]]
                q_len = CuQ[request[0] + 1] - q_start
                kv_len = CuKV[request[0] + 1] - kv_start
                q0 = q_row[0]
                causal_offset = T.alloc_var("int32", init=kv_len - q_len)
                if is_causal:
                    eff = T.alloc_var(
                        "int32",
                        init=T.max(
                            1,
                            T.min(
                                T.ceildiv(kv_len, block_N),
                                T.ceildiv(q0 + block_M + causal_offset, block_N),
                            ),
                        ),
                    )
                else:
                    eff = T.alloc_var("int32", init=T.ceildiv(kv_len, block_N))
                tx = T.get_thread_binding()

                if tx >= 256:  # ================= producer =================
                    T.set_max_nreg(24, 0)  # producer is TMA-only: release regs to consumers
                if tx >= 256 and tx < 288:  # only 1 warp issues TMA + waits (rest of WG idle)
                    T.tma_copy(
                        Q[q_start + q0 : q_start + q0 + half, by, :],
                        Qs[0, :, :],
                        barrier=q_bar,
                    )
                    T.tma_copy(
                        Q[q_start + q0 + half : q_start + q0 + block_M, by, :],
                        Qs[1, :, :],
                        barrier=q_bar,
                    )
                    T.mbarrier_arrive(q_bar)
                    for k in T.serial(eff):
                        sk = k % nsK
                        T.mbarrier_wait_parity(kfree[sk], ((k // nsK) % 2) ^ 1)
                        T.tma_copy(
                            K[kv_start + k * block_N : kv_start + (k + 1) * block_N, cv, :],
                            Ks[sk, :, :],
                            barrier=kready[sk],
                        )
                        T.mbarrier_arrive(kready[sk])
                        sv = k % nsV
                        T.mbarrier_wait_parity(vfree[sv], ((k // nsV) % 2) ^ 1)
                        T.tma_copy(
                            V[kv_start + k * block_N : kv_start + (k + 1) * block_N, cv, :],
                            Vs[sv, :, :],
                            barrier=vready[sv],
                        )
                        T.mbarrier_arrive(vready[sv])

                with T.ws(0):
                    T.set_max_nreg(240, 1)  # consumer grabs producer's released regs
                    r0 = 0 * half
                    my_bar = 1
                    nxt_bar = 2
                    acc_s = T.alloc_fragment([half, block_N], accum)
                    pcast = T.alloc_fragment([half, block_N], dtype)  # register-P (rs-wgmma)
                    acc_o = T.alloc_fragment([half, D], accum)
                    sm = T.alloc_fragment([half], accum)
                    smp = T.alloc_fragment([half], accum)
                    alpha = T.alloc_fragment([half], accum)
                    ss = T.alloc_fragment([half], accum)
                    logsum = T.alloc_fragment([half], accum)

                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(alpha, 1.0)
                    T.fill(sm, -T.infinity(accum))
                    T.mbarrier_wait_parity(q_bar, 0)
                    pass  # WG0 goes first

                    # prologue: tile 0, QK + softmax (no PV)
                    T.sync_threads(my_bar, NMMA)
                    T.mbarrier_wait_parity(kready[0], 0)
                    T.wgmma_gemm(
                        Qs[0, :, :],
                        Ks[0, :, :],
                        acc_s,
                        transpose_B=True,
                        policy=Pol,
                        clear_accum=True,
                    )
                    T.named_barrier_arrive(nxt_bar, NMMA)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(kfree[0])
                    if is_causal and q0 + r0 + causal_offset < block_N - 1:
                        mask_limit = q0 + r0 + causal_offset
                        for i, j in T.Parallel(half, block_N):
                            acc_s[i, j] = T.if_then_else(
                                mask_limit + i >= j, acc_s[i, j], -T.infinity(accum)
                            )
                    elif not is_causal and kv_len < block_N:
                        for i, j in T.Parallel(half, block_N):
                            acc_s[i, j] = T.if_then_else(
                                j < kv_len, acc_s[i, j], -T.infinity(accum)
                            )
                    if use_softcap:
                        apply_softcap(acc_s, half, block_N)
                    T.reduce_max(acc_s, sm, dim=1, clear=False)
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                    T.reduce_sum(acc_s, ss, dim=1)
                    for i in T.Parallel(half):
                        logsum[i] = ss[i]
                    T.copy(acc_s, pcast)

                    if is_causal:
                        nu = T.alloc_var(
                            "int32",
                            init=T.max(
                                1,
                                T.min(eff, T.floordiv(q0 + r0 + causal_offset + 1, block_N)),
                            ),
                        )
                    else:
                        nu = T.alloc_var("int32", init=T.max(1, T.floordiv(kv_len, block_N)))
                    for k in T.serial(1, nu):
                        sk = k % nsK
                        svp = (k - 1) % nsV
                        T.sync_threads(my_bar, NMMA)
                        T.mbarrier_wait_parity(kready[sk], (k // nsK) % 2)
                        T.wgmma_gemm(
                            Qs[0, :, :],
                            Ks[sk, :, :],
                            acc_s,
                            transpose_B=True,
                            policy=Pol,
                            clear_accum=True,
                        )
                        for i, j in T.Parallel(half, D):
                            acc_o[i, j] *= alpha[i]
                        T.mbarrier_wait_parity(vready[svp], ((k - 1) // nsV) % 2)
                        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                        T.named_barrier_arrive(nxt_bar, NMMA)
                        T.wait_wgmma(1)
                        T.mbarrier_arrive(kfree[sk])
                        if use_softcap:
                            apply_softcap(acc_s, half, block_N)
                        T.copy(sm, smp)
                        T.reduce_max(acc_s, sm, dim=1, clear=False)
                        for i in T.Parallel(half):
                            alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                        for i, j in T.Parallel(half, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                        T.reduce_sum(acc_s, ss, dim=1)
                        T.wait_wgmma(0)
                        T.mbarrier_arrive(vfree[svp])
                        for i in T.Parallel(half):
                            logsum[i] = logsum[i] * alpha[i] + ss[i]
                        T.copy(acc_s, pcast)
                    for k in T.serial(nu, eff):
                        sk = k % nsK
                        svp = (k - 1) % nsV
                        T.sync_threads(my_bar, NMMA)
                        T.mbarrier_wait_parity(kready[sk], (k // nsK) % 2)
                        T.wgmma_gemm(
                            Qs[0, :, :],
                            Ks[sk, :, :],
                            acc_s,
                            transpose_B=True,
                            policy=Pol,
                            clear_accum=True,
                        )
                        for i, j in T.Parallel(half, D):
                            acc_o[i, j] *= alpha[i]
                        T.mbarrier_wait_parity(vready[svp], ((k - 1) // nsV) % 2)
                        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                        T.named_barrier_arrive(nxt_bar, NMMA)
                        T.wait_wgmma(1)
                        T.mbarrier_arrive(kfree[sk])
                        if is_causal:
                            mask_limit_tail = q0 + r0 + causal_offset - k * block_N
                            for i, j in T.Parallel(half, block_N):
                                acc_s[i, j] = T.if_then_else(
                                    mask_limit_tail + i >= j, acc_s[i, j], -T.infinity(accum)
                                )
                        else:
                            for i, j in T.Parallel(half, block_N):
                                acc_s[i, j] = T.if_then_else(
                                    k * block_N + j < kv_len,
                                    acc_s[i, j],
                                    -T.infinity(accum),
                                )
                        if use_softcap:
                            apply_softcap(acc_s, half, block_N)
                        T.copy(sm, smp)
                        T.reduce_max(acc_s, sm, dim=1, clear=False)
                        for i in T.Parallel(half):
                            alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                        for i, j in T.Parallel(half, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                        T.reduce_sum(acc_s, ss, dim=1)
                        T.wait_wgmma(0)
                        T.mbarrier_arrive(vfree[svp])
                        for i in T.Parallel(half):
                            logsum[i] = logsum[i] * alpha[i] + ss[i]
                        T.copy(acc_s, pcast)

                    svp = (eff - 1) % nsV
                    for i, j in T.Parallel(half, D):
                        acc_o[i, j] *= alpha[i]
                    T.mbarrier_wait_parity(vready[svp], ((eff - 1) // nsV) % 2)
                    T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vfree[svp])
                    for i, j in T.Parallel(half, D):
                        if q0 + r0 + i < q_len:
                            O[q_start + q0 + r0 + i, by, j] = T.if_then_else(
                                logsum[i] > 0,
                                T.cast(acc_o[i, j] / logsum[i], dtype),
                                T.cast(0, dtype),
                            )

                with T.ws(1):
                    T.set_max_nreg(240, 1)  # consumer grabs producer's released regs
                    r0 = 1 * half
                    my_bar = 2
                    nxt_bar = 1
                    acc_s = T.alloc_fragment([half, block_N], accum)
                    pcast = T.alloc_fragment([half, block_N], dtype)  # register-P (rs-wgmma)
                    acc_o = T.alloc_fragment([half, D], accum)
                    sm = T.alloc_fragment([half], accum)
                    smp = T.alloc_fragment([half], accum)
                    alpha = T.alloc_fragment([half], accum)
                    ss = T.alloc_fragment([half], accum)
                    logsum = T.alloc_fragment([half], accum)

                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(alpha, 1.0)
                    T.fill(sm, -T.infinity(accum))
                    T.mbarrier_wait_parity(q_bar, 0)
                    T.named_barrier_arrive(1, NMMA)  # prime WG0

                    # prologue: tile 0, QK + softmax (no PV)
                    T.sync_threads(my_bar, NMMA)
                    T.mbarrier_wait_parity(kready[0], 0)
                    T.wgmma_gemm(
                        Qs[1, :, :],
                        Ks[0, :, :],
                        acc_s,
                        transpose_B=True,
                        policy=Pol,
                        clear_accum=True,
                    )
                    T.named_barrier_arrive(nxt_bar, NMMA)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(kfree[0])
                    if is_causal and q0 + r0 + causal_offset < block_N - 1:
                        mask_limit_wg1 = q0 + r0 + causal_offset
                        for i, j in T.Parallel(half, block_N):
                            acc_s[i, j] = T.if_then_else(
                                mask_limit_wg1 + i >= j, acc_s[i, j], -T.infinity(accum)
                            )
                    elif not is_causal and kv_len < block_N:
                        for i, j in T.Parallel(half, block_N):
                            acc_s[i, j] = T.if_then_else(
                                j < kv_len, acc_s[i, j], -T.infinity(accum)
                            )
                    if use_softcap:
                        apply_softcap(acc_s, half, block_N)
                    T.reduce_max(acc_s, sm, dim=1, clear=False)
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                    T.reduce_sum(acc_s, ss, dim=1)
                    for i in T.Parallel(half):
                        logsum[i] = ss[i]
                    T.copy(acc_s, pcast)

                    if is_causal:
                        nu_wg1 = T.alloc_var(
                            "int32",
                            init=T.max(
                                1,
                                T.min(eff, T.floordiv(q0 + r0 + causal_offset + 1, block_N)),
                            ),
                        )
                    else:
                        nu_wg1 = T.alloc_var("int32", init=T.max(1, T.floordiv(kv_len, block_N)))
                    for k in T.serial(1, nu_wg1):
                        sk = k % nsK
                        svp_wg1 = (k - 1) % nsV
                        T.sync_threads(my_bar, NMMA)
                        T.mbarrier_wait_parity(kready[sk], (k // nsK) % 2)
                        T.wgmma_gemm(
                            Qs[1, :, :],
                            Ks[sk, :, :],
                            acc_s,
                            transpose_B=True,
                            policy=Pol,
                            clear_accum=True,
                        )
                        for i, j in T.Parallel(half, D):
                            acc_o[i, j] *= alpha[i]
                        T.mbarrier_wait_parity(vready[svp_wg1], ((k - 1) // nsV) % 2)
                        T.wgmma_gemm(pcast, Vs[svp_wg1, :, :], acc_o, policy=Pol, clear_accum=False)
                        T.named_barrier_arrive(nxt_bar, NMMA)
                        T.wait_wgmma(1)
                        T.mbarrier_arrive(kfree[sk])
                        if use_softcap:
                            apply_softcap(acc_s, half, block_N)
                        T.copy(sm, smp)
                        T.reduce_max(acc_s, sm, dim=1, clear=False)
                        for i in T.Parallel(half):
                            alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                        for i, j in T.Parallel(half, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                        T.reduce_sum(acc_s, ss, dim=1)
                        T.wait_wgmma(0)
                        T.mbarrier_arrive(vfree[svp_wg1])
                        for i in T.Parallel(half):
                            logsum[i] = logsum[i] * alpha[i] + ss[i]
                        T.copy(acc_s, pcast)
                    for k in T.serial(nu_wg1, eff):
                        sk = k % nsK
                        svp_wg1_tail = (k - 1) % nsV
                        T.sync_threads(my_bar, NMMA)
                        T.mbarrier_wait_parity(kready[sk], (k // nsK) % 2)
                        T.wgmma_gemm(
                            Qs[1, :, :],
                            Ks[sk, :, :],
                            acc_s,
                            transpose_B=True,
                            policy=Pol,
                            clear_accum=True,
                        )
                        for i, j in T.Parallel(half, D):
                            acc_o[i, j] *= alpha[i]
                        T.mbarrier_wait_parity(vready[svp_wg1_tail], ((k - 1) // nsV) % 2)
                        T.wgmma_gemm(
                            pcast, Vs[svp_wg1_tail, :, :], acc_o, policy=Pol, clear_accum=False
                        )
                        T.named_barrier_arrive(nxt_bar, NMMA)
                        T.wait_wgmma(1)
                        T.mbarrier_arrive(kfree[sk])
                        if is_causal:
                            mask_limit_wg1_tail = q0 + r0 + causal_offset - k * block_N
                            for i, j in T.Parallel(half, block_N):
                                acc_s[i, j] = T.if_then_else(
                                    mask_limit_wg1_tail + i >= j, acc_s[i, j], -T.infinity(accum)
                                )
                        else:
                            for i, j in T.Parallel(half, block_N):
                                acc_s[i, j] = T.if_then_else(
                                    k * block_N + j < kv_len,
                                    acc_s[i, j],
                                    -T.infinity(accum),
                                )
                        if use_softcap:
                            apply_softcap(acc_s, half, block_N)
                        T.copy(sm, smp)
                        T.reduce_max(acc_s, sm, dim=1, clear=False)
                        for i in T.Parallel(half):
                            alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                        for i, j in T.Parallel(half, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                        T.reduce_sum(acc_s, ss, dim=1)
                        T.wait_wgmma(0)
                        T.mbarrier_arrive(vfree[svp_wg1_tail])
                        for i in T.Parallel(half):
                            logsum[i] = logsum[i] * alpha[i] + ss[i]
                        T.copy(acc_s, pcast)

                    svp_wg1_final = (eff - 1) % nsV
                    for i, j in T.Parallel(half, D):
                        acc_o[i, j] *= alpha[i]
                    T.mbarrier_wait_parity(vready[svp_wg1_final], ((eff - 1) // nsV) % 2)
                    T.wgmma_gemm(
                        pcast, Vs[svp_wg1_final, :, :], acc_o, policy=Pol, clear_accum=False
                    )
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vfree[svp_wg1_final])
                    for i, j in T.Parallel(half, D):
                        if q0 + r0 + i < q_len:
                            O[q_start + q0 + r0 + i, by, j] = T.if_then_else(
                                logsum[i] > 0,
                                T.cast(acc_o[i, j] / logsum[i], dtype),
                                T.cast(0, dtype),
                            )

    return main


class GQAPrefillVarlenWsKernel(VarlenKernel):
    """SM90 packed prefill using the Dense two-consumer WGMMA pipeline."""

    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call) -> bool:
        return (
            call.dtype in ATTENTION_DTYPES
            and call.dim == 128
            and not call.is_fp8
            and not call.fuse_rope
            and not uses_sliding_window(call)
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
        )

    @property
    def default_config(self) -> dict:
        return {}

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
        return self.kernel(q, k, v, cu_seqlens_q, cu_seqlens_kv)
