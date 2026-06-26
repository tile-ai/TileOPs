"""Faithful FlashInfer FA3 port WITH 2-warpgroup ping-pong (CUTLASS WarpScheduler).

  - 384 threads = 1 producer WG (TMA) + 2 consumer WG
  - each consumer WG does its OWN m=64 query rows; per-WG Qs[g] (A-operand
    offset must be 0, so split buffers instead of slicing rows)
  - register-P (rs-wgmma): P held in a fp16 fragment (pcast), fed straight to PV.
    The cast stays AFTER wait_wgmma(0) so PV(k-1) finishes reading pcast before
    it is overwritten (fusing exp->pcast earlier serializes PV -> 114us).
  - setmaxnreg: producer dealloc to 24 regs, consumers alloc 240 -> rs-P fits
    (without this, rs-P spilled/serialized and lost to smem-P).
  - WarpScheduler ping-pong via named barriers: WG0 sync(1)/arrive(2),
    WG1 sync(2)/arrive(1); WG1 primes bar1 once -> WG0 softmax overlaps WG1 mma.
  - delayed PV + wait_wgmma(1): softmax(k) overlaps in-flight PV(k-1).
"""
import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops.kernels.kernel_base import Kernel

__all__ = ["GQAPrefillFwdWsPersistentCausalKernel"]

BLOCK_M = 128
BLOCK_N = 128
NSK = 2
NSV = 2
THREADS = 384
NMMA = 256

_pc = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
       tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True}
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
@tilelang.jit(out_idx=[3], pass_configs=_pc, compile_flags=_cf)
def _gqa_prefill_fwd_fa3_kernel(B, H, Hkv, Sq, Skv, D, dtype,
                                block_M=BLOCK_M, block_N=BLOCK_N, nsK=NSK, nsV=NSV,
                                threads=THREADS):
    scale = (1.0 / D) ** 0.5 * 1.44269504
    groups = H // Hkv
    co = Skv - Sq
    accum = "float"
    half = block_M // 2
    Pol = T.GemmWarpPolicy.FullRow

    @T.prim_func
    def main(
        Q: T.Tensor([B, Sq, H, D], dtype),
        K: T.Tensor([B, Skv, Hkv, D], dtype),
        V: T.Tensor([B, Skv, Hkv, D], dtype),
        O: T.Tensor([B, Sq, H, D], dtype),
    ):
        with T.Kernel(T.ceildiv(Sq, block_M), H, B, threads=threads) as (bx, by, bz):
            Qs = T.alloc_shared([2, half, D], dtype)
            Ks = T.alloc_shared([nsK, block_N, D], dtype)
            Vs = T.alloc_shared([nsV, block_N, D], dtype)
            Os = T.alloc_shared([2, half, D], dtype)  # per-WG smem-staged output (FlashInfer epilogue)
            T.annotate_layout({Qs: make_swizzled_layout(Qs), Ks: make_swizzled_layout(Ks),
                               Vs: make_swizzled_layout(Vs)})

            q_bar = T.alloc_barrier([32])             # 1-warp producer (FlashInfer NUM_PRODUCER_THREADS=32)
            kready = T.alloc_barrier([32] * nsK)
            kfree = T.alloc_barrier([NMMA] * nsK)
            vready = T.alloc_barrier([32] * nsV)
            vfree = T.alloc_barrier([NMMA] * nsV)

            cv = by // groups
            q0 = bx * block_M
            eff = T.min(T.ceildiv(Skv, block_N),
                        T.ceildiv(q0 + block_M + co, block_N))
            tx = T.get_thread_binding()

            if tx >= 256:  # ================= producer =================
                T.set_max_nreg(24, 0)  # producer is TMA-only: release regs to consumers
            if tx >= 256 and tx < 288:  # only 1 warp issues TMA + waits (rest of WG idle)
                T.tma_copy(Q[bz, q0:q0 + half, by, :], Qs[0, :, :], barrier=q_bar)
                T.tma_copy(Q[bz, q0 + half:q0 + block_M, by, :], Qs[1, :, :], barrier=q_bar)
                T.mbarrier_arrive(q_bar)
                for k in T.serial(eff):
                    sk = k % nsK
                    T.mbarrier_wait_parity(kfree[sk], ((k // nsK) % 2) ^ 1)
                    T.tma_copy(K[bz, k * block_N:(k + 1) * block_N, cv, :], Ks[sk, :, :], barrier=kready[sk])
                    T.mbarrier_arrive(kready[sk])
                    sv = k % nsV
                    T.mbarrier_wait_parity(vfree[sv], ((k // nsV) % 2) ^ 1)
                    T.tma_copy(V[bz, k * block_N:(k + 1) * block_N, cv, :], Vs[sv, :, :], barrier=vready[sv])
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
                    T.fill(alpha, 1.0)  # pipelined rescale: carried alpha, 1st rescale no-op
                    T.fill(sm, -T.infinity(accum))
                    T.mbarrier_wait_parity(q_bar, 0)
                    pass  # WG0 goes first

                    # prologue: tile 0, QK + softmax (no PV)
                    T.sync_threads(my_bar, NMMA)
                    T.mbarrier_wait_parity(kready[0], 0)
                    T.wgmma_gemm(Qs[0, :, :], Ks[0, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True)
                    T.named_barrier_arrive(nxt_bar, NMMA)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(kfree[0])
                    T.reduce_max(acc_s, sm, dim=1, clear=False)
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                    T.reduce_sum(acc_s, ss, dim=1)
                    for i in T.Parallel(half):
                        logsum[i] = ss[i]
                    T.copy(acc_s, pcast)

                    nu = T.max(1, T.min(eff, T.floordiv(q0 + r0 + co + 1, block_N)))
                    for k in T.serial(1, nu):
                        sk = k % nsK
                        svp = (k - 1) % nsV
                        T.sync_threads(my_bar, NMMA)
                        T.mbarrier_wait_parity(kready[sk], (k // nsK) % 2)
                        T.wgmma_gemm(Qs[0, :, :], Ks[sk, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True)
                        for i, j in T.Parallel(half, D):  # pipelined rescale (prev alpha) covers QK latency
                            acc_o[i, j] *= alpha[i]
                        T.mbarrier_wait_parity(vready[svp], ((k - 1) // nsV) % 2)
                        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                        T.named_barrier_arrive(nxt_bar, NMMA)
                        T.wait_wgmma(1)
                        T.mbarrier_arrive(kfree[sk])
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
                        T.wgmma_gemm(Qs[0, :, :], Ks[sk, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True)
                        for i, j in T.Parallel(half, D):  # pipelined rescale (prev alpha)
                            acc_o[i, j] *= alpha[i]
                        T.mbarrier_wait_parity(vready[svp], ((k - 1) // nsV) % 2)
                        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                        T.named_barrier_arrive(nxt_bar, NMMA)
                        T.wait_wgmma(1)
                        T.mbarrier_arrive(kfree[sk])
                        for i, j in T.Parallel(half, block_N):
                            acc_s[i, j] = T.if_then_else(
                                q0 + r0 + i + co >= k * block_N + j, acc_s[i, j], -T.infinity(accum))
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
                    for i, j in T.Parallel(half, D):  # pipelined rescale: final alpha before last PV
                        acc_o[i, j] *= alpha[i]
                    T.mbarrier_wait_parity(vready[svp], ((eff - 1) // nsV) % 2)
                    T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vfree[svp])
                    for i, j in T.Parallel(half, D):
                        acc_o[i, j] /= logsum[i]
                    T.copy(acc_o, Os[0, :, :])                                  # registers -> smem
                    T.copy(Os[0, :, :], O[bz, q0 + r0:q0 + r0 + half, by, :])  # smem -> global (coalesced)

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
                    T.fill(alpha, 1.0)  # pipelined rescale: carried alpha, 1st rescale no-op
                    T.fill(sm, -T.infinity(accum))
                    T.mbarrier_wait_parity(q_bar, 0)
                    T.named_barrier_arrive(1, NMMA)  # prime WG0

                    # prologue: tile 0, QK + softmax (no PV)
                    T.sync_threads(my_bar, NMMA)
                    T.mbarrier_wait_parity(kready[0], 0)
                    T.wgmma_gemm(Qs[1, :, :], Ks[0, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True)
                    T.named_barrier_arrive(nxt_bar, NMMA)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(kfree[0])
                    T.reduce_max(acc_s, sm, dim=1, clear=False)
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
                    T.reduce_sum(acc_s, ss, dim=1)
                    for i in T.Parallel(half):
                        logsum[i] = ss[i]
                    T.copy(acc_s, pcast)

                    nu = T.max(1, T.min(eff, T.floordiv(q0 + r0 + co + 1, block_N)))
                    for k in T.serial(1, nu):
                        sk = k % nsK
                        svp = (k - 1) % nsV
                        T.sync_threads(my_bar, NMMA)
                        T.mbarrier_wait_parity(kready[sk], (k // nsK) % 2)
                        T.wgmma_gemm(Qs[1, :, :], Ks[sk, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True)
                        for i, j in T.Parallel(half, D):  # pipelined rescale (prev alpha) covers QK latency
                            acc_o[i, j] *= alpha[i]
                        T.mbarrier_wait_parity(vready[svp], ((k - 1) // nsV) % 2)
                        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                        T.named_barrier_arrive(nxt_bar, NMMA)
                        T.wait_wgmma(1)
                        T.mbarrier_arrive(kfree[sk])
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
                        T.wgmma_gemm(Qs[1, :, :], Ks[sk, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True)
                        for i, j in T.Parallel(half, D):  # pipelined rescale (prev alpha)
                            acc_o[i, j] *= alpha[i]
                        T.mbarrier_wait_parity(vready[svp], ((k - 1) // nsV) % 2)
                        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                        T.named_barrier_arrive(nxt_bar, NMMA)
                        T.wait_wgmma(1)
                        T.mbarrier_arrive(kfree[sk])
                        for i, j in T.Parallel(half, block_N):
                            acc_s[i, j] = T.if_then_else(
                                q0 + r0 + i + co >= k * block_N + j, acc_s[i, j], -T.infinity(accum))
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
                    for i, j in T.Parallel(half, D):  # pipelined rescale: final alpha before last PV
                        acc_o[i, j] *= alpha[i]
                    T.mbarrier_wait_parity(vready[svp], ((eff - 1) // nsV) % 2)
                    T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
                    T.wait_wgmma(0)
                    T.mbarrier_arrive(vfree[svp])
                    for i, j in T.Parallel(half, D):
                        acc_o[i, j] /= logsum[i]
                    T.copy(acc_o, Os[1, :, :])                                  # registers -> smem
                    T.copy(Os[1, :, :], O[bz, q0 + r0:q0 + r0 + half, by, :])  # smem -> global (coalesced)

    return main


class GQAPrefillFwdWsPersistentCausalKernel(Kernel):
    """Faithful FlashInfer FA3 GQA-prefill kernel (causal, fp16, dim==128, sm90).

    Warp-specialized 2-warpgroup ping-pong port matching FlashInfer's
    ``single_prefill_sm90``. Selected by the prefill op on Hopper for the fp16
    causal dim-128 path; other cases fall back to ``GQAPrefillFwdKernel``. Causal
    uses bottom-right alignment (``co = seq_len_kv - seq_len_q``).

    Note: like FlashInfer's forward kernel, this does not emit log-sum-exp; the
    op's ``(output, lse)`` contract is satisfied with ``lse=None``. A backward
    pass that needs lse must compute it separately.
    """

    supported_archs: list[int] = [90]

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len_q: int,
                 seq_len_kv: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 sm_scale: Optional[float] = None,
                 softcap: float = 0.0,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        if not is_causal:
            raise ValueError("GQAPrefillFwdWsPersistentCausalKernel only supports causal prefill.")
        if dim != 128:
            raise ValueError("GQAPrefillFwdWsPersistentCausalKernel currently requires dim == 128.")
        if dtype != torch.float16:
            raise ValueError("GQAPrefillFwdWsPersistentCausalKernel currently supports float16 only.")
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.kernel = _gqa_prefill_fwd_fa3_kernel(batch, heads, heads_kv, seq_len_q,
                                                  seq_len_kv, dim, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {}

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # No lse output (matches FlashInfer's forward); op discards the 2nd slot.
        return self.kernel(q, k, v), None
