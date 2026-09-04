"""Dense BSHD grouped-query attention kernels and preprocessing."""

import functools
import itertools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops.kernels.constants import LOG2E

from ..kernel_base import Kernel
from .call_spec import WS_ARCH
from .online_softmax import make_apply_softcap

__all__ = [
    "GQADenseCausalWsKernel",
    "GQADenseSlidingWindowKernel",
]


# Dense Q/K RoPE preprocessing.
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}
_COMPILE_FLAGS = [
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
@tilelang.jit(out_idx=[4, 5], pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS)
def _gqa_dense_rope_qk_kernel(
    B,
    H,
    Hkv,
    Sq,
    Skv,
    D,
    max_position,
    rotary_dim,
    rope_layout,
    dtype,
    threads=256,
    num_per_thread=4,
):
    half = rotary_dim // 2
    q_rows = B * Sq * H
    k_rows = B * Skv * Hkv
    q_pairs = q_rows * half
    k_pairs = k_rows * half
    tail = D - rotary_dim
    q_tail = q_rows * tail
    k_tail = k_rows * tail
    walked = max(q_pairs, k_pairs, q_tail, k_tail)
    block_size = threads * num_per_thread
    co = Skv - Sq

    @T.prim_func
    def main(
        Q: T.Tensor([q_rows * D], dtype),
        K: T.Tensor([k_rows * D], dtype),
        RopeCos: T.Tensor([max_position, half], dtype),
        RopeSin: T.Tensor([max_position, half], dtype),
        QRot: T.Tensor([q_rows * D], dtype),
        KRot: T.Tensor([k_rows * D], dtype),
    ):
        with T.Kernel(T.ceildiv(walked, block_size), threads=threads) as bx:
            for tx, it in T.Parallel(threads, num_per_thread):
                pair = (bx * threads + tx) * num_per_thread + it
                if pair < q_pairs:
                    row = pair // half
                    freq = pair % half
                    seq = (row // H) % Sq
                    d0 = freq if rope_layout == "neox" else freq * 2
                    d1 = freq + half if rope_layout == "neox" else freq * 2 + 1
                    base = row * D
                    c = T.Cast("float32", RopeCos[co + seq, freq])
                    s = T.Cast("float32", RopeSin[co + seq, freq])
                    x0 = T.Cast("float32", Q[base + d0])
                    x1 = T.Cast("float32", Q[base + d1])
                    QRot[base + d0] = T.Cast(dtype, x0 * c - x1 * s)
                    QRot[base + d1] = T.Cast(dtype, x1 * c + x0 * s)
                if pair < k_pairs:
                    row = pair // half
                    freq = pair % half
                    seq = (row // Hkv) % Skv
                    d0 = freq if rope_layout == "neox" else freq * 2
                    d1 = freq + half if rope_layout == "neox" else freq * 2 + 1
                    base = row * D
                    c = T.Cast("float32", RopeCos[seq, freq])
                    s = T.Cast("float32", RopeSin[seq, freq])
                    x0 = T.Cast("float32", K[base + d0])
                    x1 = T.Cast("float32", K[base + d1])
                    KRot[base + d0] = T.Cast(dtype, x0 * c - x1 * s)
                    KRot[base + d1] = T.Cast(dtype, x1 * c + x0 * s)
                if tail > 0:
                    if pair < q_tail:
                        row = pair // tail
                        col = pair % tail
                        idx = row * D + rotary_dim + col
                        QRot[idx] = Q[idx]
                    if pair < k_tail:
                        row = pair // tail
                        col = pair % tail
                        idx = row * D + rotary_dim + col
                        KRot[idx] = K[idx]

    return main


class DenseQKRoPEPreprocessor:
    """Rotate Dense Q and K once before an attention implementation consumes them."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len_q: int,
        seq_len_kv: int,
        dim: int,
        max_position: int,
        rotary_dim: int,
        rope_layout: str,
        dtype: str,
    ) -> None:
        self.kernel = _gqa_dense_rope_qk_kernel(
            batch,
            heads,
            heads_kv,
            seq_len_q,
            seq_len_kv,
            dim,
            max_position,
            rotary_dim,
            rope_layout,
            dtype,
        )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rope_cos is None or rope_sin is None:
            raise ValueError("fused RoPE requires rope_cos and rope_sin")
        q_rot, k_rot = self.kernel(q.reshape(-1), k.reshape(-1), rope_cos, rope_sin)
        return q_rot.reshape_as(q), k_rot.reshape_as(k)


def make_dense_qk_rope_preprocessor(
    *,
    fuse_rope: bool,
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    dtype: str,
) -> Optional[DenseQKRoPEPreprocessor]:
    """Build the shared Dense Q/K RoPE stage when requested."""
    if not fuse_rope:
        return None
    return DenseQKRoPEPreprocessor(
        batch,
        heads,
        heads_kv,
        seq_len_q,
        seq_len_kv,
        dim,
        max_position,
        rotary_dim,
        rope_layout,
        dtype,
    )


# Causal warp-specialized Dense attention.
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
@tilelang.jit(out_idx=[3], pass_configs=_pc, compile_flags=_cf)
def _gqa_dense_causal_ws_kernel(
    B,
    H,
    Hkv,
    D,
    sm_scale,
    softcap,
    dtype,
    block_M=BLOCK_M,
    block_N=BLOCK_N,
    nsK=NSK,
    nsV=NSV,
    threads=THREADS,
):
    """Build the causal WS program; its online softmax carries the previous tile's alpha."""
    score_scale = (1.0 / D) ** 0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = H // Hkv
    accum = "float"
    half = block_M // 2
    Pol = T.GemmWarpPolicy.FullRow
    seq_len_q = T.dynamic("seq_len_q")
    seq_len_kv = T.dynamic("seq_len_kv")

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
        Q: T.Tensor([B, seq_len_q, H, D], dtype),
        K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        O: T.Tensor([B, seq_len_q, H, D], dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len_q, block_M), H, B, threads=threads) as (bx, by, bz):
            Qs = T.alloc_shared([2, half, D], dtype)
            Ks = T.alloc_shared([nsK, block_N, D], dtype)
            Vs = T.alloc_shared([nsV, block_N, D], dtype)
            Os = T.alloc_shared(
                [2, half, D], dtype
            )  # per-WG smem-staged output (FlashInfer epilogue)
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

            cv = by // groups
            q0 = bx * block_M
            causal_offset = T.alloc_var("int32", init=seq_len_kv - seq_len_q)
            eff = T.alloc_var(
                "int32",
                init=T.min(
                    T.ceildiv(seq_len_kv, block_N),
                    T.ceildiv(q0 + block_M + causal_offset, block_N),
                ),
            )
            tx = T.get_thread_binding()

            if tx >= 256:  # ================= producer =================
                T.set_max_nreg(24, 0)  # producer is TMA-only: release regs to consumers
            if tx >= 256 and tx < 288:  # only 1 warp issues TMA + waits (rest of WG idle)
                T.tma_copy(Q[bz, q0 : q0 + half, by, :], Qs[0, :, :], barrier=q_bar)
                T.tma_copy(Q[bz, q0 + half : q0 + block_M, by, :], Qs[1, :, :], barrier=q_bar)
                T.mbarrier_arrive(q_bar)
                for k in T.serial(eff):
                    sk = k % nsK
                    T.mbarrier_wait_parity(kfree[sk], ((k // nsK) % 2) ^ 1)
                    T.tma_copy(
                        K[bz, k * block_N : (k + 1) * block_N, cv, :],
                        Ks[sk, :, :],
                        barrier=kready[sk],
                    )
                    T.mbarrier_arrive(kready[sk])
                    sv = k % nsV
                    T.mbarrier_wait_parity(vfree[sv], ((k // nsV) % 2) ^ 1)
                    T.tma_copy(
                        V[bz, k * block_N : (k + 1) * block_N, cv, :],
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
                    Qs[0, :, :], Ks[0, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True
                )
                T.named_barrier_arrive(nxt_bar, NMMA)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[0])
                if q0 + r0 + causal_offset < block_N - 1:
                    mask_limit = q0 + r0 + causal_offset
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            mask_limit + i >= j, acc_s[i, j], -T.infinity(accum)
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

                nu = T.alloc_var(
                    "int32",
                    init=T.max(
                        1, T.min(eff, T.floordiv(q0 + r0 + causal_offset + 1, block_N))
                    ),
                )
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
                    mask_limit_tail = q0 + r0 + causal_offset - k * block_N
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            mask_limit_tail + i >= j, acc_s[i, j], -T.infinity(accum)
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
                for i in T.Parallel(half):
                    alpha[i] = 1.0 / logsum[i]
                for i, j in T.Parallel(half, D):
                    acc_o[i, j] *= alpha[i]
                T.copy(acc_o, Os[0, :, :])
                T.copy(Os[0, :, :], O[bz, q0 + r0 : q0 + r0 + half, by, :])

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
                    Qs[1, :, :], Ks[0, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True
                )
                T.named_barrier_arrive(nxt_bar, NMMA)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[0])
                if q0 + r0 + causal_offset < block_N - 1:
                    mask_limit_wg1 = q0 + r0 + causal_offset
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            mask_limit_wg1 + i >= j, acc_s[i, j], -T.infinity(accum)
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

                nu_wg1 = T.alloc_var(
                    "int32",
                    init=T.max(
                        1, T.min(eff, T.floordiv(q0 + r0 + causal_offset + 1, block_N))
                    ),
                )
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
                    mask_limit_wg1_tail = q0 + r0 + causal_offset - k * block_N
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            mask_limit_wg1_tail + i >= j, acc_s[i, j], -T.infinity(accum)
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
                T.wgmma_gemm(pcast, Vs[svp_wg1_final, :, :], acc_o, policy=Pol, clear_accum=False)
                T.wait_wgmma(0)
                T.mbarrier_arrive(vfree[svp_wg1_final])
                for i in T.Parallel(half):
                    alpha[i] = 1.0 / logsum[i]
                for i, j in T.Parallel(half, D):
                    acc_o[i, j] *= alpha[i]
                T.copy(acc_o, Os[1, :, :])
                T.copy(Os[1, :, :], O[bz, q0 + r0 : q0 + r0 + half, by, :])

    return main


class GQADenseCausalWsKernel(Kernel):
    """Causal dense prefill using the FA3 two-consumer pipeline."""

    supported_archs: list[int] = [WS_ARCH]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len_q: int,
        seq_len_kv: int,
        dim: int,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dtype = dtype
        self.kernel = _gqa_dense_causal_ws_kernel(
            batch,
            heads,
            heads_kv,
            dim,
            dim**-0.5 if sm_scale is None else sm_scale,
            softcap,
            self.dtype_str,
        )
        self.rope = make_dense_qk_rope_preprocessor(
            fuse_rope=fuse_rope,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            dim=dim,
            max_position=max_position,
            rotary_dim=rotary_dim,
            rope_layout=rope_layout,
            dtype=self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._require_cuda(q=q, k=k, v=v)
        if self.rope is not None:
            q, k = self.rope(q, k, rope_cos, rope_sin)
        return self.kernel(q, k, v)


# Dense sliding-window attention.
@functools.lru_cache(maxsize=32)
def _gqa_sw_fwd_wgmma_pipelined_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sm_scale: Optional[float] = None,
    softcap: float = 0.0,
    dtype: str = "float16",
) -> Callable:
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = heads // heads_kv
    accum_dtype = "float"
    has_window = window_size_left >= 0 or window_size_right >= 0

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_sw_fwd_wgmma_pipelined_func(block_m, block_n, num_stages, threads):
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_m, block_n)
            if use_softcap
            else None
        )

        @T.macro
        def mma0(
            k: T.Tensor(kv_shape, dtype),
            q_shared: T.SharedBuffer([block_m, dim], dtype),
            k_shared: T.SharedBuffer([block_n, dim], dtype),
            acc_s: T.FragmentBuffer([block_m, block_n], accum_dtype),
            k_idx: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ) -> None:
            T.copy(k[bz, k_idx * block_n : (k_idx + 1) * block_n, by // groups, :], k_shared)
            if is_causal and has_window:
                for i, j in T.Parallel(block_m, block_n):
                    causal_mask = bx * block_m + i < k_idx * block_n + j
                    left_mask = (window_size_left >= 0) and (
                        k_idx * block_n + j < bx * block_m + i - window_size_left
                    )
                    acc_s[i, j] = T.if_then_else(
                        causal_mask or left_mask, -T.infinity(accum_dtype), 0
                    )
            elif is_causal:
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.if_then_else(
                        bx * block_m + i < k_idx * block_n + j, -T.infinity(accum_dtype), 0
                    )
            elif has_window:
                for i, j in T.Parallel(block_m, block_n):
                    left_mask = (window_size_left >= 0) and (
                        k_idx * block_n + j < bx * block_m + i - window_size_left
                    )
                    right_mask = (window_size_right >= 0) and (
                        k_idx * block_n + j > bx * block_m + i + window_size_right
                    )
                    acc_s[i, j] = T.if_then_else(
                        left_mask or right_mask, -T.infinity(accum_dtype), 0
                    )
            else:
                T.clear(acc_s)
            T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def mma1(
            v: T.Tensor(kv_shape, dtype),
            v_shared: T.SharedBuffer([block_n, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_m, block_n], dtype),
            acc_o: T.FragmentBuffer([block_m, dim], accum_dtype),
            k_idx: T.int32,
            by: T.int32,
            bz: T.int32,
        ) -> None:
            T.copy(v[bz, k_idx * block_n : (k_idx + 1) * block_n, by // groups, :], v_shared)
            T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.prim_func
        def _gqa_sw_fwd_wgmma_pipelined_main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(kv_shape, dtype),
            v: T.Tensor(kv_shape, dtype),
            output: T.Tensor(q_shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ) -> None:
            with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                o_shared = T.alloc_shared([block_m, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.annotate_layout({o_shared: tilelang.layout.make_swizzled_layout(o_shared)})
                T.copy(q[bz, bx * block_m : (bx + 1) * block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                if is_causal:
                    k_end = T.ceildiv(T.min(seq_len, (bx + 1) * block_m), block_n)
                elif has_window and window_size_right >= 0:
                    k_end = T.ceildiv(
                        T.min(seq_len, (bx + 1) * block_m + window_size_right), block_n
                    )
                else:
                    k_end = T.ceildiv(seq_len, block_n)

                if has_window and window_size_left >= 0:
                    k_start = T.max(0, bx * block_m - window_size_left) // block_n
                else:
                    k_start = 0

                loop_count = T.max(k_end - k_start, 0)

                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    mma0(k, q_shared, k_shared, acc_s, k_idx, bx, by, bz)
                    if use_softcap:
                        apply_softcap(acc_s)
                    # Online softmax with scores_max clamping.
                    # Clamping prevents exp2(+inf) when all block scores are -inf.
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_m):
                        # Clamp to a finite floor so exp2(prev - curr) never
                        # evaluates to exp2(-inf - (-inf)) = exp2(nan).
                        scores_max[i] = T.max(scores_max[i], T.cast(-1e38, accum_dtype))
                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]
                    mma1(v, v_shared, acc_s_cast, acc_o, k_idx, by, bz)

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                # Guard the swizzled o_shared round-trip against a shared-memory
                # race under WGMMA pipelining. Any non-zero barrier slot works;
                # a bare T.sync_threads() aliases the implicit barrier-0 and is
                # elided.
                T.sync_threads(3, threads)
                T.copy(acc_o, o_shared)
                T.sync_threads(3, threads)
                T.copy(o_shared, output[bz, bx * block_m : (bx + 1) * block_m, by, :])
                for i in T.Parallel(block_m):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_m : (bx + 1) * block_m])

        return _gqa_sw_fwd_wgmma_pipelined_main

    return _gqa_sw_fwd_wgmma_pipelined_func


@torch.library.custom_op("tileops::gqa_sw_fwd_wgmma_pipelined_wrapped_kernel", mutates_args=())
def _gqa_sw_fwd_wgmma_pipelined_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_sw_fwd_wgmma_pipelined_kernel(
        batch,
        heads,
        heads_kv,
        seq_len,
        dim,
        is_causal,
        window_size_left,
        window_size_right,
        sm_scale,
        softcap,
        dtype,
    )(block_m, block_n, num_stages, threads)(q, k, v)


@_gqa_sw_fwd_wgmma_pipelined_wrapped_kernel.register_fake
def _(
    batch,
    heads,
    heads_kv,
    seq_len,
    dim,
    is_causal,
    window_size_left,
    window_size_right,
    sm_scale,
    softcap,
    dtype,
    block_m,
    block_n,
    num_stages,
    threads,
    *inputs,
):
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class GQADenseSlidingWindowKernel(Kernel):
    """SM90 Dense sliding-window kernel with a native BSHD ABI."""

    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        window_size_left: int,
        window_size_right: int,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.rope = make_dense_qk_rope_preprocessor(
            fuse_rope=fuse_rope,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            seq_len_q=seq_len,
            seq_len_kv=seq_len,
            dim=dim,
            max_position=max_position,
            rotary_dim=rotary_dim,
            rope_layout=rope_layout,
            dtype=self.dtype_str,
        )

        self.kernel = _gqa_sw_fwd_wgmma_pipelined_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
            self.dim,
            self.is_causal,
            self.window_size_left,
            self.window_size_right,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
        )

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "num_stages": 3,
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = list(itertools.product([64, 128], [64, 128], [2, 3], [128, 256]))
        return [
            {"block_m": c[0], "block_n": c[1], "num_stages": c[2], "threads": c[3]} for c in configs
        ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._require_cuda(q=q, k=k, v=v)
        if self.rope is not None:
            q, k = self.rope(q, k, rope_cos, rope_sin)
        output, _ = _gqa_sw_fwd_wgmma_pipelined_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
            self.dim,
            self.is_causal,
            self.window_size_left,
            self.window_size_right,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
            q,
            k,
            v,
        )
        return output
