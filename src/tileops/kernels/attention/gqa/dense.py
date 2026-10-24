"""Dense BSHD grouped-query attention kernels and preprocessing."""

import functools
import itertools
import re
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch
import tvm_ffi
from tilelang.layout import make_swizzled_layout

from tileops._csrc import csrc_include
from tileops.kernels.attention.call_spec import AttentionCall, GQADenseFwdInterface
from tileops.kernels.attention.online_softmax import make_apply_softcap, make_dense_sink_scale
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_shared_memory_optin, get_sm_count

__all__ = [
    "GQADensePersistentWSKernel",
    "GQADenseSlidingWindowKernel",
    "GQADenseWSKernel",
]


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


def _anchor_value_waits(code: str) -> str:
    """Replace each ``value_wait_anchor_slot`` with a store of a few probabilities.

    The slot sits right before a key step's value-product wait; the probabilities are the
    step's score fragment as the exponential loop between the step's QK wait and the slot
    writes it. TileLang names that fragment only in the generated CUDA, so the store is
    written there. Code without a slot is returned unchanged.

    Raises:
        RuntimeError: A slot line the rewrite does not recognize, or one with no exponential
            loop since its step's QK wait.
    """
    if "tileops::value_wait_anchor_slot" not in code:
        return code
    # Under the m64nN WGMMA accumulator layout, the thread's first four probabilities of its
    # upper row (columns 2 (lane % 4) + {0, 1} and 8 more).
    entries = (0, 1, 4, 5)
    slot_line = re.compile(r"^(?P<ind>[ \t]*)tileops::value_wait_anchor_slot\((?P<ptr>.*)\);$")
    exp_target = re.compile(r"^\s*(?P<frag>\w+)\[\w+\] = exp2f\(\(\((?P=frag)\[")
    lines = code.split("\n")
    for n, line in enumerate(lines):
        if "tileops::value_wait_anchor_slot" not in line:
            continue
        slot = slot_line.match(line)
        if slot is None:
            raise RuntimeError(f"unrecognized value_wait_anchor_slot line: {line.strip()}")
        frag = None
        for prev in reversed(lines[:n]):
            if prev.strip() == "tl::wait_wgmma<1>();":
                break
            m = exp_target.match(prev)
            if m is not None:
                frag = m["frag"]
                break
        if frag is None:
            raise RuntimeError("value_wait_anchor_slot follows no exponential loop in its step")
        value = " + ".join(f"{frag}[{e}]" for e in entries)
        lines[n] = f"{slot['ind']}tileops::value_wait_anchor((float*)({slot['ptr']}), {value});"
    return "\n".join(lines)


def _register_value_wait_anchor() -> None:
    """Chain the anchor rewrite after whatever holds TileLang's CUDA post-processing slot."""
    previous = tvm_ffi.get_global_func("tilelang_callback_cuda_postproc", allow_missing=True)

    def postproc(code, target):
        if previous is not None:
            code = previous(code, target)
        return _anchor_value_waits(code)

    tilelang.register_cuda_postproc(postproc)


_register_value_wait_anchor()


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
    rope_dtype=None,
    threads=256,
    num_per_thread=4,
):
    rope_dtype = dtype if rope_dtype is None else rope_dtype
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
        RopeCos: T.Tensor([max_position, half], rope_dtype),
        RopeSin: T.Tensor([max_position, half], rope_dtype),
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


class _DenseQKRoPEPreprocessor:
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
        rope_dtype: Optional[str] = None,
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
            rope_dtype,
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
    rope_dtype: Optional[str] = None,
) -> Optional[_DenseQKRoPEPreprocessor]:
    """Build the shared Dense Q/K RoPE stage when requested."""
    if not fuse_rope:
        return None
    return _DenseQKRoPEPreprocessor(
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
        rope_dtype,
    )


# Keep O at slot 3 in both signatures: JIT output indices are shared across specializations.
@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[3], pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS)
def _gqa_dense_ws_kernel(
    B,
    H,
    Hkv,
    D,
    is_causal,
    sm_scale,
    softcap,
    dtype,
    block_M=128,
    block_N=128,
    nsK=2,
    nsV=2,
    threads=384,
    has_sinks=False,
):
    """Build the Dense WS program; its online softmax carries the previous tile's alpha."""
    score_scale = (1.0 / D) ** 0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = H // Hkv
    accum = "float"
    half = block_M // 2
    # Threads of the two MMA consumer warpgroups.
    nmma = 256
    Pol = T.GemmWarpPolicy.FullRow
    seq_len_q = T.dynamic("seq_len_q")
    seq_len_kv = T.dynamic("seq_len_kv")

    sink_scale = make_dense_sink_scale(scale, half)

    @T.macro
    def apply_softcap(acc_s, rows, cols):
        for i, j in T.Parallel(rows, cols):
            capped = T.cast(softcap, accum) * T.tanh(
                acc_s[i, j] * T.cast(score_scale / softcap, accum)
            )
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum), -T.infinity(accum), capped
            )

    @T.macro
    def attention(
        Q: T.Tensor([B, seq_len_q, H, D], dtype),
        K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        O: T.Tensor([B, seq_len_q, H, D], dtype),
        Sinks=None,
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
            kfree = T.alloc_barrier([nmma] * nsK)
            vready = T.alloc_barrier([32] * nsV)
            vfree = T.alloc_barrier([nmma] * nsV)

            cv = by // groups
            q0 = bx * block_M
            causal_offset = T.alloc_var("int32", init=seq_len_kv - seq_len_q)
            if is_causal:
                eff = T.alloc_var(
                    "int32",
                    init=T.min(
                        T.ceildiv(seq_len_kv, block_N),
                        T.ceildiv(q0 + block_M + causal_offset, block_N),
                    ),
                )
            else:
                eff = T.alloc_var("int32", init=T.ceildiv(seq_len_kv, block_N))
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
                T.set_max_nreg(240, 1)
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

                T.sync_threads(my_bar, nmma)
                T.mbarrier_wait_parity(kready[0], 0)
                T.wgmma_gemm(
                    Qs[0, :, :], Ks[0, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True
                )
                T.named_barrier_arrive(nxt_bar, nmma)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[0])
                if is_causal and q0 + r0 + causal_offset < block_N - 1:
                    mask_limit = q0 + r0 + causal_offset
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            mask_limit + i >= j, acc_s[i, j], -T.infinity(accum)
                        )
                elif not is_causal and seq_len_kv < block_N:
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            j < seq_len_kv, acc_s[i, j], -T.infinity(accum)
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
                    nu = T.alloc_var("int32", init=T.max(1, T.floordiv(seq_len_kv, block_N)))
                for k in T.serial(1, nu):
                    sk = k % nsK
                    svp = (k - 1) % nsV
                    T.sync_threads(my_bar, nmma)
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
                    T.named_barrier_arrive(nxt_bar, nmma)
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
                    T.sync_threads(my_bar, nmma)
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
                    T.named_barrier_arrive(nxt_bar, nmma)
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
                                k * block_N + j < seq_len_kv,
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
                if has_sinks:
                    sink_scale(logsum, sm, Sinks, alpha, by, 0, half)
                    for i in T.Parallel(half):
                        alpha[i] /= logsum[i]
                else:
                    for i in T.Parallel(half):
                        alpha[i] = 1.0 / logsum[i]
                for i, j in T.Parallel(half, D):
                    acc_o[i, j] *= alpha[i]
                T.copy(acc_o, Os[0, :, :])
                T.copy(Os[0, :, :], O[bz, q0 + r0 : q0 + r0 + half, by, :])

            with T.ws(1):
                T.set_max_nreg(240, 1)
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
                T.named_barrier_arrive(1, nmma)

                T.sync_threads(my_bar, nmma)
                T.mbarrier_wait_parity(kready[0], 0)
                T.wgmma_gemm(
                    Qs[1, :, :], Ks[0, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True
                )
                T.named_barrier_arrive(nxt_bar, nmma)
                T.wait_wgmma(0)
                T.mbarrier_arrive(kfree[0])
                if is_causal and q0 + r0 + causal_offset < block_N - 1:
                    mask_limit_wg1 = q0 + r0 + causal_offset
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            mask_limit_wg1 + i >= j, acc_s[i, j], -T.infinity(accum)
                        )
                elif not is_causal and seq_len_kv < block_N:
                    for i, j in T.Parallel(half, block_N):
                        acc_s[i, j] = T.if_then_else(
                            j < seq_len_kv, acc_s[i, j], -T.infinity(accum)
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
                    nu_wg1 = T.alloc_var("int32", init=T.max(1, T.floordiv(seq_len_kv, block_N)))
                for k in T.serial(1, nu_wg1):
                    sk = k % nsK
                    svp_wg1 = (k - 1) % nsV
                    T.sync_threads(my_bar, nmma)
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
                    T.named_barrier_arrive(nxt_bar, nmma)
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
                    T.sync_threads(my_bar, nmma)
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
                    T.named_barrier_arrive(nxt_bar, nmma)
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
                                k * block_N + j < seq_len_kv,
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
                T.wgmma_gemm(pcast, Vs[svp_wg1_final, :, :], acc_o, policy=Pol, clear_accum=False)
                T.wait_wgmma(0)
                T.mbarrier_arrive(vfree[svp_wg1_final])
                if has_sinks:
                    sink_scale(logsum, sm, Sinks, alpha, by, 0, half)
                    for i in T.Parallel(half):
                        alpha[i] /= logsum[i]
                else:
                    for i in T.Parallel(half):
                        alpha[i] = 1.0 / logsum[i]
                for i, j in T.Parallel(half, D):
                    acc_o[i, j] *= alpha[i]
                T.copy(acc_o, Os[1, :, :])
                T.copy(Os[1, :, :], O[bz, q0 + r0 : q0 + r0 + half, by, :])

    if has_sinks:

        @T.prim_func
        def with_sinks(
            Q: T.Tensor([B, seq_len_q, H, D], dtype),
            K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            O: T.Tensor([B, seq_len_q, H, D], dtype),
            Sinks: T.Tensor([H], "float32"),
        ):
            attention(Q, K, V, O, Sinks)

        return with_sinks

    @T.prim_func
    def main(
        Q: T.Tensor([B, seq_len_q, H, D], dtype),
        K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        O: T.Tensor([B, seq_len_q, H, D], dtype),
    ):
        attention(Q, K, V, O)

    return main


class GQADenseWSKernel(Kernel, GQADenseFwdInterface):
    """Dense attention using the FA3 two-consumer pipeline."""

    supported_archs: list[int] = [90]
    _program = staticmethod(_gqa_dense_ws_kernel)

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """The contiguous prefill region: more than one query position, no window, not FP8."""
        if call.is_fp8 or call.max_seqlen_q == 1 or call.uses_sliding_window:
            return "does not serve this call"
        return call.tensor_core_dim_refusal

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """Accepts its sequence extents at runtime unless RoPE compiles them in."""
        args = dict(
            batch=call.batch,
            heads=call.heads,
            heads_kv=call.heads_kv,
            seq_len_q=call.max_seqlen_q,
            seq_len_kv=call.seqlen_kv,
            dim=call.dim,
            is_causal=call.is_causal,
            dtype=call.dtype,
            sm_scale=call.sm_scale,
            softcap=call.softcap,
            has_sinks=call.has_sinks,
            **call.rope_args,
            device_index=call.device.index if call.device is not None else None,
        )
        runtime = () if call.fuse_rope else ("seq_len_q", "seq_len_kv")
        return tuple(v for k, v in args.items() if k not in runtime), lambda: cls(**args)

    def __init__(
        self,
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
        *,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        has_sinks: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.has_sinks = has_sinks
        self.dtype = dtype
        self.kernel = self._program(
            batch,
            heads,
            heads_kv,
            dim,
            is_causal,
            dim**-0.5 if sm_scale is None else sm_scale,
            softcap,
            self.dtype_str,
            has_sinks=self.has_sinks,
            **self._program_args(dim, device_index),
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
        self.init_config(config)

    def _program_args(self, dim: int, device_index: Optional[int]) -> dict:
        """Build arguments beyond the call's shape; the two-consumer program takes none."""
        return {}

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
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._require_devices(q=q, k=k, v=v)
        if self.rope is not None:
            q, k = self.rope(q, k, rope_cos, rope_sin)
        if self.has_sinks:
            return self.kernel(q, k, v, sinks)
        return self.kernel(q, k, v)


# Keep O at slot 3 in both signatures: JIT output indices are shared across specializations.
@functools.lru_cache(maxsize=32)
@tilelang.jit(
    out_idx=[3],
    pass_configs=_PASS_CONFIGS,
    compile_flags=[*_COMPILE_FLAGS, *csrc_include("warp_uniform.h", "value_wait_anchor.h")],
)
def _gqa_dense_ws_persistent_kernel(
    B,
    H,
    Hkv,
    D,
    is_causal,
    sm_scale,
    softcap,
    dtype,
    block_M=128,
    block_N=128,
    nsK=2,
    nsV=2,
    num_ctas=132,
    has_sinks=False,
):
    """Build the persistent Dense WS program: one CTA an SM walks its share of the query tiles.

    ``block_M`` rows split into 64-row consumer warpgroups, two or three, beside one producer
    warpgroup.
    """
    score_scale = (1.0 / D) ** 0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = H // Hkv
    accum = "float"
    half = 64
    num_consumers = block_M // half
    nmma = 128 * num_consumers
    threads = nmma + 128
    # The producer only issues TMA; 65,536 registers split over the consumers otherwise.
    producer_regs, consumer_regs = (24, 240) if num_consumers == 2 else (32, 160)
    # Each issue-turn barrier joins the group that passes the turn and the one that takes it.
    turn_threads = 256
    Pol = T.GemmWarpPolicy.FullRow
    seq_len_q = T.dynamic("seq_len_q")
    seq_len_kv = T.dynamic("seq_len_kv")

    sink_scale = make_dense_sink_scale(scale, half)
    # Three consumers take the row sums from an N=8 product of the probabilities with ones,
    # issued beside the value product. Two consumers (D128) keep per-thread partial sums: the
    # ones do not fit beside their rings, and the extra register-operand product lengthens
    # their issue turn more than the sums it saves.
    use_ones = num_consumers == 3
    sum_cols = 8  # columns of the ones operand: the narrowest WGMMA N

    @T.macro
    def apply_softcap(acc_s, rows, cols):
        for i, j in T.Parallel(rows, cols):
            capped = T.cast(softcap, accum) * T.tanh(
                acc_s[i, j] * T.cast(score_scale / softcap, accum)
            )
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum), -T.infinity(accum), capped
            )

    def zigzag(it, cta):
        """The tile a CTA takes in round ``it``."""
        return it * num_ctas + T.if_then_else(it % 2 == 0, cta, num_ctas - 1 - cta)

    def tile_coords(tile_id, num_q_tiles, causal_offset):
        """A tile's query block, head, batch, first query row and key steps.

        Heads vary fastest, then batches; a causal call takes its longest query blocks first.
        """
        by = tile_id % H
        bz = (tile_id // H) % B
        rank = tile_id // (H * B)
        bx = (num_q_tiles - 1 - rank) if is_causal else rank
        # A causal call aligns its tiles to the end of Q, rounded up to a 64-row group, so the
        # partial tile is the shortest one. Its groups before row 0 load zeros through TMA
        # and store nothing; a group never straddles row 0.
        if is_causal:
            q0 = T.ceildiv(seq_len_q, half) * half - (num_q_tiles - bx) * block_M
        else:
            q0 = bx * block_M
        if is_causal:
            eff = T.min(
                T.ceildiv(seq_len_kv, block_N), T.ceildiv(q0 + block_M + causal_offset, block_N)
            )
        else:
            eff = T.ceildiv(seq_len_kv, block_N)
        return bx, by, bz, q0, eff

    @T.macro
    def add_partial_sums(acc_s, lp):
        """Add a tile's probabilities to each thread's partial row sums.

        The first 16 columns of ``lp`` hold them: column j sums the score columns j, j + 16,
        ..., which the WGMMA accumulator layout keeps in one thread, so no step reduces across
        threads.
        """
        for t in T.unroll(block_N // 16):
            for i, j in T.Parallel(half, 16):
                lp[i, j] += acc_s[i, j + t * 16]

    @T.macro
    def kv_step(
        wg, qs, k, masked, Qs, Ks, Vs, kready, kfree, vready, vfree, ones, acc_s, pcast, acc_o,
        acc_l, lp, sm, smp, alpha, limit, my_bar, nxt_bar, g, anchor,
    ):  # fmt: skip
        """Key step k: under the issue turn, QK of key tile k and PV of tile k - 1; then the
        softmax of tile k. The output's rescale waits for that PV and stays out of the turn."""
        sk = (g + k) % nsK
        svp = (g + k - 1) % nsV
        T.sync_threads(my_bar, turn_threads)
        T.mbarrier_wait_parity(kready[sk], ((g + k) // nsK) % 2)
        T.wgmma_gemm(
            Qs[qs, wg, :, :], Ks[sk, :, :], acc_s, transpose_B=True, policy=Pol, clear_accum=True
        )
        T.mbarrier_wait_parity(vready[svp], ((g + k - 1) // nsV) % 2)
        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
        if use_ones:
            T.wgmma_gemm(pcast, ones, acc_l, transpose_B=True, policy=Pol, clear_accum=False)
        T.named_barrier_arrive(nxt_bar, turn_threads)
        T.wait_wgmma(1)
        T.mbarrier_arrive(kfree[sk])
        if masked:
            if is_causal:
                for i, j in T.Parallel(half, block_N):
                    acc_s[i, j] = T.if_then_else(
                        k * block_N + j <= limit[i], acc_s[i, j], -T.infinity(accum)
                    )
            else:
                for i, j in T.Parallel(half, block_N):
                    acc_s[i, j] = T.if_then_else(
                        k * block_N + j < seq_len_kv, acc_s[i, j], -T.infinity(accum)
                    )
        if use_softcap:
            apply_softcap(acc_s, half, block_N)
        T.copy(sm, smp)
        T.reduce_max(acc_s, sm, dim=1, clear=False)
        for i in T.Parallel(half):
            alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
        for i, j in T.Parallel(half, block_N):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
        if not use_ones:
            for i, j in T.Parallel(half, 16):
                lp[i, j] *= alpha[i]
            add_partial_sums(acc_s, lp)
        # A post-processing pass stores a few probabilities here, keeping their exponentials
        # ahead of the wait below; ptxas otherwise hoists the wait above all of them.
        T.call_extern(
            "handle", "tileops::value_wait_anchor_slot", T.access_ptr(anchor[wg * 128], "w")
        )
        T.wait_wgmma(0)
        T.mbarrier_arrive(vfree[svp])
        T.copy(acc_s, pcast)
        for i, j in T.Parallel(half, D):
            acc_o[i, j] *= alpha[i]
        if use_ones:
            for i, j in T.Parallel(half, sum_cols):
                acc_l[i, j] *= alpha[i]

    @T.macro
    def consume(
        wg, Qs, Ks, Vs, Os, O, q_bar, q_free, kready, kfree, vready, vfree, Sinks, meta, ones,
        anchor, tiles_of_cta, causal_offset,
    ):  # fmt: skip
        """One 64-row consumer warpgroup over its CTA's tiles; the groups pass the MMA issue turn
        in a ring, and the running KV step ``g`` carries the ring stages and parities across tiles."""
        T.set_max_nreg(consumer_regs, 1)
        r0 = wg * half
        my_bar = wg + 1
        nxt_bar = (wg + 1) % num_consumers + 1
        acc_s = T.alloc_fragment([half, block_N], accum)
        pcast = T.alloc_fragment([half, block_N], dtype)  # register-P (rs-wgmma)
        acc_o = T.alloc_fragment([half, D], accum)
        sm = T.alloc_fragment([half], accum)
        smp = T.alloc_fragment([half], accum)
        alpha = T.alloc_fragment([half], accum)
        logsum = T.alloc_fragment([half], accum)
        acc_l = T.alloc_fragment([half, sum_cols], accum) if use_ones else None
        lp = None if use_ones else T.alloc_fragment([half, block_N], accum)
        limit = T.alloc_fragment([half], "int32")
        q0 = T.alloc_var("int32")
        eff = T.alloc_var("int32")
        head = T.alloc_var("int32")
        batch = T.alloc_var("int32")

        if wg == num_consumers - 1:
            T.named_barrier_arrive(1, turn_threads)
        g = T.alloc_var("int32", init=0)
        for it in T.serial(tiles_of_cta):
            qs = it % 2
            T.fill(acc_o, 0)
            if use_ones:
                T.fill(acc_l, 0)
            T.fill(sm, -T.infinity(accum))
            T.mbarrier_wait_parity(q_bar[qs], (it // 2) % 2)
            # The producer wrote the tile's coordinates beside its query; they are read here,
            # before q_free hands the slot to the tile after next.
            # Broadcast from lane 0 so that ptxas keeps the ring stages, parities and the
            # descriptors derived from them in uniform registers.
            q0 = T.call_extern("int32", "tileops::warp_uniform_int", meta[qs, 0])
            eff = T.call_extern("int32", "tileops::warp_uniform_int", meta[qs, 1])
            head = T.call_extern("int32", "tileops::warp_uniform_int", meta[qs, 2])
            batch = T.call_extern("int32", "tileops::warp_uniform_int", meta[qs, 3])
            for i in T.Parallel(half):
                limit[i] = q0 + r0 + i + causal_offset

            T.sync_threads(my_bar, turn_threads)
            T.mbarrier_wait_parity(kready[g % nsK], (g // nsK) % 2)
            T.wgmma_gemm(
                Qs[qs, wg, :, :],
                Ks[g % nsK, :, :],
                acc_s,
                transpose_B=True,
                policy=Pol,
                clear_accum=True,
            )
            T.named_barrier_arrive(nxt_bar, turn_threads)
            T.wait_wgmma(0)
            T.mbarrier_arrive(kfree[g % nsK])
            if is_causal and q0 + r0 + causal_offset < block_N - 1:
                for i, j in T.Parallel(half, block_N):
                    acc_s[i, j] = T.if_then_else(j <= limit[i], acc_s[i, j], -T.infinity(accum))
            elif not is_causal and seq_len_kv < block_N:
                for i, j in T.Parallel(half, block_N):
                    acc_s[i, j] = T.if_then_else(j < seq_len_kv, acc_s[i, j], -T.infinity(accum))
            if use_softcap:
                apply_softcap(acc_s, half, block_N)
            T.reduce_max(acc_s, sm, dim=1, clear=False)
            for i, j in T.Parallel(half, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
            if not use_ones:
                T.fill(lp, 0)
                add_partial_sums(acc_s, lp)
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
                nu = T.alloc_var("int32", init=T.max(1, T.floordiv(seq_len_kv, block_N)))
            for k in T.serial(1, nu):
                kv_step(
                    wg, qs, k, False, Qs, Ks, Vs, kready, kfree, vready, vfree, ones, acc_s,
                    pcast, acc_o, acc_l, lp, sm, smp, alpha, limit, my_bar, nxt_bar, g, anchor,
                )  # fmt: skip
            for k in T.serial(nu, eff):
                kv_step(
                    wg, qs, k, True, Qs, Ks, Vs, kready, kfree, vready, vfree, ones, acc_s,
                    pcast, acc_o, acc_l, lp, sm, smp, alpha, limit, my_bar, nxt_bar, g, anchor,
                )  # fmt: skip

            svp = (g + eff - 1) % nsV
            T.mbarrier_wait_parity(vready[svp], ((g + eff - 1) // nsV) % 2)
            T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=Pol, clear_accum=False)
            if use_ones:
                T.wgmma_gemm(pcast, ones, acc_l, transpose_B=True, policy=Pol, clear_accum=False)
            T.wait_wgmma(0)
            T.mbarrier_arrive(vfree[svp])
            T.mbarrier_arrive(q_free[qs])
            if use_ones:
                # Every column of the ones product holds the row sum.
                T.reduce_max(acc_l, logsum, dim=1)
            else:
                T.reduce_sum(lp, logsum, dim=1)
            if has_sinks:
                sink_scale(logsum, sm, Sinks, alpha, head, 0, half)
                for i in T.Parallel(half):
                    alpha[i] /= logsum[i]
            else:
                for i in T.Parallel(half):
                    alpha[i] = 1.0 / logsum[i]
            for i, j in T.Parallel(half, D):
                acc_o[i, j] *= alpha[i]
            T.copy(acc_o, Os[wg, :, :])
            # The group's four warps finish their shared stores before one thread issues TMA.
            T.fence_proxy_async()
            T.sync_threads(num_consumers + 1 + wg, 128)
            if q0 + r0 >= 0:
                T.copy(Os[wg, :, :], O[batch, q0 + r0 : q0 + r0 + half, head, :])
            g += eff

    @T.macro
    def attention(
        Q: T.Tensor([B, seq_len_q, H, D], dtype),
        K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        O: T.Tensor([B, seq_len_q, H, D], dtype),
        Sinks=None,
    ):
        num_q_tiles = T.ceildiv(seq_len_q, block_M)
        num_tiles = num_q_tiles * H * B
        with T.Kernel(T.min(num_ctas, num_tiles), threads=threads) as cta:
            # Rounds alternate direction, so each CTA pairs a long causal tile with a short one;
            # the last, partial round fills from the low CTAs on even rounds, the high on odd.
            full_rounds = num_tiles // num_ctas
            partial = num_tiles % num_ctas
            in_partial = T.if_then_else(
                full_rounds % 2 == 0, cta < partial, num_ctas - 1 - cta < partial
            )
            tiles_of_cta = full_rounds + T.cast(in_partial, "int32")
            Qs = T.alloc_shared([2, num_consumers, half, D], dtype)
            Ks = T.alloc_shared([nsK, block_N, D], dtype)
            Vs = T.alloc_shared([nsV, block_N, D], dtype)
            Os = T.alloc_shared(
                [num_consumers, half, D], dtype
            )  # per-WG smem-staged output (FlashInfer epilogue)
            T.annotate_layout(
                {
                    Qs: make_swizzled_layout(Qs),
                    Ks: make_swizzled_layout(Ks),
                    Vs: make_swizzled_layout(Vs),
                }
            )

            meta = T.alloc_shared([2, 4], "int32")
            ones = T.alloc_shared([sum_cols, block_N], dtype) if use_ones else None
            # Each consumer thread's word for the value-product wait anchor.
            anchor = T.alloc_shared([nmma], accum)
            q_bar = T.alloc_barrier([32] * 2)
            q_free = T.alloc_barrier([nmma] * 2)
            kready = T.alloc_barrier([32] * nsK)
            kfree = T.alloc_barrier([nmma] * nsK)
            vready = T.alloc_barrier([32] * nsV)
            vfree = T.alloc_barrier([nmma] * nsV)

            causal_offset = seq_len_kv - seq_len_q
            tx = T.get_thread_binding()
            if use_ones:
                for u in T.serial(T.ceildiv(block_N * sum_cols, threads)):
                    e = u * threads + tx
                    if e < block_N * sum_cols:
                        ones[e % sum_cols, e // sum_cols] = T.cast(1.0, dtype)
                # The consumers read the ones through the async proxy.
                T.fence_proxy_async()
                T.sync_threads()

            if tx >= nmma:  # ================= producer =================
                T.set_max_nreg(producer_regs, 0)  # producer is TMA-only: release regs to consumers
            if tx >= nmma and tx < nmma + 32:  # only 1 warp issues TMA + waits (rest of WG idle)
                g = T.alloc_var("int32", init=0)
                for it in T.serial(tiles_of_cta):
                    _, by, bz, q0, eff = tile_coords(zigzag(it, cta), num_q_tiles, causal_offset)
                    cv = by // groups
                    qs = it % 2
                    T.mbarrier_wait_parity(q_free[qs], ((it // 2) % 2) ^ 1)
                    if tx == nmma:
                        meta[qs, 0] = q0
                        meta[qs, 1] = eff
                        meta[qs, 2] = by
                        meta[qs, 3] = bz
                    # A tile's query is read once; its key and value tiles serve every query
                    # head of the KV group, so they are kept in L2 ahead of the query.
                    for wg in T.unroll(num_consumers):
                        T.tma_copy(
                            Q[bz, q0 + wg * half : q0 + (wg + 1) * half, by, :],
                            Qs[qs, wg, :, :],
                            barrier=q_bar[qs],
                            eviction_policy="evict_first",
                        )
                    T.mbarrier_arrive(q_bar[qs])
                    for k in T.serial(eff):
                        sk = (g + k) % nsK
                        T.mbarrier_wait_parity(kfree[sk], (((g + k) // nsK) % 2) ^ 1)
                        T.tma_copy(
                            K[bz, k * block_N : (k + 1) * block_N, cv, :],
                            Ks[sk, :, :],
                            barrier=kready[sk],
                            eviction_policy="evict_last",
                        )
                        T.mbarrier_arrive(kready[sk])
                        sv = (g + k) % nsV
                        T.mbarrier_wait_parity(vfree[sv], (((g + k) // nsV) % 2) ^ 1)
                        T.tma_copy(
                            V[bz, k * block_N : (k + 1) * block_N, cv, :],
                            Vs[sv, :, :],
                            barrier=vready[sv],
                            eviction_policy="evict_last",
                        )
                        T.mbarrier_arrive(vready[sv])
                    g += eff

            with T.ws(0):
                consume(
                    0,
                    Qs,
                    Ks,
                    Vs,
                    Os,
                    O,
                    q_bar,
                    q_free,
                    kready,
                    kfree,
                    vready,
                    vfree,
                    Sinks,
                    meta,
                    ones,
                    anchor,
                    tiles_of_cta,
                    causal_offset,
                )
            with T.ws(1):
                consume(
                    1,
                    Qs,
                    Ks,
                    Vs,
                    Os,
                    O,
                    q_bar,
                    q_free,
                    kready,
                    kfree,
                    vready,
                    vfree,
                    Sinks,
                    meta,
                    ones,
                    anchor,
                    tiles_of_cta,
                    causal_offset,
                )
            if num_consumers == 3:
                with T.ws(2):
                    consume(
                        2,
                        Qs,
                        Ks,
                        Vs,
                        Os,
                        O,
                        q_bar,
                        q_free,
                        kready,
                        kfree,
                        vready,
                        vfree,
                        Sinks,
                        meta,
                        ones,
                        anchor,
                        tiles_of_cta,
                        causal_offset,
                    )

    if has_sinks:

        @T.prim_func
        def with_sinks(
            Q: T.Tensor([B, seq_len_q, H, D], dtype),
            K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            O: T.Tensor([B, seq_len_q, H, D], dtype),
            Sinks: T.Tensor([H], "float32"),
        ):
            attention(Q, K, V, O, Sinks)

        return with_sinks

    @T.prim_func
    def main(
        Q: T.Tensor([B, seq_len_q, H, D], dtype),
        K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        O: T.Tensor([B, seq_len_q, H, D], dtype),
    ):
        attention(Q, K, V, O)

    return main


class GQADensePersistentWSKernel(GQADenseWSKernel):
    """Dense prefill on the FA3 pipeline with one CTA an SM walking its share of the tiles.

    Preferred wherever its buffers fit the device's shared memory: the persistent grid
    reloads no per-CTA state between tiles and loads the next tile's query during the current
    tile's epilogue, and its key step keeps the output rescale out of the MMA issue turn.
    The WS kernel serves the wider head dimensions it refuses.
    """

    preferred_over = frozenset({"gqa_dense"})
    _program = staticmethod(_gqa_dense_ws_persistent_kernel)

    @staticmethod
    def _block_m(dim: int) -> int:
        """Query rows a tile: three 64-row consumers at D64, where the 160-register consumer
        budget still holds a 64x128 score tile, else two."""
        return 192 if dim == 64 else 128

    @classmethod
    def shared_bytes(cls, dim: int) -> int:
        """Dynamic shared memory of the program at head dimension ``dim``: two query slots,
        two key and two value stages of 128 rows, the staged output, the per-tile
        coordinates, the value-product wait anchor and, with three consumers, the ones
        operand, each placed on a kilobyte."""
        block_m = cls._block_m(dim)
        consumers = block_m // 64
        buffers = [
            2 * block_m * dim * 2,  # query slots
            2 * 128 * dim * 2,  # key stages
            2 * 128 * dim * 2,  # value stages
            block_m * dim * 2,  # staged output
            2 * 4 * 4,  # tile coordinates
            consumers * 128 * 4,  # wait anchor
        ]
        if consumers == 3:
            buffers.append(8 * 128 * 2)  # ones operand
        return sum(-(-b // 1024) * 1024 for b in buffers)

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """As the WS kernel, for head dimensions whose buffers fit the device."""
        reason = super().refusal(call)
        if reason is not None:
            return reason
        index = call.device.index if call.device is not None else None
        if cls.shared_bytes(call.dim) > get_shared_memory_optin(index):
            return "the query slots and key/value stages exceed shared memory"
        return None

    def _program_args(self, dim: int, device_index: Optional[int]) -> dict:
        return {"block_M": self._block_m(dim), "num_ctas": get_sm_count(device_index)}


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
    has_sinks: bool = False,
) -> Callable:
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = heads // heads_kv
    accum_dtype = "float"
    has_window = window_size_left >= 0 or window_size_right >= 0

    @tilelang.jit(
        out_idx=[4, 5],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_sw_fwd_wgmma_pipelined_func(block_m, block_n, num_stages, threads):
        sink_scale = make_dense_sink_scale(scale, block_m)
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
                    out_of_bounds = k_idx * block_n + j >= seq_len
                    left_mask = (window_size_left >= 0) and (
                        k_idx * block_n + j < bx * block_m + i - window_size_left
                    )
                    right_mask = (window_size_right >= 0) and (
                        k_idx * block_n + j > bx * block_m + i + window_size_right
                    )
                    acc_s[i, j] = T.if_then_else(
                        out_of_bounds or left_mask or right_mask,
                        -T.infinity(accum_dtype),
                        0,
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
            sinks: T.Tensor([heads] if has_sinks else q_shape, "float32" if has_sinks else dtype),
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
                        # Subtract before scaling: two clamped maxima then differ by exactly 0,
                        # where fast math's fused prev * scale - curr * scale leaves the
                        # product's rounding error (about 2**99) in the exponent.
                        scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * scale)
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]
                    mma1(v, v_shared, acc_s_cast, acc_o, k_idx, by, bz)

                if has_sinks:
                    sink_scale(logsum, scores_max, sinks, scores_scale, by, 0, block_m)
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]
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


@functools.lru_cache(maxsize=32)
def _gqa_sw_fwd_packed_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    window_size_left: int,
    block_m: int,
    sm_scale: Optional[float] = None,
    softcap: float = 0.0,
    dtype: str = "float16",
    has_sinks: bool = False,
) -> Callable:
    """Causal sliding window whose band fits two key tiles, one CTA a (query tile, KV head).

    The band's key and value tiles load once and serve the group's query heads in turn,
    with the next head's query tile in flight while the current one is reduced.
    """
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[4, 5],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_sw_fwd_packed_func(num_stages):
        block_n = block_m
        threads = 256
        sink_scale = make_dense_sink_scale(scale, block_m)
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_m, block_n)
            if use_softcap
            else None
        )

        @T.macro
        def band_step(
            q_shared: T.SharedBuffer([block_m, dim], dtype),
            k_shared: T.SharedBuffer([block_n, dim], dtype),
            v_shared: T.SharedBuffer([block_n, dim], dtype),
            acc_s: T.FragmentBuffer([block_m, block_n], accum_dtype),
            acc_s_cast: T.FragmentBuffer([block_m, block_n], dtype),
            acc_o: T.FragmentBuffer([block_m, dim], accum_dtype),
            scores_max: T.FragmentBuffer([block_m], accum_dtype),
            scores_max_prev: T.FragmentBuffer([block_m], accum_dtype),
            scores_scale: T.FragmentBuffer([block_m], accum_dtype),
            scores_sum: T.FragmentBuffer([block_m], accum_dtype),
            logsum: T.FragmentBuffer([block_m], accum_dtype),
            k_idx: T.int32,
            bx: T.int32,
        ) -> None:
            for i, j in T.Parallel(block_m, block_n):
                row = bx * block_m + i
                col = k_idx * block_n + j
                acc_s[i, j] = T.if_then_else(
                    (row < col) or (col < row - window_size_left), -T.infinity(accum_dtype), 0
                )
            T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            if use_softcap:
                apply_softcap(acc_s)
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_m):
                # A finite floor keeps exp2(prev - curr) defined on a fully masked tile.
                scores_max[i] = T.max(
                    T.max(scores_max[i], scores_max_prev[i]), T.cast(-1e38, accum_dtype)
                )
            for i in T.Parallel(block_m):
                scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * scale)
            for i, j in T.Parallel(block_m, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_m):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)
            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] *= scores_scale[i]
            T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.prim_func
        def _gqa_sw_fwd_packed_main(
            q: T.Tensor(q_shape, dtype),
            k: T.Tensor(kv_shape, dtype),
            v: T.Tensor(kv_shape, dtype),
            sinks: T.Tensor([heads] if has_sinks else q_shape, "float32" if has_sinks else dtype),
            output: T.Tensor(q_shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ) -> None:
            with T.Kernel(T.ceildiv(seq_len, block_m), heads_kv, batch, threads=threads) as (
                bx,
                hk,
                bz,
            ):
                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_lo = T.alloc_shared([block_n, dim], dtype)
                v_lo = T.alloc_shared([block_n, dim], dtype)
                k_hi = T.alloc_shared([block_n, dim], dtype)
                v_hi = T.alloc_shared([block_n, dim], dtype)
                o_shared = T.alloc_shared([block_m, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                T.annotate_layout({o_shared: make_swizzled_layout(o_shared)})
                hi = bx
                lo = T.max(bx - 1, 0)
                T.copy(k[bz, lo * block_n : (lo + 1) * block_n, hk, :], k_lo)
                T.copy(v[bz, lo * block_n : (lo + 1) * block_n, hk, :], v_lo)
                T.copy(k[bz, hi * block_n : (hi + 1) * block_n, hk, :], k_hi)
                T.copy(v[bz, hi * block_n : (hi + 1) * block_n, hk, :], v_hi)

                for g in T.Pipelined(groups, num_stages=num_stages):
                    head = hk * groups + g
                    T.copy(q[bz, bx * block_m : (bx + 1) * block_m, head, :], q_shared)
                    T.clear(acc_o)
                    T.clear(logsum)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    if bx > 0:
                        band_step(
                            q_shared,
                            k_lo,
                            v_lo,
                            acc_s,
                            acc_s_cast,
                            acc_o,
                            scores_max,
                            scores_max_prev,
                            scores_scale,
                            scores_sum,
                            logsum,
                            lo,
                            bx,
                        )
                    band_step(
                        q_shared,
                        k_hi,
                        v_hi,
                        acc_s,
                        acc_s_cast,
                        acc_o,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                        hi,
                        bx,
                    )
                    if has_sinks:
                        sink_scale(logsum, scores_max, sinks, scores_scale, head, 0, block_m)
                        for i, j in T.Parallel(block_m, dim):
                            acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] /= logsum[i]
                    T.sync_threads(3, threads)
                    T.copy(acc_o, o_shared)
                    T.sync_threads(3, threads)
                    T.copy(o_shared, output[bz, bx * block_m : (bx + 1) * block_m, head, :])
                    for i in T.Parallel(block_m):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                    T.copy(logsum, lse[bz, head, bx * block_m : (bx + 1) * block_m])

        return _gqa_sw_fwd_packed_main

    return _gqa_sw_fwd_packed_func


class GQADenseSlidingWindowKernel(Kernel, GQADenseFwdInterface):
    """SM90 Dense sliding-window kernel with a native BSHD ABI."""

    supported_archs: list[int] = [90]

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """The contiguous windowed region, which FP8 has its own implementation for."""
        if call.is_fp8 or not call.uses_sliding_window:
            return "does not serve this call"
        if call.max_seqlen_q != call.seqlen_kv:
            return "a sliding window requires equal Q and KV lengths"
        return call.tensor_core_dim_refusal

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """Compiles exact extents, so the query length is in the identity."""
        args = dict(
            batch=call.batch,
            heads=call.heads,
            heads_kv=call.heads_kv,
            seq_len=call.max_seqlen_q,
            dim=call.dim,
            is_causal=call.is_causal,
            window_size_left=call.window_size_left,
            window_size_right=call.window_size_right,
            dtype=call.dtype,
            sm_scale=call.sm_scale,
            softcap=call.softcap,
            has_sinks=call.has_sinks,
            **call.rope_args,
            device_index=call.device.index if call.device is not None else None,
        )
        return tuple(args.values()), lambda: cls(**args)

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
        *,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        has_sinks: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.has_sinks = has_sinks
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
            has_sinks=self.has_sinks,
        )

        self.init_config(config)

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
        configs = itertools.product([64, 128], [64, 128], [2, 3], [128, 256])
        # Each 128-thread warpgroup takes 64 rows of the FullRow GEMMs, so 256 threads
        # need a 128-row tile.
        return [
            {"block_m": m, "block_n": n, "num_stages": s, "threads": t}
            for m, n, s, t in configs
            if m >= t // 2
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
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._require_devices(q=q, k=k, v=v)
        if self.rope is not None:
            q, k = self.rope(q, k, rope_cos, rope_sin)
        output, _ = _gqa_sw_fwd_wgmma_pipelined_kernel(
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
            has_sinks=self.has_sinks,
        )(
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
        )(q, k, v, sinks if self.has_sinks else q)
        return output


class GQADenseSlidingWindowBandKernel(Kernel, GQADenseFwdInterface):
    """SM90 causal sliding window whose band fits two key tiles.

    One CTA serves a (query tile, KV head): the band's key and value tiles load once and
    serve the KV head's query heads in turn.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_dense_sliding_window"})
    _BAND_TILE = 128

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """A causal window of at most one key tile whose band fits shared memory."""
        reason = GQADenseSlidingWindowKernel.refusal(call)
        if reason is not None:
            return reason
        if not call.is_causal or not 0 <= call.window_size_left <= cls._BAND_TILE:
            return "the window reaches past one key tile"
        # Two key and two value tiles, a double-buffered query tile and the output tile:
        # seven 128 x 128 16-bit tiles, 224 KB, inside SM90's 227 KB a block may opt into.
        if call.dim > 128:
            return "the band and a double-buffered query tile exceed shared memory"
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """Compiles exact extents, so the query length is in the identity."""
        args = dict(
            batch=call.batch,
            heads=call.heads,
            heads_kv=call.heads_kv,
            seq_len=call.max_seqlen_q,
            dim=call.dim,
            window_size_left=call.window_size_left,
            dtype=call.dtype,
            sm_scale=call.sm_scale,
            softcap=call.softcap,
            has_sinks=call.has_sinks,
            **call.rope_args,
            device_index=call.device.index if call.device is not None else None,
        )
        return tuple(args.values()), lambda: cls(**args)

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        window_size_left: int,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        *,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        has_sinks: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.has_sinks = has_sinks
        self.dtype = dtype
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
        self.kernel = _gqa_sw_fwd_packed_kernel(
            batch,
            heads,
            heads_kv,
            seq_len,
            dim,
            window_size_left,
            self._BAND_TILE,
            dim**-0.5 if sm_scale is None else sm_scale,
            softcap,
            self.dtype_str,
            has_sinks=has_sinks,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"num_stages": 2}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"num_stages": stages} for stages in (1, 2)]

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
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._require_devices(q=q, k=k, v=v)
        if self.rope is not None:
            q, k = self.rope(q, k, rope_cos, rope_sin)
        program = self.kernel(self.config["num_stages"])
        output, _ = program(q, k, v, sinks if self.has_sinks else q)
        return output
