"""Dense BSHD grouped-query attention kernels and preprocessing."""

import functools
import itertools
import re
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch
import tvm_ffi
from tilelang import tvm
from tilelang.layout import make_swizzled_layout

from tileops._csrc import csrc_include
from tileops._kernels.attention.call_spec import AttentionCall, GQADenseFwdInterface
from tileops._kernels.attention.online_softmax import make_apply_softcap, make_dense_sink_scale
from tileops._kernels.constants import (
    LOG2E,
    TMA_DTYPE_BFLOAT16,
    TMA_DTYPE_FLOAT16,
    TMA_INTERLEAVE_NONE,
    TMA_L2_PROMOTION_128B,
    TMA_OOB_FILL_NONE,
    TMA_SWIZZLE_128B,
)
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._utils import get_shared_memory_optin, get_sm_count

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
    """The stage that rotates Dense Q and K once before an attention implementation consumes
    them; the kernel that owns it compiles it with :meth:`build`."""

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
        self.args = (
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

    def build(self) -> Callable:
        """The rotation, launched as ``rotate(q, k, rope_cos, rope_sin) -> (q_rot, k_rot)``."""
        program = CompiledProgram(_gqa_dense_rope_qk_kernel(*self.args))

        def rotate(
            q: torch.Tensor,
            k: torch.Tensor,
            rope_cos: Optional[torch.Tensor],
            rope_sin: Optional[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if rope_cos is None or rope_sin is None:
                raise ValueError("fused RoPE requires rope_cos and rope_sin")
            q_rot, k_rot = program(q.reshape(-1), k.reshape(-1), rope_cos, rope_sin)
            return q_rot.reshape_as(q), k_rot.reshape_as(k)

        return rotate


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
    with_lse=False,
):
    """Build the Dense WS program; its online softmax carries the previous tile's alpha.

    With ``with_lse`` the program also writes each query row's natural-log log-sum-exp
    into ``Lse``, ``[B, H, seq_len_q]``."""
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
    def store_lse(Lse, logsum, sm, bz, by, row0):
        # exp2(s * scale - sm * scale) summed to logsum: ln(logsum) + sm * scale * ln 2.
        for i in T.Parallel(half):
            if row0 + i < seq_len_q:
                Lse[bz, by, row0 + i] = T.if_then_else(
                    logsum[i] > 0,
                    T.log(logsum[i]) + sm[i] * T.cast(scale / LOG2E, accum),
                    -T.infinity(accum),
                )

    @T.macro
    def attention(
        Q: T.Tensor([B, seq_len_q, H, D], dtype),
        K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        O: T.Tensor([B, seq_len_q, H, D], dtype),
        Sinks=None,
        Lse=None,
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

            if tx >= 256:
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
                if with_lse:
                    store_lse(Lse, logsum, sm, bz, by, q0 + r0)
                for i, j in T.Parallel(half, D):
                    acc_o[i, j] *= alpha[i]
                T.copy(acc_o, Os[0, :, :])
                T.fence_proxy_async()
                T.sync_threads(3, 128)
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
                if with_lse:
                    store_lse(Lse, logsum, sm, bz, by, q0 + r0)
                for i, j in T.Parallel(half, D):
                    acc_o[i, j] *= alpha[i]
                T.copy(acc_o, Os[1, :, :])
                T.fence_proxy_async()
                T.sync_threads(4, 128)
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

    if with_lse:

        @T.prim_func
        def main_with_lse(
            Q: T.Tensor([B, seq_len_q, H, D], dtype),
            K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            O: T.Tensor([B, seq_len_q, H, D], dtype),
            Lse: T.Tensor([B, H, seq_len_q], accum),
        ):
            attention(Q, K, V, O, Lse=Lse)

        return main_with_lse

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
        if call.return_lse and call.has_sinks:
            return "does not return the log-sum-exp with sinks"
        return call.tensor_core_dim_refusal

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """Accepts its sequence extents at runtime unless RoPE compiles them in."""
        args = cls._entry_args(call)
        runtime = () if call.fuse_rope else ("seq_len_q", "seq_len_kv")
        return tuple(v for k, v in args.items() if k not in runtime), lambda: cls(**args)

    @classmethod
    def _entry_args(cls, call: AttentionCall) -> dict:
        """The constructor arguments for ``call``."""
        return dict(
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
            return_lse=call.return_lse,
            **call.rope_args,
            device_index=call.device.index if call.device is not None else None,
        )

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
        return_lse: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.has_sinks = has_sinks
        self.return_lse = return_lse
        self.dtype = dtype
        self._shape_args = (
            batch,
            heads,
            heads_kv,
            dim,
            is_causal,
            dim**-0.5 if sm_scale is None else sm_scale,
            softcap,
            self.dtype_str,
        )
        self._extra_args = self._program_args(dim, device_index)
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
        self.program(self._compile)

    def _compile(self, cfg: dict) -> tuple:
        """The attention program, and the Q/K rotation ahead of it or ``None``."""
        # Exposed as ``self.kernel`` because that is what profiling reads.
        lse = {"with_lse": True} if self.return_lse else {}
        self.kernel = kernel = self._program(
            *self._shape_args, has_sinks=self.has_sinks, **lse, **self._extra_args
        )
        rope = self.rope.build() if self.rope is not None else None
        return CompiledProgram(kernel), rope

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
        attention, rope = self.program(self._compile)
        if rope is not None:
            q, k = rope(q, k, rope_cos, rope_sin)
        if self.has_sinks:
            return attention(q, k, v, sinks)
        if self.return_lse:
            batch, seq_len_q, heads, _ = q.shape
            lse = torch.empty((batch, heads, seq_len_q), dtype=torch.float32, device=q.device)
            return attention(q, k, v, lse), lse
        return attention(q, k, v)


def _persistent_tiles(
    B, H, block_M, block_N, is_causal, num_ctas, seq_len_q, seq_len_kv, rank_group
):
    """The tile schedule of the persistent programs: ``(tiles_of_cta, tile_of, tile_coords)``.

    Round ``it`` gives CTA ``cta`` tile ``tile_of(it, cta)``; rounds alternate direction, so
    each CTA pairs a long causal tile with a short one, and the last, partial round fills from
    the low CTAs on even rounds, the high on odd. ``tile_coords`` maps a tile to its head,
    batch, first query row and key steps.
    """
    half = 64

    def tiles_of_cta(cta, num_tiles):
        full_rounds = num_tiles // num_ctas
        partial = num_tiles % num_ctas
        in_partial = T.if_then_else(
            full_rounds % 2 == 0, cta < partial, num_ctas - 1 - cta < partial
        )
        return full_rounds + T.cast(in_partial, "int32")

    def tile_of(it, cta):
        return it * num_ctas + T.if_then_else(it % 2 == 0, cta, num_ctas - 1 - cta)

    def tile_coords(tile_id, num_q_tiles, causal_offset):
        """Heads vary fastest, then batches; a causal call takes its longest query blocks
        first. With a ``rank_group`` G, each full group of G query blocks walks one head's
        blocks on neighbouring tiles, so the CTAs resident together share that head's keys and
        values; the blocks past the last full group follow heads first."""
        if rank_group == 1:
            by = tile_id % H
            bz = (tile_id // H) % B
            rank = tile_id // (H * B)
        else:
            grouped = num_q_tiles // rank_group * rank_group * (B * H)
            rest = tile_id - grouped
            head = T.if_then_else(
                tile_id < grouped, tile_id % (rank_group * B * H) // rank_group, rest % (B * H)
            )
            rank = T.if_then_else(
                tile_id < grouped,
                tile_id // (rank_group * B * H) * rank_group + tile_id % rank_group,
                num_q_tiles // rank_group * rank_group + rest // (B * H),
            )
            by = head % H
            bz = head // H
        bx = (num_q_tiles - 1 - rank) if is_causal else rank
        # A causal call aligns its tiles to the end of Q, rounded up to a 64-row group, so the
        # partial tile is the shortest one. Its groups before row 0 load zeros through TMA
        # and store nothing; a group never straddles row 0.
        if is_causal:
            q0 = T.ceildiv(seq_len_q, half) * half - (num_q_tiles - bx) * block_M
            eff = T.min(
                T.ceildiv(seq_len_kv, block_N), T.ceildiv(q0 + block_M + causal_offset, block_N)
            )
        else:
            q0 = bx * block_M
            eff = T.ceildiv(seq_len_kv, block_N)
        return by, bz, q0, eff

    return tiles_of_cta, tile_of, tile_coords


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
    # issued beside the value product. Two consumers keep per-thread partial sums: the
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

    tiles_of_cta_count, zigzag, tile_coords = _persistent_tiles(
        B, H, block_M, block_N, is_causal, num_ctas, seq_len_q, seq_len_kv, 1
    )

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
            tiles_of_cta = tiles_of_cta_count(cta, num_tiles)
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
                    by, bz, q0, eff = tile_coords(zigzag(it, cta), num_q_tiles, causal_offset)
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


@functools.lru_cache(maxsize=32)
@tilelang.jit(out_idx=[3], pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS)
def _gqa_dense_persistent_d128_kernel(
    B,
    H,
    Hkv,
    D,
    is_causal,
    sm_scale,
    softcap,
    dtype,
    num_ctas=132,
    rank_group=1,
    has_sinks=False,
    with_lse=False,
    num_splits=1,
    diag_aligned=False,
):
    """Build the persistent program at head dimension 128, written per thread.

    Warp 0 of warpgroup 2 loads by TMA; warpgroups 0 and 1 each own 64 query rows of a
    128-row tile, pass the MMA issue turn between them, and hold their accumulators in the
    m64n128 WGMMA register layout: thread ``t`` holds rows ``16 (t / 32) + 8 r + (t % 32) / 4``
    and, for element ``i``, column ``8 (i / 4) + 2 (t % 4) + i % 2`` with ``r = (i % 4) / 2``.
    Shared memory is one buffer holding two query slots, two key and two value stages and
    the staged output; every tile in it is stored as 64-column SW128 panels. With
    ``with_lse`` the program also writes each query row's natural-log log-sum-exp into
    ``Lse``, ``[B, H, seq_len_q]``.

    With ``num_splits`` S above one, each tile's key steps divide into S contiguous parts,
    each one work item: part ``s`` of batch ``b`` writes, at batch ``s B + b``, its
    unnormalized FP32 output into ``O``, ``[S B, seq_len_q, H, D]``, and each row's scaled
    base-2 maximum and sum into ``Lse``, ``[S B, H, 2, seq_len_q]``, without sinks;
    :func:`_gqa_dense_split_combine_kernel` merges the parts.

    ``diag_aligned`` promises a causal call of one part whose query and key lengths both sit
    on 128-row boundaries: then warpgroup 0's diagonal key tile is its only masked one and
    the upper 64 keys of it are masked for all its rows, so that step's QK and softmax cover
    the lower 64 keys only.
    """
    assert D == 128
    block_M = block_N = 128
    half = 64
    groups = H // Hkv
    score_scale = (1.0 / D) ** 0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    split = num_splits > 1
    # Sinks join only the merge of the parts.
    apply_sinks = has_sinks and not split
    tma_dtype = TMA_DTYPE_BFLOAT16 if dtype == "bfloat16" else TMA_DTYPE_FLOAT16
    mma_dtype = "bf16" if dtype == "bfloat16" else "fp16"
    seq_len_q = T.dynamic("seq_len_q")
    seq_len_kv = T.dynamic("seq_len_kv")
    elem_bytes = 2
    panel = 128 // elem_bytes  # columns of a 128-byte swizzle panel
    q_panel_elems = half * panel
    q_group_elems = half * D
    q_slot_elems = block_M * D
    kv_panel_elems = block_N * panel
    kv_stage_elems = block_N * D
    o_group_elems = half * D
    q_elems = 2 * q_slot_elems
    kv_elems = 2 * kv_stage_elems
    q_at = 0
    k_at = q_at + q_elems
    v_at = k_at + kv_elems
    o_at = v_at + kv_elems
    smem_elems = o_at + 2 * o_group_elems
    k_step = 16  # the K extent of one WGMMA
    stmatrix = tvm.ir.Op.get("tl.ptx_stmatrix")
    pack = tvm.ir.Op.get("tl.pack_b16")
    tma_store = tvm.ir.Op.get("tl.tma_store")
    tiles_of_cta_count, tile_of, tile_coords = _persistent_tiles(
        B, H, block_M, block_N, is_causal, num_ctas, seq_len_q, seq_len_kv, rank_group
    )

    def tma_map(X, rows, heads, box_rows):
        """A TMA map of ``X [B, rows, heads, D]`` in boxes of 64 columns by ``box_rows`` rows,
        swizzled 128B."""
        return T.create_tma_descriptor(
            tma_dtype, 4, X.data, D, rows, heads, B,
            T.int64(2), T.int64(heads * D * 2), T.int64(D * 2), T.cast(rows, "int64") * (heads * D * 2),
            panel, box_rows, 1, 1, 1, 1, 1, 1,
            TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B, TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
        )  # fmt: skip

    # Descriptors live in their own lexical allocation scope so that each issue rebuilds them
    # from constants instead of carrying them across iterations in general-purpose registers.
    @T.macro
    def qk(acc_s, smem, qs, sk, wg, steps=None, n=128):
        with T.sblock("qk"):
            T.reads()
            T.writes()
            T.sblock_attr({"lexical_alloc_scope": 1})
            desc_q = T.alloc_wgmma_desc()
            desc_k = T.alloc_wgmma_desc()
            T.initialize_wgmma_descriptor(
                desc_q, T.access_ptr(smem[q_at], "r", extent=q_elems), 1, 1, 64
            )
            T.increase_descriptor_offset(
                desc_q, (qs * q_slot_elems + wg * q_group_elems) * elem_bytes
            )
            T.initialize_wgmma_descriptor(
                desc_k, T.access_ptr(smem[k_at], "r", extent=kv_elems), 1, 1, 64
            )
            T.increase_descriptor_offset(desc_k, sk * kv_stage_elems * elem_bytes)
            T.warpgroup_fence_operand(acc_s, num_regs=n // 2)
            T.warpgroup_arrive()
            per_panel = panel // k_step
            for ki in T.unroll(D // k_step if steps is None else steps):
                k_off = ki % per_panel * k_step * elem_bytes
                T.ptx_wgmma_ss(
                    "float32", "m64n%dk16" % n, True, True, mma_dtype, mma_dtype, "fp32",
                    desc_q.data, (ki // per_panel * q_panel_elems * elem_bytes + k_off) >> 4,
                    desc_k.data, (ki // per_panel * kv_panel_elems * elem_bytes + k_off) >> 4,
                    acc_s.data, 0, T.Select(ki > 0, 1, 0), 1, 1,
                )  # fmt: skip
            T.warpgroup_commit_batch()
            T.warpgroup_fence_operand(acc_s, num_regs=n // 2)

    @T.macro
    def pv(pcast, acc_o, smem, sv, clear):
        """O += P V; with ``clear`` the product overwrites O."""
        with T.sblock("pv"):
            T.reads()
            T.writes()
            T.sblock_attr({"lexical_alloc_scope": 1})
            desc_v = T.alloc_wgmma_desc()
            T.initialize_wgmma_descriptor(
                desc_v, T.access_ptr(smem[v_at], "r", extent=kv_elems), 1, 1024, 64
            )
            T.increase_descriptor_offset(desc_v, sv * kv_stage_elems * elem_bytes)
            T.warpgroup_fence_operand(pcast, num_regs=32)
            T.warpgroup_fence_operand(acc_o, num_regs=64)
            T.warpgroup_arrive()
            for ki in T.unroll(block_N // k_step):
                T.ptx_wgmma_rs(
                    "float32", "m64n128k16", False, mma_dtype, mma_dtype, "fp32",
                    pcast.data, ki * 8, desc_v.data, (ki * k_step * panel * elem_bytes) >> 4,
                    acc_o.data, 0, T.Select(ki > 0, 1, T.Select(clear, 0, 1)), 1, 1,
                )  # fmt: skip
            T.warpgroup_commit_batch()
            T.warpgroup_fence_operand(acc_o, num_regs=64)
            T.warpgroup_fence_operand(pcast, num_regs=32)

    @T.macro
    def softmax(acc_s, k, masked, lane, limit, sm, mx, n=128):
        """Mask and cap the scores of the first ``n`` keys of key tile ``k``, fold their row
        maxima into ``sm`` and turn them into probabilities. A part may hold rows that see
        none of its keys; their maximum stays -inf and their probabilities 0."""
        if masked:
            for i in T.unroll(n // 2):
                col = k * block_N + (i // 4) * 8 + (lane % 4) * 2 + i % 2
                if is_causal:
                    acc_s[i] = T.if_then_else(
                        col <= limit[(i % 4) // 2], acc_s[i], -T.infinity("float32")
                    )
                else:
                    acc_s[i] = T.if_then_else(col < seq_len_kv, acc_s[i], -T.infinity("float32"))
        if use_softcap:
            for i in T.unroll(n // 2):
                acc_s[i] = T.if_then_else(
                    acc_s[i] == -T.infinity("float32"),
                    -T.infinity("float32"),
                    T.tanh(acc_s[i] * (score_scale / softcap)) * softcap,
                )
        # The four threads of a row hold its columns; two shuffles reduce across them.
        for r in T.unroll(2):
            mx[r] = sm[r]
            for c in T.unroll(n // 8):
                mx[r] = T.max(mx[r], T.max(acc_s[c * 4 + r * 2], acc_s[c * 4 + r * 2 + 1]))
            mx[r] = T.max(mx[r], T.shfl_xor(mx[r], 1))
            sm[r] = T.max(mx[r], T.shfl_xor(mx[r], 2))
        if split:
            for r in T.unroll(2):
                mx[r] = T.if_then_else(sm[r] == -T.infinity("float32"), 0.0, sm[r] * scale)
            for i in T.unroll(n // 2):
                acc_s[i] = T.exp2(acc_s[i] * scale - mx[(i % 4) // 2])
        else:
            for i in T.unroll(n // 2):
                acc_s[i] = T.exp2(acc_s[i] * scale - sm[(i % 4) // 2] * scale)

    @T.macro
    def add_partial_sums(acc_s, lp, n=128):
        """Each thread keeps four partial sums a row, reduced across threads once a tile."""
        for t in T.unroll(n // 16):
            for i in T.unroll(8):
                lp[i] = lp[i] + acc_s[t * 8 + (i % 4) // 2 * 4 + (i // 4) * 2 + i % 2]

    @T.macro
    def to_16bit(acc, out):
        for i in T.unroll(32):
            for v in T.vectorized(2):
                out[i * 2 + v] = T.cast(acc[i * 2 + v], dtype)

    @T.macro
    def kv_step(
        wg, k, k0, masked, g, qs, lane, my_bar, nxt_bar, smem, aux, kready, kfree, vready,
        vfree, acc_s, acc_o, pcast, lp, sm, smp, mx, alpha, limit, n=128, qk_n=128,
    ):  # fmt: skip
        """Key step k of a part starting at key tile k0: QK of key tile k and PV of tile
        k - 1 under the issue turn, then the softmax of tile k; the output rescale waits for
        that PV. With ``n`` 64 the softmax covers only the first 64 keys of tile k, and the
        others get probability zero; with ``qk_n`` 64 the QK scores only those keys. Only the
        first group waits on the K and V stages: the second takes the issue turn after the
        first has waited on the same stages."""
        j = g + k - k0  # the CTA's running key step, which picks stages and parities
        sk = j % 2
        svp = (j - 1) % 2
        T.sync_threads(my_bar, 256)
        if wg == 0:
            T.mbarrier_wait_parity(kready[sk], (j // 2) % 2)
        qk(acc_s, smem, qs, sk, wg, n=qk_n)
        if wg == 0:
            T.mbarrier_wait_parity(vready[svp], ((j - 1) // 2) % 2)
        pv(pcast, acc_o, smem, svp, k == k0 + 1)
        T.named_barrier_arrive(nxt_bar, 256)
        T.wait_wgmma(1)
        T.mbarrier_arrive(kfree[sk])
        for r in T.unroll(2):
            smp[r] = sm[r]
        softmax(acc_s, k, masked, lane, limit, sm, mx, n)
        for r in T.unroll(2):
            if split:
                # mx holds the scaled maximum, 0 for a row still without keys.
                alpha[r] = T.exp2(smp[r] * scale - mx[r])
            else:
                alpha[r] = T.exp2(smp[r] * scale - sm[r] * scale)
        for i in T.unroll(8):
            lp[i] = lp[i] * alpha[i // 4]
        add_partial_sums(acc_s, lp, n)
        # A shared store of four probabilities keeps their exponentials ahead of the wait:
        # ptxas otherwise hoists the wait above all of them.
        aux[wg * 128 + lane] = acc_s[0] + acc_s[1] + acc_s[4] + acc_s[5]
        T.wait_wgmma(0)
        T.mbarrier_arrive(vfree[svp])
        if n < block_N:
            for i in T.unroll(n // 2, block_N // 2):
                acc_s[i] = 0.0
        to_16bit(acc_s, pcast)
        for i in T.unroll(64):
            acc_o[i] = acc_o[i] * alpha[(i % 4) // 2]

    @T.macro
    def consume(
        wg, tx, smem, meta, aux, q_bar, q_free, kready, kfree, vready, vfree, O, Sinks, Lse,
        tiles_of_cta, causal_offset,
    ):  # fmt: skip
        """One 64-row consumer warpgroup over its CTA's tiles. A tile's first QK is issued in
        the turn of the previous tile's last PV, so it runs during that tile's epilogue."""
        T.set_max_nreg(240, 1)
        lane = tx % 128
        r0 = wg * half
        my_bar = wg + 1
        nxt_bar = (wg + 1) % 2 + 1
        acc_s = T.alloc_local([64], "float32")
        acc_o = T.alloc_local([64], "float32")
        pcast = T.alloc_local([64], dtype)
        lp = T.alloc_local([8], "float32")
        sm = T.alloc_local([2], "float32")
        smp = T.alloc_local([2], "float32")
        mx = T.alloc_local([2], "float32")
        alpha = T.alloc_local([2], "float32")
        logsum = T.alloc_local([2], "float32")
        limit = T.alloc_local([2], "int32")
        q0 = T.alloc_var("int32")
        eff = T.alloc_var("int32")
        head = T.alloc_var("int32")
        batch = T.alloc_var("int32")
        k0 = T.alloc_var("int32", init=0) if split else 0
        g = T.alloc_var("int32", init=0)
        if wg == 1:
            T.named_barrier_arrive(1, 256)
        for r in T.unroll(2):
            sm[r] = -T.infinity("float32")
        T.mbarrier_wait_parity(q_bar[0], 0)
        # Broadcast from lane 0, so that ptxas keeps the coordinates and everything derived
        # from them in uniform registers.
        q0 = T.shfl_sync(meta[0, 0], 0)
        eff = T.shfl_sync(meta[0, 1], 0)
        head = T.shfl_sync(meta[0, 2], 0)
        batch = T.shfl_sync(meta[0, 3], 0)
        if split:
            k0 = T.shfl_sync(meta[0, 4], 0)
        T.sync_threads(my_bar, 256)
        T.mbarrier_wait_parity(kready[0], 0)
        qk(acc_s, smem, 0, 0, wg)
        T.named_barrier_arrive(nxt_bar, 256)
        for it in T.serial(tiles_of_cta):
            qs = it % 2
            for r in T.unroll(2):
                limit[r] = q0 + r0 + (lane // 32) * 16 + r * 8 + (lane % 32) // 4 + causal_offset
            T.wait_wgmma(0)
            T.mbarrier_arrive(kfree[g % 2])
            # Key tiles below nu need no mask; eff counts the part's key steps from k0.
            if is_causal:
                softmax(
                    acc_s, k0, q0 + r0 + causal_offset < (k0 + 1) * block_N - 1, lane, limit,
                    sm, mx,
                )  # fmt: skip
                nu = T.alloc_var(
                    "int32",
                    init=T.max(k0 + 1, T.min(k0 + eff, (q0 + r0 + causal_offset + 1) // block_N)),
                )
            else:
                softmax(acc_s, k0, seq_len_kv < (k0 + 1) * block_N, lane, limit, sm, mx)
                nu = T.alloc_var(
                    "int32", init=T.max(k0 + 1, T.min(k0 + eff, seq_len_kv // block_N))
                )
            for i in T.unroll(8):
                lp[i] = 0.0
            add_partial_sums(acc_s, lp)
            to_16bit(acc_s, pcast)
            for k in T.serial(k0 + 1, nu):
                kv_step(
                    wg, k, k0, False, g, qs, lane, my_bar, nxt_bar, smem, aux, kready, kfree,
                    vready, vfree, acc_s, acc_o, pcast, lp, sm, smp, mx, alpha, limit,
                )  # fmt: skip
            if wg == 0 and diag_aligned:
                # The diagonal tile, if the tile has more than one, at 64 keys.
                for k in T.serial(nu, k0 + eff):
                    kv_step(
                        wg, k, k0, True, g, qs, lane, my_bar, nxt_bar, smem, aux, kready, kfree,
                        vready, vfree, acc_s, acc_o, pcast, lp, sm, smp, mx, alpha, limit,
                        block_N // 2, block_N // 2,
                    )  # fmt: skip
            else:
                # When the group's last needed key falls in the lower half of the last key tile,
                # that step masks and exponentiates only the lower 64 keys: the upper ones are
                # masked for every row of the group.
                if is_causal:
                    last_keys = (
                        T.min(seq_len_kv, q0 + r0 + half + causal_offset) - (k0 + eff - 1) * block_N
                    )
                else:
                    last_keys = seq_len_kv - (k0 + eff - 1) * block_N
                narrow = T.alloc_var("bool", init=(eff > 1) & (last_keys <= block_N // 2))
                for k in T.serial(nu, k0 + eff - T.cast(narrow, "int32")):
                    kv_step(
                        wg, k, k0, True, g, qs, lane, my_bar, nxt_bar, smem, aux, kready, kfree,
                        vready, vfree, acc_s, acc_o, pcast, lp, sm, smp, mx, alpha, limit,
                    )  # fmt: skip
                if narrow:
                    kv_step(
                        wg, k0 + eff - 1, k0, True, g, qs, lane, my_bar, nxt_bar, smem, aux, kready,
                        kfree, vready, vfree, acc_s, acc_o, pcast, lp, sm, smp, mx, alpha, limit,
                        block_N // 2,
                    )  # fmt: skip

            svl = (g + eff - 1) % 2
            # As in the key step, the second group's turn follows the first group's stage waits.
            if wg == 0:
                T.mbarrier_wait_parity(vready[svl], ((g + eff - 1) // 2) % 2)
            if it + 1 < tiles_of_cta:
                T.mbarrier_wait_parity(q_bar[1 - qs], ((it + 1) // 2) % 2)
                if wg == 0:
                    T.mbarrier_wait_parity(kready[(g + eff) % 2], ((g + eff) // 2) % 2)
            T.sync_threads(my_bar, 256)
            pv(pcast, acc_o, smem, svl, eff == 1)
            # Every tile ends with a QK in flight, which keeps ptxas from serializing the
            # WGMMAs; after the last tile it is one K step on whatever the shared memory
            # holds, so that the other group's last PV does not wait behind a whole one.
            if it + 1 < tiles_of_cta:
                qk(acc_s, smem, 1 - qs, (g + eff) % 2, wg)
            else:
                qk(acc_s, smem, 1 - qs, (g + eff) % 2, wg, 1)
            T.named_barrier_arrive(nxt_bar, 256)
            T.wait_wgmma(1)
            T.mbarrier_arrive(vfree[svl])
            T.mbarrier_arrive(q_free[qs])
            for r in T.unroll(2):
                mx[r] = lp[r * 4] + lp[r * 4 + 1] + lp[r * 4 + 2] + lp[r * 4 + 3]
                mx[r] = mx[r] + T.shfl_xor(mx[r], 1)
                logsum[r] = mx[r] + T.shfl_xor(mx[r], 2)
                if apply_sinks:
                    # One zero-value sink a head joins the row's softmax, in base-2 logits.
                    smp[r] = Sinks[head] * LOG2E
                    mx[r] = T.max(sm[r] * scale, smp[r])
                    alpha[r] = logsum[r] * T.exp2(sm[r] * scale - mx[r])
                    alpha[r] = (
                        T.if_then_else(
                            (logsum[r] > 0) & (smp[r] < T.infinity("float32")),
                            alpha[r] / (alpha[r] + T.exp2(smp[r] - mx[r])),
                            0.0,
                        )
                        / logsum[r]
                    )
                elif not split:
                    alpha[r] = 1.0 / logsum[r]
            if split:
                # A part keeps its unnormalized output in fp32 and its rows' scaled maximum and
                # sum apart, so that the merge weighs and cancels parts without rounding.
                if q0 + r0 >= 0:
                    for r in T.unroll(2):
                        row = q0 + r0 + (lane // 32) * 16 + r * 8 + (lane % 32) // 4
                        if row < seq_len_q:
                            for c in T.unroll(16):
                                for v in T.vectorized(2):
                                    O[batch, row, head, c * 8 + (lane % 4) * 2 + v] = acc_o[
                                        c * 4 + r * 2 + v
                                    ]
                            if lane % 4 == 0:
                                Lse[batch, head, 0, row] = T.if_then_else(
                                    sm[r] == -T.infinity("float32"),
                                    -T.infinity("float32"),
                                    sm[r] * scale,
                                )
                                Lse[batch, head, 1, row] = logsum[r]
            else:
                for i in T.unroll(64):
                    acc_o[i] = acc_o[i] * alpha[(i % 4) // 2]
                to_16bit(acc_o, pcast)

                # The previous tile's output store has read the staging buffer before it is
                # rewritten.
                T.tma_store_wait(0)
                T.sync_threads(3 + wg, 128)
                for i in T.unroll(8):
                    row = (lane // 32) * 16 + lane % 16
                    col = i * 16 + (lane % 32) // 16 * 8
                    T.call_intrin(
                        "handle", stmatrix, 0, 4,
                        T.access_ptr(
                            smem[0], "w",
                            offset=o_at + wg * o_group_elems + (col // panel) * q_panel_elems
                            + row * panel + (((col % panel) // 8) ^ (row % 8)) * 8,
                            extent=8,
                        ),
                        T.call_intrin("uint32", pack, pcast[i * 8], pcast[i * 8 + 1]),
                        T.call_intrin("uint32", pack, pcast[i * 8 + 2], pcast[i * 8 + 3]),
                        T.call_intrin("uint32", pack, pcast[i * 8 + 4], pcast[i * 8 + 5]),
                        T.call_intrin("uint32", pack, pcast[i * 8 + 6], pcast[i * 8 + 7]),
                        "m8n8",
                    )  # fmt: skip
                if with_lse and lane % 4 == 0:
                    for r in T.unroll(2):
                        aux[256 + wg * half + (lane // 32) * 16 + r * 8 + (lane % 32) // 4] = (
                            T.if_then_else(
                                logsum[r] > 0,
                                T.log(logsum[r]) + sm[r] * (scale / LOG2E),
                                -T.infinity("float32"),
                            )
                        )
                # The group's four warps finish their shared stores before one thread issues TMA.
                T.fence_proxy_async()
                T.sync_threads(3 + wg, 128)
                if q0 + r0 >= 0:
                    if with_lse and lane < 64 and q0 + r0 + lane < seq_len_q:
                        Lse[batch, head, q0 + r0 + lane] = aux[256 + wg * half + lane]
                    if T.shuffle_elect(128):
                        for p in T.unroll(D // panel):
                            T.call_intrin(
                                "handle", tma_store, tma_map(O, seq_len_q, H, half),
                                T.access_ptr(
                                    smem[0], "r", offset=o_at + wg * o_group_elems + p * q_panel_elems,
                                    extent=q_panel_elems,
                                ),
                                p * panel, q0 + r0, head, batch, 0, 0,
                            )  # fmt: skip
                        T.tma_store_arrive()
            g += eff
            for r in T.unroll(2):
                sm[r] = -T.infinity("float32")
            if it + 1 < tiles_of_cta:
                q0 = T.shfl_sync(meta[1 - qs, 0], 0)
                eff = T.shfl_sync(meta[1 - qs, 1], 0)
                head = T.shfl_sync(meta[1 - qs, 2], 0)
                batch = T.shfl_sync(meta[1 - qs, 3], 0)
                if split:
                    k0 = T.shfl_sync(meta[1 - qs, 4], 0)
        # A WGMMA still pending at exit makes ptxas serialize every WGMMA of the program.
        T.wait_wgmma(0)
        T.tma_store_wait(0)

    def item_coords(item, num_tiles, num_q_tiles, causal_offset):
        """Work item ``item``'s head, batch, first query row, first key tile, key steps and
        output batch: part ``item / num_tiles`` of tile ``item % num_tiles``."""
        if not split:
            by, bz, q0, eff = tile_coords(item, num_q_tiles, causal_offset)
            return by, bz, q0, 0, eff, bz
        by, bz, q0, steps = tile_coords(item % num_tiles, num_q_tiles, causal_offset)
        part = item // num_tiles
        k0 = part * steps // num_splits
        return by, bz, q0, k0, (part + 1) * steps // num_splits - k0, part * B + bz

    @T.macro
    def load_query(
        Q, smem, meta, q_bar, q_free, tx, cta, it, num_tiles, num_q_tiles, causal_offset
    ):
        """Load round ``it``'s query into slot ``it % 2`` once the tile two rounds back has
        released it, with the coordinates its consumers read. A tile's query is read once
        (evict-first); its key and value tiles serve every query head of the KV group
        (evict-last)."""
        by, bz, q0, k0, eff, out_batch = item_coords(
            tile_of(it, cta), num_tiles, num_q_tiles, causal_offset
        )
        qs = it % 2
        T.mbarrier_wait_parity(q_free[qs], ((it // 2) % 2) ^ 1)
        if tx == 256:
            meta[qs, 0] = q0
            meta[qs, 1] = eff
            meta[qs, 2] = by
            meta[qs, 3] = out_batch
            if split:
                meta[qs, 4] = k0
        if T.shuffle_elect(32):
            T.mbarrier_expect_tx(q_bar[qs], q_slot_elems * elem_bytes)
            for wg in T.unroll(2):
                for p in T.unroll(D // panel):
                    T.tma_load(
                        tma_map(Q, seq_len_q, H, half), q_bar[qs],
                        T.access_ptr(
                            smem[0], "w",
                            offset=q_at + qs * q_slot_elems + wg * q_group_elems + p * q_panel_elems,
                            extent=q_panel_elems,
                        ),
                        p * panel, q0 + wg * half, by, bz, 1,
                    )  # fmt: skip
            T.mbarrier_arrive(q_bar[qs])

    @T.macro
    def load(
        Q, X, x_at, ready, free, with_q, smem, meta, q_bar, q_free, tx, cta, tiles_of_cta,
        num_tiles, num_q_tiles, causal_offset,
    ):  # fmt: skip
        """One producer warp: each work item's key (or value) tiles through the two-stage
        ring at ``x_at``; with ``with_q`` also the queries, the next tile's with the third key
        tile, or the item's last if it has fewer: its slot's last tile has then finished (the
        first key tile's slot was freed after it), and the load does not meet every CTA's next
        keys at the tile's end."""
        if with_q:
            load_query(
                Q, smem, meta, q_bar, q_free, tx, cta, 0, num_tiles, num_q_tiles, causal_offset
            )
        g = T.alloc_var("int32", init=0)
        for it in T.serial(tiles_of_cta):
            by, bz, q0, k0, eff, out_batch = item_coords(
                tile_of(it, cta), num_tiles, num_q_tiles, causal_offset
            )
            cv = by // groups
            for k in T.serial(eff):
                sk = (g + k) % 2
                T.mbarrier_wait_parity(free[sk], (((g + k) // 2) % 2) ^ 1)
                if T.shuffle_elect(32):
                    T.mbarrier_expect_tx(ready[sk], kv_stage_elems * elem_bytes)
                    for p in T.unroll(D // panel):
                        T.tma_load(
                            tma_map(X, seq_len_kv, Hkv, block_N), ready[sk],
                            T.access_ptr(
                                smem[0], "w", offset=x_at + sk * kv_stage_elems + p * kv_panel_elems,
                                extent=kv_panel_elems,
                            ),
                            p * panel, (k0 + k) * block_N, cv, bz, 2,
                        )  # fmt: skip
                    T.mbarrier_arrive(ready[sk])
                if with_q and k == T.min(2, eff - 1) and it + 1 < tiles_of_cta:
                    load_query(
                        Q, smem, meta, q_bar, q_free, tx, cta, it + 1, num_tiles, num_q_tiles,
                        causal_offset,
                    )  # fmt: skip
            g += eff

    @T.macro
    def attention(Q, K, V, O, Sinks, Lse):
        num_q_tiles = T.ceildiv(seq_len_q, block_M)
        num_tiles = num_q_tiles * H * B
        # Work item ``i`` is part ``i / num_tiles`` of tile ``i % num_tiles``.
        num_items = num_tiles * num_splits
        with T.Kernel(T.min(num_ctas, num_items), threads=384) as cta:
            tiles_of_cta = tiles_of_cta_count(cta, num_items)
            smem = T.alloc_shared([smem_elems], dtype)
            meta = T.alloc_shared([2, 5 if split else 4], "int32")
            # Words 0-255: each consumer thread's wait anchor, never read; 256-383: the two
            # groups' log-sum-exp rows.
            aux = T.alloc_shared([2 * 128 + 2 * half], "float32")
            q_bar = T.alloc_barrier([1] * 2)
            q_free = T.alloc_barrier([256] * 2)
            kready = T.alloc_barrier([1] * 2)
            kfree = T.alloc_barrier([256] * 2)
            vready = T.alloc_barrier([1] * 2)
            vfree = T.alloc_barrier([256] * 2)
            causal_offset = seq_len_kv - seq_len_q
            tx = T.get_thread_binding()
            if tx >= 256:
                T.set_max_nreg(24, 0)
            if tx >= 256 and tx < 288:
                load(
                    Q, K, k_at, kready, kfree, True, smem, meta, q_bar, q_free, tx, cta,
                    tiles_of_cta, num_tiles, num_q_tiles, causal_offset,
                )  # fmt: skip
            # Values come from their own warp: a key load never queues behind a value-stage wait.
            if tx >= 288 and tx < 320:
                load(
                    Q, V, v_at, vready, vfree, False, smem, meta, q_bar, q_free, tx, cta,
                    tiles_of_cta, num_tiles, num_q_tiles, causal_offset,
                )  # fmt: skip
            if tx < 128:
                consume(
                    0, tx, smem, meta, aux, q_bar, q_free, kready, kfree, vready, vfree, O,
                    Sinks, Lse, tiles_of_cta, causal_offset,
                )  # fmt: skip
            if tx >= 128 and tx < 256:
                consume(
                    1, tx, smem, meta, aux, q_bar, q_free, kready, kfree, vready, vfree, O,
                    Sinks, Lse, tiles_of_cta, causal_offset,
                )  # fmt: skip

    if split:

        @T.prim_func
        def parts(
            Q: T.Tensor([B, seq_len_q, H, D], dtype),
            K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            O: T.Tensor([B * num_splits, seq_len_q, H, D], "float32"),
            Lse: T.Tensor([B * num_splits, H, 2, seq_len_q], "float32"),
        ):
            attention(Q, K, V, O, None, Lse)

        return parts

    if with_lse:

        @T.prim_func
        def main_with_lse(
            Q: T.Tensor([B, seq_len_q, H, D], dtype),
            K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            O: T.Tensor([B, seq_len_q, H, D], dtype),
            Lse: T.Tensor([B, H, seq_len_q], "float32"),
        ):
            attention(Q, K, V, O, None, Lse)

        return main_with_lse

    if has_sinks:

        @T.prim_func
        def with_sinks(
            Q: T.Tensor([B, seq_len_q, H, D], dtype),
            K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
            O: T.Tensor([B, seq_len_q, H, D], dtype),
            Sinks: T.Tensor([H], "float32"),
        ):
            attention(Q, K, V, O, Sinks, None)

        return with_sinks

    @T.prim_func
    def main(
        Q: T.Tensor([B, seq_len_q, H, D], dtype),
        K: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        V: T.Tensor([B, seq_len_kv, Hkv, D], dtype),
        O: T.Tensor([B, seq_len_q, H, D], dtype),
    ):
        attention(Q, K, V, O, None, None)

    return main


@functools.lru_cache(maxsize=32)
@tilelang.jit(pass_configs=_PASS_CONFIGS, compile_flags=_COMPILE_FLAGS)
def _gqa_dense_split_combine_kernel(B, H, D, num_splits, dtype, has_sinks=False, with_lse=False):
    """Build the merge of the parts :func:`_gqa_dense_persistent_d128_kernel` writes with
    ``num_splits`` S. Part ``s`` holds a row's unnormalized output ``o_s``, its scaled
    base-2 maximum ``m_s`` and its sum ``l_s``; with ``M`` the largest maximum (the sink's
    included), the row's output is ``sum_s 2^(m_s - M) o_s`` over
    ``sum_s 2^(m_s - M) l_s + 2^(sink log2 e - M)``. With ``with_lse`` it also writes the
    merged natural-log log-sum-exp. Sixteen threads take a row, eight columns each.
    """
    seq_len_q = T.dynamic("seq_len_q")
    vec = 8
    lanes = D // vec
    rows = 128 // lanes

    @T.macro
    def merge(Op, Lp, O, Sinks, Lse):
        with T.Kernel(T.ceildiv(seq_len_q, rows), H, B, threads=128) as (bx, by, bz):
            tx = T.get_thread_binding()
            row = bx * rows + tx // lanes
            col = tx % lanes * vec
            top_s = T.alloc_local([num_splits], "float32")
            part = T.alloc_local([vec], "float32")
            acc = T.alloc_local([vec], "float32")
            top = T.alloc_var("float32", init=-T.infinity("float32"))
            total = T.alloc_var("float32", init=0.0)
            if row < seq_len_q:
                for s in T.unroll(num_splits):
                    top_s[s] = Lp[s * B + bz, by, 0, row]
                    top = T.max(top, top_s[s])
                if has_sinks:
                    top = T.max(top, Sinks[by] * LOG2E)
                # A row no part saw a key for keeps a -inf maximum; its weights are then 0.
                top = T.if_then_else(top == -T.infinity("float32"), 0.0, top)
                for i in T.unroll(vec):
                    acc[i] = 0.0
                for s in T.unroll(num_splits):
                    # Maxima and sums are kept apart, so the weights lose nothing at any logit.
                    top_s[s] = T.exp2(top_s[s] - top)
                    total = total + top_s[s] * Lp[s * B + bz, by, 1, row]
                    for i in T.vectorized(vec):
                        part[i] = Op[s * B + bz, row, by, col + i]
                    for i in T.unroll(vec):
                        acc[i] = acc[i] + top_s[s] * part[i]
                if has_sinks:
                    total = total + T.exp2(Sinks[by] * LOG2E - top)
                # An infinite sink takes the whole row: the total is then NaN and O zero.
                total = T.if_then_else(total > 0, total, T.infinity("float32"))
                for i in T.vectorized(vec):
                    O[bz, row, by, col + i] = T.cast(acc[i] / total, dtype)
                if with_lse and tx % lanes == 0:
                    Lse[bz, by, row] = T.log(total) + top * (1.0 / LOG2E)

    if has_sinks:

        @T.prim_func
        def merge_with_sinks(
            Op: T.Tensor([B * num_splits, seq_len_q, H, D], "float32"),
            Lp: T.Tensor([B * num_splits, H, 2, seq_len_q], "float32"),
            Sinks: T.Tensor([H], "float32"),
            O: T.Tensor([B, seq_len_q, H, D], dtype),
        ):
            merge(Op, Lp, O, Sinks, None)

        return merge_with_sinks

    if with_lse:

        @T.prim_func
        def merge_with_lse(
            Op: T.Tensor([B * num_splits, seq_len_q, H, D], "float32"),
            Lp: T.Tensor([B * num_splits, H, 2, seq_len_q], "float32"),
            O: T.Tensor([B, seq_len_q, H, D], dtype),
            Lse: T.Tensor([B, H, seq_len_q], "float32"),
        ):
            merge(Op, Lp, O, None, Lse)

        return merge_with_lse

    @T.prim_func
    def main(
        Op: T.Tensor([B * num_splits, seq_len_q, H, D], "float32"),
        Lp: T.Tensor([B * num_splits, H, 2, seq_len_q], "float32"),
        O: T.Tensor([B, seq_len_q, H, D], dtype),
    ):
        merge(Op, Lp, O, None, None)

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
        coordinates, the value-product wait anchor (at D128 with each row's log-sum-exp beside
        it) and, with three consumers, the ones operand, each placed on a kilobyte."""
        block_m = cls._block_m(dim)
        consumers = block_m // 64
        buffers = [
            2 * block_m * dim * 2,  # query slots
            2 * 128 * dim * 2,  # key stages
            2 * 128 * dim * 2,  # value stages
            block_m * dim * 2,  # staged output
            2 * 4 * 4,  # tile coordinates
            (consumers * 128 + (block_m if dim == 128 else 0)) * 4,  # wait anchor, LSE rows
        ]
        if consumers == 3:
            buffers.append(8 * 128 * 2)  # ones operand
        return sum(-(-b // 1024) * 1024 for b in buffers)

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """As the WS kernel, for head dimensions whose buffers fit the device; only the D128
        program writes the log-sum-exp."""
        if call.return_lse and call.dim != 128:
            return "returns the log-sum-exp only at head dimension 128"
        reason = super().refusal(call)
        if reason is not None:
            return reason
        index = call.device.index if call.device is not None else None
        if cls.shared_bytes(call.dim) > get_shared_memory_optin(index):
            return "the query slots and key/value stages exceed shared memory"
        return None

    @classmethod
    def _entry_args(cls, call: AttentionCall) -> dict:
        """At D128, a ``rank_group`` of 16 runs a head's query blocks on neighbouring tiles when
        the keys and values of every head exceed half of L2, which a heads-first round reads
        once and the next round again; ``num_splits`` comes from :meth:`_num_splits`; a causal
        call of one part with both lengths on 128-row boundaries is ``diag_aligned``."""
        args = super()._entry_args(call)
        if call.dim != 128:
            return args
        index = call.device.index if call.device is not None else None
        device = torch.cuda.current_device() if index is None else index
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        kv_bytes = call.batch * call.heads_kv * call.seqlen_kv * call.dim * 2 * 2
        splits = cls._num_splits(call)
        return {
            **args,
            "rank_group": 16 if kv_bytes > l2_bytes // 2 else 1,
            "num_splits": splits,
            "diag_aligned": splits == 1
            and call.is_causal
            and call.max_seqlen_q % 128 == 0
            and (call.seqlen_kv - call.max_seqlen_q) % 128 == 0,
        }

    @staticmethod
    def _num_splits(call: AttentionCall) -> int:
        """Key parts a 128-row tile divides into, so that a call of few tiles fills the SMs:
        ``sm_count // tiles``, at most the key steps of the shortest tile so that every part
        has one; one unless the parts take at least two key steps off the longest tile, as the
        merge costs about one."""
        q_tiles = -(-call.max_seqlen_q // 128)
        kv_steps = -(-call.seqlen_kv // 128)
        fewest = most = kv_steps
        if call.is_causal:
            # Tiles align to the end of Q in 64-row groups; the first is the shortest.
            first_q0 = -(-call.max_seqlen_q // 64) * 64 - q_tiles * 128
            offset = call.seqlen_kv - call.max_seqlen_q
            fewest = min(kv_steps, -(-(first_q0 + 128 + offset) // 128))
        splits = min(call.sm_count // (q_tiles * call.heads * call.batch), fewest)
        if splits < 2 or most - -(-most // splits) < 2:
            return 1
        return splits

    def __init__(
        self, *args, rank_group: int = 1, num_splits: int = 1, diag_aligned: bool = False, **kwargs
    ) -> None:
        self.rank_group = rank_group
        self.num_splits = num_splits
        self.diag_aligned = diag_aligned
        super().__init__(*args, **kwargs)

    def _program_args(self, dim: int, device_index: Optional[int]) -> dict:
        return {"block_M": self._block_m(dim), "num_ctas": get_sm_count(device_index)}

    def _compile(self, cfg: dict) -> tuple:
        """Head dimension 128 runs the per-thread program; the others, the tile-level one."""
        if self._shape_args[3] != 128:
            return super()._compile(cfg)
        self.kernel = kernel = _gqa_dense_persistent_d128_kernel(
            *self._shape_args,
            num_ctas=self._extra_args["num_ctas"],
            rank_group=self.rank_group,
            has_sinks=self.has_sinks,
            with_lse=self.return_lse,
            num_splits=self.num_splits,
            diag_aligned=self.diag_aligned,
        )
        rope = self.rope.build() if self.rope is not None else None
        return CompiledProgram(kernel), rope

    def _compile_merge(self, cfg: dict) -> Callable:
        """The merge of the key parts."""
        batch, heads, _, dim = self._shape_args[:4]
        return _gqa_dense_split_combine_kernel(
            batch, heads, dim, self.num_splits, self.dtype_str, self.has_sinks, self.return_lse
        )

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
        """As the WS kernel; with key parts, the parts' outputs are merged into the result."""
        if self.num_splits == 1:
            return super().forward(q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin, sinks)
        self._require_devices(q=q, k=k, v=v)
        attention, rope = self.program(self._compile)
        merge = self.program(self._compile_merge)
        if rope is not None:
            q, k = rope(q, k, rope_cos, rope_sin)
        batch, seq_len_q, heads, _ = q.shape
        part_stats = torch.empty(
            (self.num_splits * batch, heads, 2, seq_len_q), dtype=torch.float32, device=q.device
        )
        parts = attention(q, k, v, part_stats)
        out = torch.empty_like(q)
        if self.has_sinks:
            merge(parts, part_stats, sinks, out)
            return out
        if self.return_lse:
            lse = torch.empty((batch, heads, seq_len_q), dtype=torch.float32, device=q.device)
            merge(parts, part_stats, out, lse)
            return out, lse
        merge(parts, part_stats, out)
        return out


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
        if call.return_lse:
            return "does not return the log-sum-exp"
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
        attention, rope = self.program(self._compile)
        if rope is not None:
            q, k = rope(q, k, rope_cos, rope_sin)
        output, _ = attention(q, k, v, sinks if self.has_sinks else q)
        return output

    def _compile(self, cfg: dict) -> tuple:
        """The attention program for *cfg*, and the Q/K rotation ahead of it or ``None``."""
        attention = self.kernel(cfg["block_m"], cfg["block_n"], cfg["num_stages"], cfg["threads"])
        rope = self.rope.build() if self.rope is not None else None
        return CompiledProgram(attention), rope


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
        if call.return_lse:
            return "does not return the log-sum-exp"
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
        attention, rope = self.program(self._compile)
        if rope is not None:
            q, k = rope(q, k, rope_cos, rope_sin)
        output, _ = attention(q, k, v, sinks if self.has_sinks else q)
        return output

    def _compile(self, cfg: dict) -> tuple:
        """The attention program for *cfg*, and the Q/K rotation ahead of it or ``None``."""
        rope = self.rope.build() if self.rope is not None else None
        return CompiledProgram(self.kernel(cfg["num_stages"])), rope
