"""SM90 warp-specialized DSA decode: a gathering producer and two seesaw consumers."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops.kernels.attention.call_spec import DSADecodeCall
from tileops.kernels.attention.dsa.decode import DSADecodeKernelBase
from tileops.kernels.constants import LOG2E

__all__ = ["DSADecodeWSKernel"]

_HEADS = 64  # query heads per CTA: one m64 WGMMA row block
_KEYS = 64  # selected keys per block; blocks are taken in pairs
_HALF = 256  # value dims each consumer accumulates
_TILE = 64  # columns one 128-byte swizzle atom holds
_CONSUMERS = 256  # two consumer warpgroups; the producer warpgroup follows them
_MAX_INIT = -1.0e30  # a finite start, so a fully masked block weighs 0 instead of NaN
_MASKED = 1.0e38  # a dead slot's bias is -_MASKED: scaled, it falls far below _MAX_INIT

# A pair's shared-memory parts, each with its own ready and free barrier: K0L / K0R hold
# value dims [0, 256) / [256, 512) (and the key tail) of the even block, K1L / K1R of the
# odd block. Each consumer scores both parts of its own block; consumer 0 sums over the
# L parts, consumer 1 over the R parts.
_K0L, _K0R, _K1L, _K1R = 0, 1, 2, 3


@functools.lru_cache(maxsize=32)
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
def _dsa_decode_ws_kernel(
    batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride, q_start, kv_group,
    sm_scale, dtype,
):  # fmt: skip
    """One CTA per (token, 64-head block); the selected keys are taken in block pairs.

    Consumer 0 scores the even block of a pair and consumer 1 the odd one. Each turns its
    scores into weights, keeps them in registers for its own half of the value dims and
    hands them to the other through smem. The row max is one chain through both blocks:
    consumer 1 continues from consumer 0's max, and consumer 0 rescales its weights to the
    pair's final max before handing them over.
    """
    dq = dim + tail_dim
    group_heads = heads // kv_group
    head_blocks = group_heads // _HEADS
    pairs = topk // (2 * _KEYS)
    scale = (dq**-0.5 if sm_scale is None else sm_scale) * LOG2E
    accum = "float"
    has_tail = tail_dim > 0

    @T.macro
    def copy_cols(dst, row, src, i0, i1, i2, col0, ntiles, lane8, live):
        """Columns [col0, col0 + 64 * ntiles) of one source row into `dst` row `row`,
        16 bytes a lane. A dead row is zero-filled without a read: every dead slot names
        row 0, and reading it would send them all to the same L2 lines."""
        for t in T.unroll(ntiles):
            T.ptx_cp_async(
                T.access_ptr(dst[row, t * _TILE + lane8 * 8], "w", 8),
                T.access_ptr(src[i0, i1, i2, col0 + t * _TILE + lane8 * 8], "r", 8),
                8,
                live,
            )

    @T.macro
    def load_left(dst, src_rows, live_rows, half, KV, b, g, group, lane8, k_ready, part):
        for r in T.unroll(4):
            copy_cols(
                dst, 16 * r + group, KV, b, src_rows[half, r], g, 0, 4, lane8, live_rows[half, r]
            )
        T.cp_async_barrier_noinc(k_ready[part])

    @T.macro
    def load_right(dst, dst_tail, src_rows, live_rows, half, KV, b, g, group, lane8, k_ready, part):
        for r in T.unroll(4):
            row = 16 * r + group
            copy_cols(dst, row, KV, b, src_rows[half, r], g, _HALF, 4, lane8, live_rows[half, r])
            if has_tail:
                copy_cols(
                    dst_tail, row, KV, b, src_rows[half, r], g, dim, 1, lane8, live_rows[half, r]
                )
        T.cp_async_barrier_noinc(k_ready[part])

    @T.macro
    def producer(Q, KV, Indices, QL, QR, QT, K0L, K0R, K0T, K1L, K1R, K1T, Bias, q_full, k_ready, k_free, bias_ready, b, s, g, h0, p):  # fmt: skip
        """Eight lanes per 128-byte tile row: thread p copies 16 bytes of each tile of rows
        16 * r + p // 8, r in [0, 4), so a warp request covers whole sectors."""
        group = p // 8
        lane8 = p % 8
        for r in T.unroll(4):
            row = 16 * r + group
            copy_cols(QL, row, Q, b, s, h0 + row, 0, 4, lane8, True)
            copy_cols(QR, row, Q, b, s, h0 + row, _HALF, 4, lane8, True)
            if has_tail:
                copy_cols(QT, row, Q, b, s, h0 + row, dim, 1, lane8, True)
        T.cp_async_barrier_noinc(q_full[0])

        src_rows = T.alloc_local([2, 4], "int32")
        live_rows = T.alloc_local([2, 4], "bool")
        limit = q_start + s - kv_stride + 1
        for pair in T.serial(pairs):
            free_phase = (pair % 2) ^ 1
            # A dead row names row 0 and is zero-filled; its bias makes it weigh nothing.
            for half in T.unroll(2):
                for r in T.unroll(4):
                    idx = Indices[b, s, g, (2 * pair + half) * _KEYS + 16 * r + group]
                    live_rows[half, r] = idx >= 0 and idx < seq_len_kv and idx * kv_stride <= limit
                    src_rows[half, r] = idx * T.Cast("int32", live_rows[half, r])
            # Refill each part in the order its last reader releases it.
            T.mbarrier_wait_parity(k_free[_K0L], free_phase)
            load_left(K0L, src_rows, live_rows, 0, KV, b, g, group, lane8, k_ready, _K0L)
            T.mbarrier_wait_parity(k_free[_K1R], free_phase)
            load_right(K1R, K1T, src_rows, live_rows, 1, KV, b, g, group, lane8, k_ready, _K1R)
            T.mbarrier_wait_parity(k_free[_K0R], free_phase)
            load_right(K0R, K0T, src_rows, live_rows, 0, KV, b, g, group, lane8, k_ready, _K0R)
            T.mbarrier_wait_parity(k_free[_K1L], free_phase)
            load_left(K1L, src_rows, live_rows, 1, KV, b, g, group, lane8, k_ready, _K1L)
            # Every reader of the last pair's mask has released K1L by now. One index per
            # thread: thread p masks slot p of the pair.
            idx = Indices[b, s, g, 2 * pair * _KEYS + p]
            live = idx >= 0 and idx < seq_len_kv and idx * kv_stride <= limit
            Bias[p // _KEYS, p % _KEYS] = (T.Cast(accum, live) - 1.0) * _MASKED
            T.mbarrier_arrive(bias_ready[0])

    @T.macro
    def score_left(QL, KL, acc, clear):
        T.wgmma_gemm(QL, KL, acc, transpose_B=True, clear_accum=clear)

    @T.macro
    def score_right(QR, QT, KR, KT, acc, clear):
        T.wgmma_gemm(QR, KR, acc, transpose_B=True, clear_accum=clear)
        if has_tail:
            T.wgmma_gemm(QT, KT, acc, transpose_B=True)

    @T.macro
    def softmax(rP, rS, rO, rM, rL, prev, cur, keep, Bias, half, sM):
        """Mask, continue the max chain from `prev`, rescale rO and rL by the step, and
        turn rP into weights (f32 in rP, bf16 in rS). Publish the new max in sM."""
        for i, j in T.Parallel(_HEADS, _KEYS):
            rP[i, j] = rP[i, j] + Bias[half, j]
        T.reduce_max(rP, cur, dim=1, clear=True)
        for i in T.Parallel(_HEADS):
            cur[i] = T.max(prev[i], cur[i] * scale)
            keep[i] = T.exp2(rM[i] - cur[i])
            rM[i] = cur[i]
        for i, j in T.Parallel(_HEADS, _HALF):
            rO[i, j] *= keep[i]
        for i, j in T.Parallel(_HEADS, _KEYS):
            rP[i, j] = T.exp2(rP[i, j] * scale - cur[i])
            rS[i, j] = rP[i, j]
        T.reduce_sum(rP, cur, dim=1)
        for i in T.Parallel(_HEADS):
            rL[i] = rL[i] * keep[i] + cur[i]
            sM[i] = rM[i]

    @T.macro
    def finish(w, rO, rL, sL, out, staged, O, b, s, h0):
        """Add the two consumers' row sums, stage this consumer's value dims in its Q
        tile, which every score has finished reading, and store them with TMA."""
        for i in T.Parallel(_HEADS):
            sL[w, i] = rL[i]
        T.sync_threads(5, _CONSUMERS)
        # A row with a live key sums to at least 1 (its max weighs 1); a row with none
        # has rO = 0.
        for i in T.Parallel(_HEADS):
            rL[i] = 1 / T.max(sL[0, i] + sL[1, i], 1.0)
        for i, j in T.Parallel(_HEADS, _HALF):
            out[i, j] = rO[i, j] * rL[i]
        T.copy(out, staged)
        T.fence_proxy_async()
        T.sync_threads(6 + w, 128)
        T.copy(staged, O[b, s, h0 : h0 + _HEADS, w * _HALF : (w + 1) * _HALF])

    @T.macro
    def consumer0(O, QL, QR, QT, K0L, K0R, K0T, K1L, S0, S1, Bias, sM, sL, q_full, k_ready, k_free, bias_ready, b, s, h0):  # fmt: skip
        """Scores the even blocks; accumulates value dims [0, 256)."""
        T.set_max_nreg(216, 1)
        rP = T.alloc_fragment([_HEADS, _KEYS], accum)
        rS = T.alloc_fragment([_HEADS, _KEYS], dtype)
        rO = T.alloc_fragment([_HEADS, _HALF], accum)
        out = T.alloc_fragment([_HEADS, _HALF], dtype)
        rM = T.alloc_fragment([_HEADS], accum)
        rL = T.alloc_fragment([_HEADS], accum)
        cur = T.alloc_fragment([_HEADS], accum)
        keep = T.alloc_fragment([_HEADS], accum)
        T.fill(rO, 0)
        T.fill(rL, 0)
        T.fill(rM, _MAX_INIT)
        T.mbarrier_wait_parity(q_full[0], 0)
        T.mbarrier_wait_parity(k_ready[_K0L], 0)
        score_left(QL, K0L, rP, True)
        T.mbarrier_wait_parity(k_ready[_K0R], 0)
        score_right(QR, QT, K0R, K0T, rP, False)
        T.wait_wgmma(0)
        for pair in T.serial(pairs):
            phase = pair % 2
            T.mbarrier_wait_parity(bias_ready[0], phase)
            softmax(rP, rS, rO, rM, rL, rM, cur, keep, Bias, 0, sM)
            T.named_barrier_arrive(1, _CONSUMERS)
            T.wgmma_gemm(rS, K0L, rO)
            T.wait_wgmma(0)
            T.mbarrier_arrive(k_free[_K0L])
            # Consumer 1 continued the chain: bring this block's weights to its max.
            T.sync_threads(2, _CONSUMERS)
            for i in T.Parallel(_HEADS):
                keep[i] = T.exp2(rM[i] - sM[i])
                rM[i] = sM[i]
                rL[i] *= keep[i]
            for i, j in T.Parallel(_HEADS, _KEYS):
                rS[i, j] = rP[i, j] * keep[i]
            T.copy(rS, S0)
            T.fence_proxy_async()
            T.named_barrier_arrive(3, _CONSUMERS)
            T.sync_threads(4, _CONSUMERS)
            for i, j in T.Parallel(_HEADS, _HALF):
                rO[i, j] *= keep[i]
            T.wgmma_gemm(S1, K1L, rO)
            if pair + 1 < pairs:
                T.mbarrier_wait_parity(k_ready[_K0L], phase ^ 1)
                score_left(QL, K0L, rP, True)
                T.wait_wgmma(1)
                T.mbarrier_arrive(k_free[_K1L])
                T.mbarrier_wait_parity(k_ready[_K0R], phase ^ 1)
                score_right(QR, QT, K0R, K0T, rP, False)
                T.wait_wgmma(0)
            else:
                T.wait_wgmma(0)
                T.mbarrier_arrive(k_free[_K1L])
        finish(0, rO, rL, sL, out, QL, O, b, s, h0)

    @T.macro
    def consumer1(O, QL, QR, QT, K0R, K1L, K1R, K1T, S0, S1, Bias, sM, sL, q_full, k_ready, k_free, bias_ready, b, s, h0):  # fmt: skip
        """Scores the odd blocks; accumulates value dims [256, 512)."""
        T.set_max_nreg(216, 1)
        rP = T.alloc_fragment([_HEADS, _KEYS], accum)
        rS = T.alloc_fragment([_HEADS, _KEYS], dtype)
        rO = T.alloc_fragment([_HEADS, _HALF], accum)
        out = T.alloc_fragment([_HEADS, _HALF], dtype)
        rM = T.alloc_fragment([_HEADS], accum)
        rL = T.alloc_fragment([_HEADS], accum)
        prev = T.alloc_fragment([_HEADS], accum)
        cur = T.alloc_fragment([_HEADS], accum)
        keep = T.alloc_fragment([_HEADS], accum)
        T.fill(rO, 0)
        T.fill(rL, 0)
        T.fill(rM, _MAX_INIT)
        T.mbarrier_wait_parity(q_full[0], 0)
        for pair in T.serial(pairs):
            phase = pair % 2
            T.mbarrier_wait_parity(k_ready[_K1R], phase)
            score_right(QR, QT, K1R, K1T, rP, True)
            T.mbarrier_wait_parity(k_ready[_K1L], phase)
            score_left(QL, K1L, rP, False)
            T.wait_wgmma(0)
            T.mbarrier_wait_parity(bias_ready[0], phase)
            # Consumer 0 has scored the even block: continue its max.
            T.sync_threads(1, _CONSUMERS)
            for i in T.Parallel(_HEADS):
                prev[i] = sM[i]
            softmax(rP, rS, rO, rM, rL, prev, cur, keep, Bias, 1, sM)
            T.named_barrier_arrive(2, _CONSUMERS)
            T.wgmma_gemm(rS, K1R, rO)
            T.copy(rS, S1)
            T.sync_threads(3, _CONSUMERS)
            T.wgmma_gemm(S0, K0R, rO)
            T.fence_proxy_async()
            T.named_barrier_arrive(4, _CONSUMERS)
            T.wait_wgmma(1)
            T.mbarrier_arrive(k_free[_K1R])
            T.wait_wgmma(0)
            T.mbarrier_arrive(k_free[_K0R])
        finish(1, rO, rL, sL, out, QR, O, b, s, h0)

    @T.prim_func
    def main(
        Q: T.Tensor([batch, seq_len, heads, dq], dtype),
        KV: T.Tensor([batch, seq_len_kv, kv_group, dq], dtype),
        Indices: T.Tensor([batch, seq_len, kv_group, topk], "int32"),
        O: T.Tensor([batch, seq_len, heads, dim], dtype),
    ):
        with T.Kernel(seq_len * head_blocks, kv_group, batch, threads=_CONSUMERS + 128) as (
            bx,
            g,
            b,
        ):
            s = bx // head_blocks
            h0 = g * group_heads + (bx % head_blocks) * _HEADS
            QL = T.alloc_shared([_HEADS, _HALF], dtype)
            QR = T.alloc_shared([_HEADS, _HALF], dtype)
            K0L = T.alloc_shared([_KEYS, _HALF], dtype)
            K0R = T.alloc_shared([_KEYS, _HALF], dtype)
            K1L = T.alloc_shared([_KEYS, _HALF], dtype)
            K1R = T.alloc_shared([_KEYS, _HALF], dtype)
            S1 = T.alloc_shared([_HEADS, _KEYS], dtype)
            parts = [QL, QR, K0L, K0R, K1L, K1R, S1]
            if has_tail:
                QT = T.alloc_shared([_HEADS, tail_dim], dtype)
                K0T = T.alloc_shared([_KEYS, tail_dim], dtype)
                K1T = T.alloc_shared([_KEYS, tail_dim], dtype)
                parts += [QT, K0T, K1T]
                # The even block's weights take its key tail's place once it is scored.
                S0 = K0T
            else:
                QT, K0T, K1T = QR, K0R, K1R
                S0 = T.alloc_shared([_HEADS, _KEYS], dtype)
                parts.append(S0)
            T.annotate_layout({buf: make_swizzled_layout(buf) for buf in parts})
            Bias = T.alloc_shared([2, _KEYS], accum)
            sM = T.alloc_shared([_HEADS], accum)
            sL = T.alloc_shared([2, _HEADS], accum)
            q_full = T.alloc_barrier([128])
            k_ready = T.alloc_barrier([128] * 4)
            k_free = T.alloc_barrier([128] * 4)
            bias_ready = T.alloc_barrier([128])
            tx = T.get_thread_binding()
            if tx >= _CONSUMERS:
                T.set_max_nreg(72, 0)
                producer(
                    Q, KV, Indices, QL, QR, QT, K0L, K0R, K0T, K1L, K1R, K1T, Bias, q_full,
                    k_ready, k_free, bias_ready, b, s, g, h0, tx - _CONSUMERS,
                )  # fmt: skip
            with T.ws(0):
                consumer0(
                    O, QL, QR, QT, K0L, K0R, K0T, K1L, S0, S1, Bias, sM, sL, q_full, k_ready,
                    k_free, bias_ready, b, s, h0,
                )  # fmt: skip
            with T.ws(1):
                consumer1(
                    O, QL, QR, QT, K0R, K1L, K1R, K1T, S0, S1, Bias, sM, sL, q_full, k_ready,
                    k_free, bias_ready, b, s, h0,
                )  # fmt: skip

    return main


class DSADecodeWSKernel(DSADecodeKernelBase):
    """SM90 sparse MLA decode: value dim 512, a key tail of 0 or 64, 64-head blocks."""

    supported_archs: list[int] = [90]
    # Both consumers score and both accumulate, where the older warp-specialized kernel
    # leaves the scores, the softmax and half the sums to one of them.
    preferred_over: ClassVar[frozenset[str]] = frozenset({"dsa_decode_kernel"})

    @classmethod
    def applies(cls, call: DSADecodeCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: DSADecodeCall) -> Optional[str]:
        if not call.is_causal:
            return "requires the causal mask"
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16"
        if (call.heads // call.kv_group) % _HEADS != 0:
            return f"requires a multiple of {_HEADS} heads per KV group"
        if call.topk % (2 * _KEYS) != 0:
            return f"requires topk a multiple of {2 * _KEYS}"
        if call.dim != 2 * _HALF or call.tail_dim not in (0, _TILE):
            return f"requires dim {2 * _HALF} and tail_dim 0 or {_TILE}"
        return None

    def __init__(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        heads: int,
        dim: int,
        tail_dim: int,
        dtype: torch.dtype,
        topk: int,
        kv_stride: int,
        q_start_index_s: int,
        kv_group: int = 1,
        sm_scale: float = None,
        is_causal: bool = True,
        cp0: bool = True,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.dim = dim
        self.tail_dim = tail_dim
        self.dtype = dtype
        self.topk = topk
        self.kv_stride = kv_stride
        self.q_start_index_s = q_start_index_s
        self.kv_group = kv_group
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.kernel = _dsa_decode_ws_kernel(
            batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride,
            q_start_index_s, kv_group, sm_scale, self.dtype_str,
        )  # fmt: skip

    @property
    def default_config(self) -> dict:
        return {}

    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        self._require_cuda(q=q, kv=kv, indices=indices)
        out = torch.empty(
            (self.batch, self.seq_len, self.heads, self.dim), dtype=q.dtype, device=q.device
        )
        self.kernel(q, kv, indices, out)
        return out
