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
    # The tiles below are fixed; DSADecodeWSKernel.refusal states the shapes they cover.
    block_h = 64  # query heads per CTA: one m64 WGMMA row block
    block_k = 64  # selected keys per block; blocks are taken in pairs
    half = 256  # value dims each consumer accumulates
    atom = 64  # columns one 128-byte swizzle atom holds
    tail_atoms = 1  # the key tail, when present, is one atom
    vec = 8  # elements one 16-byte cp.async moves
    lanes = atom // vec  # producer lanes per tile row
    warpgroup = 128
    consumer_threads = 2 * warpgroup  # tx < 256; the producer warpgroup follows
    producer_threads = warpgroup
    rows_per_pass = producer_threads // lanes
    # 2 x 128 x 216 + 128 x 72 = 64512 of the 65536 registers a CTA may hold.
    consumer_regs = 216
    producer_regs = 72
    max_init = -1.0e30  # a finite start, so a fully masked block weighs 0 instead of NaN
    masked = 1.0e38  # a dead slot's bias is -masked: scaled, it falls far below max_init

    # A pair's shared-memory parts, each with its own ready and free barrier: K0L / K0R hold
    # value dims [0, 256) / [256, 512) (and the key tail) of the even block, K1L / K1R of the
    # odd block. Each consumer scores both parts of its own block; consumer 0 sums over the
    # L parts, consumer 1 over the R parts.
    part_0l, part_0r, part_1l, part_1r = 0, 1, 2, 3
    # Named barriers the consumer warpgroups sync on; id 0 is __syncthreads.
    even_max_ready = 1  # consumer 0 published the even block's max in sM
    pair_max_ready = 2  # consumer 1 continued it to the pair's max
    s0_ready = 3  # consumer 0 wrote the even block's weights to S0
    s1_ready = 4  # consumer 1 wrote the odd block's weights to S1
    sums_ready = 5  # both consumers' row sums are in sL
    staged_ready = 6  # + w: consumer w staged its output half

    dq = dim + tail_dim
    group_heads = heads // kv_group
    head_blocks = group_heads // block_h
    pairs = topk // (2 * block_k)
    scale = (dq**-0.5 if sm_scale is None else sm_scale) * LOG2E
    accum = "float"
    has_tail = tail_dim > 0

    @T.macro
    def copy_cols(dst, row, src, i0, i1, i2, col0, ntiles, lane, live):
        """Copy columns [col0, col0 + atom * ntiles) of one source row into `dst` row `row`.

        Each lane moves one vector of each atom. A dead row is zero-filled without a read:
        every dead slot names row 0, and reading it would send them all to the same L2 lines.
        """
        for t in T.unroll(ntiles):
            T.ptx_cp_async(
                T.access_ptr(dst[row, t * atom + lane * vec], "w", vec),
                T.access_ptr(src[i0, i1, i2, col0 + t * atom + lane * vec], "r", vec),
                vec,
                live,
            )

    @T.macro
    def load_part(dst, dst_tail, with_tail, col0, src_rows, live_rows, blk, KV, b, g, group, lane, k_ready, part):  # fmt: skip
        """Gather block `blk`'s value dims from `col0`, and its key tail, then mark `part`."""
        for r in T.unroll(block_k // rows_per_pass):
            row = rows_per_pass * r + group
            copy_cols(
                dst, row, KV, b, src_rows[blk, r], g, col0, half // atom, lane, live_rows[blk, r]
            )
            if with_tail and has_tail:
                copy_cols(
                    dst_tail, row, KV, b, src_rows[blk, r], g, dim, tail_atoms, lane,
                    live_rows[blk, r],
                )  # fmt: skip
        T.cp_async_barrier_noinc(k_ready[part])

    @T.macro
    def producer(Q, KV, Indices, QL, QR, QT, K0L, K0R, K0T, K1L, K1R, K1T, Bias, q_full, k_ready, k_free, bias_ready, b, s, g, h0, p):  # fmt: skip
        """Load Q once, then gather each pair's parts and mask.

        `lanes` lanes per 128-byte tile row: thread p copies one vector of each atom of rows
        rows_per_pass * r + p // lanes, so a warp request covers whole sectors.
        """
        group = p // lanes
        lane = p % lanes
        for r in T.unroll(block_h // rows_per_pass):
            row = rows_per_pass * r + group
            copy_cols(QL, row, Q, b, s, h0 + row, 0, half // atom, lane, True)
            copy_cols(QR, row, Q, b, s, h0 + row, half, half // atom, lane, True)
            if has_tail:
                copy_cols(QT, row, Q, b, s, h0 + row, dim, tail_atoms, lane, True)
        T.cp_async_barrier_noinc(q_full[0])

        src_rows = T.alloc_local([2, block_k // rows_per_pass], "int32")
        live_rows = T.alloc_local([2, block_k // rows_per_pass], "bool")
        limit = q_start + s - kv_stride + 1
        for pair in T.serial(pairs):
            free_phase = (pair % 2) ^ 1
            # A dead row names row 0 and is zero-filled; its bias makes it weigh nothing.
            for blk in T.unroll(2):
                for r in T.unroll(block_k // rows_per_pass):
                    idx = Indices[b, s, g, (2 * pair + blk) * block_k + rows_per_pass * r + group]
                    live_rows[blk, r] = idx >= 0 and idx < seq_len_kv and idx * kv_stride <= limit
                    src_rows[blk, r] = idx * T.Cast("int32", live_rows[blk, r])
            # Refill each part in the order its last reader releases it.
            T.mbarrier_wait_parity(k_free[part_0l], free_phase)
            load_part(
                K0L, K0T, False, 0, src_rows, live_rows, 0, KV, b, g, group, lane, k_ready, part_0l
            )
            T.mbarrier_wait_parity(k_free[part_1r], free_phase)
            load_part(
                K1R, K1T, True, half, src_rows, live_rows, 1, KV, b, g, group, lane, k_ready,
                part_1r,
            )  # fmt: skip
            T.mbarrier_wait_parity(k_free[part_0r], free_phase)
            load_part(
                K0R, K0T, True, half, src_rows, live_rows, 0, KV, b, g, group, lane, k_ready,
                part_0r,
            )  # fmt: skip
            T.mbarrier_wait_parity(k_free[part_1l], free_phase)
            load_part(
                K1L, K1T, False, 0, src_rows, live_rows, 1, KV, b, g, group, lane, k_ready, part_1l
            )
            # Every reader of the last pair's mask has released K1L by now. One index per
            # thread: the producer's threads number the pair's 2 * block_k slots.
            idx = Indices[b, s, g, 2 * pair * block_k + p]
            live = idx >= 0 and idx < seq_len_kv and idx * kv_stride <= limit
            Bias[p // block_k, p % block_k] = (T.Cast(accum, live) - 1.0) * masked
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
    def softmax(rP, rS, rO, rM, rL, prev, cur, keep, Bias, blk, sM):
        """Turn block `blk`'s scores in rP into weights, continuing the max chain from `prev`.

        Masks, rescales rO and rL by the step, leaves the weights in rP (f32) and rS (storage
        dtype), and publishes the new max in sM.
        """
        for i, j in T.Parallel(block_h, block_k):
            rP[i, j] = rP[i, j] + Bias[blk, j]
        T.reduce_max(rP, cur, dim=1, clear=True)
        for i in T.Parallel(block_h):
            cur[i] = T.max(prev[i], cur[i] * scale)
            keep[i] = T.exp2(rM[i] - cur[i])
            rM[i] = cur[i]
        for i, j in T.Parallel(block_h, half):
            rO[i, j] *= keep[i]
        for i, j in T.Parallel(block_h, block_k):
            rP[i, j] = T.exp2(rP[i, j] * scale - cur[i])
            rS[i, j] = rP[i, j]
        T.reduce_sum(rP, cur, dim=1)
        for i in T.Parallel(block_h):
            rL[i] = rL[i] * keep[i] + cur[i]
            sM[i] = rM[i]

    @T.macro
    def finish(w, rO, rL, sL, out, staged, O, b, s, h0):
        """Normalize consumer `w`'s value dims by both row sums and store them with TMA.

        The output is staged in this consumer's Q tile, which every score has finished reading.
        """
        for i in T.Parallel(block_h):
            sL[w, i] = rL[i]
        T.sync_threads(sums_ready, consumer_threads)
        # A row with a live key sums to at least 1 (its max weighs 1); a row with none
        # has rO = 0.
        for i in T.Parallel(block_h):
            rL[i] = 1 / T.max(sL[0, i] + sL[1, i], 1.0)
        for i, j in T.Parallel(block_h, half):
            out[i, j] = rO[i, j] * rL[i]
        T.copy(out, staged)
        T.fence_proxy_async()
        T.sync_threads(staged_ready + w, warpgroup)
        T.copy(staged, O[b, s, h0 : h0 + block_h, w * half : (w + 1) * half])

    @T.macro
    def consumer0(O, QL, QR, QT, K0L, K0R, K0T, K1L, S0, S1, Bias, sM, sL, q_full, k_ready, k_free, bias_ready, b, s, h0):  # fmt: skip
        """Scores the even blocks; accumulates value dims [0, 256)."""
        T.set_max_nreg(consumer_regs, 1)
        rP = T.alloc_fragment([block_h, block_k], accum)
        rS = T.alloc_fragment([block_h, block_k], dtype)
        rO = T.alloc_fragment([block_h, half], accum)
        out = T.alloc_fragment([block_h, half], dtype)
        rM = T.alloc_fragment([block_h], accum)
        rL = T.alloc_fragment([block_h], accum)
        cur = T.alloc_fragment([block_h], accum)
        keep = T.alloc_fragment([block_h], accum)
        T.fill(rO, 0)
        T.fill(rL, 0)
        T.fill(rM, max_init)
        T.mbarrier_wait_parity(q_full[0], 0)
        T.mbarrier_wait_parity(k_ready[part_0l], 0)
        score_left(QL, K0L, rP, True)
        T.mbarrier_wait_parity(k_ready[part_0r], 0)
        score_right(QR, QT, K0R, K0T, rP, False)
        T.wait_wgmma(0)
        for pair in T.serial(pairs):
            phase = pair % 2
            T.mbarrier_wait_parity(bias_ready[0], phase)
            softmax(rP, rS, rO, rM, rL, rM, cur, keep, Bias, 0, sM)
            T.named_barrier_arrive(even_max_ready, consumer_threads)
            T.wgmma_gemm(rS, K0L, rO)
            T.wait_wgmma(0)
            T.mbarrier_arrive(k_free[part_0l])
            # Consumer 1 continued the chain: bring this block's weights to its max.
            T.sync_threads(pair_max_ready, consumer_threads)
            for i in T.Parallel(block_h):
                keep[i] = T.exp2(rM[i] - sM[i])
                rM[i] = sM[i]
                rL[i] *= keep[i]
            for i, j in T.Parallel(block_h, block_k):
                rS[i, j] = rP[i, j] * keep[i]
            T.copy(rS, S0)
            T.fence_proxy_async()
            T.named_barrier_arrive(s0_ready, consumer_threads)
            T.sync_threads(s1_ready, consumer_threads)
            for i, j in T.Parallel(block_h, half):
                rO[i, j] *= keep[i]
            T.wgmma_gemm(S1, K1L, rO)
            if pair + 1 < pairs:
                T.mbarrier_wait_parity(k_ready[part_0l], phase ^ 1)
                score_left(QL, K0L, rP, True)
                T.wait_wgmma(1)
                T.mbarrier_arrive(k_free[part_1l])
                T.mbarrier_wait_parity(k_ready[part_0r], phase ^ 1)
                score_right(QR, QT, K0R, K0T, rP, False)
                T.wait_wgmma(0)
            else:
                T.wait_wgmma(0)
                T.mbarrier_arrive(k_free[part_1l])
        finish(0, rO, rL, sL, out, QL, O, b, s, h0)

    @T.macro
    def consumer1(O, QL, QR, QT, K0R, K1L, K1R, K1T, S0, S1, Bias, sM, sL, q_full, k_ready, k_free, bias_ready, b, s, h0):  # fmt: skip
        """Scores the odd blocks; accumulates value dims [256, 512)."""
        T.set_max_nreg(consumer_regs, 1)
        rP = T.alloc_fragment([block_h, block_k], accum)
        rS = T.alloc_fragment([block_h, block_k], dtype)
        rO = T.alloc_fragment([block_h, half], accum)
        out = T.alloc_fragment([block_h, half], dtype)
        rM = T.alloc_fragment([block_h], accum)
        rL = T.alloc_fragment([block_h], accum)
        prev = T.alloc_fragment([block_h], accum)
        cur = T.alloc_fragment([block_h], accum)
        keep = T.alloc_fragment([block_h], accum)
        T.fill(rO, 0)
        T.fill(rL, 0)
        T.fill(rM, max_init)
        T.mbarrier_wait_parity(q_full[0], 0)
        for pair in T.serial(pairs):
            phase = pair % 2
            T.mbarrier_wait_parity(k_ready[part_1r], phase)
            score_right(QR, QT, K1R, K1T, rP, True)
            T.mbarrier_wait_parity(k_ready[part_1l], phase)
            score_left(QL, K1L, rP, False)
            T.wait_wgmma(0)
            T.mbarrier_wait_parity(bias_ready[0], phase)
            # Consumer 0 has scored the even block: continue its max.
            T.sync_threads(even_max_ready, consumer_threads)
            for i in T.Parallel(block_h):
                prev[i] = sM[i]
            softmax(rP, rS, rO, rM, rL, prev, cur, keep, Bias, 1, sM)
            T.named_barrier_arrive(pair_max_ready, consumer_threads)
            T.wgmma_gemm(rS, K1R, rO)
            T.copy(rS, S1)
            T.sync_threads(s0_ready, consumer_threads)
            T.wgmma_gemm(S0, K0R, rO)
            T.fence_proxy_async()
            T.named_barrier_arrive(s1_ready, consumer_threads)
            T.wait_wgmma(1)
            T.mbarrier_arrive(k_free[part_1r])
            T.wait_wgmma(0)
            T.mbarrier_arrive(k_free[part_0r])
        finish(1, rO, rL, sL, out, QR, O, b, s, h0)

    @T.prim_func
    def main(
        Q: T.Tensor([batch, seq_len, heads, dq], dtype),
        KV: T.Tensor([batch, seq_len_kv, kv_group, dq], dtype),
        Indices: T.Tensor([batch, seq_len, kv_group, topk], "int32"),
        O: T.Tensor([batch, seq_len, heads, dim], dtype),
    ):
        with T.Kernel(
            seq_len * head_blocks, kv_group, batch, threads=consumer_threads + producer_threads
        ) as (
            bx,
            g,
            b,
        ):
            s = bx // head_blocks
            h0 = g * group_heads + (bx % head_blocks) * block_h
            QL = T.alloc_shared([block_h, half], dtype)
            QR = T.alloc_shared([block_h, half], dtype)
            K0L = T.alloc_shared([block_k, half], dtype)
            K0R = T.alloc_shared([block_k, half], dtype)
            K1L = T.alloc_shared([block_k, half], dtype)
            K1R = T.alloc_shared([block_k, half], dtype)
            S1 = T.alloc_shared([block_h, block_k], dtype)
            parts = [QL, QR, K0L, K0R, K1L, K1R, S1]
            if has_tail:
                QT = T.alloc_shared([block_h, tail_dim], dtype)
                K0T = T.alloc_shared([block_k, tail_dim], dtype)
                K1T = T.alloc_shared([block_k, tail_dim], dtype)
                parts += [QT, K0T, K1T]
                # The even block's weights take its key tail's place once it is scored.
                S0 = K0T
            else:
                QT, K0T, K1T = QR, K0R, K1R
                S0 = T.alloc_shared([block_h, block_k], dtype)
                parts.append(S0)
            T.annotate_layout({buf: make_swizzled_layout(buf) for buf in parts})
            Bias = T.alloc_shared([2, block_k], accum)
            sM = T.alloc_shared([block_h], accum)
            sL = T.alloc_shared([2, block_h], accum)
            q_full = T.alloc_barrier([producer_threads])
            k_ready = T.alloc_barrier([producer_threads] * 4)
            k_free = T.alloc_barrier([warpgroup] * 4)
            bias_ready = T.alloc_barrier([producer_threads])
            tx = T.get_thread_binding()
            if tx >= consumer_threads:
                T.set_max_nreg(producer_regs, 0)
                producer(
                    Q, KV, Indices, QL, QR, QT, K0L, K0R, K0T, K1L, K1R, K1T, Bias, q_full,
                    k_ready, k_free, bias_ready, b, s, g, h0, tx - consumer_threads,
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
        # The builder's tiles: 64-head blocks, pairs of 64-key blocks, two 256-wide value
        # halves and a key tail of at most one 64-column swizzle atom.
        if (call.heads // call.kv_group) % 64 != 0:
            return "requires a multiple of 64 heads per KV group"
        if call.topk % 128 != 0:
            return "requires topk a multiple of 128"
        if call.dim != 512 or call.tail_dim not in (0, 64):
            return "requires dim 512 and tail_dim 0 or 64"
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
        sm_scale: Optional[float] = None,
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
