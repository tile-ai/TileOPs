"""SM90 DeepSeek Sparse Attention decode over a paged FP8 latent cache.

The cache row is FlashMLA's DeepSeek-V3.2 FP8 layout: 512 float8_e4m3fn latent values, one
float32 scale per 128 of them, then 64 bfloat16 rope values, 656 bytes in all.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops._csrc import csrc_include
from tileops.kernels.attention.call_spec import DSAPagedCall, DSAPagedFwdInterface
from tileops.kernels.attention.dsa.decode_ws import _ROLE_INDEX_CALL
from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["DSAPagedClusterKernel", "DSAPagedKernelBase", "DSAPagedWSKernel"]

# Both the attention kernel and the split merge compile with these flags.
_COMPILE_FLAGS = [
    "-O3",
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
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        # Rows are in range by the block table's contract; the guard splits fp8 vectors.
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        # Buffers of different warp roles are live at once.
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
    },
    compile_flags=[*_COMPILE_FLAGS, *csrc_include("warp_uniform.h", "row_gather.h")],
)
def _dsa_paged_kernel(batch, seq_len, heads, topk, page_size, split, sm_scale, cluster):
    """One thread block (CTA) per (query row, 64-head block, key split).

    Two 64-slot key buffers alternate. Warpgroup 0 scores and accumulates value dims [0, 256); warpgroup 1 gathers the next
    block and accumulates [256, 512); the producer warpgroup dequantizes in place. With
    ``cluster == 2``, each CTA of a pair dequantizes half of every block for both.
    """
    block_h = 64  # query heads per CTA: one m64 WGMMA row block
    block_k = 64  # selected slots per block
    stages = 2  # key-block buffers: one read while the other is filled
    half = 256  # value dims each consumer warpgroup accumulates
    # FP8 row: 512 fp8 latent, f32 scales from byte 512, 64 bf16 rope from byte 528.
    dim = 512
    rope = 64
    groups = dim // 128
    scale_col = dim // 4  # in float32 elements
    rope_col = (dim + 4 * groups) // 2  # in bfloat16 elements
    row_bytes = dim + 4 * groups + 2 * rope
    dq = dim + rope
    atom = 64  # bf16 columns one 128-byte swizzle atom holds
    vec = 8  # bf16 elements one 16-byte copy moves
    rows_mine = block_k // cluster  # the rows of a block this CTA gathers and dequantizes
    rows_per_warp = 8  # rows a producer warp dequantizes a pass: 64 registers
    passes = rows_mine // 4 // rows_per_warp
    row_threads = 128 // rows_mine  # warpgroup 1 threads gathering one row's latent
    chunk = 8  # values a lane dequantizes per row and half: 32 lanes cover 256 columns
    tail_lanes = rope // vec  # the rope tile's 16-byte vectors per row
    warpgroup = 128
    consumer_threads = 2 * warpgroup  # tx < 256; the producer warpgroup follows
    producer_threads = warpgroup
    # 384 threads launch at 168 registers; the producer's spare 64 fund the scorer's 56.
    scorer_regs = 224
    mixer_regs = 168
    producer_regs = 104
    max_init = -1.0e30  # a finite start, so a fully masked block weighs 0 instead of NaN
    masked = 1.0e38  # a dead slot's bias, far below max_init

    # Named barriers; id 0 is __syncthreads.
    s_ready = 1  # warpgroup 0 wrote the block's weights and row rescale
    sums_ready = 2  # warpgroup 0 wrote each row's normalizer
    staged_ready = 3  # + w: consumer w staged its output half
    rows_ready = 5  # warpgroup 1 published a block's rows

    tokens = batch * seq_len
    head_blocks = heads // block_h
    span = topk // split
    blocks = span // block_k
    scale = (dq**-0.5 if sm_scale is None else sm_scale) * LOG2E
    dtype = "bfloat16"
    accum = "float"
    out_dtype = dtype if split == 1 else accum
    num_pages = T.dynamic("num_pages")
    table_width = T.dynamic("table_width")
    cache_rows = num_pages * page_size

    @T.macro
    def resolve(Table, length, b, pos, src):
        """Resolve request `b`'s position `pos` to its cache row, or -1 when dead."""
        live = pos >= 0 and pos < length
        live_pos = pos * T.Cast("int32", live)
        src[0] = T.if_then_else(
            live, Table[b, live_pos // page_size] * page_size + live_pos % page_size, -1
        )

    @T.macro
    def issue(Rows16, Scales, KL, KR, KT, sScales, Rows, landed, stage, base, p):
        """Start gathering this CTA's rows of a block; dead rows land as zeros."""
        # 16-byte latent chunk c of row r lands raw in the back half of r's own row:
        # tile KL or KR by c // 16, swizzle atom 2 + c % 16 // 8.
        r = base + p // row_threads
        row = Rows[stage, r]
        for i in T.unroll(dim // 16 // row_threads):
            c = p % row_threads + row_threads * i
            col = (2 + c % 16 // 8) * atom + (c % 8 ^ r % 8) * vec
            if c < 16:
                T.call_extern(
                    "handle", "tileops::cp_async16_l2_256", T.access_ptr(KL[stage, r, col], "w"),
                    T.access_ptr(Rows16[T.max(row, 0), c * vec], "r"), row >= 0,
                )  # fmt: skip
            else:
                T.call_extern(
                    "handle", "tileops::cp_async16_l2_256", T.access_ptr(KR[stage, r, col], "w"),
                    T.access_ptr(Rows16[T.max(row, 0), c * vec], "r"), row >= 0,
                )  # fmt: skip
        if p >= block_k and p < block_k + rows_mine:
            srow = Rows[stage, base + p - block_k]
            T.ptx_cp_async(
                T.access_ptr(sScales[stage, base + p - block_k, 0], "w", groups),
                T.access_ptr(Scales[T.max(srow, 0), scale_col], "r", groups),
                groups,
                srow >= 0,
            )
        for k in T.unroll(block_k * tail_lanes // producer_threads):
            slot = k * (producer_threads // tail_lanes) + p // tail_lanes
            trow = Rows[stage, slot]
            T.ptx_cp_async(
                T.access_ptr(KT[stage, slot, p % tail_lanes * vec], "w", vec),
                T.access_ptr(Rows16[T.max(trow, 0), rope_col + p % tail_lanes * vec], "r", vec),
                vec,
                trow >= 0,
            )
        T.cp_async_barrier_noinc(landed[stage])

    @T.macro
    def convert(KL, KR, sScales, remote, stage, held, outv, warp, lane, base, peer):
        """Dequantize warp `warp`'s rows in place and, in a cluster, store them into the peer.

        A row's fp8 bytes sit in the row itself, so its warp reads them all before writing.
        """
        for n in T.unroll(passes):
            for i in T.unroll(rows_per_warp):
                r = base + (warp * passes + n) * rows_per_warp + i
                col = (2 + lane // 16) * atom + (lane // 2 % 8 ^ r % 8) * vec + lane % 2 * 4
                for v in T.vectorized(4):
                    held[0, i, v] = KL[stage, r, col + v]
                for v in T.vectorized(4):
                    held[1, i, v] = KR[stage, r, col + v]
            T.sync_warp()
            for i in T.unroll(rows_per_warp):
                r = base + (warp * passes + n) * rows_per_warp + i
                for h in T.unroll(2):
                    T.call_extern(
                        "handle", "tileops::dequant8", T.address_of(outv[h, 0]),
                        T.address_of(held[h, i, 0]), sScales[stage, r, 2 * h + lane // 16],
                    )  # fmt: skip
                for v in T.vectorized(chunk):
                    KL[stage, r, lane * chunk + v] = outv[0, v]
                for v in T.vectorized(chunk):
                    KR[stage, r, lane * chunk + v] = outv[1, v]
                if cluster == 2:
                    T.call_extern(
                        "handle", "tileops::send16", T.address_of(KL[stage, r, lane * chunk]),
                        T.address_of(remote[stage]), peer, T.address_of(outv[0, 0]),
                    )  # fmt: skip
                    T.call_extern(
                        "handle", "tileops::send16", T.address_of(KR[stage, r, lane * chunk]),
                        T.address_of(remote[stage]), peer, T.address_of(outv[1, 0]),
                    )  # fmt: skip

    @T.macro
    def release(k_free, stage, lane, peer):
        """Arrive on the stage's free barrier here and, in a cluster, in the peer."""
        if lane == 0:
            T.mbarrier_arrive(k_free[stage])
            if cluster == 2:
                T.call_extern("handle", "tileops::arrive_peer", T.address_of(k_free[stage]), peer)

    @T.macro
    def producer(Q, QL, QR, QT, KL, KR, sScales, q_full, k_landed, k_remote, k_ready, t, h0, count, base, peer, p):  # fmt: skip
        """Load Q, then dequantize this CTA's share of each landed block."""
        q_lanes = atom // vec
        q_rows = producer_threads // q_lanes
        for r in T.unroll(block_h // q_rows):
            row = q_rows * r + p // q_lanes
            col = p % q_lanes * vec
            for c in T.unroll(half // atom):
                T.ptx_cp_async(
                    T.access_ptr(QL[row, c * atom + col], "w", vec),
                    T.access_ptr(Q[t, h0 + row, c * atom + col], "r", vec),
                    vec,
                )
                T.ptx_cp_async(
                    T.access_ptr(QR[row, c * atom + col], "w", vec),
                    T.access_ptr(Q[t, h0 + row, half + c * atom + col], "r", vec),
                    vec,
                )
            T.ptx_cp_async(
                T.access_ptr(QT[row, col], "w", vec),
                T.access_ptr(Q[t, h0 + row, dim + col], "r", vec),
                vec,
            )
        T.cp_async_barrier_noinc(q_full[0])

        warp = p // 32
        lane = p % 32
        held = T.alloc_local([2, rows_per_warp, 4], dtype)  # 8 staged fp8 values, as bf16 bits
        outv = T.alloc_local([2, chunk], dtype)
        for j in T.serial(blocks):
            if j < count:
                stage = j % stages
                T.mbarrier_wait_parity(k_landed[stage], (j // stages) % 2)
                # Only after landing has the stage's previous phase completed.
                if cluster == 2 and p == 0:
                    T.call_extern(
                        "handle", "tileops::expect_bytes", T.address_of(k_remote[stage]),
                        rows_mine * dim * 2,
                    )  # fmt: skip
                convert(KL, KR, sScales, k_remote, stage, held, outv, warp, lane, base, peer)
                T.fence_proxy_async()
                T.mbarrier_arrive(k_ready[stage])

    @T.macro
    def finish(w, rO, sInv, out, staged, O, sp, t, h0):
        """Normalize consumer `w`'s value dims and store them."""
        if split == 1:
            for i, k in T.Parallel(block_h, half):
                out[i, k] = rO[i, k] * sInv[i]
            T.copy(out, staged)
            T.fence_proxy_async()
            T.sync_threads(staged_ready + w, warpgroup)
            T.copy(staged, O[0, t, h0 : h0 + block_h, w * half : (w + 1) * half])
        else:
            # In place: a second float32 fragment would not fit.
            for i, k in T.Parallel(block_h, half):
                rO[i, k] *= sInv[i]
            T.copy(rO, O[sp, t, h0 : h0 + block_h, w * half : (w + 1) * half])

    @T.macro
    def scorer(O, LSE, QL, QR, QT, KL, KR, KT, Rows, sScale, sInv, q_full, k_ready, k_remote, k_free, sp, t, h0, count, tx, peer):  # fmt: skip
        """Warpgroup 0: scores, softmax and value dims [0, 256)."""
        T.set_max_nreg(scorer_regs, 1)
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
        for j in T.serial(blocks):
            if j < count:
                stage = j % stages
                T.wait_wgmma(0)
                if j > 0:
                    release(k_free, (j + stages - 1) % stages, tx % 32, peer)
                T.mbarrier_wait_parity(k_ready[stage], (j // stages) % 2)
                if cluster == 2:
                    T.mbarrier_wait_parity(k_remote[stage], (j // stages) % 2)
                T.wgmma_gemm(QL, KL[stage, :, :], rP, transpose_B=True, clear_accum=True)
                T.wgmma_gemm(QR, KR[stage, :, :], rP, transpose_B=True)
                T.wgmma_gemm(QT, KT[stage, :, :], rP, transpose_B=True)
                T.wait_wgmma(0)
                # Bias after scaling, so any sign of scale masks dead slots.
                for i, k in T.Parallel(block_h, block_k):
                    rP[i, k] = rP[i, k] * scale - T.Cast(accum, Rows[stage, k] < 0) * masked
                T.reduce_max(rP, cur, dim=1, clear=True)
                for i in T.Parallel(block_h):
                    cur[i] = T.max(rM[i], cur[i])
                    keep[i] = T.exp2(rM[i] - cur[i])
                    rM[i] = cur[i]
                for i, k in T.Parallel(block_h, block_k):
                    rP[i, k] = T.exp2(rP[i, k] - cur[i])
                    rS[i, k] = rP[i, k]
                T.reduce_sum(rP, cur, dim=1)
                for i in T.Parallel(block_h):
                    rL[i] = rL[i] * keep[i] + cur[i]
                # The weights reuse the scored rope tile.
                T.copy(rS, KT[stage, :, :])
                for i in T.Parallel(block_h):
                    sScale[stage, i] = keep[i]
                T.fence_proxy_async()
                T.named_barrier_arrive(s_ready, consumer_threads)
                for i, k in T.Parallel(block_h, half):
                    rO[i, k] *= keep[i]
                T.wgmma_gemm(rS, KL[stage, :, :], rO)
        T.wait_wgmma(0)
        # A row without a live key has rL = 0: output 0, lse -inf.
        for i in T.Parallel(block_h):
            LSE[sp, t, h0 + i] = (rM[i] + T.log2(rL[i])) / LOG2E
            sInv[i] = 1 / T.max(rL[i], 1.0)
        T.named_barrier_arrive(sums_ready, consumer_threads)
        finish(0, rO, sInv, out, QL, O, sp, t, h0)

    @T.macro
    def gather(Rows16, Scales, Indices, Table, KL, KR, KT, sScales, Rows, k_landed, src, pos, length, t, b, k0, j, base, x):  # fmt: skip
        """Publish block j's rows, start its gathers, and look up the next block's rows."""
        stage = j % stages
        slot = x % block_k
        if x < block_k:
            Rows[stage, slot] = src[0]
        T.sync_threads(rows_ready, warpgroup)
        issue(Rows16, Scales, KL, KR, KT, sScales, Rows, k_landed, stage, base, x)
        resolve(Table, length, b, pos[0], src)
        pos[0] = Indices[t, T.min(k0 + (j + 2) * block_k + slot, topk - 1)]

    @T.macro
    def mixer(O, QR, KL, KR, KT, Rows16, Scales, Indices, Table, Lens, sScales, Rows, sScale, sInv, k_landed, k_free, sp, t, b, h0, k0, count, base, peer, x):  # fmt: skip
        """Warpgroup 1: gathers a block ahead and accumulates value dims [256, 512)."""
        # Equals the launch allocation, but still sets this role's register budget.
        T.set_max_nreg(mixer_regs, 0)
        lane = x % 32
        src = T.alloc_local([1], "int32")
        pos = T.alloc_local([1], "int32")
        rO = T.alloc_fragment([block_h, half], accum)
        out = T.alloc_fragment([block_h, half], dtype)
        T.fill(rO, 0)
        length = Lens[b]
        # Positions load two blocks ahead, rows resolve one block ahead.
        resolve(Table, length, b, Indices[t, T.min(k0 + x % block_k, topk - 1)], src)
        pos[0] = Indices[t, T.min(k0 + block_k + x % block_k, topk - 1)]
        if count > 0:
            gather(
                Rows16, Scales, Indices, Table, KL, KR, KT, sScales, Rows, k_landed, src, pos,
                length, t, b, k0, 0, base, x,
            )  # fmt: skip
        for j in T.serial(blocks):
            if j < count:
                stage = j % stages
                if j + 1 < count:
                    # Block j + 1 takes the buffer block j - 1 frees.
                    T.mbarrier_wait_parity(k_free[(j + 1) % stages], ((j + 1) // stages) % 2 ^ 1)
                    gather(
                        Rows16, Scales, Indices, Table, KL, KR, KT, sScales, Rows, k_landed, src,
                        pos, length, t, b, k0, j + 1, base, x,
                    )  # fmt: skip
                T.sync_threads(s_ready, consumer_threads)
                for i, k in T.Parallel(block_h, half):
                    rO[i, k] *= sScale[stage, i]
                T.wgmma_gemm(KT[stage, :, :], KR[stage, :, :], rO)
                T.wait_wgmma(0)
                release(k_free, stage, lane, peer)
        T.sync_threads(sums_ready, consumer_threads)
        finish(1, rO, sInv, out, QR, O, sp, t, h0)

    @T.prim_func
    def main(
        Q: T.Tensor([tokens, heads, dq], dtype),
        Rows16: T.Tensor([cache_rows, row_bytes // 2], dtype),
        Scales: T.Tensor([cache_rows, row_bytes // 4], accum),
        Table: T.Tensor([batch, table_width], "int32"),
        Lens: T.Tensor([batch], "int32"),
        Indices: T.Tensor([tokens, topk], "int32"),
        O: T.Tensor([split, tokens, heads, dim], out_dtype),
        LSE: T.Tensor([split, tokens, heads], accum),
    ):
        with T.ClusterKernel(
            head_blocks * split, tokens, cluster_dims=(cluster, 1, 1),
            threads=consumer_threads + producer_threads,
        ) as (bx, t):  # fmt: skip
            hb = bx % head_blocks
            sp = bx // head_blocks
            b = t // seq_len
            h0 = hb * block_h
            # Splits share the blocks up to the row's last live slot; the rest are skipped.
            live_end = T.alloc_shared([1], "int32")
            tx = T.get_thread_binding()
            if tx == 0:
                live_end[0] = 0
            T.sync_threads()
            length = Lens[b]
            for k in T.serial(T.ceildiv(topk, consumer_threads + producer_threads)):
                slot = k * (consumer_threads + producer_threads) + tx
                if slot < topk:
                    pos = Indices[t, slot]
                    if pos >= 0 and pos < length:
                        T.atomic_max(live_end[0], slot + 1)
            T.sync_threads()
            live_blocks = T.ceildiv(live_end[0], block_k)
            per_split = T.ceildiv(live_blocks, split)
            count = T.max(T.min(per_split, live_blocks - sp * per_split), 0)
            k0 = sp * per_split * block_k
            rank = bx % cluster
            peer = rank ^ 1
            base = rank * rows_mine
            QL = T.alloc_shared([block_h, half], dtype)
            QR = T.alloc_shared([block_h, half], dtype)
            QT = T.alloc_shared([block_h, rope], dtype)
            KL = T.alloc_shared([stages, block_k, half], dtype)
            KR = T.alloc_shared([stages, block_k, half], dtype)
            KT = T.alloc_shared([stages, block_k, rope], dtype)
            T.annotate_layout({buf: make_swizzled_layout(buf) for buf in [QL, QR, QT, KL, KR, KT]})
            sScales = T.alloc_shared([stages, block_k, groups], accum)
            Rows = T.alloc_shared([stages, block_k], "int32")
            sScale = T.alloc_shared([stages, block_h], accum)
            sInv = T.alloc_shared([block_h], accum)
            q_full = T.alloc_barrier([producer_threads])
            k_landed = T.alloc_barrier([producer_threads] * stages)
            k_ready = T.alloc_barrier([producer_threads] * stages)
            k_remote = T.alloc_barrier([1] * stages)
            # One arrival a consumer warp, here and from the peer.
            k_free = T.alloc_barrier([8 * cluster] * stages)
            if cluster == 2:
                # Both CTAs' barriers are initialized before either touches the other's.
                T.cluster_arrive()
                T.cluster_wait()
            # Marker for decode_ws's uniform-role rewrite; it must stay a bare call.
            T.evaluate(
                T.call_extern("int32", _ROLE_INDEX_CALL)  # lint: extern-target allow: decode_ws
            )
            if tx >= consumer_threads:
                T.set_max_nreg(producer_regs, 0)
                producer(
                    Q, QL, QR, QT, KL, KR, sScales, q_full, k_landed, k_remote, k_ready, t, h0,
                    count, base, peer, tx - consumer_threads,
                )  # fmt: skip
            with T.ws(0):
                scorer(
                    O, LSE, QL, QR, QT, KL, KR, KT, Rows, sScale, sInv, q_full, k_ready,
                    k_remote, k_free, sp, t, h0, count, tx, peer,
                )  # fmt: skip
            with T.ws(1):
                mixer(
                    O, QR, KL, KR, KT, Rows16, Scales, Indices, Table, Lens, sScales, Rows,
                    sScale, sInv, k_landed, k_free, sp, t, b, h0, k0, count, base, peer,
                    tx - warpgroup,
                )  # fmt: skip
            if cluster == 2:
                # Neither CTA exits while the peer may still write into it.
                T.cluster_arrive()
                T.cluster_wait()

    return main


@functools.lru_cache(maxsize=32)
@tilelang.jit(compile_flags=_COMPILE_FLAGS)
def _dsa_paged_combine_kernel(tokens, heads, split):
    """Merge each row's `split` partial outputs by their log-sum-exps."""
    dim = 512  # value dims
    threads = 128
    cols = dim // threads

    @T.prim_func
    def combine(
        Part: T.Tensor([split, tokens, heads, dim], "float"),
        PartLSE: T.Tensor([split, tokens, heads], "float"),
        O: T.Tensor([tokens, heads, dim], "bfloat16"),
        LSE: T.Tensor([tokens, heads], "float"),
    ):
        with T.Kernel(heads, tokens, threads=threads) as (h, t):
            tx = T.get_thread_binding()
            peak = T.alloc_var("float", init=-T.infinity("float"))
            total = T.alloc_var("float", init=0.0)
            acc = T.alloc_local([cols], "float")
            for s in T.serial(split):
                peak = T.max(peak, PartLSE[s, t, h])
            for c in T.serial(cols):
                acc[c] = 0.0
            # No live key: peak -inf, output 0. Otherwise the peak split weighs 1.
            if peak > -T.infinity("float"):
                for s in T.serial(split):
                    w = T.exp2((PartLSE[s, t, h] - peak) * LOG2E)
                    total += w
                    for c in T.vectorized(cols):
                        acc[c] += w * Part[s, t, h, tx * cols + c]
            for c in T.vectorized(cols):
                O[t, h, tx * cols + c] = T.Cast("bfloat16", acc[c] / T.max(total, 1.0))
            if tx == 0:
                LSE[t, h] = T.if_then_else(total > 0, peak + T.log(total), peak)

    return combine


class DSAPagedKernelBase(Kernel, DSAPagedFwdInterface):
    """SM90 sparse Multi-head Latent Attention (MLA) decode over a paged FP8 cache."""

    supported_archs: list[int] = [90]
    cluster: ClassVar[int]

    @classmethod
    def refusal(cls, call: DSAPagedCall) -> Optional[str]:
        if call.heads % (64 * cls.cluster) != 0:
            return f"requires a multiple of {64 * cls.cluster} heads"
        if call.topk % 64 != 0:
            return "requires topk a multiple of 64"
        return None

    @classmethod
    def entry_for(cls, call: DSAPagedCall) -> Entry:
        index = call.device.index if call.device is not None else None
        sms = torch.cuda.get_device_properties(
            call.device if call.device is not None else "cuda"
        ).multi_processor_count
        split = cls.split_for(call.batch * call.seq_len * (call.heads // 64), call.topk, sms)
        args = (
            call.batch,
            call.seq_len,
            call.heads,
            call.topk,
            call.page_size,
            split,
            call.sm_scale,
        )
        return (*args, index), lambda: cls(*args, device_index=index)

    @staticmethod
    def split_for(ctas: int, topk: int, sms: int) -> int:
        """The most key splits whose CTAs still fit one wave, each a whole number of blocks."""
        split = 1
        while ctas * split * 2 <= sms and topk % (64 * split * 2) == 0:
            split *= 2
        return split

    def __init__(self, batch, seq_len, heads, topk, page_size, split, sm_scale, device_index=None):
        super().__init__(device_index=device_index)
        self.batch, self.seq_len, self.heads = batch, seq_len, heads
        self.split = split
        self.kernel = _dsa_paged_kernel(
            batch, seq_len, heads, topk, page_size, split, sm_scale, self.cluster
        )
        self.combine = (
            _dsa_paged_combine_kernel(batch * seq_len, heads, split) if split > 1 else None
        )

    def forward(self, q, kv_cache, block_table, cache_seqlens, indices):
        self._require_devices(
            q=q, kv_cache=kv_cache, block_table=block_table, cache_seqlens=cache_seqlens,
            indices=indices,
        )  # fmt: skip
        # The kernel addresses the cache's bfloat16 view with int32 offsets.
        if kv_cache.numel() // 2 >= 2**31:
            raise ValueError(
                f"kv_cache holds {kv_cache.numel()} bytes; this kernel addresses below 2^32"
            )
        tokens = self.batch * self.seq_len
        dim = 512  # value dims
        rows = kv_cache.view(-1, kv_cache.shape[-1])
        part_dtype = torch.bfloat16 if self.split == 1 else torch.float32
        out = torch.empty((self.split, tokens, self.heads, dim), dtype=part_dtype, device=q.device)
        lse = torch.empty((self.split, tokens, self.heads), dtype=torch.float32, device=q.device)
        self.kernel(
            q.view(tokens, self.heads, -1),
            rows.view(torch.bfloat16),
            rows.view(torch.float32),
            block_table,
            cache_seqlens,
            indices.view(tokens, -1),
            out,
            lse,
        )
        if self.split == 1:
            o, row_lse = out[0], lse[0]
        else:
            o = torch.empty((tokens, self.heads, dim), dtype=q.dtype, device=q.device)
            row_lse = torch.empty((tokens, self.heads), dtype=torch.float32, device=q.device)
            self.combine(out, lse, o, row_lse)
        shape = (self.batch, self.seq_len, self.heads)
        return o.view(*shape, dim), row_lse.view(shape)


class DSAPagedWSKernel(DSAPagedKernelBase):
    """Each 64-head block of a query row gathers and dequantizes every key row itself."""

    cluster: ClassVar[int] = 1


class DSAPagedClusterKernel(DSAPagedKernelBase):
    """A query row's two 64-head blocks share each key block's dequantization as a cluster."""

    cluster: ClassVar[int] = 2
    preferred_over: ClassVar[frozenset[str]] = frozenset({"dsa_paged_ws_kernel"})
