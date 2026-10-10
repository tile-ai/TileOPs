"""Packed variable-length GQA prefill over FP8 Q/K/V with per-request scales.

Inputs use THD layout:
  q: [T_q, H, D]  k/v: [T_kv, H_kv, D], all ``float8_e4m3fn``
  q_scale/k_scale/v_scale: float32 [B, H_kv], one per request and KV head
  output: [T_q, H, D] in a 16-bit dtype

``cu_seqlens_q`` and ``cu_seqlens_kv`` delimit each request; causal masking and the
sliding window use bottom-right alignment per request.

A block serves one request and one KV head, so the dequantization is two scalars
rather than a per-element promotion: ``q_scale * k_scale`` enters the score scale the
masking pass already applies, and ``v_scale`` enters the epilogue's row reciprocal.
TileFoundry prices the promoting alternative at 14.05 MB of shared-memory reads
against 7.53 MB, and moves the block from the memory bound to the compute bound.
"""

import functools
import itertools
from typing import Callable, ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops.kernels.attention.call_spec import ATTENTION_DTYPES, AttentionCall
from tileops.kernels.attention.fp8_fa3_layouts import (
    fa3_acc_fragment,
    fa3_qk_acc_column,
    fa3_qk_row_fragment,
)
from tileops.kernels.attention.online_softmax import (
    make_online_softmax_with_mask_guard,
    make_rescale,
    make_varlen_sink_scale,
)
from tileops.kernels.attention.varlen import VarlenKernel
from tileops.kernels.attention.varlen_rope import make_varlen_query_rope
from tileops.kernels.constants import (
    LOG2E,
    SHARED_BUFFER_ALIGN_BYTES,
    TMA_DTYPE_UINT8,
    TMA_INTERLEAVE_NONE,
    TMA_L2_PROMOTION_128B,
    TMA_OOB_FILL_NONE,
    TMA_SWIZZLE_128B,
    VECTOR_ACCESS_BYTES,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.utils import get_shared_memory_optin, get_sm_count

__all__ = ["GQAVarlenFP8FwdKernel", "GQAVarlenFP8WSFwdKernel"]

_FP8_DTYPE = "float8_e4m3fn"
# A call without rotation still hands the program its tables, so the placeholder's
# innermost extent is a whole vectorized access, which the packed ABI requires.
_ROPE_PLACEHOLDER_WIDTH = VECTOR_ACCESS_BYTES // 2
# The work counter the persistent CTAs claim from: the next item to hand out, and the
# CTAs that have stopped claiming. The program leaves both at zero, so one buffer serves
# every launch and no call has to clear it.
_CLAIM_SLOTS = 2


@functools.lru_cache(maxsize=32)
def _gqa_varlen_fp8_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    softcap: float = 0.0,
    out_dtype: str = "float16",
    window_size_left: int = -1,
    window_size_right: int = -1,
    has_sinks: bool = False,
) -> Callable:
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    has_left = window_size_left >= 0
    has_right = window_size_right >= 0
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[9],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_varlen_fp8_fwd_func(
        block_m: int, block_n: int, num_stages: int, threads: int
    ) -> Callable:
        total_q = T.dynamic("total_q")
        total_kv = T.dynamic("total_kv")
        q_shape = (total_q, heads, dim)
        kv_shape = (total_kv, heads_kv, dim)
        scale_shape = (batch, heads_kv)
        # The score scale carries LOG2E so the softmax reduces with exp2; the
        # per-request dequantization multiplies it at run time.
        exp2_scale = score_scale * LOG2E
        online_softmax = make_online_softmax_with_mask_guard(1.0, accum_dtype, block_m, block_n)
        rescale = make_rescale(block_m, dim)
        sink_scale = make_varlen_sink_scale(1.0, block_m)
        q_tiling = GroupTiling(batch, block_m)
        num_q_tiles = q_tiling.tile_upper_bound(total_q)

        @T.prim_func
        def _gqa_varlen_fp8_fwd_main(
            q: T.Tensor(q_shape, _FP8_DTYPE),  # type: ignore
            k: T.Tensor(kv_shape, _FP8_DTYPE),  # type: ignore
            v: T.Tensor(kv_shape, _FP8_DTYPE),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
            q_descale: T.Tensor(scale_shape, accum_dtype),  # type: ignore
            k_descale: T.Tensor(scale_shape, accum_dtype),  # type: ignore
            v_descale: T.Tensor(scale_shape, accum_dtype),  # type: ignore
            sinks: T.Tensor(
                [heads] if has_sinks else q_shape, "float32" if has_sinks else _FP8_DTYPE
            ),
            output: T.Tensor(q_shape, out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(num_q_tiles, heads, threads=threads) as (q_tile, by):
                q_shared = T.alloc_shared([block_m, dim], _FP8_DTYPE)
                k_shared = T.alloc_shared([block_n, dim], _FP8_DTYPE)
                v_shared = T.alloc_shared([block_n, dim], _FP8_DTYPE)
                p_shared = T.alloc_shared([block_m, block_n], _FP8_DTYPE)
                o_shared = T.alloc_shared([block_m, dim], out_dtype)
                tile_cum = T.alloc_shared([batch + 1], "int32")
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)
                inv_logsum = T.alloc_fragment([block_m], accum_dtype)
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                q_row = T.alloc_local([1], "int32")
                request = T.alloc_local([1], "int32")

                q_tiling.cumsum_offsets(cu_seqlens_q, tile_cum)
                if q_tile < tile_cum[batch]:
                    q_tiling.decode(q_tile, tile_cum, lo, hi, request, q_row)

                    q_start = cu_seqlens_q[request[0]]
                    kv_start = cu_seqlens_kv[request[0]]
                    q_len = cu_seqlens_q[request[0] + 1] - q_start
                    kv_len = cu_seqlens_kv[request[0] + 1] - kv_start
                    causal_offset = kv_len - q_len
                    cur_kv_head = by // groups
                    # Both scales are read once per block: the score scale folds
                    # q_scale * k_scale, and the epilogue folds v_scale.
                    qk_scale = T.alloc_var(
                        accum_dtype,
                        init=q_descale[request[0], cur_kv_head]
                        * k_descale[request[0], cur_kv_head]
                        * T.cast(score_scale if use_softcap else exp2_scale, accum_dtype),
                    )
                    value_scale = T.alloc_var(accum_dtype, init=v_descale[request[0], cur_kv_head])

                    for i, d in T.Parallel(block_m, dim):
                        q_pos = q_row[0] + i
                        q_shared[i, d] = T.if_then_else(
                            q_pos < q_len, q[q_start + q_pos, by, d], T.cast(0, _FP8_DTYPE)
                        )

                    T.clear(acc_o)
                    T.clear(logsum)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                    if is_causal:
                        loop_range = T.max(
                            0,
                            T.ceildiv(T.min(kv_len, causal_offset + q_row[0] + block_m), block_n),
                        )
                    elif has_right:
                        loop_range = T.max(
                            0,
                            T.ceildiv(
                                T.min(
                                    kv_len,
                                    causal_offset + q_row[0] + block_m + window_size_right,
                                ),
                                block_n,
                            ),
                        )
                    else:
                        loop_range = T.ceildiv(kv_len, block_n)
                    # Key tiles wholly left of the window are skipped, not masked.
                    if has_left:
                        k_first = T.max(0, causal_offset + q_row[0] - window_size_left) // block_n
                        loop_range = T.max(0, loop_range - k_first)
                    else:
                        k_first = 0

                    for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                        tile_start = (k_first + k_idx) * block_n
                        tile_end = tile_start + block_n
                        if tile_end <= kv_len:
                            T.copy(
                                k[kv_start + tile_start : kv_start + tile_end, cur_kv_head, :],
                                k_shared,
                            )
                            T.copy(
                                v[kv_start + tile_start : kv_start + tile_end, cur_kv_head, :],
                                v_shared,
                            )
                        else:
                            for j, d in T.Parallel(block_n, dim):
                                kv_pos = tile_start + j
                                if kv_pos < kv_len:
                                    k_shared[j, d] = k[kv_start + kv_pos, cur_kv_head, d]
                                    v_shared[j, d] = v[kv_start + kv_pos, cur_kv_head, d]
                                else:
                                    k_shared[j, d] = T.cast(0, _FP8_DTYPE)
                                    v_shared[j, d] = T.cast(0, _FP8_DTYPE)

                        T.gemm(
                            q_shared,
                            k_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=True,
                        )
                        for i, j in T.Parallel(block_m, block_n):
                            q_pos = q_row[0] + i
                            kv_pos = tile_start + j
                            if is_causal:
                                valid = (
                                    (q_pos < q_len)
                                    & (kv_pos < kv_len)
                                    & (kv_pos <= q_pos + causal_offset)
                                )
                            elif has_right:
                                valid = (
                                    (q_pos < q_len)
                                    & (kv_pos < kv_len)
                                    & (kv_pos <= q_pos + causal_offset + window_size_right)
                                )
                            else:
                                valid = (q_pos < q_len) & (kv_pos < kv_len)
                            if has_left:
                                valid = valid & (kv_pos >= q_pos + causal_offset - window_size_left)
                            if use_softcap:
                                logit = T.cast(softcap * LOG2E, accum_dtype) * T.tanh(
                                    acc_s[i, j] * qk_scale / T.cast(softcap, accum_dtype)
                                )
                            else:
                                logit = acc_s[i, j] * qk_scale
                            acc_s[i, j] = T.if_then_else(valid, logit, -T.infinity(accum_dtype))
                        online_softmax(
                            acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                        )
                        T.copy(acc_s, p_shared)
                        rescale(acc_o, scores_scale)
                        T.gemm(p_shared, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                    if has_sinks:
                        sink_scale(logsum, scores_max, sinks, scores_scale, by)
                        rescale(acc_o, scores_scale)
                    for i in T.Parallel(block_m):
                        inv_logsum[i] = T.if_then_else(
                            logsum[i] > 0,
                            value_scale / logsum[i],
                            T.cast(0, accum_dtype),
                        )
                    if q_row[0] + block_m <= q_len:
                        for i, j in T.Parallel(block_m, dim):
                            o_shared[i, j] = T.cast(acc_o[i, j] * inv_logsum[i], out_dtype)
                        T.copy(
                            o_shared,
                            output[q_start + q_row[0] : q_start + q_row[0] + block_m, by, :],
                            disable_tma=True,
                        )
                    else:
                        for i, j in T.Parallel(block_m, dim):
                            q_pos = q_row[0] + i
                            if q_pos < q_len:
                                output[q_start + q_pos, by, j] = T.cast(
                                    acc_o[i, j] * inv_logsum[i], out_dtype
                                )

        return _gqa_varlen_fp8_fwd_main

    return _gqa_varlen_fp8_fwd_func


@functools.lru_cache(maxsize=32)
def _gqa_varlen_fp8_ws_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: Optional[float],
    softcap: float,
    out_dtype: str,
    window_size_left: int,
    window_size_right: int,
    fuse_rope: bool,
    max_position: int,
    rotary_dim: int,
    rope_layout: str,
    rope_dtype: str,
    num_ctas: int,
    has_sinks: bool = False,
    can_promote: bool = False,
) -> Callable:
    """Build the warp-specialized FP8 packed-varlen program.

    A persistent CTA per SM walks the query tiles of the whole call in a fixed stride,
    so its TMA warpgroup and its two compute warpgroups agree on the item without a
    shared claim. The compute warpgroups issue the FA3 WGMMA pair through the CUTE
    helpers of ``fp8_gqa_helper.h``: TileLang's own ``T.gemm`` waits on every WGMMA it
    issues, and ``T.wgmma_gemm`` has no FP8 lowering, so the key contraction of one tile
    cannot otherwise overlap the value contraction of the one before it.
    """
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"
    # The CUTE helpers are written for the FA3 tile: a 64-row WGMMA per warpgroup,
    # a 224-key block and a 128-wide head.
    block_n = 224
    half_m = WGMMA_ROWS
    block_m = 2 * half_m
    # Threads of the two compute warpgroups, which the turn-passing barriers count.
    compute_threads = 2 * WARPGROUP_THREADS
    # Named barriers in use: 1 and 2 pass the turn between the compute warpgroups, 3 and
    # 4 close each one's query tile and epilogue, 5 and 6 bracket the claim.
    block_threads = 3 * WARPGROUP_THREADS
    claim_read_barrier = 5
    claim_write_barrier = 6
    # Key and value tiles in flight: one more than either compute warpgroup needs, so
    # one of them can be a tile ahead of the other. A fourth does not fit beside the
    # query and output buffers; ``_shared_bytes`` is what prices that.
    stages = 3
    # A tile holds the group's query heads side by side, so it spans this many query
    # positions and its causal bound advances once every ``groups`` rows. The tiling
    # therefore counts positions, not rows: FlashAttention-3's ``PackGQA`` traversal.
    positions_per_tile = block_m // groups
    positions_per_half = half_m // groups
    packed_store = f"tileops::fp8_fa3_o_smem_store_global_packed_64x128<{positions_per_half}>"
    packed_store_tail = (
        f"tileops::fp8_fa3_o_smem_store_global_packed_64x128_tail<{positions_per_half}>"
    )
    attention_scale = dim**-0.5 if sm_scale is None else sm_scale
    scale = attention_scale * LOG2E
    use_softcap = softcap > 0.0
    capped_softmax_scale = softcap * LOG2E
    sink_scale = make_varlen_sink_scale(
        capped_softmax_scale if use_softcap else scale, half_m, groups
    )
    has_left = window_size_left >= 0
    has_right = window_size_right >= 0
    q_tiling = GroupTiling(batch, positions_per_tile)
    # The program always takes the tables so one body serves both calls; a call without
    # rotation hands it a one-entry placeholder it never reads.
    rope_half = rotary_dim // 2 if fuse_rope else _ROPE_PLACEHOLDER_WIDTH
    # The query tile is rotated where it lands, in the warpgroup that owns it; the keys
    # were rotated in their own launch, because every query tile of a request reads them.
    rotate_query_tile = (
        make_varlen_query_rope(
            half_m,
            rotary_dim,
            rope_layout,
            max_position,
            _FP8_DTYPE,
            rope_dtype,
            positions=positions_per_half,
        )
        if fuse_rope
        else None
    )

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
        compile_flags=[
            "-O3",
            "-DENABLE_BF16",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            *csrc_include("fp8_gqa_helper.h"),
        ],
    )
    def func():
        @T.macro
        def pv_tile(acc_s, values, slot, acc_o, extent):
            # Hopper FP8 tensor-core accumulation loses small contributions in
            # long reductions. Reuse the consumed score registers for one PV
            # tile, then promote its sum in FP32 before issuing another QK.
            if can_promote and extent >= 16:
                T.call_extern(
                    "handle",
                    "tileops::fp8_pv_ptx_unit_begin_accumulate_fa3_raw_64x128x224",
                    acc_s.data,
                    T.access_ptr(values[slot, 0, 0], "r"),
                    acc_s.data,
                    True,
                )
                T.wait_wgmma(0)
                T.warpgroup_fence_operand(acc_s, num_regs=64)
                T.call_extern(
                    "handle",
                    "tileops::fp8_fa3_raw_acc_promote_64x128",
                    acc_o.data,
                    acc_s.data,
                )
            else:
                T.call_extern(
                    "handle",
                    "tileops::fp8_pv_ptx_unit_begin_accumulate_fa3_raw_64x128x224",
                    acc_s.data,
                    T.access_ptr(values[slot, 0, 0], "r"),
                    acc_o.data,
                )

        total_q = T.dynamic("total_q")
        total_kv = T.dynamic("total_kv")

        @T.macro
        def online_softmax(acc_s, sm, smp, ss, ssum, logsum, qk_descale):
            """Fold one scaled score tile into the running max and the lane-local row sums."""
            if use_softcap:
                T.copy(sm, smp)
                T.fill(sm, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, sm, dim=1, clear=False)
                for i in T.Parallel(half_m):
                    sm[i] = T.max(T.max(sm[i], smp[i]), -1e38)
                    ss[i] = T.exp2((smp[i] - sm[i]) * T.cast(capped_softmax_scale, accum_dtype))
                for i, j in T.Parallel(half_m, block_n):
                    acc_s[i, j] = T.exp2(
                        (acc_s[i, j] - sm[i]) * T.cast(capped_softmax_scale, accum_dtype)
                    )
            else:
                score_scale_softmax = qk_descale * scale
                T.copy(sm, smp)
                T.fill(sm, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, sm, dim=1, clear=False)
                for i in T.Parallel(half_m):
                    sm[i] = T.max(T.max(sm[i] * qk_descale, smp[i]), -1e38)
                    ss[i] = T.exp2(smp[i] * scale - sm[i] * scale)
                for i, j in T.Parallel(half_m, block_n):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * score_scale_softmax - sm[i] * scale)
            for i in T.Parallel(half_m):
                logsum[i] = logsum[i] * ss[i]
            # Lane-local row sums; the quad reduction is deferred to the epilogue.
            T.call_extern(
                "handle", "tileops::fp8_partial_row_sum_raw_acc_64x224", acc_s.data, ssum.data
            )
            for i in T.Parallel(half_m):
                logsum[i] = logsum[i] + ssum[i]

        @T.macro
        def transpose_value_tile(slot, v_smem, v_raw_full, v_full, phase):
            """Turn the staged value tile into the K-major operand the WGMMA reads.

            The transform runs where the transfer landed: the helper's in-place form
            fences between its read and its write, which frees the whole second copy of
            the ring and is what pays for its third stage.
            """
            T.barrier_wait(v_raw_full[slot], phase)
            T.call_extern(
                "handle",
                "tileops::fp8_transpose_v_128x224_fa3_src_ldsm_stsm_barrier_each_iter",
                T.access_ptr(v_smem[slot, 0, 0], "rw"),
                T.access_ptr(v_smem[slot, 0, 0], "w"),
            )
            T.barrier_arrive(v_full[slot])

        @T.macro
        def claim_item(claimed, Claim):
            """Draw the next item off the shared counter into *claimed* for the whole CTA.

            One thread draws it so the transfer warpgroup and the two compute warpgroups
            reach the same item; the first barrier holds the draw until every role has
            read the previous one. Both are per item rather than per key tile, so a CTA
            crosses them once per item it runs and not once per tile it scans.
            """
            T.sync_threads(barrier_id=claim_read_barrier, arrive_count=block_threads)
            if T.get_thread_binding() == 0:
                claimed[0] = T.atomic_add(Claim[0], 1, return_prev=True)
            T.sync_threads(barrier_id=claim_write_barrier, arrive_count=block_threads)

        @T.macro
        def locate(work, CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta):
            """Fill meta for item *work*: KV head, offsets, lengths, first position, key tiles."""
            # Items are handed out from the last query tile of the call. Under the causal
            # mask a later tile scans more keys, so the longest go first and the closing
            # wave carries the short ones; claiming in the opposite order costs 8%.
            q_tiling.decode(
                tile_cum[batch] - 1 - work // heads_kv, tile_cum, lo, hi, request, q_row
            )
            meta[0] = work % heads_kv
            meta[1] = CuQ[request[0]] + q_row[0]
            meta[2] = CuKV[request[0]]
            meta[3] = CuQ[request[0] + 1] - CuQ[request[0]]
            meta[4] = CuKV[request[0] + 1] - meta[2]
            meta[5] = q_row[0]
            # Key tiles wholly left of the window are skipped, not masked.
            if has_left:
                meta[8] = T.max(0, meta[4] - meta[3] + meta[5] - window_size_left) // block_n
            else:
                meta[8] = 0
            if is_causal:
                last = T.min(
                    T.ceildiv(meta[4], block_n),
                    T.ceildiv(meta[5] + positions_per_tile + meta[4] - meta[3], block_n),
                )
            elif has_right:
                last = T.min(
                    T.ceildiv(meta[4], block_n),
                    T.ceildiv(
                        meta[5] + positions_per_tile + meta[4] - meta[3] + window_size_right,
                        block_n,
                    ),
                )
            else:
                last = T.ceildiv(meta[4], block_n)
            meta[6] = T.max(1, last - meta[8])
            meta[7] = request[0]

        @T.prim_func
        def main(
            Q: T.Tensor([total_q, heads, dim], _FP8_DTYPE),  # type: ignore
            K: T.Tensor([total_kv, heads_kv, dim], _FP8_DTYPE),  # type: ignore
            V: T.Tensor([total_kv, heads_kv, dim], _FP8_DTYPE),  # type: ignore
            CuQ: T.Tensor([batch + 1], "int32"),  # type: ignore
            CuKV: T.Tensor([batch + 1], "int32"),  # type: ignore
            QD: T.Tensor([batch, heads_kv], accum_dtype),  # type: ignore
            KD: T.Tensor([batch, heads_kv], accum_dtype),  # type: ignore
            VD: T.Tensor([batch, heads_kv], accum_dtype),  # type: ignore
            RopeCos: T.Tensor([max_position, rope_half], rope_dtype),  # type: ignore
            RopeSin: T.Tensor([max_position, rope_half], rope_dtype),  # type: ignore
            Claim: T.Tensor([_CLAIM_SLOTS], "int32"),  # type: ignore
            Sinks: T.Tensor(
                [heads] if has_sinks else [total_q, heads, dim],
                "float32" if has_sinks else _FP8_DTYPE,
            ),
            O: T.Tensor([total_q, heads, dim], out_dtype),  # type: ignore
        ) -> None:
            # A CTA's items come off the claim counter, so the block index names nothing.
            with T.Kernel(num_ctas, threads=3 * WARPGROUP_THREADS) as _cta:
                q_shared_1 = T.alloc_shared([half_m, dim], _FP8_DTYPE)
                q_shared_2 = T.alloc_shared([half_m, dim], _FP8_DTYPE)
                k_smem = T.alloc_shared([stages, block_n, dim], _FP8_DTYPE)
                v_smem = T.alloc_shared([stages, dim, block_n], _FP8_DTYPE)
                o_shared_1 = T.alloc_shared([half_m, dim], out_dtype)
                o_shared_2 = T.alloc_shared([half_m, dim], out_dtype)
                ss_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ss_shared_2 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_1 = T.alloc_shared([half_m], accum_dtype)
                ls_shared_2 = T.alloc_shared([half_m], accum_dtype)
                tile_cum = T.alloc_shared([batch + 1], "int32")
                claimed = T.alloc_shared([1], "int32")
                acc_s_1 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_o_1 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_1 = T.alloc_fragment([half_m], accum_dtype)
                smp_1 = T.alloc_fragment([half_m], accum_dtype)
                ss_1 = T.alloc_fragment([half_m], accum_dtype)
                ssum_1 = T.alloc_fragment([half_m], accum_dtype)
                ls_1 = T.alloc_fragment([half_m], accum_dtype)
                acc_s_2 = T.alloc_fragment([half_m, block_n], accum_dtype)
                acc_o_2 = T.alloc_fragment([half_m, dim], accum_dtype)
                sm_2 = T.alloc_fragment([half_m], accum_dtype)
                smp_2 = T.alloc_fragment([half_m], accum_dtype)
                ss_2 = T.alloc_fragment([half_m], accum_dtype)
                ssum_2 = T.alloc_fragment([half_m], accum_dtype)
                ls_2 = T.alloc_fragment([half_m], accum_dtype)
                k_full = T.alloc_barrier([WARPGROUP_THREADS] * stages)
                k_empty = T.alloc_barrier([compute_threads] * stages)
                v_raw_full = T.alloc_barrier([WARPGROUP_THREADS] * stages)
                v_full = T.alloc_barrier([WARPGROUP_THREADS] * stages)
                v_empty = T.alloc_barrier([compute_threads] * stages)
                lo = T.alloc_local([1], "int32")
                hi = T.alloc_local([1], "int32")
                q_row = T.alloc_local([1], "int32")
                request = T.alloc_local([1], "int32")
                meta = T.alloc_local([9], "int32")
                T.annotate_layout(
                    {
                        q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                        q_shared_2: tilelang.layout.make_swizzled_layout(q_shared_2),
                        k_smem: tilelang.layout.make_swizzled_layout(k_smem),
                        acc_s_1: fa3_acc_fragment(block_n, 128),
                        acc_s_2: fa3_acc_fragment(block_n, 256),
                        sm_1: fa3_qk_row_fragment(128),
                        smp_1: fa3_qk_row_fragment(128),
                        ss_1: fa3_qk_row_fragment(128),
                        ssum_1: fa3_qk_row_fragment(128),
                        ls_1: fa3_qk_row_fragment(128),
                        sm_2: fa3_qk_row_fragment(256),
                        smp_2: fa3_qk_row_fragment(256),
                        ss_2: fa3_qk_row_fragment(256),
                        ssum_2: fa3_qk_row_fragment(256),
                        ls_2: fa3_qk_row_fragment(256),
                        acc_o_1: fa3_acc_fragment(dim, 128),
                        acc_o_2: fa3_acc_fragment(dim, 256),
                    }
                )
                q_tiling.cumsum_offsets(CuQ, tile_cum)
                T.sync_threads()
                tx = T.get_thread_binding()
                if tx < 128:
                    T.dec_max_nreg(24)
                    issued = T.alloc_var("int32", init=0)
                    folded = T.alloc_var("int32", init=0)
                    work = T.alloc_var("int32", init=0)
                    claim_item(claimed, Claim)
                    work = claimed[0]
                    while work < tile_cum[batch] * heads_kv:
                        locate(work, CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta)
                        head_kv = meta[0]
                        kv_start = meta[2]
                        first = meta[8]
                        for n_idx in T.Pipelined(meta[6], num_stages=0):
                            if issued >= stages:
                                T.barrier_wait(
                                    v_empty[issued % stages], ((issued // stages) % 2) ^ 1
                                )
                            if tx == 0:
                                T.mbarrier_expect_tx(v_raw_full[issued % stages], dim * block_n)
                                v_desc = T.create_tma_descriptor(
                                    TMA_DTYPE_UINT8,
                                    4,
                                    V.data,
                                    dim,
                                    heads_kv,
                                    total_kv,
                                    1,
                                    1,
                                    dim,
                                    heads_kv * dim,
                                    total_kv * heads_kv * dim,
                                    dim,
                                    1,
                                    block_n,
                                    1,
                                    1,
                                    1,
                                    1,
                                    1,
                                    TMA_INTERLEAVE_NONE,
                                    TMA_SWIZZLE_128B,
                                    TMA_L2_PROMOTION_128B,
                                    TMA_OOB_FILL_NONE,
                                )
                                T.call_extern(
                                    "handle",
                                    "tileops::fp8_tma_load_4d_ptx",
                                    v_desc,
                                    v_raw_full[issued % stages],
                                    T.access_ptr(v_smem[issued % stages, 0, 0], "w"),
                                    0,
                                    head_kv,
                                    kv_start + (first + n_idx) * block_n,
                                    0,
                                )
                            T.barrier_arrive(v_raw_full[issued % stages])
                            if issued >= stages:
                                T.barrier_wait(
                                    k_empty[issued % stages], ((issued // stages) % 2) ^ 1
                                )
                            T.tma_copy(
                                K[
                                    kv_start + (first + n_idx) * block_n : kv_start
                                    + (first + n_idx + 1) * block_n,
                                    head_kv,
                                    :,
                                ],
                                k_smem[issued % stages, :, :],
                                barrier=k_full[issued % stages],
                            )
                            T.barrier_arrive(k_full[issued % stages])
                            issued = issued + 1
                            # Both descriptors are out first, so this tile's transfers are
                            # in flight while the previous tile's value is transformed.
                            if issued - folded >= stages:
                                transpose_value_tile(
                                    folded % stages, v_smem, v_raw_full, v_full, (folded // stages) % 2
                                )  # fmt: skip
                                folded = folded + 1
                        # Every tile this item staged is turned into its operand before the
                        # item ends. Carrying one over would hold a compute warpgroup on
                        # ``v_full`` while this warpgroup waits at the claim, and the claim
                        # is what would have issued the tiles that released it.
                        while folded < issued:
                            transpose_value_tile(
                                folded % stages, v_smem, v_raw_full, v_full, (folded // stages) % 2
                            )  # fmt: skip
                            folded = folded + 1
                        claim_item(claimed, Claim)
                        work = claimed[0]
                    while folded < issued:
                        transpose_value_tile(
                            folded % stages, v_smem, v_raw_full, v_full, (folded // stages) % 2
                        )  # fmt: skip
                        folded = folded + 1
                    # A CTA touches the counter for the last time before it arrives here,
                    # so the CTA that arrives last leaves both slots zeroed for the next
                    # launch and no caller has to clear them.
                    if tx == 0:
                        arrived = T.atomic_add(Claim[1], 1, return_prev=True)
                        if arrived == num_ctas - 1:
                            Claim[0] = 0
                            Claim[1] = 0
                elif tx < 256:
                    T.inc_max_nreg(240)
                    gi_k = T.alloc_var("int32", init=0)
                    gi_v = T.alloc_var("int32", init=0)
                    work = T.alloc_var("int32", init=0)
                    claim_item(claimed, Claim)
                    work = claimed[0]
                    while work < tile_cum[batch] * heads_kv:
                        locate(work, CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta)
                        head_kv = meta[0]
                        head_base = head_kv * groups
                        q_at = meta[1] + 0
                        kv_start = meta[2]
                        q_len = meta[3]
                        kv_len = meta[4]
                        pos_base = meta[5] + 0
                        eff = meta[6]
                        first = meta[8]
                        causal_offset = kv_len - q_len
                        qk_descale = T.alloc_var(
                            accum_dtype, init=QD[meta[7], head_kv] * KD[meta[7], head_kv]
                        )
                        value_descale = T.alloc_var(accum_dtype, init=VD[meta[7], head_kv])
                        # The packed tile is a gather, so it is staged with ``cp.async``
                        # rather than one bulk transfer. Its rows run head by head: a head's
                        # rows are then one contiguous run of the request's tokens, which is
                        # the widest region one transfer can carry. Rows past the call's last
                        # token are predicated away and never stored.
                        for g in T.serial(groups):
                            T.async_copy(
                                Q[q_at : q_at + positions_per_half, head_base + g, :],
                                q_shared_1[
                                    g * positions_per_half : (g + 1) * positions_per_half, :
                                ],
                            )
                        T.ptx_wait_group(0)
                        T.sync_threads(barrier_id=3, arrive_count=128)
                        T.fence_proxy_async()
                        if fuse_rope:
                            # Query token i of a request sits at position kv_len - q_len + i.
                            rotate_query_tile(
                                q_shared_1, RopeCos, RopeSin, causal_offset + pos_base
                            )
                        T.clear(acc_o_1)
                        T.clear(ls_1)
                        T.fill(sm_1, -T.infinity(accum_dtype))
                        for n_idx in T.Pipelined(eff, num_stages=0):
                            # The turn to issue passes between the compute warpgroups, so one
                            # folds its score tile while the other issues its contraction.
                            T.sync_threads(1, compute_threads)
                            T.barrier_wait(k_full[gi_k % stages], (gi_k // stages) % 2)
                            T.call_extern(
                                "handle",
                                "tileops::fp8_qk_cute_grouped_fa3_raw_64x224x128",
                                q_shared_1.access_ptr("r"),
                                T.access_ptr(k_smem[gi_k % stages, 0, 0], "r"),
                                acc_s_1.data,
                            )
                            T.named_barrier_arrive(2, compute_threads)
                            if n_idx > 0:
                                # The previous tile's value contraction runs under this tile's
                                # key contraction; only the older of the two is waited on.
                                T.wait_wgmma(1)
                                T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                                T.barrier_arrive(v_empty[gi_v % stages])
                                gi_v = gi_v + 1
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_s_1, num_regs=112)
                            T.barrier_arrive(k_empty[gi_k % stages])
                            gi_k = gi_k + 1
                            if use_softcap:
                                T.call_extern(
                                    "handle",
                                    "tileops::fp8_apply_softcap_raw_acc_64x224",
                                    acc_s_1.data,
                                    qk_descale * attention_scale / softcap,
                                )
                            # Only a tile the request's end or a mask boundary cuts
                            # needs the per-element pass; one wholly inside the
                            # visible range keeps every score it computed.
                            tile_end = (first + n_idx + 1) * block_n
                            cut = tile_end > kv_len
                            if is_causal:
                                cut = cut | (tile_end > causal_offset + pos_base + 1)
                            elif has_right:
                                cut = cut | (
                                    tile_end > causal_offset + pos_base + window_size_right + 1
                                )
                            if has_left:
                                cut = cut | (
                                    (first + n_idx) * block_n
                                    < causal_offset
                                    + pos_base
                                    + positions_per_half
                                    - window_size_left
                                )
                            if cut:
                                for i, j in T.Parallel(half_m, block_n):
                                    kv_pos = (first + n_idx) * block_n + fa3_qk_acc_column(j)
                                    limit = causal_offset + pos_base + i % positions_per_half
                                    if is_causal:
                                        visible = (kv_pos < kv_len) & (kv_pos <= limit)
                                    elif has_right:
                                        visible = (kv_pos < kv_len) & (
                                            kv_pos <= limit + window_size_right
                                        )
                                    else:
                                        visible = kv_pos < kv_len
                                    if has_left:
                                        visible = visible & (kv_pos >= limit - window_size_left)
                                    acc_s_1[i, j] = T.if_then_else(
                                        visible, acc_s_1[i, j], -T.infinity(accum_dtype)
                                    )
                            online_softmax(acc_s_1, sm_1, smp_1, ss_1, ssum_1, ls_1, qk_descale)
                            T.copy(ss_1, ss_shared_1)
                            T.call_extern(
                                "handle",
                                "tileops::fp8_fa3_raw_acc_rescale_keep_ptx_layout_64x128",
                                acc_o_1.data,
                                ss_shared_1.access_ptr("r"),
                            )
                            T.barrier_wait(v_full[gi_v % stages], (gi_v // stages) % 2)
                            pv_tile(
                                acc_s_1,
                                v_smem,
                                gi_v % stages,
                                acc_o_1,
                                meta[6],
                            )
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(acc_o_1, num_regs=64)
                        T.barrier_arrive(v_empty[gi_v % stages])
                        gi_v = gi_v + 1
                        # The lane partials of the row sum combine once, here.
                        for i in T.Parallel(half_m):
                            ls_1[i] = ls_1[i] + T.shfl_xor(ls_1[i], 1)
                            ls_1[i] = ls_1[i] + T.shfl_xor(ls_1[i], 2)
                        if has_sinks:
                            sink_scale(ls_1, sm_1, Sinks, ss_1, head_base)
                            T.copy(ss_1, ss_shared_1)
                            T.sync_threads(barrier_id=3, arrive_count=128)
                            T.call_extern(
                                "handle",
                                "tileops::fp8_fa3_raw_acc_rescale_keep_ptx_layout_64x128",
                                acc_o_1.data,
                                ss_shared_1.access_ptr("r"),
                            )
                        T.copy(ls_1, ls_shared_1)
                        T.call_extern(
                            "handle",
                            "tileops::fp8_fa3_raw_acc_finalize_store_smem_cute_64x128",
                            acc_o_1.data,
                            ls_shared_1.access_ptr("r"),
                            4,
                            value_descale,
                            o_shared_1.access_ptr("w"),
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=3, arrive_count=128)
                        if pos_base + positions_per_half <= q_len:
                            T.call_extern(
                                "handle",
                                packed_store,
                                o_shared_1.access_ptr("r"),
                                T.address_of(O[q_at, head_base, 0]),
                                heads * dim,
                            )
                        elif pos_base < q_len:
                            T.call_extern(
                                "handle",
                                packed_store_tail,
                                o_shared_1.access_ptr("r"),
                                T.address_of(O[q_at, head_base, 0]),
                                heads * dim,
                                q_len - pos_base,
                            )
                        claim_item(claimed, Claim)
                        work = claimed[0]
                else:
                    T.inc_max_nreg(240)
                    gi_k = T.alloc_var("int32", init=0)
                    gi_v = T.alloc_var("int32", init=0)
                    work = T.alloc_var("int32", init=0)
                    claim_item(claimed, Claim)
                    work = claimed[0]
                    T.named_barrier_arrive(
                        1, compute_threads
                    )  # let warpgroup 0 take the first turn
                    while work < tile_cum[batch] * heads_kv:
                        locate(work, CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta)
                        head_kv = meta[0]
                        head_base = head_kv * groups
                        q_at = meta[1] + positions_per_half
                        kv_start = meta[2]
                        q_len = meta[3]
                        kv_len = meta[4]
                        pos_base = meta[5] + positions_per_half
                        eff = meta[6]
                        first = meta[8]
                        causal_offset = kv_len - q_len
                        qk_descale = T.alloc_var(
                            accum_dtype, init=QD[meta[7], head_kv] * KD[meta[7], head_kv]
                        )
                        value_descale = T.alloc_var(accum_dtype, init=VD[meta[7], head_kv])
                        # The packed tile is a gather, so it is staged with ``cp.async``
                        # rather than one bulk transfer. Its rows run head by head: a head's
                        # rows are then one contiguous run of the request's tokens, which is
                        # the widest region one transfer can carry. Rows past the call's last
                        # token are predicated away and never stored.
                        for g in T.serial(groups):
                            T.async_copy(
                                Q[q_at : q_at + positions_per_half, head_base + g, :],
                                q_shared_2[
                                    g * positions_per_half : (g + 1) * positions_per_half, :
                                ],
                            )
                        T.ptx_wait_group(0)
                        T.sync_threads(barrier_id=4, arrive_count=128)
                        T.fence_proxy_async()
                        if fuse_rope:
                            # Query token i of a request sits at position kv_len - q_len + i.
                            rotate_query_tile(
                                q_shared_2, RopeCos, RopeSin, causal_offset + pos_base
                            )
                        T.clear(acc_o_2)
                        T.clear(ls_2)
                        T.fill(sm_2, -T.infinity(accum_dtype))
                        for n_idx in T.Pipelined(eff, num_stages=0):
                            # The turn to issue passes between the compute warpgroups, so one
                            # folds its score tile while the other issues its contraction.
                            T.sync_threads(2, compute_threads)
                            T.barrier_wait(k_full[gi_k % stages], (gi_k // stages) % 2)
                            T.call_extern(
                                "handle",
                                "tileops::fp8_qk_cute_grouped_fa3_raw_64x224x128",
                                q_shared_2.access_ptr("r"),
                                T.access_ptr(k_smem[gi_k % stages, 0, 0], "r"),
                                acc_s_2.data,
                            )
                            T.named_barrier_arrive(1, compute_threads)
                            if n_idx > 0:
                                # The previous tile's value contraction runs under this tile's
                                # key contraction; only the older of the two is waited on.
                                T.wait_wgmma(1)
                                T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                                T.barrier_arrive(v_empty[gi_v % stages])
                                gi_v = gi_v + 1
                            T.wait_wgmma(0)
                            T.warpgroup_fence_operand(acc_s_2, num_regs=112)
                            T.barrier_arrive(k_empty[gi_k % stages])
                            gi_k = gi_k + 1
                            if use_softcap:
                                T.call_extern(
                                    "handle",
                                    "tileops::fp8_apply_softcap_raw_acc_64x224",
                                    acc_s_2.data,
                                    qk_descale * attention_scale / softcap,
                                )
                            # Only a tile the request's end or a mask boundary cuts
                            # needs the per-element pass; one wholly inside the
                            # visible range keeps every score it computed.
                            tile_end = (first + n_idx + 1) * block_n
                            cut = tile_end > kv_len
                            if is_causal:
                                cut = cut | (tile_end > causal_offset + pos_base + 1)
                            elif has_right:
                                cut = cut | (
                                    tile_end > causal_offset + pos_base + window_size_right + 1
                                )
                            if has_left:
                                cut = cut | (
                                    (first + n_idx) * block_n
                                    < causal_offset
                                    + pos_base
                                    + positions_per_half
                                    - window_size_left
                                )
                            if cut:
                                for i, j in T.Parallel(half_m, block_n):
                                    kv_pos = (first + n_idx) * block_n + fa3_qk_acc_column(j)
                                    limit = causal_offset + pos_base + i % positions_per_half
                                    if is_causal:
                                        visible = (kv_pos < kv_len) & (kv_pos <= limit)
                                    elif has_right:
                                        visible = (kv_pos < kv_len) & (
                                            kv_pos <= limit + window_size_right
                                        )
                                    else:
                                        visible = kv_pos < kv_len
                                    if has_left:
                                        visible = visible & (kv_pos >= limit - window_size_left)
                                    acc_s_2[i, j] = T.if_then_else(
                                        visible, acc_s_2[i, j], -T.infinity(accum_dtype)
                                    )
                            online_softmax(acc_s_2, sm_2, smp_2, ss_2, ssum_2, ls_2, qk_descale)
                            T.copy(ss_2, ss_shared_2)
                            T.call_extern(
                                "handle",
                                "tileops::fp8_fa3_raw_acc_rescale_keep_ptx_layout_64x128",
                                acc_o_2.data,
                                ss_shared_2.access_ptr("r"),
                            )
                            T.barrier_wait(v_full[gi_v % stages], (gi_v // stages) % 2)
                            pv_tile(
                                acc_s_2,
                                v_smem,
                                gi_v % stages,
                                acc_o_2,
                                meta[6],
                            )
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(acc_o_2, num_regs=64)
                        T.barrier_arrive(v_empty[gi_v % stages])
                        gi_v = gi_v + 1
                        # The lane partials of the row sum combine once, here.
                        for i in T.Parallel(half_m):
                            ls_2[i] = ls_2[i] + T.shfl_xor(ls_2[i], 1)
                            ls_2[i] = ls_2[i] + T.shfl_xor(ls_2[i], 2)
                        if has_sinks:
                            sink_scale(ls_2, sm_2, Sinks, ss_2, head_base)
                            T.copy(ss_2, ss_shared_2)
                            T.sync_threads(barrier_id=4, arrive_count=128)
                            T.call_extern(
                                "handle",
                                "tileops::fp8_fa3_raw_acc_rescale_keep_ptx_layout_64x128",
                                acc_o_2.data,
                                ss_shared_2.access_ptr("r"),
                            )
                        T.copy(ls_2, ls_shared_2)
                        T.call_extern(
                            "handle",
                            "tileops::fp8_fa3_raw_acc_finalize_store_smem_cute_64x128",
                            acc_o_2.data,
                            ls_shared_2.access_ptr("r"),
                            4,
                            value_descale,
                            o_shared_2.access_ptr("w"),
                        )
                        T.fence_proxy_async()
                        T.sync_threads(barrier_id=4, arrive_count=128)
                        if pos_base + positions_per_half <= q_len:
                            T.call_extern(
                                "handle",
                                packed_store,
                                o_shared_2.access_ptr("r"),
                                T.address_of(O[q_at, head_base, 0]),
                                heads * dim,
                            )
                        elif pos_base < q_len:
                            T.call_extern(
                                "handle",
                                packed_store_tail,
                                o_shared_2.access_ptr("r"),
                                T.address_of(O[q_at, head_base, 0]),
                                heads * dim,
                                q_len - pos_base,
                            )
                        claim_item(claimed, Claim)
                        work = claimed[0]

        return main

    return func


class GQAVarlenFP8FwdKernel(VarlenKernel):
    """Packed prefill over FP8 Q/K/V with one dequantization scale per request and KV head."""

    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call: AttentionCall) -> bool:
        return cls.refusal(call) is None

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Why this class does not serve *call*, or ``None`` when it does."""
        if not call.is_fp8:
            return "requires float8_e4m3fn q, k and v"
        if call.dim != 128:
            return "requires head dimension 128"
        if call.fuse_rope:
            return "does not serve fused RoPE"
        return None

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        window_size_left: int = -1,
        window_size_right: int = -1,
        accum_dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        has_sinks: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        if dtype not in ATTENTION_DTYPES:
            raise ValueError("FP8 packed-varlen GQA outputs float16 or bfloat16")
        self._rope_placeholder: Optional[torch.Tensor] = None
        super().__init__(
            batch,
            heads,
            heads_kv,
            dim,
            is_causal,
            dtype,
            sm_scale=sm_scale,
            softcap=softcap,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            accum_dtype=accum_dtype,
            config=config,
            tune=tune,
            fuse_rope=fuse_rope,
            max_position=max_position,
            rotary_dim=rotary_dim,
            rope_layout=rope_layout,
            has_sinks=has_sinks,
            device_index=device_index,
        )

    def _rope_tables(
        self, rope_cos: Optional[torch.Tensor], rope_sin: Optional[torch.Tensor]
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """The rotation tables, or the placeholder a call without rotation passes instead."""
        if self.key_rope is None:
            if rope_cos is not None or rope_sin is not None:
                raise ValueError(f"{type(self).__name__} was not built for fused RoPE")
            if self._rope_placeholder is None:
                self._rope_placeholder = torch.zeros(
                    self.max_position,
                    _ROPE_PLACEHOLDER_WIDTH,
                    dtype=self.dtype,
                    device=torch.device("cuda", self.device_index or 0),
                )
            return self._rope_placeholder, self._rope_placeholder
        if rope_cos is None or rope_sin is None:
            raise ValueError("fused RoPE requires rope_cos and rope_sin")
        return rope_cos, rope_sin

    @property
    def rotated_dtype_str(self) -> str:
        """The rotation reads and writes the FP8 inputs, not the 16-bit output.

        Rotating is linear, so the stored values are rotated under the per-request scales
        they already carry and the attention program is the one a call without RoPE runs.
        """
        return _FP8_DTYPE

    def _make_kernel(self) -> Callable:
        return _gqa_varlen_fp8_fwd_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            self.window_size_left,
            self.window_size_right,
            self.has_sinks,
        )

    def _make_supply_prog(self) -> Callable:
        """Supply FP8 packed tensors, offsets and scales while autotuning."""
        from tilelang.utils.device import get_current_device

        batch, heads, heads_kv, dim = self.batch, self.heads, self.heads_kv, self.dim
        # Two 128-token tiles exercise the tiled loop while keeping every candidate
        # probe bounded; this is a tuning point, not a claim about any packed total.
        tokens_per_request = 256
        total = batch * tokens_per_request

        def supply_prog(params):
            if len(params) != 9:
                raise RuntimeError(
                    f"autotuning {type(self).__name__} expects q, k, v, two cumulative-length "
                    f"inputs, three scales and a sink slot, got {len(params)} parameters"
                )
            device = get_current_device()
            cu_seqlens = torch.arange(
                0, total + 1, tokens_per_request, dtype=torch.int32, device=device
            )
            fp8 = [
                (torch.randn(total, h, dim, device=device) * 0.2).to(torch.float8_e4m3fn)
                for h in (heads, heads_kv, heads_kv)
            ]
            scales = [
                torch.ones(batch, heads_kv, dtype=torch.float32, device=device) for _ in range(3)
            ]
            return [
                *fp8,
                cu_seqlens,
                cu_seqlens.clone(),
                *scales,
                torch.zeros(heads, device=device),
            ]

        return supply_prog

    @property
    def default_config(self) -> dict:
        """The first candidate the device's shared memory holds."""
        wide = {"block_m": 128, "block_n": 128, "num_stages": 2, "threads": 256}
        candidates = [
            wide,
            {**wide, "num_stages": 1},
            {"block_m": 64, "block_n": 64, "num_stages": 1, "threads": 128},
        ]
        cap = get_shared_memory_optin(self.device_index)
        return next((c for c in candidates if self._shared_bytes(c) <= cap), candidates[-1])

    def _shared_bytes(self, config: dict) -> int:
        """Shared memory a *config* allocates, buffer by buffer as TileLang does."""
        block_m, block_n = config["block_m"], config["block_n"]
        stages = config["num_stages"]
        tile = block_n * self.dim
        buffers = [
            block_m * self.dim,
            stages * tile,
            stages * tile,
            block_m * block_n,
            block_m * self.dim * self.dtype.itemsize,
            4 * (self.batch + 1),
        ]
        # TileLang stages the score tile through shared memory when a warpgroup
        # holds fewer than one WGMMA atom's rows.
        if block_m // (config["threads"] // WARPGROUP_THREADS) < WGMMA_ROWS:
            buffers += [4 * config["threads"], 4 * config["threads"]]
        align = SHARED_BUFFER_ALIGN_BYTES
        return sum(-(-b // align) * align for b in buffers)

    @property
    def autotune_configs(self) -> list[dict]:
        configs = itertools.product([64, 128], [64, 128, 192], [1, 2, 3], [128, 256])
        return [
            {"block_m": c[0], "block_n": c[1], "num_stages": c[2], "threads": c[3]} for c in configs
        ]

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
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q_scale is None or k_scale is None or v_scale is None:
            raise ValueError(f"{type(self).__name__} requires q_scale, k_scale and v_scale")
        if rope_cos is not None or rope_sin is not None:
            raise ValueError(f"{type(self).__name__} does not serve fused RoPE")
        return self.kernel(
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
        )(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            q_scale,
            k_scale,
            v_scale,
            sinks if sinks is not None else q,
        )


class GQAVarlenFP8WSFwdKernel(GQAVarlenFP8FwdKernel):
    """SM90 warp-specialized FP8 packed prefill: a persistent CTA per SM.

    Each CTA walks the call's query tiles at a fixed stride, so its transfer warpgroup
    and its two compute warpgroups reach the same item without a shared claim; the
    transfer warpgroup streams that item's keys and values through mbarriers and the
    compute warpgroups take 64 rows each.
    """

    preferred_over: ClassVar[frozenset[str]] = frozenset({"gqa_varlen_fp8"})
    # Which CTA runs which item follows the order the claims land in, which is not fixed
    # between launches. An item owns a disjoint slice of the output, so the result does
    # not depend on that order.
    _claim: Optional[torch.Tensor] = None
    # The staged FA3 buffers take 217 KB of the 227 KB an SM90 block may use, and the
    # per-request tile prefix takes four bytes a request out of what is left. Recompute
    # it from the buffer list in the program if any shared buffer grows.
    _MAX_BATCH: int = 2048

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Why this class does not serve *call*, or ``None`` when it does."""
        base = GQAVarlenFP8FwdKernel.refusal(call)
        # The rotation is this class's alone: it rotates the query tile inside the CTA
        # that owns it, which the pipelined program has no warpgroup split to do.
        if base is not None and base != "does not serve fused RoPE":
            return base
        if call.empty_kv:
            return "requires a key in at least one request"
        if WGMMA_ROWS % (call.heads // call.heads_kv):
            return (
                "packs a KV group's query heads into one tile, so the group must divide "
                f"the {WGMMA_ROWS} rows a warpgroup holds"
            )
        if call.batch > cls._MAX_BATCH:
            return (
                f"holds the per-request tile prefix in shared memory, so batch <= {cls._MAX_BATCH}"
            )
        return None

    def _make_kernel(self, can_promote: bool = False) -> Callable:
        return _gqa_varlen_fp8_ws_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            self.window_size_left,
            self.window_size_right,
            self.key_rope is not None,
            self.max_position,
            self.rotary_dim,
            self.rope_layout,
            self.rope_table_dtype_str,
            get_sm_count(self.device_index),
            self.has_sinks,
            can_promote,
        )

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> Optional[list[dict]]:
        return None

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
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q_scale is None or k_scale is None or v_scale is None:
            raise ValueError(f"{type(self).__name__} requires q_scale, k_scale and v_scale")
        cos, sin = self._rope_tables(rope_cos, rope_sin)
        if self.key_rope is not None:
            k = self.key_rope(k, cu_seqlens_kv, rope_cos, rope_sin)
        out = torch.empty(q.shape, dtype=self.dtype, device=q.device)
        if self._claim is None:
            self._claim = torch.zeros(_CLAIM_SLOTS, dtype=torch.int32, device=q.device)
        # Packed total is a host-visible upper bound on each request's length.
        # Short calls compile out the promotion branch altogether; long totals
        # still check each work item's actual visible tile count on the device.
        kernel = self._make_kernel(True) if k.shape[0] > 15 * 224 else self.kernel
        kernel()(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            q_scale,
            k_scale,
            v_scale,
            cos,
            sin,
            self._claim,
            sinks if sinks is not None else q,
            out,
        )
        return out
