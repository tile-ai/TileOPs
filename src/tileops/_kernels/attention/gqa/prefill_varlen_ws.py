"""SM90 warp-specialized packed variable-length GQA prefill kernel."""

from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops._csrc import csrc_include
from tileops._kernels.attention.call_spec import ATTENTION_DTYPES
from tileops._kernels.attention.online_softmax import finite_max_floor, make_sink_scale
from tileops._kernels.attention.varlen import VarlenKernel
from tileops._kernels.attention.varlen_rope import make_varlen_query_rope
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import LOG2E
from tileops._kernels.grouped_tiling import GroupTiling
from tileops._utils import get_sm_count

__all__ = ["GQAPrefillVarlenWSFwdKernel", "GQAVarlenPackedWSFwdKernel"]


@target_cache(maxsize=32)
# No out_idx: resolving a symbolic output shape costs about 170 us of host time per call.
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        # A packed tile's row r stores position (64 wg + r) // groups of head r % groups,
        # which is injective in r; the checker cannot prove it and reports the store as a race.
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
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
        *csrc_include("attn_wgmma.h"),
    ],
)
def _gqa_prefill_varlen_ws_kernel(
    batch, heads, heads_kv, dim, is_causal, sm_scale, softcap, dtype, block_n, stages, num_ctas,
    fuse_rope=False, max_position=1, rotary_dim=0, rope_layout="neox", rope_dtype="",
    window_left=-1, window_right=-1, has_sinks=False, pack=False, chunked=False,
    wide=True, tail_n=0, tail_p=1,
):  # fmt: skip
    """A persistent CTA per SM: a TMA producer warp claims work, two consumer warpgroups run it.

    With *pack*, a request's tiles may hold its KV head's group position by position."""
    score_scale = (1.0 / dim) ** 0.5 if sm_scale is None else sm_scale
    has_window = window_left >= 0 or window_right >= 0
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    max_floor = finite_max_floor(scale)
    groups = heads // heads_kv
    accum = "float"
    half = 64  # rows per consumer warpgroup: one m64 WGMMA
    sink_scale = make_sink_scale(scale, accum, half)
    # Key steps over 128 channels issue their products through attn_wgmma.h.
    step_wgmma = dim == 128 and block_n == 128
    # A K or V stage's size in descriptor address units (16 bytes).
    stage_desc = block_n * dim * 2 // 16
    wgmma_dtype = "tl::DataType::kBFloat16" if dtype == "bfloat16" else "tl::DataType::kFloat16"
    block_m = 2 * half
    consumers = 256  # threads of the two consumer warpgroups
    policy = T.GemmWarpPolicy.FullRow
    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    q_tiling = GroupTiling(batch, block_m)
    # Quadratic ranking prioritizes long key scans with few query tiles.
    order_max_batch = 64
    ordered = is_causal and not has_window and batch <= order_max_batch
    # Packed row r of tile t maps to position (128*t + r)//groups, head r%groups.
    # Whole positions must fit each warpgroup for contiguous TMA query loads.
    packs = pack and groups > 1 and half % groups == 0
    # The packed program ranks with one thread a request, over keys it reads once.
    if packs:
        ordered = is_causal and not has_window and batch <= 2 * order_max_batch
    positions_per_wg = half // groups if packs else half
    sink_scale_packed = make_sink_scale(scale, accum, half, head_group=groups) if packs else None
    # Metadata: head, Q/KV starts, Q/KV lengths, first query, key-tile count,
    # then packed flag and first key tile. Packed heads index KV groups.
    packed_at = 7
    first_at = packed_at + int(packs)
    meta_count = first_at + int(has_window or packs)
    # Split metadata: part count, (request, KV head) group, part index, base slot, void flag.
    # Part p uses workspace slot base + p*heads_kv; the last arrival merges it. A void claim
    # (flag 2) is a tail part of an item too short to split: the producer skips it.
    splits = packs and ordered
    # Splitting a request by chunk of keys: compiled only for a call that may leave SMs idle.
    chunks = splits and chunked
    parts_at = meta_count
    if splits:
        meta_count += 5
    # Tail split: the last tail_n wide claims of a call whose items fill the SMs each scan
    # their keys in tail_p parts, merged like a split request's (producer decides).
    tail_split = splits and tail_n > 0 and tail_p > 1
    # Each part scans four key tiles or more, enough to outweigh merging its partial.
    tail_min = 4 * tail_p
    # Per-request item prefix for tile/head decoding.
    item_tiling = GroupTiling(batch, 1)

    def packed_request(q_len):
        """Whether a request of *q_len* queries packs: it needs fewer packed tiles than
        one-head tiles; a tie keeps one head a tile."""
        if not packs:
            return False
        return T.ceildiv(q_len * groups, block_m) < groups * T.ceildiv(q_len, block_m)

    def request_tiles(q_len, packed):
        """Query tiles of a request; packed, the item heads are KV heads."""
        if not packs:
            return T.ceildiv(q_len, block_m)
        return T.if_then_else(packed, T.ceildiv(q_len * groups, block_m), T.ceildiv(q_len, block_m))

    def request_items(q_len):
        packed = packed_request(q_len)
        if not packs:
            return request_tiles(q_len, packed) * heads
        return request_tiles(q_len, packed) * T.if_then_else(packed, heads_kv, heads)

    def request_parts(q_len, kv_len, chunk):
        """Parts each item of a request splits into: a packed request whose rows fit one
        tile scans every key, in parts of *chunk* key tiles when *chunk* is set."""
        if not chunks:
            return 1
        one_tile = packed_request(q_len) & (q_len * groups <= block_m) & (chunk > 0)
        return T.if_then_else(
            one_tile, T.ceildiv(T.max(1, T.ceildiv(kv_len, block_n)), T.max(chunk, 1)), 1
        )

    def tile_positions(packed):
        """Query positions a tile spans."""
        return T.if_then_else(packed, block_m // groups, block_m) if packs else block_m

    def offsets_items(offsets, r):
        return request_items(offsets[r + 1] - offsets[r])

    items_prefix = item_tiling.cumsum_by(offsets_items)
    rope_half = rotary_dim // 2
    table_dtype = rope_dtype or dtype
    # Vectorized RoPE loads precede arithmetic to overlap memory latency.
    rope_vec = 8  # 16-bit channels in one 16-byte shared access
    neox = rope_layout == "neox"
    # Channels a run of one rotation pairs with another run: neox pairs the halves of the
    # rotated width, interleaved the neighbors inside one run.
    rope_width = rope_half if neox else rotary_dim
    rope_runs = rope_width // rope_vec
    # The runs must tile the width and split evenly over a warpgroup's 128 threads.
    vector_rope = fuse_rope and rope_runs * rope_vec == rope_width and half * rope_runs % 128 == 0
    # Table entries a run reads: one a channel under neox, one a channel pair interleaved.
    table_vec = rope_vec if neox else rope_vec // 2
    rope_units = half * rope_runs // 128  # runs each consumer thread moves
    rotate_query_tile = (
        make_varlen_query_rope(
            half, rotary_dim, rope_layout, max_position, dtype, rope_dtype, tile_axes=2
        )
        if fuse_rope and not vector_rope
        else None
    )
    rotate_packed_tile = (
        make_varlen_query_rope(
            half,
            rotary_dim,
            rope_layout,
            max_position,
            dtype,
            rope_dtype,
            tile_axes=2,
            rows_per_position=groups,
        )  # fmt: skip
        if fuse_rope and not vector_rope and packs
        else None
    )

    @T.macro
    def load_table_run(dst, at, table, pos, freq0):
        # The tables are in the call's 16-bit dtype, so a run's entries are one access.
        for v in T.vectorized(table_vec):
            dst[at + v] = table[pos, freq0 + v]

    @T.macro
    def rotate_half_tile(Qs, rope_cos, rope_sin, base, slot, wg, xa, xb, ca, sa, packed=False):
        """Rotate warpgroup *wg*'s 64 rows of Qs[slot] in place; row i sits at base + i, or at
        base + i // groups when the tile packs."""
        lane = T.get_thread_binding(0)
        for u in T.unroll(rope_units):
            idx = u * 128 + lane
            i = idx // rope_runs
            ch = idx % rope_runs
            # Rows past the request hold padding whose rotation is never read.
            pos = T.max(
                0,
                T.min(
                    base + (T.if_then_else(packed, i // groups, i) if packs else i),
                    max_position - 1,
                ),
            )
            for v in T.vectorized(rope_vec):
                xa[u * rope_vec + v] = Qs[slot, wg, i, ch * rope_vec + v]
            if neox:
                for v in T.vectorized(rope_vec):
                    xb[u * rope_vec + v] = Qs[slot, wg, i, rope_half + ch * rope_vec + v]
            load_table_run(ca, u * table_vec, rope_cos, pos, ch * table_vec)
            load_table_run(sa, u * table_vec, rope_sin, pos, ch * table_vec)
        for u in T.unroll(rope_units):
            idx = u * 128 + lane
            i = idx // rope_runs
            ch = idx % rope_runs
            if neox:
                for v in T.unroll(rope_vec):
                    x0 = T.Cast(accum, xa[u * rope_vec + v])
                    x1 = T.Cast(accum, xb[u * rope_vec + v])
                    c = T.Cast(accum, ca[u * table_vec + v])
                    s = T.Cast(accum, sa[u * table_vec + v])
                    xa[u * rope_vec + v] = T.Cast(dtype, x0 * c - x1 * s)
                    xb[u * rope_vec + v] = T.Cast(dtype, x1 * c + x0 * s)
                for v in T.vectorized(rope_vec):
                    Qs[slot, wg, i, rope_half + ch * rope_vec + v] = xb[u * rope_vec + v]
            else:
                for f in T.unroll(table_vec):
                    x0 = T.Cast(accum, xa[u * rope_vec + 2 * f])
                    x1 = T.Cast(accum, xa[u * rope_vec + 2 * f + 1])
                    c = T.Cast(accum, ca[u * table_vec + f])
                    s = T.Cast(accum, sa[u * table_vec + f])
                    xa[u * rope_vec + 2 * f] = T.Cast(dtype, x0 * c - x1 * s)
                    xa[u * rope_vec + 2 * f + 1] = T.Cast(dtype, x1 * c + x0 * s)
            for v in T.vectorized(rope_vec):
                Qs[slot, wg, i, ch * rope_vec + v] = xa[u * rope_vec + v]

    @T.macro
    def desc_of(buf, offset, leading_byte_offset):
        return T.call_extern(
            "uint64",
            f"tileops::attn_desc<{leading_byte_offset}>",
            buf.access_ptr("r", offset=offset),
        )

    @T.macro
    def apply_softcap(acc_s):
        for i, j in T.Parallel(half, block_n):
            capped = T.cast(softcap, accum) * T.tanh(
                acc_s[i, j] * T.cast(score_scale / softcap, accum)
            )
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum), -T.infinity(accum), capped
            )

    def position(row, qpos, i):
        """The request position of the warpgroup's row i: row + i, or qpos[i] in the packed
        program."""
        return row + i if qpos is None else qpos[i]

    @T.macro
    def mask(acc_s, key0, row, offset, kv_len, qpos=None):
        for i, j in T.Parallel(half, block_n):
            visible = key0 + j < kv_len
            if is_causal:
                visible = visible & (key0 + j <= position(row, qpos, i) + offset)
            if window_left >= 0:
                visible = visible & (key0 + j >= position(row, qpos, i) + offset - window_left)
            if window_right >= 0:
                visible = visible & (key0 + j <= position(row, qpos, i) + offset + window_right)
            acc_s[i, j] = T.if_then_else(visible, acc_s[i, j], -T.infinity(accum))

    @T.macro
    def row_max(acc_s, red, sm):
        """sm = max(sm, row max): per-thread partials over the 16 column groups, then one
        cross-thread reduce."""
        T.reduce_max(T.reshape(acc_s, [half, block_n // 8, 8]), red, dim=1, clear=True)
        T.reduce_max(red, sm, dim=1, clear=False, batch=2)

    @T.macro
    def softmax_step(acc_s, sm, smp, alpha, ss, red):
        """Fold one score tile into the running max, rescale factor, and row sums."""
        if use_softcap:
            apply_softcap(acc_s)
        T.copy(sm, smp)
        row_max(acc_s, red, sm)
        for i in T.Parallel(half):
            sm[i] = T.if_then_else((sm[i] - smp[i]) * scale > 8.0, sm[i], smp[i])
            alpha[i] = 1.0
            if sm[i] != smp[i]:
                alpha[i] = T.exp2(smp[i] * scale - sm[i] * scale)
        for i, j in T.Parallel(half, block_n):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
        T.reduce_sum(acc_s, ss, dim=1, batch=2)

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
        red,
        logsum,
        my_bar,
        nxt_bar,
        k,
        n,
        row,
        causal_offset,
        kv_len,
        tail: bool,
        qpos=None,
        descs=None,
    ):
        """QK of KV tile k overlapped with PV of tile k - 1; n counts tiles across work items."""
        sk = n % stages
        svp = (n - 1) % stages
        if step_wgmma:
            # The tile's descriptors of stage 0 plus this step's stage offset.
            qd, kd0, vd0 = descs
            kd = kd0 + T.cast(sk, "uint64") * stage_desc
            vd = vd0 + T.cast(svp, "uint64") * stage_desc
        T.sync_threads(my_bar, consumers)
        T.mbarrier_wait_parity(kready[sk], (n // stages) % 2)
        if step_wgmma:
            T.call_extern(
                "handle", f"tileops::attn_qk_64x128_d128<{wgmma_dtype}>", acc_s.data, qd, kd
            )
        else:
            T.wgmma_gemm(
                q_tile, Ks[sk, :, :], acc_s, transpose_B=True, policy=policy, clear_accum=True
            )
        rescale = T.alloc_var("int32", init=0)
        for i in T.Parallel(half):
            if alpha[i] != 1.0:
                rescale = 1
        if rescale == 1:
            for i, j in T.Parallel(half, dim):
                acc_o[i, j] *= alpha[i]
        T.mbarrier_wait_parity(vready[svp], ((n - 1) // stages) % 2)
        if step_wgmma:
            T.call_extern(
                "handle",
                f"tileops::attn_pv_64x128_d128<{wgmma_dtype}>",
                acc_o.data,
                pcast.data,
                vd,
            )
        else:
            T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
        T.named_barrier_arrive(nxt_bar, consumers)
        T.wait_wgmma(1)
        T.mbarrier_arrive(kfree[sk])
        if tail:
            if has_window:
                mask(acc_s, k * block_n, row, causal_offset, kv_len, qpos)
            elif qpos is not None:
                if is_causal:
                    limit = causal_offset - k * block_n
                    for i, j in T.Parallel(half, block_n):
                        acc_s[i, j] = T.if_then_else(
                            limit + qpos[i] >= j, acc_s[i, j], -T.infinity(accum)
                        )
                else:
                    for i, j in T.Parallel(half, block_n):
                        acc_s[i, j] = T.if_then_else(
                            k * block_n + j < kv_len, acc_s[i, j], -T.infinity(accum)
                        )
            else:
                if is_causal:
                    limit = row + causal_offset - k * block_n
                    for i, j in T.Parallel(half, block_n):
                        acc_s[i, j] = T.if_then_else(
                            limit + i >= j, acc_s[i, j], -T.infinity(accum)
                        )
                else:
                    for i, j in T.Parallel(half, block_n):
                        acc_s[i, j] = T.if_then_else(
                            k * block_n + j < kv_len, acc_s[i, j], -T.infinity(accum)
                        )
        # A call of long requests waits on the 128-channel PV and releases its V stage before
        # the softmax, since ptxas moves a release placed after it to the end of the step.
        # Short requests keep the PV beside the softmax of their single warpgroup.
        early_v = step_wgmma and wide
        if early_v:
            T.wait_wgmma(0)
            T.mbarrier_arrive(vfree[svp])
        softmax_step(acc_s, sm, smp, alpha, ss, red)
        if not early_v:
            T.wait_wgmma(0)
            T.mbarrier_arrive(vfree[svp])
        for i in T.Parallel(half):
            logsum[i] = logsum[i] * alpha[i] + ss[i]
        T.copy(acc_s, pcast)

    @T.macro
    def request_key(CuQ, CuKV, r):
        """Sort key of request *r*: fewer query tiles first, then a longer last tile."""
        q_len = CuQ[r + 1] - CuQ[r]
        kv_len = CuKV[r + 1] - CuKV[r]
        tiles = T.ceildiv(q_len, block_m)
        # Key tiles the request's last query tile scans, as locate counts them.
        last = T.max(
            1,
            T.min(
                T.ceildiv(kv_len, block_n),
                T.ceildiv(tiles * block_m + kv_len - q_len, block_n),
            ),
        )
        # Tile count in the high half; the low half falls as the last tile grows.
        return T.min(tiles, 32767) * 65536 + 65535 - T.min(last, 65535)

    @T.macro
    def rank_requests(CuQ, CuKV, tile_cum, order, lane):
        """Fill the claim order and its tile prefix sum: a lane ranks each of its requests
        against every request and sums the tiles placed before it."""
        for i in T.unroll(T.ceildiv(batch, 32)):
            r = i * 32 + lane
            if r < batch:
                key = request_key(CuQ, CuKV, r)
                rank = T.alloc_var("int32", init=0)
                start = T.alloc_var("int32", init=0)
                every = T.alloc_var("int32", init=0)
                for j in T.serial(batch):
                    other = request_key(CuQ, CuKV, j)
                    tiles = T.ceildiv(CuQ[j + 1] - CuQ[j], block_m)
                    ahead = (other < key) | ((other == key) & (j < r))
                    rank += T.if_then_else(ahead, 1, 0)
                    # Items are claimed from the last position backwards.
                    start += T.if_then_else(ahead | (j == r), 0, tiles)
                    every += tiles
                order[batch - 1 - rank] = r
                tile_cum[batch - 1 - rank] = start
                if r == 0:
                    tile_cum[batch] = every
        T.sync_warp()

    @T.macro
    def locate(work, CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta, order=None):
        """Fill meta (head, q_start, kv_start, q_len, kv_len, q0, eff) for work item *work*."""
        q_tiling.decode(tile_cum[batch] - 1 - work // heads, tile_cum, lo, hi, request, q_row)
        if ordered:
            request[0] = order[request[0]]
        meta[0] = work % heads
        meta[1] = CuQ[request[0]]
        meta[2] = CuKV[request[0]]
        meta[3] = CuQ[request[0] + 1] - meta[1]
        meta[4] = CuKV[request[0] + 1] - meta[2]
        meta[5] = q_row[0]
        if has_window:
            last_q = T.min(meta[3] - 1, meta[5] + block_m - 1) + meta[4] - meta[3]
            key_end = T.alloc_var("int32", init=meta[4])
            if is_causal:
                key_end = T.min(key_end, last_q + 1)
            if window_right >= 0:
                key_end = T.min(key_end, last_q + window_right + 1)
            first = T.alloc_var("int32", init=0)
            if window_left >= 0:
                first = T.max(0, meta[5] + meta[4] - meta[3] - window_left) // block_n
            meta[7] = first
            # The first transfer must still be issued for an entirely masked query tile.
            meta[6] = T.max(1, T.ceildiv(key_end, block_n) - first)
        else:
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
    def packed_request_key(CuQ, CuKV, r):
        """Sort key of request *r*: wide requests before narrow ones, then fewer query tiles,
        then a longer last tile."""
        q_len = CuQ[r + 1] - CuQ[r]
        kv_len = CuKV[r + 1] - CuKV[r]
        packed = packed_request(q_len)
        tiles = request_tiles(q_len, packed)
        # Key tiles the request's last query tile scans, as locate counts them.
        last = T.max(
            1,
            T.min(
                T.ceildiv(kv_len, block_n),
                T.ceildiv(tiles * tile_positions(packed) + kv_len - q_len, block_n),
            ),
        )
        # Class in the top bit, tile count below it; the low half falls as the last tile grows.
        narrow = T.if_then_else(packed & (q_len * groups <= half), 1 << 30, 0) if packs else 0
        return narrow + T.min(tiles, 16383) * 65536 + 65535 - T.min(last, 65535)

    @T.macro
    def rank_packed_requests(CuQ, CuKV, tile_cum, order, keys, tx, chunk):
        """Rank requests and build the item prefix; reuse keys for split workspace offsets."""
        r = tx
        if r < batch:
            keys[r] = packed_request_key(CuQ, CuKV, r)
        T.sync_threads()
        slot = T.alloc_var("int32", init=0)
        if r < batch:
            key = keys[r]
            rank = T.alloc_var("int32", init=0)
            start = T.alloc_var("int32", init=0)
            every = T.alloc_var("int32", init=0)
            narrow = T.alloc_var("int32", init=0)
            for j in T.serial(batch):
                other = keys[j]
                q_len = CuQ[j + 1] - CuQ[j]
                parts = request_parts(q_len, CuKV[j + 1] - CuKV[j], chunk)
                items = request_items(q_len) * parts
                ahead = (other < key) | ((other == key) & (j < r))
                rank += T.if_then_else(ahead, 1, 0)
                # Items are claimed from the last position backwards.
                start += T.if_then_else(ahead | (j == r), 0, items)
                every += items
                # The narrow class is the top bit of the key.
                narrow += T.if_then_else(other >= (1 << 30), items, 0)
                # A split request takes a slot an item, in request order.
                slot += T.if_then_else((j < r) & (parts > 1), items, 0)
            order[batch - 1 - rank] = r
            tile_cum[batch - 1 - rank] = start
            if r == 0:
                tile_cum[batch] = every
                tile_cum[batch + 1] = narrow
        if splits:
            T.sync_threads()
            if r < batch:
                keys[r] = slot
        T.sync_threads()

    def claimed_item(work, tile_cum):
        """Interleave wide and narrow items in reverse rank order to balance SM work."""
        total = tile_cum[batch]
        if not (ordered and packs):
            return total - 1 - work
        narrow = tile_cum[batch + 1]
        wide = total - narrow
        both = T.min(wide, narrow)
        rest = work - 2 * both
        return T.if_then_else(
            work < 2 * both,
            T.if_then_else(work % 2 == 0, total - 1 - work // 2, narrow - 1 - work // 2),
            T.if_then_else(wide > narrow, total - 1 - both - rest, narrow - 1 - both - rest),
        )

    @T.macro
    def locate_packed(
        work, CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta, order=None, chunk=0, slots=None,
        tail_base=0, tail_nt=0, item_index=None, tails=True,
    ):  # fmt: skip
        """Fill meta (head, q_start, kv_start, q_len, kv_len, q0, eff[, packed][, first]) for
        work item *work*: a request's items run from its last tile back, heads innermost."""
        if tail_split and tails:
            # Claim work maps to an item; a tail claim also to its part tpart and tail index tidx.
            total = tile_cum[batch]
            narrow = tile_cum[batch + 1]
            wide = total - narrow
            # The call's last tail_nt wide items split (the producer decides tail_nt).
            nt = tail_nt
            plain = wide - nt
            tpart = T.alloc_var("int32", init=-1)
            tidx = T.alloc_var("int32", init=0)
            item = T.alloc_var("int32", init=claimed_item(work, tile_cum))
            if item_index is not None:
                item = item_index
            elif nt > 0:
                # With a split tail, narrow claims spread evenly over the claims before it.
                pre = plain + narrow
                before = T.min(work, pre) * narrow // pre
                is_wide = (work >= pre) | ((T.min(work, pre) + 1) * narrow // pre == before)
                u = T.if_then_else(is_wide, work - before, before)
                if is_wide:
                    if u < plain:
                        item = total - 1 - u
                    else:
                        tidx = (u - plain) // tail_p
                        tpart = (u - plain) % tail_p
                        item = total - 1 - plain - tidx
                else:
                    item = narrow - 1 - u
            item_tiling.decode(item, tile_cum, lo, hi, request, q_row)
        else:
            item_tiling.decode(claimed_item(work, tile_cum), tile_cum, lo, hi, request, q_row)
        if ordered:
            request[0] = order[request[0]]
        meta[1] = CuQ[request[0]]
        meta[2] = CuKV[request[0]]
        meta[3] = CuQ[request[0] + 1] - meta[1]
        meta[4] = CuKV[request[0] + 1] - meta[2]
        packed = packed_request(meta[3])
        if packs:
            # Two divisions by constants and a select cost less than one by a runtime count.
            meta[0] = T.if_then_else(packed, q_row[0] % heads_kv, q_row[0] % heads)
            meta[5] = T.if_then_else(
                packed, q_row[0] // heads_kv * (block_m // groups), q_row[0] // heads * block_m
            )
            meta[packed_at] = T.if_then_else(packed, 1, 0)
        else:
            meta[0] = q_row[0] % heads
            meta[5] = q_row[0] // heads * block_m
        positions = tile_positions(packed)
        if has_window:
            last_q = T.min(meta[3] - 1, meta[5] + positions - 1) + meta[4] - meta[3]
            key_end = T.alloc_var("int32", init=meta[4])
            if is_causal:
                key_end = T.min(key_end, last_q + 1)
            if window_right >= 0:
                key_end = T.min(key_end, last_q + window_right + 1)
            first = T.alloc_var("int32", init=0)
            if window_left >= 0:
                first = T.max(0, meta[5] + meta[4] - meta[3] - window_left) // block_n
            meta[first_at] = first
            # The first transfer must still be issued for an entirely masked query tile.
            meta[6] = T.max(1, T.ceildiv(key_end, block_n) - first)
        else:
            meta[first_at] = 0
            if is_causal:
                meta[6] = T.max(
                    1,
                    T.min(
                        T.ceildiv(meta[4], block_n),
                        T.ceildiv(meta[5] + positions + meta[4] - meta[3], block_n),
                    ),
                )
            else:
                # At least one tile: the consumers always wait on tile 0.
                meta[6] = T.max(1, T.ceildiv(meta[4], block_n))
        if splits:
            parts = request_parts(meta[3], meta[4], chunk)
            # A split request has one tile; part p of KV head h is its item p * heads_kv + h.
            part = q_row[0] // heads_kv % parts
            meta[parts_at] = parts
            meta[parts_at + 1] = request[0] * heads_kv + meta[0]
            meta[parts_at + 2] = part
            meta[parts_at + 3] = slots[request[0]] + meta[0]
            meta[parts_at + 4] = 0
            if parts > 1:
                meta[5] = 0
                meta[first_at] = part * chunk
                meta[6] = T.min(chunk, meta[6] - part * chunk)
            if tail_split and tails:
                # Each part keeps at least four key tiles; a shorter item runs whole as part 0
                # and its other parts are void.
                t_short = meta[6] < tail_min
                if (tpart > 0) & t_short:
                    meta[parts_at + 4] = 2
                if (tpart >= 0) & ~t_short:
                    # Part q scans key tiles [q eff // p, (q + 1) eff // p).
                    t_lo = tpart * meta[6] // tail_p
                    t_hi = (tpart + 1) * meta[6] // tail_p
                    meta[parts_at] = tail_p
                    meta[parts_at + 1] = batch * heads_kv + tidx
                    meta[parts_at + 2] = tpart
                    meta[parts_at + 3] = (
                        tail_base + tidx // heads_kv * tail_p * heads_kv + tidx % heads_kv
                    )
                    meta[first_at] = meta[first_at] + t_lo
                    meta[6] = t_hi - t_lo

    @T.macro
    def fetch(Sched, lane, work):
        """Claim the next work item into work[0]: lane 0 claims, the warp shares it."""
        if lane == 0:
            work[0] = T.atomic_add(Sched[0], 1, return_prev=True)
        work[0] = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), work[0], 0, 32, 32)

    @T.macro
    def load_packed_q(Q, Qs, slot, q_bar, q_start, q_end, kv_head):
        """Load a warpgroup's positions as one box of (position, head, 64 channels): its rows
        are the packed rows in order. A warpgroup the request does not fill loads its
        positions one by one."""
        for wg in T.unroll(2):
            base = q_start + wg * positions_per_wg
            if q_end - base >= positions_per_wg:
                for c in T.unroll(dim // 64):
                    T.tma_copy(
                        Q[
                            base : base + positions_per_wg,
                            kv_head * groups : (kv_head + 1) * groups,
                            c * 64 : (c + 1) * 64,
                        ],
                        T.reshape(Qs, [2, 2, positions_per_wg, groups, dim])[
                            slot, wg, :, :, c * 64 : (c + 1) * 64
                        ],
                        barrier=q_bar[slot],
                    )
            else:
                for p in T.serial(T.max(0, q_end - base)):
                    for c in T.unroll(dim // 64):
                        T.tma_copy(
                            Q[
                                base + p,
                                kv_head * groups : (kv_head + 1) * groups,
                                c * 64 : (c + 1) * 64,
                            ],
                            Qs[slot, wg, p * groups : (p + 1) * groups, c * 64 : (c + 1) * 64],
                            barrier=q_bar[slot],
                        )

    @T.macro
    def load_q(Q, Qs, slot, q_bar, q_start, head):
        """TMA 128 positions of one head, 64 rows a warpgroup."""
        T.tma_copy(Q[q_start : q_start + half, head, :], Qs[slot, 0, :, :], barrier=q_bar[slot])
        T.tma_copy(
            Q[q_start + half : q_start + block_m, head, :],
            Qs[slot, 1, :, :],
            barrier=q_bar[slot],
        )

    @T.macro
    def claim_items(
        Q, K, V, CuQ, CuKV, Sched, Qs, Ks, Vs, item_slot, q_bar, qfree, kready, kfree, vready, vfree, tile_cum, lo, hi, request, q_row, meta, lane, order, chunk, slots, tail_base, tail_nt, claims,
        tails: bool,
    ):  # fmt: skip
        """Claim work items until none remain; load each one's Q, then its KV tiles. With
        *tails*, the claims of a split tail map to their parts."""
        issued = T.alloc_var("int32", init=0)
        loaded = T.alloc_var("int32", init=0)
        work = T.alloc_local([1], "int32")
        fetch(Sched, lane, work)
        while work[0] < claims:
            if packs:
                locate_packed(
                    work[0], CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta, order, chunk,
                    slots, tail_base, tail_nt, tails=tails,
                )  # fmt: skip
            else:
                locate(work[0], CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta, order)
            # A void tail part (its item too short to split) loads nothing.
            if (meta[parts_at + 4] != 2) if tail_split and tails else True:
                head = meta[0]
                q_start = meta[1] + meta[5]
                kv_start = meta[2] + (meta[first_at] * block_n if has_window or packs else 0)
                cv = (
                    T.if_then_else(meta[packed_at] == 1, head, head // groups)
                    if packs
                    else head // groups
                )
                slot = loaded % 2
                T.mbarrier_wait_parity(qfree[slot], ((loaded // 2) % 2) ^ 1)
                # The consumers read the item's geometry here rather than decode it again.
                for i in T.unroll(meta_count):
                    item_slot[slot, i] = meta[i]
                item_slot[slot, meta_count] = work[0]
                if packs:
                    if meta[packed_at] == 1:
                        load_packed_q(Q, Qs, slot, q_bar, q_start, meta[1] + meta[3], head)
                    else:
                        load_q(Q, Qs, slot, q_bar, q_start, head)
                else:
                    load_q(Q, Qs, slot, q_bar, q_start, head)
                T.mbarrier_arrive(q_bar[slot])
                if packs:
                    # Values trail keys by one tile, the order the consumers take them in.
                    for k in T.serial(meta[6] + 1):
                        if k < meta[6]:
                            sk = (issued + k) % stages
                            T.mbarrier_wait_parity(kfree[sk], (((issued + k) // stages) % 2) ^ 1)
                            key0 = kv_start + k * block_n
                            T.tma_copy(
                                K[key0 : key0 + block_n, cv, :],
                                Ks[sk, :, :],
                                barrier=kready[sk],
                            )
                            T.mbarrier_arrive(kready[sk])
                        if k > 0:
                            sv = (issued + k - 1) % stages
                            T.mbarrier_wait_parity(
                                vfree[sv], (((issued + k - 1) // stages) % 2) ^ 1
                            )
                            val0 = kv_start + (k - 1) * block_n
                            T.tma_copy(
                                V[val0 : val0 + block_n, cv, :],
                                Vs[sv, :, :],
                                barrier=vready[sv],
                            )
                            T.mbarrier_arrive(vready[sv])
                else:
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
        T.mbarrier_wait_parity(qfree[loaded % 2], ((loaded // 2) % 2) ^ 1)
        item_slot[loaded % 2, meta_count] = -1
        T.mbarrier_arrive(q_bar[loaded % 2])
        if lane == 0:
            finished = T.atomic_add(Sched[1], 1, return_prev=True)
            if finished == num_ctas - 1:
                Sched[0] = 0
                Sched[1] = 0

    @T.macro
    def producer(
        Q, K, V, CuQ, CuKV, Sched, Qs, Ks, Vs, item_slot, q_bar, qfree, kready, kfree, vready,
        vfree, total, tile_cum, lo, hi, request, q_row, meta, lane, order=None, chunk=0,
        slots=None, tail_base=0,
    ):  # fmt: skip
        """Decide the split tail, then claim and load work items until none remain."""
        claims = T.alloc_var("int32", init=total)
        tail_nt = T.alloc_var("int32", init=0)
        if tail_split:
            # A call whose items fill the SMs, with fewer narrow items than SMs, splits its last
            # tail_n wide claims, if the longest of them holds tail_min key tiles.
            narrow = tile_cum[batch + 1]
            nt0 = T.if_then_else(
                ((chunk > 0) if chunks else False) | (narrow == 0) | (2 * narrow >= num_ctas),
                0,
                T.min(total - narrow, tail_n),
            )
            if nt0 > 0:
                locate_packed(
                    0, CuQ, CuKV, tile_cum, lo, hi, request, q_row, meta, order, chunk, slots,
                    tail_base, 0, total - 1 - (total - narrow - nt0),
                )  # fmt: skip
                if meta[6] >= tail_min:
                    tail_nt = nt0
            claims = total + tail_nt * (tail_p - 1)
        claim = (
            Q, K, V, CuQ, CuKV, Sched, Qs, Ks, Vs, item_slot, q_bar, qfree, kready, kfree, vready, vfree, tile_cum, lo, hi, request, q_row, meta, lane, order, chunk, slots, tail_base, tail_nt, claims,
        )  # fmt: skip
        if tail_split:
            # A call with no split tail claims without the tail mapping.
            if tail_nt > 0:
                claim_items(*claim, True)
            else:
                claim_items(*claim, False)
        else:
            claim_items(*claim, False)

    @T.macro
    def consumer(
        wg: int, Qs, Ks, Vs, Os, O, item_slot, q_bar, qfree, kready, kfree, vready, vfree, meta,
        Sinks, rope_cos=None, rope_sin=None,
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
        red = T.alloc_fragment([half, 8], accum)
        logsum = T.alloc_fragment([half], accum)
        if vector_rope:
            # The thread's runs of channels and their table entries, all loaded at once.
            rope_xa = T.alloc_local([rope_units * rope_vec], dtype)
            rope_xb = T.alloc_local([rope_units * rope_vec], dtype)
            rope_ca = T.alloc_local([rope_units * table_vec], table_dtype)
            rope_sa = T.alloc_local([rope_units * table_vec], table_dtype)
        full = T.alloc_var("int32", init=0)
        if has_window:
            first_full = T.alloc_var("int32", init=0)
        # Finished KV tiles and items: they carry the barrier phases.
        done = T.alloc_var("int32", init=0)
        served = T.alloc_var("int32", init=0)
        work = T.alloc_var("int32", init=0)
        T.mbarrier_wait_parity(q_bar[0], 0)
        work = item_slot[0, meta_count]
        while work >= 0:
            for i in T.unroll(meta_count):
                meta[i] = item_slot[served % 2, i]
            head = meta[0]
            q_start = meta[1]
            q_len = meta[3]
            kv_len = meta[4]
            causal_offset = meta[4] - meta[3]
            row = meta[5] + wg * half
            eff = meta[6]
            first = meta[first_at] if has_window else 0
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(alpha, 1.0)
            T.fill(sm, -T.infinity(accum))
            if rope_cos is not None:
                # Query token i of a request sits at position kv_len - q_len + i. A
                # warpgroup rotates its own half, which the named-barrier sync before the
                # first WGMMA fences.
                if vector_rope:
                    rotate_half_tile(
                        Qs, rope_cos, rope_sin, causal_offset + row, served % 2, wg,
                        rope_xa, rope_xb, rope_ca, rope_sa,
                    )  # fmt: skip
                else:
                    rotate_query_tile(Qs, rope_cos, rope_sin, causal_offset + row, served % 2, wg)
            if (wg == 1) & (served == 0):
                T.named_barrier_arrive(1, consumers)  # let warpgroup 0 go first

            # KV tile 0: QK and softmax only.
            s0 = done % stages
            T.sync_threads(my_bar, consumers)
            T.mbarrier_wait_parity(kready[s0], (done // stages) % 2)
            T.wgmma_gemm(
                Qs[served % 2, wg, :, :],
                Ks[s0, :, :],
                acc_s,
                transpose_B=True,
                policy=policy,
                clear_accum=True,
            )
            T.named_barrier_arrive(nxt_bar, consumers)
            T.wait_wgmma(0)
            T.mbarrier_arrive(kfree[s0])
            if has_window:
                first_full = 0
                if window_left >= 0:
                    last_q = T.min(row + half - 1, q_len - 1) + causal_offset
                    first_full = T.min(
                        eff, T.max(0, T.ceildiv(last_q - window_left, block_n) - first)
                    )
                full = T.min(eff, kv_len // block_n - first)
                if is_causal:
                    full = T.min(full, T.max(0, (row + causal_offset + 1) // block_n - first))
                if window_right >= 0:
                    full = T.min(
                        full, T.max(0, (row + causal_offset + window_right + 1) // block_n - first)
                    )
                full = T.max(first_full, full)
                if first_full > 0 or full < 1:
                    mask(acc_s, first * block_n, row, causal_offset, kv_len)
            else:
                if is_causal and row + causal_offset < block_n - 1:
                    limit = row + causal_offset
                    for i, j in T.Parallel(half, block_n):
                        acc_s[i, j] = T.if_then_else(
                            limit + i >= j, acc_s[i, j], -T.infinity(accum)
                        )
                elif not is_causal and kv_len < block_n:
                    for i, j in T.Parallel(half, block_n):
                        acc_s[i, j] = T.if_then_else(j < kv_len, acc_s[i, j], -T.infinity(accum))
            if use_softcap:
                apply_softcap(acc_s)
            T.reduce_max(acc_s, sm, dim=1, clear=False)
            if has_window or has_sinks:
                for i in T.Parallel(half):
                    sm[i] = T.max(sm[i], max_floor)
            for i, j in T.Parallel(half, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
            T.reduce_sum(acc_s, ss, dim=1, batch=2)
            for i in T.Parallel(half):
                logsum[i] = ss[i]
            T.copy(acc_s, pcast)

            # Only tiles intersecting either window boundary need masking.
            if not has_window:
                if is_causal:
                    full = T.max(1, T.min(eff, T.floordiv(row + causal_offset + 1, block_n)))
                else:
                    full = T.max(1, T.floordiv(kv_len, block_n))
            step = (Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp)
            descs = None
            if step_wgmma:
                qd = T.alloc_var("uint64")
                kd0 = T.alloc_var("uint64")
                vd0 = T.alloc_var("uint64")
                qd = desc_of(Qs, ((served % 2) * 2 + wg) * half * dim, 1)
                kd0 = desc_of(Ks, 0, 1)
                vd0 = desc_of(Vs, 0, 1024)
                descs = (qd, kd0, vd0)
            if has_window:
                # One loop masks the tiles at both window edges: a short scan stays in the
                # instruction cache.
                for k in T.serial(1, eff):
                    kv_step(
                        Qs[served % 2, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                        first + k, done + k, row, causal_offset, kv_len,
                        (k < T.max(1, first_full)) | (k >= T.max(1, full)), descs=descs,
                    )  # fmt: skip
            else:
                for k in T.serial(1, T.max(1, full)):
                    kv_step(
                        Qs[served % 2, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar, first + k,
                        done + k, row, causal_offset, kv_len, False, descs=descs,
                    )  # fmt: skip
                for k in T.serial(T.max(1, full), eff):
                    kv_step(
                        Qs[served % 2, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar, first + k,
                        done + k, row, causal_offset, kv_len, True, descs=descs,
                    )  # fmt: skip
            # Every QK is done, so the producer may reload Q.
            T.mbarrier_arrive(qfree[served % 2])

            # PV of the last tile, then normalize and store.
            svp = (done + eff - 1) % stages
            for i, j in T.Parallel(half, dim):
                acc_o[i, j] *= alpha[i]
            T.mbarrier_wait_parity(vready[svp], ((done + eff - 1) // stages) % 2)
            T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
            T.wait_wgmma(0)
            T.mbarrier_arrive(vfree[svp])
            if has_sinks:
                sink_scale(sm, logsum, Sinks, head, alpha)
            # Every row sees a key: causal needs kv_len >= q_len, non-causal kv_len > 0.
            if row + half <= q_len and kv_len >= (q_len if is_causal else 1):
                if not has_sinks:
                    for i in T.Parallel(half):
                        if has_window:
                            alpha[i] = T.if_then_else(logsum[i] > 0, 1.0 / logsum[i], 0)
                        else:
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
                        if has_sinks:
                            O[q_start + row + i, head, j] = T.cast(acc_o[i, j] * alpha[i], dtype)
                        else:
                            O[q_start + row + i, head, j] = T.if_then_else(
                                logsum[i] > 0,
                                T.cast(acc_o[i, j] / logsum[i], dtype),
                                T.cast(0, dtype),
                            )
            done += eff
            served += 1
            T.mbarrier_wait_parity(q_bar[served % 2], (served // 2) % 2)
            work = item_slot[served % 2, meta_count]

    @T.macro
    def scan(
        wg: int, slot, Qs, Ks, Vs, qfree, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm,
        smp, alpha, ss, red, logsum, done, eff, first, q_len, kv_len, causal_offset, row,
        first_q, last_q, qpos,
    ):  # fmt: skip
        """The warpgroup's scan of an item's key tiles, through the last product. Its rows'
        positions run from *first_q* to *last_q*."""
        my_bar = 1 + wg
        nxt_bar = 2 - wg
        full = T.alloc_var("int32", init=0)
        if has_window:
            first_full = T.alloc_var("int32", init=0)
        s0 = done % stages
        T.sync_threads(my_bar, consumers)
        T.mbarrier_wait_parity(kready[s0], (done // stages) % 2)
        T.wgmma_gemm(
            Qs[slot, wg, :, :],
            Ks[s0, :, :],
            acc_s,
            transpose_B=True,
            policy=policy,
            clear_accum=True,
        )
        T.named_barrier_arrive(nxt_bar, consumers)
        T.wait_wgmma(0)
        T.mbarrier_arrive(kfree[s0])
        if has_window:
            first_full = 0
            if window_left >= 0:
                first_full = T.min(
                    eff,
                    T.max(
                        0,
                        T.ceildiv(T.min(last_q, q_len - 1) + causal_offset - window_left, block_n)
                        - first,
                    ),
                )
            full = T.min(eff, kv_len // block_n - first)
            if is_causal:
                full = T.min(full, T.max(0, (first_q + causal_offset + 1) // block_n - first))
            if window_right >= 0:
                full = T.min(
                    full,
                    T.max(0, (first_q + causal_offset + window_right + 1) // block_n - first),
                )
            full = T.max(first_full, full)
            if first_full > 0 or full < 1:
                mask(acc_s, first * block_n, row, causal_offset, kv_len, qpos)
        else:
            # The scan's first tile is key tile `first`: a split item's part starts past 0.
            if is_causal and first_q + causal_offset < (first + 1) * block_n - 1:
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(
                        causal_offset + position(row, qpos, i) - first * block_n >= j,
                        acc_s[i, j],
                        -T.infinity(accum),
                    )
            elif not is_causal and kv_len < (first + 1) * block_n:
                for i, j in T.Parallel(half, block_n):
                    acc_s[i, j] = T.if_then_else(
                        j < kv_len - first * block_n, acc_s[i, j], -T.infinity(accum)
                    )
        if use_softcap:
            apply_softcap(acc_s)
        T.reduce_max(acc_s, sm, dim=1, clear=False)
        # The store scales rows, so a NaN would survive a zero factor.
        for i in T.Parallel(half):
            sm[i] = T.max(sm[i], max_floor)
        for i, j in T.Parallel(half, block_n):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - sm[i] * scale)
        T.reduce_sum(acc_s, ss, dim=1, batch=2)
        for i in T.Parallel(half):
            logsum[i] = ss[i]
        T.copy(acc_s, pcast)

        # Only tiles intersecting either window boundary need masking.
        if not has_window:
            if is_causal:
                full = T.max(
                    1, T.min(eff, T.floordiv(first_q + causal_offset + 1, block_n) - first)
                )
            else:
                full = T.max(1, T.floordiv(kv_len, block_n) - first)
        step = (Ks, Vs, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm, smp)
        descs = None
        if step_wgmma:
            qd = T.alloc_var("uint64")
            kd0 = T.alloc_var("uint64")
            vd0 = T.alloc_var("uint64")
            qd = desc_of(Qs, (slot * 2 + wg) * half * dim, 1)
            kd0 = desc_of(Ks, 0, 1)
            vd0 = desc_of(Vs, 0, 1024)
            descs = (qd, kd0, vd0)
        if has_window:
            for k in T.serial(1, T.max(1, first_full)):
                kv_step(
                    Qs[slot, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                    first + k, done + k, row, causal_offset, kv_len, True, qpos, descs=descs,
                )  # fmt: skip
        start = T.max(1, first_full) if has_window else 1
        # Unmasked tiles run without the mask branch; an idle warpgroup 1 scans only eff tiles.
        stop = T.min(eff, T.max(1, full))
        for k in T.serial(start, stop):
            kv_step(
                Qs[slot, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                first + k, done + k, row, causal_offset, kv_len, False, qpos, descs=descs,
            )  # fmt: skip
        for k in T.serial(stop, eff):
            kv_step(
                Qs[slot, wg, :, :], *step, alpha, ss, red, logsum, my_bar, nxt_bar,
                first + k, done + k, row, causal_offset, kv_len, True, qpos, descs=descs,
            )  # fmt: skip
        # Every QK is done, so the producer may reload Q.
        T.mbarrier_arrive(qfree[slot])

        svp = (done + eff - 1) % stages
        for i, j in T.Parallel(half, dim):
            acc_o[i, j] *= alpha[i]
        T.mbarrier_wait_parity(vready[svp], ((done + eff - 1) // stages) % 2)
        T.wgmma_gemm(pcast, Vs[svp, :, :], acc_o, policy=policy, clear_accum=False)
        T.wait_wgmma(0)
        T.mbarrier_arrive(vfree[svp])

    @T.macro
    def finish_part(
        wg: int, slot, item_slot, Part, Stat, Merge, acc_o, sm, logsum, fa, fb, meta, first_q,
        q_len,
    ):  # fmt: skip
        """Publish partial statistics; the last arrival merges and resets the counter.

        Only warpgroups containing valid query rows participate.
        """
        parts = meta[parts_at]
        group = meta[parts_at + 1]
        own = meta[parts_at + 2]
        base = meta[parts_at + 3]
        if first_q < q_len:
            for i, d in T.Parallel(half, dim):
                Part[base + own * heads_kv, wg * half + i, d] = acc_o[i, d]
            for i in T.Parallel(half):
                Stat[base + own * heads_kv, wg * half + i, 0] = sm[i]
                Stat[base + own * heads_kv, wg * half + i, 1] = logsum[i]
            # The barrier orders every thread's stores before the count, which releases them
            # to the part that counts last and acquires the earlier parts' for it.
            T.sync_threads(3 + wg, 128)
            if T.get_thread_binding(0) == 0:
                item_slot[slot, meta_count + 1 + wg] = T.atomic_add(
                    Merge[group * 2 + wg], 1, memory_order="acq_rel", return_prev=True
                )
            T.sync_threads(3 + wg, 128)
            if item_slot[slot, meta_count + 1 + wg] == parts - 1:
                if T.get_thread_binding(0) == 0:
                    Merge[group * 2 + wg] = 0
                for c in T.serial(parts):
                    if c != own:
                        for i in T.Parallel(half):
                            peak = T.max(sm[i], Stat[base + c * heads_kv, wg * half + i, 0])
                            fa[i] = T.exp2((sm[i] - peak) * scale)
                            fb[i] = T.exp2(
                                (Stat[base + c * heads_kv, wg * half + i, 0] - peak) * scale
                            )
                            logsum[i] = (
                                logsum[i] * fa[i]
                                + Stat[base + c * heads_kv, wg * half + i, 1] * fb[i]
                            )
                            sm[i] = peak
                        for i, d in T.Parallel(half, dim):
                            acc_o[i, d] = (
                                acc_o[i, d] * fa[i]
                                + Part[base + c * heads_kv, wg * half + i, d] * fb[i]
                            )

    @T.macro
    def store(
        Os, O, Sinks, acc_o, sm, logsum, alpha, wg: int, head, q_start, q_len, q0, row, packed
    ):
        """Normalize and store the warpgroup's rows: positions row .. row + 63 of query head
        *head*, or, packed, position q0 + (64 wg + r) // groups of head r % groups of KV head
        *head*'s group for row r. Rows past the request are not stored."""
        if has_sinks:
            if packs:
                if packed:
                    sink_scale_packed(sm, logsum, Sinks, head * groups, alpha)
                else:
                    sink_scale(sm, logsum, Sinks, head, alpha)
            else:
                sink_scale(sm, logsum, Sinks, head, alpha)
        else:
            # A row with no visible key is written as zero.
            for i in T.Parallel(half):
                alpha[i] = T.if_then_else(logsum[i] > 0, 1.0 / logsum[i], 0)
        for i, j in T.Parallel(half, dim):
            acc_o[i, j] *= alpha[i]
        # The warpgroup has stored the previous item before Os is reused.
        T.sync_threads(3 + wg, 128)
        T.copy(acc_o, Os[wg, :, :])
        T.sync_threads(3 + wg, 128)
        # Whole rows of one head: one TMA store.
        whole = row + half <= q_len
        if packs:
            whole = T.if_then_else(packed, False, whole)
        if whole:
            T.copy(Os[wg, :, :], O[q_start + row : q_start + row + half, head, :])
        else:
            # A thread stores 16-byte runs of the rows inside the request.
            lane = T.get_thread_binding(0)
            runs = dim // 8
            for u in T.serial(half * runs // 128):
                i = (u * 128 + lane) // runs
                c = (u * 128 + lane) % runs
                pos = (
                    T.if_then_else(packed, q0 + (wg * half + i) // groups, row + i)
                    if packs
                    else row + i
                )
                if pos < q_len:
                    h = T.if_then_else(packed, head * groups + i % groups, head) if packs else head
                    for v in T.vectorized(8):
                        O[q_start + pos, h, c * 8 + v] = Os[wg, i, c * 8 + v]

    @T.macro
    def consumer_item(
        wg: int, Qs, Ks, Vs, Os, O, item_slot, qfree, kready, kfree, vready, vfree, meta, Sinks, rope_cos, rope_sin, Part, Stat, Merge, acc_s, pcast, acc_o, sm, smp, alpha, ss, red, logsum, rope_xa, rope_xb, rope_ca, rope_sa, done, served, head, q_start, q_len, kv_len, causal_offset, q0, row, eff, first, slot, packed, qpos,
    ):  # fmt: skip
        """Warpgroup *wg*'s work on one claimed item: its scan, then its merge or store.
        *packed* and *qpos* are compile-time for a known item kind."""
        # Request-relative query bounds of this warpgroup.
        first_q = T.if_then_else(packed, q0 + wg * positions_per_wg, row) if packs else row
        last_q = (
            T.if_then_else(packed, first_q + positions_per_wg - 1, row + half - 1)
            if packs
            else row + half - 1
        )
        if qpos is not None:
            for i in T.Parallel(half):
                qpos[i] = first_q + T.if_then_else(packed, i // groups, i)
        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(alpha, 1.0)
        T.fill(sm, -T.infinity(accum))
        if rope_cos is not None:
            # Query token i of a request sits at position kv_len - q_len + i. A
            # warpgroup rotates its own half, which the named-barrier sync before the
            # first WGMMA fences.
            if vector_rope:
                rotate_half_tile(
                    Qs, rope_cos, rope_sin, causal_offset + first_q, slot, wg,
                    rope_xa, rope_xb, rope_ca, rope_sa, packed,
                )  # fmt: skip
            elif packs:
                if packed:
                    rotate_packed_tile(Qs, rope_cos, rope_sin, causal_offset + first_q, slot, wg)
                else:
                    rotate_query_tile(Qs, rope_cos, rope_sin, causal_offset + row, slot, wg)
            else:
                rotate_query_tile(Qs, rope_cos, rope_sin, causal_offset + row, slot, wg)
        if (wg == 1) & (served == 0):
            T.named_barrier_arrive(1, consumers)  # let warpgroup 0 go first
        # The scan runs unconditionally: under a branch ptxas keeps the WGMMA descriptors
        # out of uniform registers. Warpgroup 1 of a narrow item, which holds no row, scans
        # the first key tile only and passes on the rest.
        idle = ((wg == 1) & packed & (q_len * groups <= half)) if packs else False
        scan(
            wg, slot, Qs, Ks, Vs, qfree, kready, kfree, vready, vfree, acc_s, pcast, acc_o, sm,
            smp, alpha, ss, red, logsum, done, T.if_then_else(idle, 1, eff) if packs else eff,
            first, q_len, kv_len, causal_offset, row, first_q, last_q, qpos,
        )  # fmt: skip
        if packs:
            # The product turns of the key tiles it skips, released without computing.
            for k in T.serial(1, T.if_then_else(idle, eff, 1)):
                n = done + k
                T.sync_threads(2, consumers)
                T.mbarrier_wait_parity(kready[n % stages], (n // stages) % 2)
                T.named_barrier_arrive(1, consumers)
                T.mbarrier_arrive(kfree[n % stages])
                T.mbarrier_wait_parity(vready[n % stages], (n // stages) % 2)
                T.mbarrier_arrive(vfree[n % stages])
        if splits:
            if meta[parts_at] > 1:
                # Only the part that finishes a warpgroup's rows last stores them.
                finish_part(
                    wg, slot, item_slot, Part, Stat, Merge, acc_o, sm, logsum, alpha, ss,
                    meta, first_q, q_len,
                )  # fmt: skip
                if (first_q < q_len) & (item_slot[slot, meta_count + 1 + wg] == meta[parts_at] - 1):
                    store(
                        Os,
                        O,
                        Sinks,
                        acc_o,
                        sm,
                        logsum,
                        alpha,
                        wg,
                        head,
                        q_start,
                        q_len,
                        q0,
                        row,
                        packed,
                    )
            else:
                store(
                    Os,
                    O,
                    Sinks,
                    acc_o,
                    sm,
                    logsum,
                    alpha,
                    wg,
                    head,
                    q_start,
                    q_len,
                    q0,
                    row,
                    packed,
                )
        else:
            store(
                Os,
                O,
                Sinks,
                acc_o,
                sm,
                logsum,
                alpha,
                wg,
                head,
                q_start,
                q_len,
                q0,
                row,
                packed,
            )

    @T.macro
    def packed_consumer(
        wg: int, Q, Qs, Ks, Vs, Os, O, item_slot, q_bar, qfree, kready, kfree, vready, vfree,
        meta, Sinks, rope_cos=None, rope_sin=None, Part=None, Stat=None, Merge=None,
    ):  # fmt: skip
        """Consumer warpgroup *wg*: its 64 rows of each claimed tile."""
        T.set_max_nreg(240, 1)
        acc_s = T.alloc_fragment([half, block_n], accum)
        pcast = T.alloc_fragment([half, block_n], dtype)
        acc_o = T.alloc_fragment([half, dim], accum)
        sm = T.alloc_fragment([half], accum)
        smp = T.alloc_fragment([half], accum)
        alpha = T.alloc_fragment([half], accum)
        ss = T.alloc_fragment([half], accum)
        red = T.alloc_fragment([half, 8], accum)
        logsum = T.alloc_fragment([half], accum)
        # Query positions repeat across the heads of a packed group.
        qpos = T.alloc_fragment([half], "int32") if packs else None
        if vector_rope:
            # The thread's runs of channels and their table entries, all loaded at once.
            rope_xa = T.alloc_local([rope_units * rope_vec], dtype)
            rope_xb = T.alloc_local([rope_units * rope_vec], dtype)
            rope_ca = T.alloc_local([rope_units * table_vec], table_dtype)
            rope_sa = T.alloc_local([rope_units * table_vec], table_dtype)
        rope_regs = (rope_xa, rope_xb, rope_ca, rope_sa) if vector_rope else (None,) * 4
        # Finished KV tiles and items: they carry the barrier phases.
        done = T.alloc_var("int32", init=0)
        served = T.alloc_var("int32", init=0)
        work = T.alloc_var("int32", init=0)
        T.mbarrier_wait_parity(q_bar[0], 0)
        work = item_slot[0, meta_count]
        while work >= 0:
            for i in T.unroll(meta_count):
                meta[i] = item_slot[served % 2, i]
            head = meta[0]
            q_start = meta[1]
            q_len = meta[3]
            kv_len = meta[4]
            causal_offset = meta[4] - meta[3]
            q0 = meta[5]
            row = q0 + wg * half
            eff = meta[6]
            first = meta[first_at]
            slot = served % 2
            packed = meta[packed_at] == 1 if packs else False
            item = (
                wg, Qs, Ks, Vs, Os, O, item_slot, qfree, kready, kfree, vready, vfree, meta, Sinks, rope_cos, rope_sin, Part, Stat, Merge, acc_s, pcast, acc_o, sm, smp, alpha, ss, red, logsum, *rope_regs, done, served, head, q_start, q_len, kv_len, causal_offset, q0, row, eff, first, slot,
            )  # fmt: skip
            if wide:
                # Each item kind runs its own copy of the item code.
                if packed:
                    consumer_item(*item, True, qpos)
                else:
                    consumer_item(*item, False, None)
            else:
                consumer_item(*item, packed, qpos)
            done += eff
            served += 1
            T.mbarrier_wait_parity(q_bar[served % 2], (served // 2) % 2)
            work = item_slot[served % 2, meta_count]

    @T.macro
    def launch(
        Q, K, V, CuQ, CuKV, O, Sched, Sinks, rope_cos=None, rope_sin=None, Part=None, Stat=None,
        Merge=None, chunk=0, tail_base=0,
    ):  # fmt: skip
        """One persistent CTA: the TMA producer warp, then the two consumer warpgroups."""
        Qs = T.alloc_shared([2, 2, half, dim], dtype)
        Ks = T.alloc_shared([stages, block_n, dim], dtype)
        Vs = T.alloc_shared([stages, block_n, dim], dtype)
        Os = T.alloc_shared([2, half, dim], dtype)
        # A ranked packed program keeps the narrow items' total past the prefix.
        tile_cum = T.alloc_shared([batch + 2 if ordered and packs else batch + 1], "int32")
        order = T.alloc_shared([batch], "int32") if ordered else None
        # The packed program's ranking keys, then each split request's first workspace slot.
        keys = T.alloc_shared([batch], "int32") if ordered and packs else None
        # Past an item's geometry and claim, a split item's two warpgroups read their part count.
        item_slot = T.alloc_shared([2, meta_count + 1 + 2 * int(splits)], "int32")
        T.annotate_layout(
            {
                Qs: make_swizzled_layout(Qs),
                Ks: make_swizzled_layout(Ks),
                Vs: make_swizzled_layout(Vs),
                Os: make_swizzled_layout(Os),
            }
        )
        q_bar = T.alloc_barrier([32, 32])
        qfree = T.alloc_barrier([consumers, consumers])
        kready = T.alloc_barrier([32] * stages)
        kfree = T.alloc_barrier([consumers] * stages)
        vready = T.alloc_barrier([32] * stages)
        vfree = T.alloc_barrier([consumers] * stages)
        lo = T.alloc_local([1], "int32")
        hi = T.alloc_local([1], "int32")
        q_row = T.alloc_local([1], "int32")
        request = T.alloc_local([1], "int32")
        meta = T.alloc_local([meta_count], "int32")

        tx = T.get_thread_binding(1) * 128 + T.get_thread_binding(0)
        # Only the producer warp reads the tile prefix sum; the ordered one it builds itself.
        if not ordered:
            if packs:
                items_prefix(CuQ, tile_cum)
            else:
                q_tiling.cumsum_offsets(CuQ, tile_cum)
            T.sync_threads()
        if chunks:
            # The call splits keys only if its unsplit items leave SMs idle.
            unsplit = T.alloc_shared([1], "int32")
            if tx == 0:
                unsplit[0] = 0
            T.sync_threads()
            if tx < batch:
                T.atomic_add(unsplit[0], request_items(CuQ[tx + 1] - CuQ[tx]))
            T.sync_threads()
            chunk = T.if_then_else(unsplit[0] < num_ctas, chunk, 0)
        if packs:
            if ordered:
                rank_packed_requests(CuQ, CuKV, tile_cum, order, keys, tx, chunk)
        elif ordered and tx >= consumers and tx < consumers + 32:
            rank_requests(CuQ, CuKV, tile_cum, order, tx - consumers)
        if tx >= consumers:
            T.set_max_nreg(24, 0)  # the producer only issues TMA
        if tx >= consumers and tx < consumers + 32:
            total_items = tile_cum[batch] if packs else tile_cum[batch] * heads
            producer(
                Q, K, V, CuQ, CuKV, Sched, Qs, Ks, Vs, item_slot, q_bar, qfree, kready, kfree,
                vready, vfree, total_items, tile_cum, lo, hi, request, q_row, meta,
                tx - consumers, order, chunk, keys, tail_base,
            )  # fmt: skip
        args = (
            Qs,
            Ks,
            Vs,
            Os,
            O,
            item_slot,
            q_bar,
            qfree,
            kready,
            kfree,
            vready,
            vfree,
            meta,
            Sinks,
        )
        if rope_cos is not None:
            args = (*args, rope_cos, rope_sin)
        # One consumer body serves both warpgroups: warpgroup wg is threadIdx.y, broadcast
        # from lane 0 so ptxas keeps what derives from it in uniform registers.
        wg = T.call_extern("int32", "tileops::warp_uniform_int", T.get_thread_binding(1))
        if wg < 2:
            if packs:
                packed_consumer(wg, Q, *args, Part=Part, Stat=Stat, Merge=Merge)
            else:
                consumer(wg, *args)

    if splits:
        part_shape = [T.dynamic("slots"), block_m]
        if fuse_rope:

            @T.prim_func
            def main(
                Q: T.Tensor([total_q, heads, dim], dtype),
                K: T.Tensor([total_kv, heads_kv, dim], dtype),
                V: T.Tensor([total_kv, heads_kv, dim], dtype),
                CuQ: T.Tensor([batch + 1], "int32"),
                CuKV: T.Tensor([batch + 1], "int32"),
                RopeCos: T.Tensor([max_position, rope_half], table_dtype),
                RopeSin: T.Tensor([max_position, rope_half], table_dtype),
                O: T.Tensor([total_q, heads, dim], dtype),
                Sched: T.Tensor([2], "int32"),
                Sinks: T.Tensor([heads], "float32"),
                Part: T.Tensor([*part_shape, dim], accum),
                Stat: T.Tensor([*part_shape, 2], accum),
                Merge: T.Tensor([(batch * heads_kv + tail_n) * 2], "int32"),
                chunk: T.int32,
                tail_base: T.int32,
            ):
                with T.Kernel(num_ctas, threads=[128, 3]):
                    launch(
                        Q, K, V, CuQ, CuKV, O, Sched, Sinks, RopeCos, RopeSin, Part, Stat,
                        Merge, chunk, tail_base,
                    )  # fmt: skip

            return main

        @T.prim_func
        def main(
            Q: T.Tensor([total_q, heads, dim], dtype),
            K: T.Tensor([total_kv, heads_kv, dim], dtype),
            V: T.Tensor([total_kv, heads_kv, dim], dtype),
            CuQ: T.Tensor([batch + 1], "int32"),
            CuKV: T.Tensor([batch + 1], "int32"),
            O: T.Tensor([total_q, heads, dim], dtype),
            Sched: T.Tensor([2], "int32"),
            Sinks: T.Tensor([heads], "float32"),
            Part: T.Tensor([*part_shape, dim], accum),
            Stat: T.Tensor([*part_shape, 2], accum),
            Merge: T.Tensor([(batch * heads_kv + tail_n) * 2], "int32"),
            chunk: T.int32,
            tail_base: T.int32,
        ):
            with T.Kernel(num_ctas, threads=[128, 3]):
                launch(
                    Q,
                    K,
                    V,
                    CuQ,
                    CuKV,
                    O,
                    Sched,
                    Sinks,
                    None,
                    None,
                    Part,
                    Stat,
                    Merge,
                    chunk,
                    tail_base,
                )

        return main

    if fuse_rope:

        @T.prim_func
        def main(
            Q: T.Tensor([total_q, heads, dim], dtype),
            K: T.Tensor([total_kv, heads_kv, dim], dtype),
            V: T.Tensor([total_kv, heads_kv, dim], dtype),
            CuQ: T.Tensor([batch + 1], "int32"),
            CuKV: T.Tensor([batch + 1], "int32"),
            RopeCos: T.Tensor([max_position, rope_half], table_dtype),
            RopeSin: T.Tensor([max_position, rope_half], table_dtype),
            O: T.Tensor([total_q, heads, dim], dtype),
            Sched: T.Tensor([2], "int32"),
            Sinks: T.Tensor([heads], "float32"),
        ):
            with T.Kernel(num_ctas, threads=[128, 3]):
                launch(Q, K, V, CuQ, CuKV, O, Sched, Sinks, RopeCos, RopeSin)

        return main

    @T.prim_func
    def main(
        Q: T.Tensor([total_q, heads, dim], dtype),
        K: T.Tensor([total_kv, heads_kv, dim], dtype),
        V: T.Tensor([total_kv, heads_kv, dim], dtype),
        CuQ: T.Tensor([batch + 1], "int32"),
        CuKV: T.Tensor([batch + 1], "int32"),
        O: T.Tensor([total_q, heads, dim], dtype),
        Sched: T.Tensor([2], "int32"),
        Sinks: T.Tensor([heads], "float32"),
    ):
        with T.Kernel(num_ctas, threads=[128, 3]):
            launch(Q, K, V, CuQ, CuKV, O, Sched, Sinks)

    return main


class GQAPrefillVarlenWSFwdKernel(VarlenKernel):
    """SM90 warp-specialized packed prefill for heads of dimension 64 or 128.

    Dimension 256 stays on the general kernel: its shared buffers would need 448 KB,
    about twice the 227 KB an SM90 block may use.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_varlen_sliding_window"})
    _DIMS: tuple[int, ...] = (64, 128)
    _BLOCK_N: int = 128
    _STAGES: int = 2
    # Leave room for per-request metadata within SM90's 227 KB shared-memory limit.
    _MAX_BATCH: int = 448
    _PACK: bool = False

    @classmethod
    def refusal(cls, call) -> "str | None":
        if call.dtype not in ATTENTION_DTYPES:
            return f"requires float16 or bfloat16, got {call.dtype}"
        if call.dim not in cls._DIMS:
            return f"requires head dim in {sorted(cls._DIMS)}, got {call.dim}"
        if call.is_fp8:
            return "does not serve FP8"
        if call.sm_scale is not None and call.sm_scale <= 0:
            return "requires a positive score scale"
        if call.empty_kv:
            return "does not serve an empty KV sequence"
        if call.batch > cls._MAX_BATCH:
            return f"serves at most {cls._MAX_BATCH} sequences, got {call.batch}"
        return super().refusal(call)

    def _make_kernel(
        self, chunked: bool = False, wide: bool = True, tail_n: int = 0, tail_p: int = 1
    ):
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
            # At 64 channels four K/V stages fit shared memory; calls of short requests take
            # them: one warpgroup scans a narrow item with no second warpgroup's products
            # hiding its key loads.
            self._STAGES if wide or self.dim > 64 else 4,
            get_sm_count(self.device_index),
            self.fuse_rope,
            self.max_position,
            self.rotary_dim,
            self.rope_layout,
            self.rope_table_dtype_str,
            self.window_size_left,
            self.window_size_right,
            self.has_sinks,
            self._PACK,
            chunked,
            wide,
            tail_n,
            tail_p,
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
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._require_devices(q=q, k=k, v=v)
        if torch.cuda.is_current_stream_capturing():
            # Each captured graph owns a counter from its private pool.
            counter = torch.zeros(2, dtype=torch.int32, device=q.device)
        else:
            stream = torch.cuda.current_stream(q.device).cuda_stream
            counter = self._counters.get(stream)
            if counter is None:
                counter = torch.zeros(2, dtype=torch.int32, device=q.device)
                self._counters[stream] = counter
        out = torch.empty_like(q)
        program = self.program(self._compile, *self._program_facts(q))
        if self.key_rope is None:
            program(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                out,
                counter,
                sinks if self.has_sinks else self._unused_sinks,
                *self._split_args(q, k),
            )
            return out
        k_rot = self.key_rope(k, cu_seqlens_kv, rope_cos, rope_sin)
        program(
            q,
            k_rot,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            rope_cos,
            rope_sin,
            out,
            counter,
            sinks if self.has_sinks else self._unused_sinks,
            *self._split_args(q, k),
        )
        return out

    def _split_args(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """Optional split-key workspace arguments."""
        return ()

    def _wide(self, q: torch.Tensor) -> bool:
        """Whether the call's mean request holds two 128-row tiles a KV head or more: its
        work is mostly wide items that both consumer warpgroups scan."""
        return q.shape[0] * (self.heads // self.heads_kv) >= 2 * 128 * self.batch

    def _program_facts(self, q: torch.Tensor) -> tuple:
        """What selects among this geometry's programs on a call."""
        return (False, self._wide(q))

    def _compile(self, cfg: dict, chunked: bool = False, wide: bool = True):
        """Return the persistent program for this geometry."""
        if not chunked and wide:
            return self.kernel
        return self._make_kernel(chunked=chunked, wide=wide)


class GQAVarlenPackedWSFwdKernel(GQAPrefillVarlenWSFwdKernel):
    """The warp-specialized program for packings of short requests: a request that needs
    fewer packed tiles than one-head tiles fills each tile with its KV head's group, position
    by position, and reads its keys once a KV head rather than once a query head.

    It serves every call whose group divides a warpgroup's 64 rows; a request that does not
    pack runs one-head tiles, so whether requests are short need not be known on the host.
    Other groups keep the one-head program.
    """

    preferred_over = frozenset({"gqa_varlen_ws", "gqa_varlen_sliding_window"})
    _PACK = True
    # A split tail item scans its keys in this many parts; the last arrival merges their partials.
    _TAIL_PARTS: int = 2

    @classmethod
    def refusal(cls, call) -> "str | None":
        if call.heads_kv <= 0 or call.heads % call.heads_kv != 0:
            return "requires heads a multiple of heads_kv"
        group = call.heads // call.heads_kv
        if group == 1 or 64 % group != 0:
            return "requires a query group that divides a warpgroup's 64 rows"
        return super().refusal(call)

    @property
    def _splits(self) -> bool:
        """Whether the program may split items: causal, unwindowed calls the ranking serves."""
        return (
            self.is_causal
            and self.window_size_left < 0
            and self.window_size_right < 0
            # Ranked calls only: the parts follow the ranking's order of key scans.
            and self.batch <= 128
        )

    @property
    def _tail_n(self) -> int:
        """How many of a call's last wide claims may split: half the SMs."""
        return get_sm_count(self.device_index) // 2

    def _make_kernel(self, chunked: bool = False, wide: bool = True):
        return super()._make_kernel(chunked, wide, self._tail_n, self._TAIL_PARTS)

    def _program_facts(self, q: torch.Tensor) -> tuple:
        """Whether the call may split requests by chunk of keys, then whether it is wide. A tile
        holds at most 128 query rows, so a call has at least ``total_queries * heads / 128``
        unsplit items, and it splits only when they number fewer than the SMs."""
        sms = get_sm_count(self.device_index)
        return (self._splits and q.shape[0] * self.heads < 128 * sms, self._wide(q))

    def _split_args(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """Allocate partial statistics and stream-local merge counters.

        Splitting requires fewer unsplit items than SMs, bounding split requests by
        SMs // heads_kv. Chunk sizing targets two items per SM; workspace includes
        per-request rounding. Each (request, KV head, warpgroup) counter resets on
        its last arrival.
        """
        if not self._splits:
            return ()
        sms = get_sm_count(self.device_index)
        key_tiles = -(-k.shape[0] // self._BLOCK_N)
        chunk = max(8, -(-key_tiles * self.heads_kv // (2 * sms)))
        # A request of t key tiles splits into ceil(t / chunk) parts, and its tiles count at
        # most one more than its keys fill.
        split = min(self.batch, sms // self.heads_kv)
        slots = self.heads_kv * ((key_tiles + split) // chunk + split)
        # The tail split's slots follow: _TAIL_PARTS a tail item, heads_kv-strided per item block.
        tail_n = self._tail_n
        tail_base = slots
        slots += -(-tail_n // self.heads_kv) * self._TAIL_PARTS * self.heads_kv
        part = torch.empty(slots, 128, self.dim, dtype=torch.float32, device=q.device)
        stat = torch.empty(slots, 128, 2, dtype=torch.float32, device=q.device)
        merge_size = (self.batch * self.heads_kv + tail_n) * 2
        if torch.cuda.is_current_stream_capturing():
            merge = torch.zeros(merge_size, dtype=torch.int32, device=q.device)
        else:
            stream = torch.cuda.current_stream(q.device).cuda_stream
            merge = self._merge_counters.get(stream)
            if merge is None:
                merge = self._merge_counters[stream] = torch.zeros(
                    merge_size, dtype=torch.int32, device=q.device
                )
        return part, stat, merge, chunk, tail_base

    def __init__(self, *args, **kwargs) -> None:
        self._merge_counters: dict = {}
        super().__init__(*args, **kwargs)
