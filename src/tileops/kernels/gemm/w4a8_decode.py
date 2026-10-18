"""W4A8 GEMM for at most 32 tokens over the unified INT4 weight order.

The weight rows are the WGMMA M extent and the tokens its N extent. A *unit* is
``unit_warpgroups`` consumer warpgroups and one producer warp sharing a ring of stages; its
*tile* is ``unit_warpgroups * 64`` weight rows, and an *item* is one tile's ``block_k`` chunk
of K. Each CTA runs a contiguous range of items, split over its units; the partial sums of a
tile held by several units or CTAs are added by the holder of the tile's last item.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops.kernels.constants import (
    SHARED_BUFFER_ALIGN_BYTES,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops.kernels.gemm.call_spec import GemmW4A8Call, GemmW4A8FwdInterface
from tileops.kernels.gemm.w4a8_repack import GROUP_SIZE, W4A8_LAYOUT
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import WARP_LANES, get_shared_memory_optin, get_sm_count

__all__ = ["GemmW4A8DecodeKernel"]

# Token tiles from this size stage their output through shared memory; the shared-memory
# estimate and the builder both branch on it.
_STAGED_MIN_TOKENS = 16
# Row stride in bf16 of a warpgroup's staged output tile, read by the shared-memory estimate and
# the builder: the 8-element pad staggers successive token rows by 4 banks.
_STAGE_STRIDE = WGMMA_ROWS + 8
# Partial-sum slots per consumer warpgroup, for the head and the tail of its unit's item range,
# the only tiles a range can hold in part; the estimate, the builder and the workspace size
# them alike.
_PARTIAL_SLOTS = 2


def _shared_bytes(m_tile: int, nw: int, uw: int, block_k: int, stages: int) -> tuple[int, int]:
    """Shared memory of a config's buffers, each rounded up to the buffer alignment.

    Returns:
        The bytes without and with the in-CTA partial-sum slots.
    """
    nu = nw // uw
    groups = block_k // GROUP_SIZE

    def footprint(nbytes: int) -> int:
        return -(-nbytes // SHARED_BUFFER_ALIGN_BYTES) * SHARED_BUFFER_ALIGN_BYTES

    ring = (
        footprint(nu * stages * groups * m_tile * GROUP_SIZE)
        + footprint(nu * stages * uw * (block_k // W4A8_LAYOUT.tile_k) * W4A8_LAYOUT.tile_bytes)
        + footprint(nu * stages * WGMMA_ROWS * uw * groups)
    )
    staged = (
        footprint(nw * m_tile * _STAGE_STRIDE * torch.bfloat16.itemsize)
        if m_tile >= _STAGED_MIN_TOKENS
        else 0
    )
    slot_bytes = WGMMA_ROWS * m_tile * torch.float32.itemsize
    slots = footprint(nw * _PARTIAL_SLOTS * slot_bytes) if nu > 1 else 0
    base = ring + staged + footprint(m_tile * torch.float32.itemsize)
    return base, base + slots


def _check_config(
    m: int, n: int, k: int, cfg: dict, smem_budget: int, sm_count: int
) -> Optional[str]:
    """Return why the builder cannot run *cfg* for ``(m, n, k)``, or ``None``."""
    nt, nw, uw = cfg["block_tokens"], cfg["warpgroups"], cfg["unit_warpgroups"]
    bk, stages, ctas = cfg["block_k"], cfg["num_stages"], cfg["num_ctas"]
    if nt not in GemmW4A8DecodeKernel.token_tiles or nt < m:
        return f"block_tokens {nt} must be one of {GemmW4A8DecodeKernel.token_tiles} and at least m={m}"
    if uw not in (1, 2) or nw not in range(uw, 5, uw):
        return f"warpgroups {nw} must be 1..4 and a multiple of unit_warpgroups {uw} (1 or 2)"
    if uw == 2 and n % (2 * WGMMA_ROWS):
        return f"unit_warpgroups 2 needs n a multiple of {2 * WGMMA_ROWS}, got {n}"
    block_ks = tuple(GROUP_SIZE * groups for groups in (1, 2, 4))
    if bk not in block_ks:
        return f"block_k {bk} must be one of {block_ks}"
    if stages < 2:
        return f"num_stages {stages} must be at least 2"
    if _shared_bytes(nt, nw, uw, bk, stages)[0] > smem_budget:
        return f"stages of {cfg} exceed {smem_budget} bytes of shared memory"
    items = -(-n // (WGMMA_ROWS * uw)) * -(-k // bk)
    if not 1 <= ctas <= min(sm_count, items):
        return f"num_ctas {ctas} must be in 1..{min(sm_count, items)}"
    return None


def _grid(n: int, k: int, uw: int, block_k: int, sm_count: int, fill: tuple[int, int]) -> int:
    """One CTA per tile above ``fill`` of the SMs and at most one wave; otherwise split K."""
    tiles = -(-n // (WGMMA_ROWS * uw))
    numerator, denominator = fill
    if denominator * tiles >= numerator * sm_count and tiles <= sm_count:
        return tiles
    return min(sm_count, tiles * -(-k // block_k))


def _select_config(m: int, n: int, k: int, smem_budget: int, sm_count: int) -> dict:
    """Select a config that passes the builder's single legality check.

    Raises:
        ValueError: No stage count of the selected family fits the device.
    """
    # Every threshold below was measured on H200 over the manifest's W4A8 rows.
    # 8-token tiles: a weight past 32 MiB runs three warpgroups on three stages, a smaller
    # one two on four.
    small_weight_bytes, small_units, small_stages = 32 * 1024 * 1024, (2, 3), (4, 3)
    # 16- and 32-token tiles: two warpgroups on three stages.
    wide_units, wide_stages = 2, 3
    # 512-K chunks where K allows, else 256-K chunks with twice the stages in flight.
    narrow_k, wide_k = 256, 512
    nt = next(t for t in GemmW4A8DecodeKernel.token_tiles if t >= m)
    if nt == GemmW4A8DecodeKernel.token_tiles[0]:
        large_weight = n * k // 2 > small_weight_bytes
        nw = small_units[large_weight]
        uw = 1
        stages = small_stages[large_weight]
    else:
        nw = wide_units
        uw = 2 if n % (2 * WGMMA_ROWS) == 0 else 1
        stages = wide_stages
    bk = wide_k if k % wide_k == 0 else narrow_k
    if bk == narrow_k:
        stages *= wide_k // narrow_k
    # One CTA per tile once the tiles fill three quarters of the SMs within one wave.
    ctas = _grid(n, k, uw, bk, sm_count, (3, 4))
    tiles = -(-n // (WGMMA_ROWS * uw))
    nchunk = -(-k // bk)
    items = tiles * nchunk
    balanced = 2 * tiles
    # At short K, two contributors per tile pay when they keep five sixths of the SMs busy
    # (with the tile conditions below). The K test passes up to K = 3200, and up to K = 3584
    # only when the extra K work stays bounded, which keeps the original grid on the
    # K = 3456 shapes it measured faster on.
    fill_num, fill_den = 5, 6
    if (
        k <= 3584
        and nt >= _STAGED_MIN_TOKENS
        and uw == 2
        and n % (WGMMA_ROWS * uw) == 0
        and fill_num * sm_count <= fill_den * balanced <= fill_den * sm_count
        and balanced <= items
        and (
            k <= 3200
            or (
                bk == wide_k
                and -(-nchunk // 2) <= -(-items // ctas)
                and (k // GROUP_SIZE) * (ctas - balanced) <= ctas
            )
        )
    ):
        ctas = balanced
    for st in range(stages, 1, -1):
        cfg = dict(
            block_tokens=nt,
            warpgroups=nw,
            unit_warpgroups=uw,
            block_k=bk,
            num_stages=st,
            num_ctas=ctas,
        )
        if _check_config(m, n, k, cfg, smem_budget, sm_count) is None:
            return cfg
    raise ValueError(f"no legal W4A8 decode config for m={m}, n={n}, k={k}")


@functools.lru_cache(maxsize=32)
def _gemm_w4a8_decode_kernel(
    m: int,
    n: int,
    k: int,
    block_tokens: int,
    warpgroups: int,
    unit_warpgroups: int,
    block_k: int,
    num_stages: int,
    num_ctas: int,
    shared_slots: bool,
) -> Callable:
    """Compile the decode GEMM for one shape and config.

    The tensors are ``(activation, full_blocks, compact_block, group_scale, channel_scale,
    activation_scale, partials, flags, unit_partials, out)``: the packed weight's whole 64-row
    blocks as ``[n // 64, k // 64, 256]`` uint64 and its compact last block (``n % 64`` rows) as
    1-D uint64, each a placeholder when absent. *shared_slots* keeps the in-CTA partial sums in
    shared memory instead of *unit_partials*.
    """
    nt, nw, uw, bk, st = block_tokens, warpgroups, unit_warpgroups, block_k, num_stages
    nu = nw // uw
    groups_per_chunk = bk // GROUP_SIZE
    k_groups = k // GROUP_SIZE
    unit_k = W4A8_LAYOUT.unit_k
    k_slices = k // unit_k
    chunk_slices = bk // unit_k
    nchunk = -(-k // bk)
    full_blocks = n // W4A8_LAYOUT.tile_rows
    compact_units = (n % W4A8_LAYOUT.tile_rows) // W4A8_LAYOUT.unit_rows
    ragged = compact_units > 0
    tiles = -(-n // (WGMMA_ROWS * uw))
    items = tiles * nchunk
    unit_scale_rows = WGMMA_ROWS * uw
    stage_scale_bytes = unit_scale_rows * groups_per_chunk
    # A lane loads the group scales of these rows of its unit's tile, one 32-bit word per row.
    scale_words = unit_scale_rows // WARP_LANES
    # FP32 accumulator registers per thread of one m64 x nt WGMMA.
    acc_regs = WGMMA_ROWS * nt // WARPGROUP_THREADS
    # Thread-contiguous FP32 partial sums of one warpgroup.
    slot = WARPGROUP_THREADS * acc_regs
    warps_per_warpgroup = WARPGROUP_THREADS // WARP_LANES
    threads = WARPGROUP_THREADS * nw + WARP_LANES * nu
    staged = nt >= _STAGED_MIN_TOKENS
    # A one-unit CTA runs its items rotated, the tail it hands to a later CTA first and the head
    # whose earlier parts it receives last.
    rotated = nu == 1
    # WGMMA descriptors address shared memory in 16-byte units.
    desc_unit = 16
    act_stage16 = groups_per_chunk * nt * GROUP_SIZE // desc_unit
    # TileLang numbers its own named barriers upward from 3 (3 and 4 in this program); each
    # consumer warpgroup takes 8 + its index, below the hardware's 16.
    bar_base = 8
    unit_rows = W4A8_LAYOUT.unit_rows
    # A thread's accumulator rows are srow0 and srow0 + row_pair.
    row_pair = unit_rows // 2
    units_per_block = W4A8_LAYOUT.tile_rows // unit_rows
    # Bytes of one row block's K slice; the weight views hold them as uint64.
    slice_bytes = W4A8_LAYOUT.k_half_bytes
    slice_words = slice_bytes // torch.uint64.itemsize
    unit_bytes = W4A8_LAYOUT.unit_bytes
    unit_words = unit_bytes // torch.uint64.itemsize
    # A lane's share of one unit: the 16 bytes one ld.shared.v4 of load_slice returns.
    lane_words = unit_bytes // WARP_LANES // torch.uint32.itemsize
    # K of one WGMMA instruction for FP8 operands.
    mma_k = 32
    steps_per_group = GROUP_SIZE // mma_k
    steps_per_slice = unit_k // mma_k
    slices_per_group = GROUP_SIZE // unit_k
    # Two K64 slices of A fragments: decoding one overlaps the other's WGMMAs.
    ring_slices = 2
    ring = ring_slices * steps_per_slice
    # Registers of one plus/minus A fragment pair, each m64 x k32 FP8 fragment one byte per
    # element over the warpgroup's 32-bit registers.
    frag_regs = 2 * WGMMA_ROWS * mma_k // (WARPGROUP_THREADS * torch.uint32.itemsize)
    # In-CTA partial-sum workspace; a placeholder, never read, when the slots are shared.
    ws_floats = num_ctas * nw * _PARTIAL_SLOTS * slot if not shared_slots else 4
    load_slice = f"tileops::w4a8::load_slice<{uw}, {chunk_slices}, {units_per_block}, {unit_bytes}>"

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16", *csrc_include("w4a8_decode_helper.h")],
    )
    def build():
        def clamp_row(row):
            return T.min(row, n - 1) if ragged else row

        def channel_scale(ch, tile, row):
            return T.call_extern(
                "float32",
                "tileops::w4a8::load_early",
                T.address_of(ch[0]),
                clamp_row(tile * WGMMA_ROWS * uw + row),
            )

        def units_of(item):
            # A slice of the compact last tile holds only its ragged units.
            if not ragged:
                return units_per_block
            return T.if_then_else(item // nchunk < full_blocks, units_per_block, compact_units)

        def item_of(kk, gb, ge, he, ts):
            # Rotated order: [ts, ge), then [he, ts), then [gb, he).
            tail = ge - ts
            return T.if_then_else(
                kk < tail,
                ts + kk,
                T.if_then_else(kk < ge - he, he + kk - tail, gb + kk - (ge - he)),
            )

        @T.macro
        def item_body(
            flip,
            acc,
            frag,
            tables,
            group,
            vnext,
            vcur,
            desc,
            wring,
            empty,
            stage,
            atom,
            kc,
            fresh,
            release,
            lane,
            units_per_slice,
        ):
            fl = "true" if flip else "false"
            T.call_extern(
                "handle",
                load_slice,
                vnext.data,
                wring.access_ptr("r"),
                stage,
                atom,
                0,
                units_per_slice,
            )
            for gi in T.unroll(groups_per_chunk):
                if (not flip) or gi * GROUP_SIZE < kc:
                    T.call_extern(
                        "handle", f"tileops::w4a8::group_tables<{fl}>", group.data, tables.data, gi
                    )
                    for h2 in T.unroll(slices_per_group):
                        sl = gi * slices_per_group + h2
                        for q in T.unroll(lane_words):
                            vcur[q] = vnext[q]
                        if sl + 1 < chunk_slices:
                            T.call_extern(
                                "handle",
                                load_slice,
                                vnext.data,
                                wring.access_ptr("r"),
                                stage,
                                atom,
                                sl + 1,
                                units_per_slice,
                            )
                        for pp in T.unroll(steps_per_slice):
                            T.call_extern(
                                "handle",
                                f"tileops::w4a8::dequant_step<{fl}>",
                                T.address_of(
                                    frag[((sl * steps_per_slice + pp) % ring) * frag_regs]
                                ),
                                vcur.data,
                                group.data,
                                pp,
                            )
                        T.warpgroup_fence_operand(acc, num_regs=acc_regs)
                        T.warpgroup_arrive()
                        for pp in T.unroll(steps_per_slice):
                            s = sl * steps_per_slice + pp
                            boff = (
                                stage * act_stage16
                                + (
                                    (s // steps_per_group) * nt * GROUP_SIZE
                                    + (s % steps_per_group) * mma_k
                                )
                                // desc_unit
                            )
                            T.ptx_wgmma_rs(
                                "float32",
                                f"m{WGMMA_ROWS}n{nt}k{mma_k}",
                                True,
                                "e4m3",
                                "e4m3",
                                "fp32",
                                frag.data,
                                (s % ring) * frag_regs,
                                desc.data,
                                boff,
                                acc.data,
                                0,
                                T.if_then_else(fresh != 0 and s == 0, 0, 1),
                                True,
                                True,
                            )
                            T.ptx_wgmma_rs(
                                "float32",
                                f"m{WGMMA_ROWS}n{nt}k{mma_k}",
                                True,
                                "e4m3",
                                "e4m3",
                                "fp32",
                                frag.data,
                                (s % ring) * frag_regs + frag_regs // 2,
                                desc.data,
                                boff,
                                acc.data,
                                0,
                                1,
                                False,
                                True,
                            )
                        T.warpgroup_commit_batch()
                        T.warpgroup_fence_operand(acc, num_regs=acc_regs)
                        T.warpgroup_wait(ring_slices - 1)
                        # The previous item's stage is free once its WGMMAs have retired.
                        if sl == ring_slices - 1 and release >= 0:
                            T.sync_warp()
                            if lane == 0:
                                T.barrier_arrive(empty[release])

        @T.macro
        def consume(fixed, stage, kc, fresh, release, units_per_slice):
            acc, frag, tables, group, vnext, vcur, desc, negative, wring, scring, empty = fixed[:11]
            atom, srow0, lane = fixed[11:]
            # item_tables fills the tables both branches read, so it is not part of the test.
            negative[0] = T.call_extern(
                "int32",
                f"tileops::w4a8::item_tables<{groups_per_chunk}>",
                tables.data,
                scring.access_ptr("r"),
                stage * stage_scale_bytes,
                srow0,
            )
            body = (acc, frag, tables, group, vnext, vcur, desc, wring, empty, stage, atom, kc)
            body += (fresh, release, lane, units_per_slice)
            if negative[0] == 0 and kc == bk:
                item_body(False, *body)
            else:
                item_body(True, *body)

        @T.macro
        def drain(acc):
            T.warpgroup_wait(0)
            T.warpgroup_fence_operand(acc, num_regs=acc_regs)

        @T.macro
        def store_tile(v, tile, ch0, ch1, fixed):
            out, sub, wi, g, srow0, as_s, stage_out, wg, lane = fixed
            if staged:
                T.call_extern(
                    "handle",
                    f"tileops::w4a8::store_staged<{nt}, {_STAGE_STRIDE}>",
                    v.data,
                    T.address_of(out[0, 0]),
                    m,
                    n,
                    T.int64(tile * WGMMA_ROWS * uw + sub * WGMMA_ROWS),
                    wi * W4A8_LAYOUT.unit_rows + g,
                    ch0,
                    ch1,
                    T.address_of(as_s[0]),
                    T.address_of(stage_out[wg * nt * _STAGE_STRIDE]),
                    bar_base + wg,
                )
            else:
                # WGMMA accumulator layout: element c of lane 4 g + t is row g + 8 ((c >> 1) & 1),
                # token 8 (c >> 2) + 2 t + (c & 1).
                for c in T.unroll(acc_regs):
                    tok = (c // 4) * 8 + 2 * (lane % 4) + c % 2
                    row = tile * WGMMA_ROWS * uw + srow0 + row_pair * ((c // 2) % 2)
                    if tok < m and row < n:
                        out[tok, row] = T.cast(
                            v[c] * T.if_then_else((c // 2) % 2 == 1, ch1, ch0) * as_s[tok],
                            "bfloat16",
                        )

        @T.prim_func
        def main(
            act: T.Tensor((m, k), "float8_e4m3"),  # type: ignore
            wfull: T.Tensor((max(full_blocks, 1), k_slices, slice_words), "uint64"),  # type: ignore
            wtail: T.Tensor((max(compact_units, 1) * unit_words * k_slices,), "uint64"),  # type: ignore
            gs: T.Tensor((n, k_groups), "uint8"),  # type: ignore
            ch: T.Tensor((n,), "float32"),  # type: ignore
            asc: T.Tensor((m,), "float32"),  # type: ignore
            xs: T.Tensor((num_ctas * uw * slot,), "float32"),  # type: ignore
            xf: T.Tensor((num_ctas * uw,), "int32"),  # type: ignore
            ws: T.Tensor((ws_floats,), "float32"),  # type: ignore
            out: T.Tensor((m, n), "bfloat16"),  # type: ignore
        ):
            with T.Kernel(num_ctas, threads=threads) as cid:
                aring = T.alloc_shared((nu * st * groups_per_chunk, nt, GROUP_SIZE), "float8_e4m3")
                wring = T.alloc_shared((nu * st, uw, chunk_slices, slice_words), "uint64")
                scring = T.alloc_shared((nu * st * stage_scale_bytes,), "uint8")
                slots = (
                    T.alloc_shared((nw * _PARTIAL_SLOTS * slot if not rotated else 4,), "float32")
                    if shared_slots
                    else ws
                )
                sbase = 0 if shared_slots else cid * (nw * _PARTIAL_SLOTS * slot)
                stage_out = T.alloc_shared((nw * nt * _STAGE_STRIDE if staged else 8,), "bfloat16")
                as_s = T.alloc_shared((nt,), "float32")
                full = T.alloc_barrier([1] * (nu * st))
                # Each consumer warp of the unit arrives once.
                empty = T.alloc_barrier([warps_per_warpgroup * uw] * (nu * st))
                T.annotate_layout({aring: tilelang.layout.make_swizzled_layout(aring)})

                tx = T.get_thread_binding()
                warp = tx // WARP_LANES
                lane = tx % WARP_LANES
                gb = T.alloc_var("int32", init=0)
                ge = T.alloc_var("int32", init=0)
                count = T.alloc_var("int32", init=0)
                gb = cid * items // num_ctas
                ge = (cid + 1) * items // num_ctas
                count = ge - gb
                he = T.alloc_var("int32", init=0)
                ts = T.alloc_var("int32", init=0)
                he = T.if_then_else(
                    gb % nchunk == 0 or (gb // nchunk + 1) * nchunk > ge,
                    gb,
                    (gb // nchunk + 1) * nchunk,
                )
                ts = T.if_then_else(
                    ((ge - 1) // nchunk + 1) * nchunk <= ge,
                    ge,
                    T.max(he, ((ge - 1) // nchunk) * nchunk),
                )

                if tx == 0:
                    for i in T.serial(m):
                        as_s[i] = asc[i]
                    for i in T.serial(nt - m):
                        as_s[m + i] = T.float32(0)

                if warp >= warps_per_warpgroup * nw:
                    # The warps past the consumer warpgroups are the producers, one per unit.
                    pu = warp - warps_per_warpgroup * nw
                    ib = T.alloc_var("int32", init=0)
                    ie = T.alloc_var("int32", init=0)
                    ib = gb + pu * count // nu
                    ie = gb + (pu + 1) * count // nu
                    nx = T.alloc_local((scale_words,), "uint32")
                    cur = T.alloc_local((scale_words,), "uint32")
                    wsa = T.alloc_var("uint32", init=0)
                    if ragged:
                        wsa = T.call_extern(
                            "uint32", "tileops::w4a8::shared_address", wring.access_ptr("r")
                        )
                    if ib < ie:
                        T.call_extern(
                            "handle",
                            f"tileops::w4a8::load_item_scales<{unit_scale_rows}, {groups_per_chunk}>",
                            nx.data,
                            T.address_of(gs[0, 0]),
                            k_groups,
                            item_of(0, gb, ge, he, ts) if rotated else ib,
                            nchunk,
                            n,
                        )
                    T.sync_threads()
                    for kk in T.serial(count // nu + 1):
                        i = item_of(kk, gb, ge, he, ts) if rotated else ib + kk
                        if (kk < count) if rotated else (i < ie):
                            s = kk % st
                            T.barrier_wait(empty[pu * st + s], ((kk // st) & 1) ^ 1)
                            tile = i // nchunk
                            c = i % nchunk
                            if not ragged or tile < full_blocks:
                                T.tma_copy(
                                    wfull[
                                        tile * uw : tile * uw + uw,
                                        c * chunk_slices : c * chunk_slices + chunk_slices,
                                        0:slice_words,
                                    ],
                                    wring[pu * st + s, :, :, :],
                                    barrier=full[pu * st + s],
                                    leader_scope_threads=WARP_LANES,
                                )
                            else:
                                if lane == 0:
                                    T.call_extern(
                                        "handle",
                                        f"tileops::w4a8::load_compact_block<{unit_bytes}>",
                                        wsa,
                                        T.int64((pu * st + s) * uw * chunk_slices * slice_bytes),
                                        wtail.access_ptr("r"),
                                        c * chunk_slices,
                                        T.min(chunk_slices, k_slices - c * chunk_slices),
                                        compact_units,
                                        T.address_of(full[pu * st + s]),
                                    )
                            for bx in T.unroll(groups_per_chunk):
                                k0 = c * bk + bx * GROUP_SIZE
                                T.tma_copy(
                                    act[0:nt, k0 : k0 + GROUP_SIZE],
                                    aring[(pu * st + s) * groups_per_chunk + bx, :, :],
                                    barrier=full[pu * st + s],
                                    leader_scope_threads=WARP_LANES,
                                )
                            for h in T.unroll(scale_words):
                                cur[h] = nx[h]
                            if (kk + 1 < count) if rotated else (i + 1 < ie):
                                T.call_extern(
                                    "handle",
                                    f"tileops::w4a8::load_item_scales<{unit_scale_rows}, {groups_per_chunk}>",
                                    nx.data,
                                    T.address_of(gs[0, 0]),
                                    k_groups,
                                    item_of(kk + 1, gb, ge, he, ts) if rotated else i + 1,
                                    nchunk,
                                    n,
                                )
                            T.call_extern(
                                "handle",
                                f"tileops::w4a8::store_item_scales<{unit_scale_rows}, {groups_per_chunk}>",
                                T.address_of(scring[(pu * st + s) * stage_scale_bytes]),
                                cur.data,
                            )
                            T.sync_warp()
                            if lane == 0:
                                T.barrier_arrive(full[pu * st + s])
                    if not rotated:
                        # The consumers' two CTA barriers: partial sums parked, then read.
                        T.sync_threads()
                        T.sync_threads()
                else:
                    # Consumer warpgroup wg computes rows [sub * 64, sub * 64 + 64) of each tile
                    # of unit ul.
                    wg = T.alloc_var("int32", init=0)
                    wg = T.call_extern("int32", "tileops::w4a8::warpgroup_uniform")
                    ul = wg // uw
                    sub = wg % uw
                    wi = warp % warps_per_warpgroup
                    g = lane // 4
                    srow0 = sub * WGMMA_ROWS + wi * unit_rows + g
                    ib = T.alloc_var("int32", init=0)
                    ie = T.alloc_var("int32", init=0)
                    ib = gb + ul * count // nu
                    ie = gb + (ul + 1) * count // nu
                    acc = T.alloc_local((acc_regs,), "float32")
                    frag = T.alloc_local((ring * frag_regs,), "uint32")
                    # The word counts item_tables and group_tables fill, which their array
                    # parameters fix.
                    tables = T.alloc_local((5,), "uint32")
                    group = T.alloc_local((8,), "uint32")
                    vnext = T.alloc_local((lane_words,), "uint32")
                    vcur = T.alloc_local((lane_words,), "uint32")
                    negative = T.alloc_local((1,), "int32")
                    desc = T.alloc_wgmma_desc()
                    T.call_extern(
                        "handle",
                        "tileops::w4a8::swizzled_descriptor",
                        desc[0],
                        aring.access_ptr("r"),
                    )
                    park0 = T.alloc_local((acc_regs,), "float32")
                    park1 = T.alloc_local((acc_regs,), "float32")
                    ct_cur = T.alloc_var("int32", init=-1)
                    seg_begin = T.alloc_var("int32", init=0)
                    prev_st = T.alloc_var("int32", init=-1)
                    fresh = T.alloc_var("int32", init=1)
                    parked0 = T.alloc_var("int32", init=0)
                    parked1 = T.alloc_var("int32", init=0)
                    pch0 = T.alloc_var("float32", init=0)
                    pch1 = T.alloc_var("float32", init=0)
                    stores = (out, sub, wi, g, srow0, as_s, stage_out, wg, lane)
                    consumes = (
                        acc,
                        frag,
                        tables,
                        group,
                        vnext,
                        vcur,
                        desc,
                        negative,
                        wring,
                        scring,
                    )
                    consumes += (empty, sub * units_per_block + wi, srow0, lane)
                    T.sync_threads()

                    if rotated:
                        seg_end = T.alloc_var("int32", init=0)
                        for kk in T.serial(count + 1):
                            i = item_of(kk, gb, ge, he, ts)
                            # -2 is no tile and not ct_cur's initial -1: kk == count flushes.
                            ct = T.if_then_else(kk < count, i // nchunk, -2)
                            if ct != ct_cur:
                                if ct_cur >= 0:
                                    drain(acc)
                                    if seg_end < (ct_cur + 1) * nchunk:
                                        # A part of a tile another CTA owns: handed on at once.
                                        T.call_extern(
                                            "handle",
                                            f"tileops::w4a8::publish<{nt}>",
                                            T.address_of(xs[(cid * uw + sub) * slot]),
                                            T.address_of(xf[cid * uw + sub]),
                                            acc.data,
                                            bar_base + wg,
                                        )
                                    else:
                                        if seg_begin > ct_cur * nchunk:
                                            T.call_extern(
                                                "handle",
                                                f"tileops::w4a8::receive_all<{nt}>",
                                                acc.data,
                                                T.address_of(xs[0]),
                                                T.address_of(xf[0]),
                                                ct_cur * nchunk,
                                                items,
                                                num_ctas,
                                                cid,
                                                uw,
                                                sub,
                                                bar_base + wg,
                                            )
                                        store_tile(acc, ct_cur, pch0, pch1, stores)
                                if kk < count:
                                    ct_cur = ct
                                    seg_begin = i
                                    pch0 = channel_scale(ch, ct, srow0)
                                    pch1 = channel_scale(ch, ct, srow0 + row_pair)
                                    fresh = 1
                            if kk < count:
                                s = kk % st
                                T.barrier_wait(full[s], (kk // st) & 1)
                                c = i % nchunk
                                consume(
                                    consumes,
                                    s,
                                    T.min(bk, k - c * bk),
                                    fresh,
                                    T.if_then_else(prev_st >= 0, prev_st, -1),
                                    units_of(i),
                                )
                                fresh = 0
                                prev_st = s
                                seg_end = i + 1
                        drain(acc)
                        if prev_st >= 0 and lane == 0:
                            T.barrier_arrive(empty[prev_st])
                    else:
                        for kk in T.serial(count // nu + 1):
                            i = ib + kk
                            if i < ie:
                                ct = i // nchunk
                                if ct != ct_cur:
                                    if ct_cur >= 0:
                                        drain(acc)
                                        if (
                                            seg_begin == ct_cur * nchunk
                                            and i == (ct_cur + 1) * nchunk
                                        ):
                                            store_tile(acc, ct_cur, pch0, pch1, stores)
                                        elif seg_begin == ib:
                                            for q in T.unroll(acc_regs):
                                                park0[q] = acc[q]
                                            parked0 = 1
                                        else:
                                            for q in T.unroll(acc_regs):
                                                park1[q] = acc[q]
                                            parked1 = 1
                                    ct_cur = ct
                                    seg_begin = i
                                    pch0 = channel_scale(ch, ct, srow0)
                                    pch1 = channel_scale(ch, ct, srow0 + row_pair)
                                    fresh = 1
                                s = kk % st
                                T.barrier_wait(full[ul * st + s], (kk // st) & 1)
                                c = i % nchunk
                                consume(
                                    consumes,
                                    ul * st + s,
                                    T.min(bk, k - c * bk),
                                    fresh,
                                    T.if_then_else(prev_st >= 0, ul * st + prev_st, -1),
                                    units_of(i),
                                )
                                fresh = 0
                                prev_st = s
                        drain(acc)
                        if ct_cur >= 0 and prev_st >= 0 and lane == 0:
                            T.barrier_arrive(empty[ul * st + prev_st])
                        if ct_cur >= 0:
                            if seg_begin == ct_cur * nchunk and ie == (ct_cur + 1) * nchunk:
                                store_tile(acc, ct_cur, pch0, pch1, stores)
                            elif seg_begin == ib:
                                for q in T.unroll(acc_regs):
                                    park0[q] = acc[q]
                                parked0 = 1
                            else:
                                for q in T.unroll(acc_regs):
                                    park1[q] = acc[q]
                                parked1 = 1
                        # Each warpgroup parks its head and tail partial sums in its own two slots.
                        if parked0 == 1:
                            T.call_extern(
                                "handle",
                                f"tileops::w4a8::park<{nt}>",
                                T.address_of(slots[sbase + (wg * _PARTIAL_SLOTS) * slot]),
                                park0.data,
                            )
                        if parked1 == 1:
                            T.call_extern(
                                "handle",
                                f"tileops::w4a8::park<{nt}>",
                                T.address_of(slots[sbase + (wg * _PARTIAL_SLOTS + 1) * slot]),
                                park1.data,
                            )
                        T.sync_threads()
                        if ib < ie:
                            cto = ib // nchunk
                            lfirst = T.max(gb, cto * nchunk)
                            llast = T.min(ge, (cto + 1) * nchunk) - 1
                            tl_last = (ge - 1) // nchunk
                            # This CTA's part of its last tile goes out first; publishing never waits.
                            if ie == ge and tl_last != cto and (tl_last + 1) * nchunk > ge:
                                T.call_extern(
                                    "handle",
                                    f"tileops::w4a8::publish<{nt}>",
                                    T.address_of(xs[(cid * uw + sub) * slot]),
                                    T.address_of(xf[cid * uw + sub]),
                                    park1.data,
                                    bar_base + wg,
                                )
                            if llast < ie and (ib > cto * nchunk or (cto + 1) * nchunk > ge):
                                # This unit holds tile cto's last in-CTA item: add the earlier units'.
                                for v in T.serial(nu):
                                    if v < ul:
                                        vb = gb + v * count // nu
                                        ve = gb + (v + 1) * count // nu
                                        if vb < ve and ve > lfirst:
                                            T.call_extern(
                                                "handle",
                                                f"tileops::w4a8::add_parked<{nt}>",
                                                park0.data,
                                                T.address_of(
                                                    slots[
                                                        sbase
                                                        + (
                                                            (v * uw + sub) * _PARTIAL_SLOTS
                                                            + T.if_then_else(
                                                                vb >= cto * nchunk, 0, 1
                                                            )
                                                        )
                                                        * slot
                                                    ]
                                                ),
                                            )
                                # Tile cto starts in an earlier CTA: add every earlier CTA's part.
                                if (cto + 1) * nchunk <= ge and cto * nchunk < gb:
                                    T.call_extern(
                                        "handle",
                                        f"tileops::w4a8::receive_all<{nt}>",
                                        park0.data,
                                        T.address_of(xs[0]),
                                        T.address_of(xf[0]),
                                        cto * nchunk,
                                        items,
                                        num_ctas,
                                        cid,
                                        uw,
                                        sub,
                                        bar_base + wg,
                                    )
                                if (cto + 1) * nchunk > ge:
                                    T.call_extern(
                                        "handle",
                                        f"tileops::w4a8::publish<{nt}>",
                                        T.address_of(xs[(cid * uw + sub) * slot]),
                                        T.address_of(xf[cid * uw + sub]),
                                        park0.data,
                                        bar_base + wg,
                                    )
                                else:
                                    c0 = channel_scale(ch, cto, srow0)
                                    c1 = channel_scale(ch, cto, srow0 + row_pair)
                                    store_tile(park0, cto, c0, c1, stores)
                        T.sync_threads()

        return main

    return build()


class GemmW4A8DecodeKernel(Kernel, GemmW4A8FwdInterface):
    """W4A8 NT GEMM for at most 32 tokens over the prepacked weight.

    Each instance owns its cross-CTA flag workspace, so one instance must not run on two
    streams at once; the flags are zero again when a launch ends, so CUDA graph replays need
    no reset.

    Args:
        m: Token rows, 1..32.
        n: Output columns, a multiple of 16.
        k: Contraction dim, a multiple of 128.
        config: Optional explicit config; defaults to :attr:`default_config`.
        device_index: CUDA device the kernel is built for.
    """

    token_tiles = (8, 16, 32)

    # WGMMA, TMA and mbarrier transaction counts.
    supported_archs: list[int] = [90]
    # The tuner times one JIT launch, while a config here also sizes the workspaces.
    autotune_configs = None

    @classmethod
    def refusal(cls, call: GemmW4A8Call) -> Optional[str]:
        """Serve calls of at most 32 tokens with 128-wide groups that some config runs."""
        if call.m > GemmW4A8DecodeKernel.token_tiles[-1]:
            return f"serves m <= {GemmW4A8DecodeKernel.token_tiles[-1]}, got {call.m}"
        if call.group_size != GROUP_SIZE:
            return f"requires group_size {GROUP_SIZE}"
        if (
            call.m < 1
            or call.n <= 0
            or call.k <= 0
            or call.n % W4A8_LAYOUT.unit_rows
            or call.k % W4A8_LAYOUT.tile_k
        ):
            return f"no config covers m={call.m}, n={call.n}, k={call.k}"
        try:
            _select_config(call.m, call.n, call.k, call.smem_budget, call.sm_count)
        except ValueError as exc:
            return str(exc)
        return None

    @classmethod
    def entry_for(cls, call: GemmW4A8Call) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (call.m, call.n, call.k, call.group_size, index)
        return identity, lambda: cls(call.m, call.n, call.k, device_index=index)

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.m, self.n, self.k = m, n, k
        self.config = {}
        self.init_config(config)
        smem = get_shared_memory_optin(self.device_index)
        reason = _check_config(m, n, k, self.config, smem, get_sm_count(self.device_index))
        if reason is not None:
            raise ValueError(reason)
        cfg = self.config
        nt, nw, uw, ctas = (
            cfg["block_tokens"],
            cfg["warpgroups"],
            cfg["unit_warpgroups"],
            cfg["num_ctas"],
        )
        shared_slots = _shared_bytes(nt, nw, uw, cfg["block_k"], cfg["num_stages"])[1] <= smem
        self.kernel = _gemm_w4a8_decode_kernel(
            m, n, k, nt, nw, uw, cfg["block_k"], cfg["num_stages"], ctas, shared_slots
        )
        device = (
            torch.device("cuda", self.device_index) if self.device_index is not None else "cuda"
        )
        # FP32 partial sums of one warpgroup's m64 x nt accumulator.
        slot = WGMMA_ROWS * nt
        self._partials = torch.empty(ctas * uw * slot, dtype=torch.float32, device=device)
        # Zero between launches: a publisher sets its flag to 1 and its one reader clears it.
        self._flags = torch.zeros(ctas * uw, dtype=torch.int32, device=device)
        self._unit_partials = torch.empty(
            4 if shared_slots else ctas * nw * _PARTIAL_SLOTS * slot,
            dtype=torch.float32,
            device=device,
        )
        layout = W4A8_LAYOUT
        full, compact = n // layout.tile_rows, (n % layout.tile_rows) // layout.unit_rows
        slice_words = layout.k_half_bytes // torch.uint64.itemsize
        unit_words = layout.unit_bytes // torch.uint64.itemsize
        k_slices = k // layout.unit_k
        # Views the kernel takes for a block kind the weight does not have; never read.
        self._full_placeholder = (
            None
            if full
            else torch.zeros((1, k_slices, slice_words), dtype=torch.uint64, device=device)
        )
        self._compact_placeholder = (
            None
            if compact
            else torch.zeros(unit_words * k_slices, dtype=torch.uint64, device=device)
        )

    @property
    def default_config(self) -> dict:
        """A legal config for this shape and device."""
        return _select_config(
            self.m,
            self.n,
            self.k,
            get_shared_memory_optin(self.device_index),
            get_sm_count(self.device_index),
        )

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_group_scale: torch.Tensor,
        weight_channel_scale: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply FP8 activations by the prepacked, group-scaled INT4 weight.

        Args:
            activation: ``[M, K]`` ``float8_e4m3fn``.
            packed_weight: ``[N, K / 2]`` ``uint8`` in the repacked order.
            weight_group_scale: ``[N, K / 128]`` ``float8_e4m3fn``.
            weight_channel_scale: ``[N]`` ``float32``.
            activation_scale: ``[M]`` ``float32``.

        Returns:
            A new ``[M, N]`` ``bfloat16`` tensor.
        """
        n, k, layout = self.n, self.k, W4A8_LAYOUT
        full = n // layout.tile_rows
        flat = packed_weight.reshape(-1)
        # Whole row blocks of tile_rows * K / 2 bytes, then the compact last block.
        full_bytes = full * (k // layout.tile_k) * layout.tile_bytes
        wfull = (
            flat[:full_bytes].view(torch.uint64).view(full, k // layout.unit_k, -1)
            if full
            else self._full_placeholder
        )
        wtail = (
            flat[full_bytes:].view(torch.uint64)
            if self._compact_placeholder is None
            else self._compact_placeholder
        )
        out = torch.empty((self.m, n), dtype=torch.bfloat16, device=activation.device)
        self.kernel(
            activation,
            wfull,
            wtail,
            weight_group_scale.view(torch.uint8),
            weight_channel_scale,
            activation_scale,
            self._partials,
            self._flags,
            self._unit_partials,
            out,
        )
        return out
