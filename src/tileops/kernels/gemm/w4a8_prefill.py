"""W4A8 GEMM for prefill token counts over the unified INT4 weight order."""

import functools
from dataclasses import dataclass
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops.kernels.constants import (
    SHARED_BUFFER_ALIGN_BYTES,
    TMA_DTYPE_UINT8,
    TMA_INTERLEAVE_NONE,
    TMA_L2_PROMOTION_128B,
    TMA_OOB_FILL_NONE,
    TMA_SWIZZLE_128B,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops.kernels.gemm.call_spec import GemmW4A8Call, GemmW4A8FwdInterface
from tileops.kernels.gemm.w4a8_repack import GROUP_SIZE, W4A8_LAYOUT
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import WARP_LANES, get_shared_memory_optin, get_sm_count

__all__ = ["GemmW4A8PrefillKernel"]


@dataclass(frozen=True)
class W4A8PrefillLayout:
    """Extents of the CTA plan that the schedule, the config check and the builder share.

    An n-tile is ``block_n`` weight rows: one unified-order row block per consumer warpgroup.
    """

    consumer_warpgroups: int = 2
    # Row width of the 2D view of the packed weight the TMA reads: the widest TMA box row
    # (boxDim <= 256), and a divisor of the 1024 bytes of a tile's 16 rows.
    tma_row_bytes: int = 256
    # A row's table for one K step: the e4m3 value of each of the 16 SM2 codes.
    table_bytes: int = 16
    # Token rows of one activation TMA box: half the largest token tile, so a 2-CTA cluster
    # multicasts one box from each CTA.
    act_box_tokens: int = 128

    @property
    def block_n(self) -> int:
        """Weight rows of one n-tile."""
        return self.consumer_warpgroups * W4A8_LAYOUT.tile_rows

    @property
    def consumer_threads(self) -> int:
        """Threads of the consumer warpgroups."""
        return self.consumer_warpgroups * WARPGROUP_THREADS

    @property
    def rom_bytes(self) -> int:
        """Bytes of the scale ROM: one table per value of the e4m3 scale byte."""
        return 256 * self.table_bytes


LAYOUT = W4A8PrefillLayout()
# Two workers per 128-row output tile balance a short wave once they keep five sixths of
# the SMs busy; _worker_count and default_config must test the same fill.
_BALANCED_FILL = (5, 6)


def _schedule(m: int, n: int, k: int, sms: int, cluster: int, stream_k: int, block_m: int) -> dict:
    """Plan the persistent grid: the K-step range of each tile that each worker runs.

    A tile is ``block_m`` tokens by ``block_n`` weight rows, and a worker is one CTA, or the
    pair of a 2-CTA cluster running two adjacent n-tiles. With ``tiles % workers`` non-zero and
    ``stream_k`` set, tile ids below ``sk_tiles`` (that remainder, plus one wave of ``workers``
    tiles when ``stream_k`` is 2 and ``tiles > workers``) are split over all workers in units of
    two K steps; an odd K-step count makes step 0 of each tile a unit of its own, so only a
    segment starting at step 0 has odd length. Worker ``w`` runs the other tiles whole, tile
    ``sk_tiles + w + workers * i`` in wave ``i``.

    Returns:
        The grid extents (``tiles_m``, ``tiles_n``, ``workers``, ``tiles``, ``sk_tiles``,
        ``dp_waves``), ``segments[w]``: ``max(1, max_segs)`` rows of (tile, first K step,
        end K step, contributor slot), empty rows padding; ``contributor_counts[t]``: the
        workers holding a non-empty range of stream-K tile ``t``; ``partial_tiles``: one
        counter per CTA of each stream-K tile, at least one; and ``workspace_floats``: one
        ``block_m x block_n`` FP32 partial per counter and contributor slot.
    """
    kg = k // W4A8_LAYOUT.tile_k
    tiles_m = -(-m // block_m)
    tiles_n = -(-n // LAYOUT.block_n)
    workers = sms // cluster
    tiles = tiles_m * (tiles_n // cluster)
    if stream_k == 0 or tiles % workers == 0:
        sk_tiles = 0
    else:
        sk_tiles = tiles % workers + (workers if (stream_k == 2 and tiles > workers) else 0)
    dp_waves = -(-(tiles - sk_tiles) // workers)
    units_per_tile = (kg + 1) // 2
    units = sk_tiles * units_per_tile

    def unit_start(u: int) -> int:
        return 2 * u if kg % 2 == 0 else max(0, 2 * u - 1)

    rows = [[] for _ in range(workers)]
    contributors = [[] for _ in range(sk_tiles)]
    if units:
        for w in range(workers):
            u0, u1 = w * units // workers, (w + 1) * units // workers
            if u1 <= u0:
                continue
            for t in range(u0 // units_per_tile, (u1 - 1) // units_per_tile + 1):
                gb = max(u0, t * units_per_tile) - t * units_per_tile
                ge = min(u1, (t + 1) * units_per_tile) - t * units_per_tile
                rows[w].append((t, unit_start(gb), unit_start(ge), len(contributors[t])))
                contributors[t].append(w)
    max_segs = max((len(r) for r in rows), default=0)
    max_contributors = max([1] + [len(c) for c in contributors])
    partial_tiles = max(1, sk_tiles * cluster)
    return {
        "tiles_m": tiles_m,
        "tiles_n": tiles_n,
        "workers": workers,
        "tiles": tiles,
        "sk_tiles": sk_tiles,
        "dp_waves": dp_waves,
        "max_segs": max_segs,
        "max_contributors": max_contributors,
        "segments": [r + [(0, 0, 0, 0)] * (max(1, max_segs) - len(r)) for r in rows],
        "contributor_counts": [len(c) for c in contributors] or [0],
        "partial_tiles": partial_tiles,
        "workspace_floats": partial_tiles * max_contributors * block_m * LAYOUT.block_n,
    }


def _weight_view(n: int, k: int) -> tuple:
    """Shape of the ``tma_row_bytes``-wide view the TMA reads of a ``[N, K/2]`` unified weight.

    ``[row block, K step x tile row, byte]`` when ``block_n`` divides N; otherwise flat rows,
    since the last n-tile then holds a missing or compact row block, which the TMA copies alone.
    """
    rows = n * k // 2 // LAYOUT.tma_row_bytes
    if n % LAYOUT.block_n:
        return (rows, LAYOUT.tma_row_bytes)
    blocks = n // W4A8_LAYOUT.tile_rows
    return (blocks, rows // blocks, LAYOUT.tma_row_bytes)


def _check_config(n: int, config: dict, shared_limit: int) -> None:
    """Reject a config the program cannot run.

    Raises:
        ValueError: A key is off its domain, the cluster does not fit the tile, or the
            config's shared memory exceeds ``shared_limit`` bytes.
    """
    block_m, cluster = config["block_m"], config["cluster"]
    if block_m not in GemmW4A8PrefillKernel.token_tiles:
        raise ValueError(
            f"block_m must be one of {GemmW4A8PrefillKernel.token_tiles}, got {block_m}"
        )
    if (
        config["epilogue_tokens"] not in (32, 48, 64, 80, 96, 128)
        or block_m % config["epilogue_tokens"]
    ):
        raise ValueError(f"epilogue_tokens must be 32, 48, 64, 80, 96 or 128 and divide {block_m}")
    if config["epilogue_buffers"] not in (1, 2) or config["stream_k"] not in (0, 1, 2):
        raise ValueError("epilogue_buffers must be 1 or 2 and stream_k 0, 1 or 2")
    if config["num_stages"] < 2 or config["group_m"] < 1:
        raise ValueError("num_stages must be at least 2 and group_m at least 1")
    if cluster not in (1, 2):
        raise ValueError(f"cluster must be 1 or 2, got {cluster}")
    if cluster > 1 and (
        block_m != cluster * LAYOUT.act_box_tokens or n % (cluster * LAYOUT.block_n)
    ):
        # Each CTA multicasts one activation box of the tile, and the cluster takes whole
        # n-tiles.
        raise ValueError(
            f"cluster {cluster} needs block_m {cluster * LAYOUT.act_box_tokens} and n a multiple"
            f" of {cluster * LAYOUT.block_n}, got {block_m}, {n}"
        )

    n_pairs = -(-n // LAYOUT.block_n) // cluster
    # tile_coords stores group_m and group_m * n_pairs in signed 32-bit TIR indices.
    group_limit = torch.iinfo(torch.int32).max // n_pairs
    if config["group_m"] > group_limit:
        raise ValueError(f"group_m must be at most {group_limit} for signed 32-bit tile indices")

    def aligned(nbytes: int) -> int:
        return -(-nbytes // SHARED_BUFFER_ALIGN_BYTES) * SHARED_BUFFER_ALIGN_BYTES

    # An upper bound: every shared buffer the builder allocates, each rounded up to the
    # alignment, the last term the stream-K flag.
    need = (
        aligned(config["num_stages"] * block_m * W4A8_LAYOUT.tile_k)
        + aligned(config["num_stages"] * LAYOUT.consumer_warpgroups * W4A8_LAYOUT.tile_bytes)
        + aligned(config["num_stages"] * LAYOUT.block_n * LAYOUT.table_bytes)
        + config["epilogue_buffers"]
        * aligned(config["epilogue_tokens"] * LAYOUT.block_n * torch.bfloat16.itemsize)
        + aligned(LAYOUT.rom_bytes)
        + aligned(block_m * torch.float32.itemsize)
        + SHARED_BUFFER_ALIGN_BYTES
    )
    if need > shared_limit:
        raise ValueError(f"config needs {need} bytes of shared memory, over {shared_limit}")


def _worker_count(
    m: int, n: int, k: int, sms: int, block_m: int, cluster: int, stream_k: int
) -> int:
    """Choose the resident worker count shared by the launch and workspace plan."""
    # The balanced grid was measured faster only for 48- and 64-token tiles from K = 3072,
    # and only while the idle SMs would leave at most 22 K groups each.
    if not (32 < m <= block_m and block_m in (48, 64) and cluster == 1 and stream_k != 0):
        return sms
    if k < 3072 or n % LAYOUT.block_n:
        return sms
    balanced = 2 * (n // LAYOUT.block_n)
    numerator, denominator = _BALANCED_FILL
    if (
        numerator * sms <= denominator * balanced <= denominator * sms
        and (k // W4A8_LAYOUT.tile_k) * (sms - balanced) <= 22 * sms
    ):
        return balanced
    return sms


@functools.lru_cache(maxsize=32)
def _gemm_w4a8_prefill_kernel(m: int, n: int, k: int, sms: int) -> Callable:
    step_k, block_n = W4A8_LAYOUT.tile_k, LAYOUT.block_n
    kg = k // step_k
    # An n-tile holds rows past N, whose scale reads clamp, only when block_n does not divide N.
    ragged = n % block_n != 0
    full_blocks = n // W4A8_LAYOUT.tile_rows
    ragged_units = (n % W4A8_LAYOUT.tile_rows) // W4A8_LAYOUT.unit_rows
    rows_per_tma = W4A8_LAYOUT.tile_bytes // LAYOUT.tma_row_bytes
    # A tile's 16-row slice: one unit from each 64-K half.
    slice_bytes = 2 * W4A8_LAYOUT.unit_bytes
    rows_per_slice = slice_bytes // LAYOUT.tma_row_bytes
    # Scale bytes are prefetched four K steps ahead into an eight-register ring per lane,
    # measured deep enough to hide their load behind the decode.
    depth = 4
    odd_k = kg % 2 == 1
    clamp = "true" if ragged else "false"
    # One producer warpgroup: the TMA warp, then the table warps.
    threads = WARPGROUP_THREADS + LAYOUT.consumer_threads
    # Each table lane builds the tables of two rows WARP_LANES apart (the helper's ring).
    table_warps = block_n // (2 * WARP_LANES)
    # setmaxnreg split: the producer warpgroup issues TMA and builds tables in 40 registers;
    # each consumer warpgroup holds an accumulator of up to 128 registers in 232.
    # 128 * 40 + 256 * 232 fits the 64K-register file.
    producer_reg, consumer_reg = 40, 232

    @tilelang.jit(
        out_idx=[-1],
        compile_flags=[
            "-O3",
            "--use_fast_math",
            "-DENABLE_BF16",
            *csrc_include("w4a8_prefill_helper.h"),
        ],
    )
    def _gemm_w4a8_prefill_func(
        block_m: int,
        num_stages: int,
        group_m: int,
        stream_k: int,
        cluster: int,
        epilogue_tokens: int,
        epilogue_buffers: int,
    ):
        plan = _schedule(m, n, k, sms, cluster, stream_k, block_m)
        tiles_m, workers, tiles = plan["tiles_m"], plan["workers"], plan["tiles"]
        sk_tiles, dp_waves, max_segs = plan["sk_tiles"], plan["dp_waves"], plan["max_segs"]
        max_contributors, partial_tiles = plan["max_contributors"], plan["partial_tiles"]
        n_pairs = plan["tiles_n"] // cluster
        acc_regs = WGMMA_ROWS * block_m // WARPGROUP_THREADS
        # Three 64-row boxes cover 192 tokens; a 128-row box would leave 64 rows unread.
        sub_m = 64 if block_m == 192 else min(block_m, LAYOUT.act_box_tokens)
        nsub = block_m // sub_m
        passes = block_m // epilogue_tokens
        consumers = LAYOUT.consumer_threads
        # The first warp of the last consumer warpgroup issues the output stores.
        store_warp = (threads - WARPGROUP_THREADS) // WARP_LANES
        mma = f"tileops::w4a8_decode_mma<{block_m}>"
        epilogue = f"tileops::w4a8_epilogue<{block_m}, {epilogue_tokens}>"
        arrive_cluster = f"tileops::w4a8_arrive_cluster<{cluster}>"

        @T.macro
        def tile_coords(tid, mt, nt):
            # Tile id -> (M tile, n-tile of the worker): ids run over group_m M tiles first,
            # then over the n-tiles, then on to the next group_m M tiles.
            per_group = T.int32(group_m * n_pairs)
            first_m = tid // per_group * T.int32(group_m)
            gm = T.min(T.int32(group_m), T.int32(tiles_m) - first_m)
            mt[0] = first_m + (tid % per_group) % gm
            nt[0] = (tid % per_group) // gm

        @T.macro
        def segment(w, s, seg, segments):
            # seg = [tile, first K step, end K step, stream-K tile or -1, contributor slot].
            if s < max_segs:
                seg[0] = segments[w, s, 0]
                seg[1] = segments[w, s, 1]
                seg[2] = segments[w, s, 2]
                seg[3] = segments[w, s, 0]
                seg[4] = segments[w, s, 3]
            else:
                seg[0] = sk_tiles + w + workers * (s - max_segs)
                seg[1] = 0
                seg[2] = T.if_then_else(seg[0] < tiles, kg, 0)
                seg[3] = -1
                seg[4] = 0

        @T.macro
        def release_stage(empty, ci):
            # Frees the stage step ci - 1 read, across the cluster when the activation is shared.
            if cluster > 1:
                T.call_extern(
                    "handle",
                    arrive_cluster,
                    T.address_of(empty[(ci + num_stages - 1) % num_stages]),
                )
            else:
                T.barrier_arrive(empty[(ci + num_stages - 1) % num_stages])

        @T.macro
        def k_step(kk, a_cur, a_prev, ops, ci, kfirst):
            acc, a_s, w_s, tab_s, full, empty, wg, tx, hs = ops
            slot = ci % num_stages
            T.barrier_wait(full[slot], (ci // num_stages) & 1)
            T.call_extern(
                "handle",
                mma,
                acc.data,
                a_cur.data,
                T.address_of(w_s[slot, wg, 0, 0]),
                T.address_of(tab_s[slot, 0, 0]),
                wg,
                T.address_of(a_s[0, 0, 0]),
                slot * (block_m * step_k),
                T.if_then_else(kk == kfirst, 1, 0),
                hs[0] if ragged else W4A8_LAYOUT.tile_bytes // 2,
            )
            # Leaves this step's WGMMAs in flight and retires the previous step's, which read
            # a_prev and the previous stage.
            T.wait_wgmma(1)
            T.call_extern("handle", "tileops::w4a8_keep16", a_prev.data)
            if kk > kfirst and tx % WARP_LANES == 0:
                release_stage(empty, ci)

        @T.prim_func
        def _gemm_w4a8_prefill_main(
            act: T.Tensor((m, k), "float8_e4m3"),  # type: ignore
            wq: T.Tensor(_weight_view(n, k), "uint8"),  # type: ignore
            gscale: T.Tensor((n, kg), "float8_e4m3"),  # type: ignore
            cscale: T.Tensor((n,), "float32"),  # type: ignore
            ascale: T.Tensor((m,), "float32"),  # type: ignore
            ws: T.Tensor((plan["workspace_floats"],), "float32"),  # type: ignore
            cnt: T.Tensor((partial_tiles,), "int32"),  # type: ignore
            segments: T.Tensor((workers, max(1, max_segs), 4), "int32"),  # type: ignore
            contributor_counts: T.Tensor((max(1, sk_tiles),), "int32"),  # type: ignore
            out: T.Tensor((m, n), "bfloat16"),  # type: ignore
        ) -> None:
            with T.ClusterKernel(sms, cluster_dims=cluster, threads=threads) as pid:
                rank = pid % cluster
                w = pid // cluster
                a_s = T.alloc_shared((num_stages * nsub, sub_m, step_k), "float8_e4m3")
                w_s = T.alloc_shared(
                    (num_stages, LAYOUT.consumer_warpgroups, rows_per_tma, LAYOUT.tma_row_bytes),
                    "uint8",
                )
                tab_s = T.alloc_shared((num_stages, block_n, LAYOUT.table_bytes), "uint8")
                o_s = T.alloc_shared((epilogue_tokens, block_n), "bfloat16")
                o_t = (
                    T.alloc_shared((epilogue_tokens, block_n), "bfloat16")
                    if epilogue_buffers == 2
                    else o_s
                )
                flag_s = T.alloc_shared((1,), "int32")
                as_s = T.alloc_shared((block_m,), "float32")
                rom_s = T.alloc_shared((LAYOUT.rom_bytes,), "uint8")
                T.annotate_layout(
                    {
                        a_s: tilelang.layout.make_swizzled_layout(a_s),
                        w_s: tilelang.layout.make_linear_layout(w_s),
                        tab_s: tilelang.layout.make_linear_layout(tab_s),
                        o_s: tilelang.layout.make_swizzled_layout(o_s),
                        **(
                            {o_t: tilelang.layout.make_swizzled_layout(o_t)}
                            if epilogue_buffers == 2
                            else {}
                        ),
                    }
                )
                acc = T.alloc_local((acc_regs,), "float32")
                # The A fragment w4a8_decode_mma writes: 16 words of e4m3 weights per thread.
                a0 = T.alloc_local((16,), "uint32")
                a1 = T.alloc_local((16,), "uint32")
                # The TMA lane and every table lane arrive once per stage.
                full = T.alloc_barrier([1 + WARP_LANES * table_warps] * num_stages)
                # Every consumer warp of every CTA in the cluster arrives once per stage.
                empty = T.alloc_barrier([consumers // WARP_LANES * cluster] * num_stages)
                ring = T.alloc_local((depth * 2,), "uint32")
                p_idx = T.alloc_var("int32", init=0)
                c_idx = T.alloc_var("int32", init=0)
                mt = T.alloc_local((1,), "int32")
                nt = T.alloc_local((1,), "int32")
                seg = T.alloc_local((5,), "int32")
                do_epi = T.alloc_var("int32", init=1)
                sc = T.alloc_local((2,), "float32")
                hs = T.alloc_local((1,), "int32")
                tx = T.get_thread_binding()
                if cluster > 1:
                    T.cluster_sync()
                # The activation as m rows of k bytes, read in sub_m x step_k boxes.
                act_desc = T.create_tma_descriptor(
                    TMA_DTYPE_UINT8,
                    2,
                    act.data,
                    k,
                    m,
                    1,
                    k,
                    step_k,
                    sub_m,
                    1,
                    1,
                    TMA_INTERLEAVE_NONE,
                    TMA_SWIZZLE_128B,
                    TMA_L2_PROMOTION_128B,
                    TMA_OOB_FILL_NONE,
                )
                T.call_extern("handle", "tileops::w4a8_scale_rom_fill", T.address_of(rom_s[0]))
                T.sync_threads()
                if tx < WARPGROUP_THREADS:
                    T.dec_max_nreg(producer_reg)
                    if tx < WARP_LANES:
                        # TMA warp: the activation and the packed weight of each K step.
                        for s in T.serial(max_segs + dp_waves):
                            segment(w, s, seg, segments)
                            tile_coords(seg[0], mt, nt)
                            m0 = mt[0] * block_m
                            rb0 = (nt[0] * cluster + rank) * LAYOUT.consumer_warpgroups
                            for kk in T.serial(seg[1], seg[2]):
                                slot = p_idx % num_stages
                                T.barrier_wait(empty[slot], ((p_idx // num_stages) & 1) ^ 1)
                                if cluster > 1:
                                    if tx == 0:
                                        # This CTA's barrier counts the whole tile: its own box
                                        # and the boxes the other CTAs multicast to it.
                                        T.call_extern(
                                            "handle",
                                            "tileops::w4a8_expect_tx",
                                            T.address_of(full[slot]),
                                            block_m * step_k,
                                        )
                                        T.call_extern(
                                            "handle",
                                            "tileops::w4a8_tma_multicast_2d",
                                            act_desc,
                                            T.address_of(full[slot]),
                                            T.address_of(a_s[0, 0, 0]),
                                            (slot * nsub + rank) * (sub_m * step_k),
                                            (1 << cluster) - 1,
                                            kk * step_k,
                                            m0 + rank * sub_m,
                                        )
                                else:
                                    for sb in T.unroll(nsub, explicit=True):
                                        T.tma_copy(
                                            act[
                                                m0 + sb * sub_m : m0 + (sb + 1) * sub_m,
                                                kk * step_k : (kk + 1) * step_k,
                                            ],
                                            a_s[slot * nsub + sb, :, :],
                                            barrier=full[slot],
                                        )
                                if ragged:
                                    # Per 64-row block: a whole tile, the compact last tile, or
                                    # nothing past N.
                                    for b in T.unroll(LAYOUT.consumer_warpgroups, explicit=True):
                                        if rb0 + b < full_blocks:
                                            T.tma_copy(
                                                wq[
                                                    ((rb0 + b) * kg + kk) * rows_per_tma : (
                                                        (rb0 + b) * kg + kk
                                                    )
                                                    * rows_per_tma
                                                    + rows_per_tma,
                                                    0 : LAYOUT.tma_row_bytes,
                                                ],
                                                w_s[slot, b, :, :],
                                                barrier=full[slot],
                                            )
                                        if ragged_units > 0 and rb0 + b == full_blocks:
                                            T.tma_copy(
                                                wq[
                                                    full_blocks * kg * rows_per_tma
                                                    + kk
                                                    * ragged_units
                                                    * rows_per_slice : full_blocks
                                                    * kg
                                                    * rows_per_tma
                                                    + (kk + 1) * ragged_units * rows_per_slice,
                                                    0 : LAYOUT.tma_row_bytes,
                                                ],
                                                w_s[
                                                    slot,
                                                    b,
                                                    0 : ragged_units * rows_per_slice,
                                                    :,
                                                ],
                                                barrier=full[slot],
                                            )
                                else:
                                    T.tma_copy(
                                        wq[
                                            rb0 : rb0 + LAYOUT.consumer_warpgroups,
                                            kk * rows_per_tma : (kk + 1) * rows_per_tma,
                                            0 : LAYOUT.tma_row_bytes,
                                        ],
                                        w_s[slot, :, :, :],
                                        barrier=full[slot],
                                    )
                                if tx == 0:
                                    T.barrier_arrive(full[slot])
                                p_idx = p_idx + 1
                    elif tx < WARP_LANES + WARP_LANES * table_warps:
                        # Table warps: each lane builds the tables of rows row0 and row0 + 32
                        # from the raw scale bytes, prefetched `depth` K steps ahead.
                        row0 = (tx // WARP_LANES - 1) * (block_n // table_warps) + tx % WARP_LANES
                        for s in T.serial(max_segs + dp_waves):
                            segment(w, s, seg, segments)
                            tile_coords(seg[0], mt, nt)
                            n0 = (nt[0] * cluster + rank) * block_n
                            rlim = T.min(n - n0, block_n)
                            if seg[2] > seg[1]:
                                T.call_extern(
                                    "handle",
                                    f"tileops::w4a8_ring_fill<{depth}, {clamp}>",
                                    ring.data,
                                    T.address_of(gscale[n0, 0]),
                                    kg,
                                    row0,
                                    seg[1],
                                    seg[2],
                                    rlim,
                                )
                            if odd_k:
                                # Slot d always serves step base + d, so a loaded byte is first
                                # read `depth` steps later and never moved.
                                for kq in T.serial((seg[2] - seg[1] + depth - 1) // depth):
                                    for d in T.unroll(depth, explicit=True):
                                        if seg[1] + depth * kq + d < seg[2]:
                                            slot = p_idx % num_stages
                                            T.barrier_wait(
                                                empty[slot], ((p_idx // num_stages) & 1) ^ 1
                                            )
                                            T.call_extern(
                                                "handle",
                                                "tileops::w4a8_ring_tables",
                                                T.address_of(tab_s[slot, 0, 0]),
                                                row0,
                                                ring.data,
                                                T.address_of(rom_s[0]),
                                                d,
                                            )
                                            T.barrier_arrive(full[slot])
                                            T.call_extern(
                                                "handle",
                                                f"tileops::w4a8_ring_load<{clamp}>",
                                                ring.data,
                                                T.address_of(gscale[n0, 0]),
                                                kg,
                                                row0,
                                                seg[1] + depth * kq + d + depth,
                                                seg[2],
                                                rlim,
                                                d,
                                            )
                                            p_idx = p_idx + 1
                            else:
                                for kk in T.serial(seg[1], seg[2]):
                                    slot = p_idx % num_stages
                                    T.barrier_wait(empty[slot], ((p_idx // num_stages) & 1) ^ 1)
                                    T.call_extern(
                                        "handle",
                                        "tileops::w4a8_ring_tables",
                                        T.address_of(tab_s[slot, 0, 0]),
                                        row0,
                                        ring.data,
                                        T.address_of(rom_s[0]),
                                        0,
                                    )
                                    T.barrier_arrive(full[slot])
                                    T.call_extern(
                                        "handle",
                                        f"tileops::w4a8_ring_shift<{depth}, {clamp}>",
                                        ring.data,
                                        T.address_of(gscale[n0, 0]),
                                        kg,
                                        row0,
                                        kk + depth,
                                        seg[2],
                                        rlim,
                                    )
                                    p_idx = p_idx + 1
                else:
                    T.inc_max_nreg(consumer_reg)
                    wg = tx // WARPGROUP_THREADS - 1
                    lt = tx - WARPGROUP_THREADS
                    # This thread's accumulator rows are r0 and r0 + 8 of its warpgroup's 64,
                    # four lanes to a row: the layout w4a8_epilogue reads.
                    r0 = (
                        tx % WARPGROUP_THREADS
                    ) // WARP_LANES * W4A8_LAYOUT.unit_rows + tx % WARP_LANES // 4
                    ops = (acc, a_s, w_s, tab_s, full, empty, wg, tx, hs)
                    for s in T.serial(max_segs + dp_waves):
                        segment(w, s, seg, segments)
                        if seg[2] > seg[1]:
                            tile_coords(seg[0], mt, nt)
                            m0 = mt[0] * block_m
                            n0 = (nt[0] * cluster + rank) * block_n
                            nrow = n0 + wg * WGMMA_ROWS + r0
                            # Every consumer thread has passed barrier 2 of the previous tile's
                            # last epilogue pass, after its last read of as_s.
                            if lt < block_m:
                                as_s[lt] = T.if_then_else(
                                    m0 + lt < m, ascale[T.min(m0 + lt, m - 1)], T.float32(0)
                                )
                            if ragged:
                                sc[0] = cscale[T.min(nrow, n - 1)]
                                sc[1] = cscale[T.min(nrow + 8, n - 1)]
                                # The second 64-K half of this warpgroup's tile: half a 4 KB
                                # tile, or half the compact last tile.
                                hs[0] = T.if_then_else(
                                    n0 // W4A8_LAYOUT.tile_rows + wg == full_blocks,
                                    ragged_units * slice_bytes // 2,
                                    W4A8_LAYOUT.tile_bytes // 2,
                                )
                            else:
                                sc[0] = cscale[nrow]
                                sc[1] = cscale[nrow + 8]
                            nk = seg[2] - seg[1]
                            if not odd_k:
                                if nk == kg:
                                    # A whole tile runs a compile-time trip count.
                                    for kp in T.serial(kg // 2):
                                        k_step(2 * kp, a0, a1, ops, c_idx, 0)
                                        c_idx = c_idx + 1
                                        k_step(2 * kp + 1, a1, a0, ops, c_idx, 0)
                                        c_idx = c_idx + 1
                                # A stream-K segment has even length; a branch around a single
                                # WGMMA makes ptxas serialize every WGMMA.
                                for kp in T.serial(T.if_then_else(nk == kg, 0, nk // 2)):
                                    k_step(seg[1] + 2 * kp, a0, a1, ops, c_idx, seg[1])
                                    c_idx = c_idx + 1
                                    k_step(seg[1] + 2 * kp + 1, a1, a0, ops, c_idx, seg[1])
                                    c_idx = c_idx + 1
                            else:
                                # Step 0 runs ahead of the pairs: straight-line for a whole tile,
                                # a 0/1-trip loop for a segment (odd only when it starts at 0).
                                if nk == kg:
                                    k_step(0, a1, a0, ops, c_idx, 0)
                                    c_idx = c_idx + 1
                                    for kp in T.serial(kg // 2):
                                        k_step(2 * kp + 1, a0, a1, ops, c_idx, 0)
                                        c_idx = c_idx + 1
                                        k_step(2 * kp + 2, a1, a0, ops, c_idx, 0)
                                        c_idx = c_idx + 1
                                if sk_tiles > 0:
                                    for kt in T.serial(T.if_then_else(nk == kg, 0, nk % 2)):
                                        k_step(seg[1] + kt, a1, a0, ops, c_idx, seg[1])
                                        c_idx = c_idx + 1
                                    for kp in T.serial(T.if_then_else(nk == kg, 0, nk // 2)):
                                        k_step(seg[1] + nk % 2 + 2 * kp, a0, a1, ops, c_idx, seg[1])
                                        c_idx = c_idx + 1
                                        k_step(
                                            seg[1] + nk % 2 + 2 * kp + 1, a1, a0, ops, c_idx, seg[1]
                                        )
                                        c_idx = c_idx + 1
                            T.wait_wgmma(0)
                            if tx % WARP_LANES == 0:
                                release_stage(empty, c_idx)
                            # Stream-K: every contributor stores its partial; the last to arrive
                            # adds the others' and runs the epilogue.
                            do_epi = 1
                            if seg[3] >= 0 and (seg[1] > 0 or seg[2] < kg):
                                t = seg[3]
                                nc = contributor_counts[t]
                                wsi = t * cluster + rank
                                base = wsi * max_contributors * acc_regs * consumers
                                T.call_extern(
                                    "handle",
                                    f"tileops::w4a8_store_partial<{acc_regs}, {consumers}>",
                                    acc.data,
                                    ws.data,
                                    base + seg[4] * acc_regs * consumers + lt * 4,
                                )
                                # Named barriers 1, 2, 5 and 6 span the consumers. TileLang keeps
                                # 1 and 2 for reductions, none here, and numbers the barriers it
                                # inserts from 3 (3 and 4 at most in this program).
                                T.sync_threads(barrier_id=5, arrive_count=consumers)
                                if tx == WARPGROUP_THREADS:
                                    flag_s[0] = T.call_extern(
                                        "int32", "tileops::w4a8_signal", cnt.data, wsi, nc
                                    )
                                T.sync_threads(barrier_id=6, arrive_count=consumers)
                                do_epi = flag_s[0]
                                # The last arriving CTA keeps its accumulator, so FP32
                                # addition order can vary between launches (see the op API).
                                if do_epi == 1:
                                    for c in T.serial(nc):
                                        if c != seg[4]:
                                            T.call_extern(
                                                "handle",
                                                f"tileops::w4a8_add_partial<{acc_regs}, {consumers}>",
                                                acc.data,
                                                ws.data,
                                                base + c * acc_regs * consumers + lt * 4,
                                            )
                            if do_epi == 1:
                                for p in T.unroll(passes, explicit=True):
                                    # The buffer this pass writes is free once at most
                                    # epilogue_buffers - 1 stores are in flight.
                                    if tx // WARP_LANES == store_warp:
                                        # An odd number of double-buffered passes ends on
                                        # the buffer the next tile starts with. Drain that
                                        # store before the first pass overwrites it.
                                        if passes % epilogue_buffers != 0 and p == 0:
                                            T.tma_store_wait(0)
                                        else:
                                            T.tma_store_wait(epilogue_buffers - 1)
                                    T.sync_threads(barrier_id=1, arrive_count=consumers)
                                    for bsel in T.unroll(epilogue_buffers, explicit=True):
                                        if p % epilogue_buffers == bsel:
                                            T.call_extern(
                                                "handle",
                                                epilogue,
                                                acc.data,
                                                T.if_then_else(
                                                    bsel == 0,
                                                    T.address_of(o_s[0, 0]),
                                                    T.address_of(o_t[0, 0]),
                                                ),
                                                wg,
                                                p,
                                                T.address_of(as_s[p * epilogue_tokens]),
                                                sc[0],
                                                sc[1],
                                                m - (m0 + p * epilogue_tokens),
                                            )
                                    T.fence_proxy_async()
                                    T.sync_threads(barrier_id=2, arrive_count=consumers)
                                    if tx // WARP_LANES == store_warp:
                                        if p % epilogue_buffers == 0:
                                            T.tma_copy(
                                                o_s[:, :],
                                                out[
                                                    m0 + p * epilogue_tokens : m0
                                                    + (p + 1) * epilogue_tokens,
                                                    n0 : n0 + block_n,
                                                ],
                                            )
                                        else:
                                            T.tma_copy(
                                                o_t[:, :],
                                                out[
                                                    m0 + p * epilogue_tokens : m0
                                                    + (p + 1) * epilogue_tokens,
                                                    n0 : n0 + block_n,
                                                ],
                                            )
                    if tx // WARP_LANES == store_warp:
                        T.tma_store_wait(0)
                if cluster > 1:
                    T.cluster_sync()

        return _gemm_w4a8_prefill_main

    return _gemm_w4a8_prefill_func


class GemmW4A8PrefillKernel(Kernel, GemmW4A8FwdInterface):
    """W4A8 GEMM for prefill token counts over the unified INT4 weight order.

    Calls of one instance must run in stream order: stream-K tiles meet on counters and a
    workspace the instance owns.

    Args:
        m: Token rows.
        n: Output columns, a multiple of 16.
        k: Contraction dim, a multiple of 128.
        config: Keys of :attr:`default_config` to override; the others keep their default.
        device_index: CUDA device the kernel is built for.
    """

    token_tiles = (32, 48, 64, 80, 96, 128, 192, 256)

    # WGMMA, setmaxnreg, TMA multicast and thread-block clusters.
    supported_archs: list[int] = [90]

    @classmethod
    def refusal(cls, call: GemmW4A8Call) -> Optional[str]:
        """Serve prefill token counts with 128-wide groups, N a multiple of 16 and K of 128."""
        if call.m <= 32:
            return f"m={call.m} is a decode token count (serves m > 32)"
        if call.group_size != GROUP_SIZE:
            return f"requires group_size {GROUP_SIZE}, got {call.group_size}"
        if call.n <= 0 or call.k <= 0:
            return f"requires positive N and K, got n={call.n}, k={call.k}"
        if call.k % W4A8_LAYOUT.tile_k:
            return f"requires k a multiple of {W4A8_LAYOUT.tile_k}, got {call.k}"
        if call.n % W4A8_LAYOUT.unit_rows:
            return f"requires n a multiple of {W4A8_LAYOUT.unit_rows}, got {call.n}"
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
        self.m = m
        self.n = n
        self.k = k
        sms = get_sm_count(device_index)
        self.init_config(config)
        workers = _worker_count(
            m,
            n,
            k,
            sms,
            self.config["block_m"],
            self.config["cluster"],
            self.config["stream_k"],
        )
        self.kernel = _gemm_w4a8_prefill_kernel(m, n, k, workers)
        _check_config(n, self.config, get_shared_memory_optin(device_index))
        plan = _schedule(
            m,
            n,
            k,
            workers,
            self.config["cluster"],
            self.config["stream_k"],
            self.config["block_m"],
        )
        device = torch.device("cuda", device_index) if device_index is not None else "cuda"
        self._ws = torch.empty(plan["workspace_floats"], dtype=torch.float32, device=device)
        # Each tile's counter returns to zero when its last contributor arrives.
        self._counters = torch.zeros(plan["partial_tiles"], dtype=torch.int32, device=device)
        self._segments = torch.tensor(plan["segments"], dtype=torch.int32, device=device)
        self._contributors = torch.tensor(
            plan["contributor_counts"], dtype=torch.int32, device=device
        )

    @property
    def default_config(self) -> dict:
        """A legal tile and schedule for this shape and device."""
        m, n, k = self.m, self.n, self.k
        hardware_sms = get_sm_count(self.device_index)
        block_m = next(
            (t for t in GemmW4A8PrefillKernel.token_tiles if m <= t),
            GemmW4A8PrefillKernel.token_tiles[-1],
        )
        if m > 128 and -(-m // 192) == -(-m // 256):
            block_m = 192
        # Thresholds below were measured on H200 over the manifest's W4A8 rows. Up to K = 3584
        # (the TP8 down projections) a tile has few K steps, so an extra segment or a
        # stream-K split costs more than it saves; from K = 7168 it can pay.
        short_k, long_k = 3584, 7168
        base_workers = _worker_count(m, n, k, hardware_sms, block_m, 1, 2)
        one_segment = False
        if block_m < 192 and k <= short_k:
            one_segment = _schedule(m, n, k, base_workers, 1, 2, block_m)["max_segs"] <= 1
        short_dp = False
        balanced = 2 * -(-n // LAYOUT.block_n)
        fill_num, fill_den = _BALANCED_FILL
        if (
            k <= short_k
            and n % LAYOUT.block_n == 0
            and fill_num * hardware_sms <= fill_den * balanced <= fill_den * hardware_sms
            and not one_segment
        ):
            trial = next((t for t in GemmW4A8PrefillKernel.token_tiles if -(-m // 2) <= t), 256)
            if trial <= 128 and -(-m // trial) == 2:
                block_m, short_dp = trial, True
        # Larger token tiles leave shared memory for fewer stages.
        num_stages = 8 if block_m <= 80 else 6 if block_m <= 128 else 4
        # 192- and 256-token tiles store their output in two 64-token buffers.
        epilogue_tokens = block_m if block_m <= 128 else 64
        epilogue_buffers = 2 if block_m in (192, 256) else 1
        # A two-CTA cluster multicasts the activation; measured faster only on outputs at
        # most 2304 wide.
        cluster = 2 if block_m == 256 and n % (2 * LAYOUT.block_n) == 0 and n <= 2304 else 1
        group_m = 16 if cluster == 2 else 8
        if short_dp:
            stream_k = 0
        elif block_m < 192:
            # short_dp is false here, so block_m is the one base_workers was counted for.
            sms = base_workers
            stream_k = 2
            # Stream-K is dropped where the idle SMs would leave at most 16 K groups each and
            # it would still split a tile into several segments.
            if block_m in (48, 64) and k <= short_k:
                tiles = -(-n // LAYOUT.block_n)
                if tiles < sms and (k // W4A8_LAYOUT.tile_k) * (sms - tiles) <= 16 * sms:
                    plan = _schedule(m, n, k, sms, 1, stream_k, block_m)
                    if plan["max_segs"] > 1:
                        stream_k = 0
        else:
            # A worker owns one CTA, or a pair of CTAs for cluster 2. Use the same
            # units as _schedule for both the grid and the hardware capacity.
            workers = hardware_sms // cluster
            tiles = -(-m // block_m) * (-(-n // LAYOUT.block_n) // cluster)
            waves, tail = divmod(tiles, workers)
            # Below long_k a 192-token tile runs data-parallel; from long_k stream-K pays when
            # under three quarters of the workers have a tile and the idle ones would have at
            # least 28 K groups each, or when one wave ends at most half full.
            if block_m == 192 and k < long_k:
                stream_k = 0
            elif k >= long_k and (
                (
                    4 * tiles < 3 * workers
                    and (k // W4A8_LAYOUT.tile_k) * (workers - tiles) >= 28 * workers
                )
                or (waves == 1 and 0 < tail <= workers // 2)
            ):
                stream_k = 2
            else:
                stream_k = 1 if 2 <= waves <= 4 and 0 < tail <= workers // 2 else 0
        return {
            "block_m": block_m,
            "num_stages": num_stages,
            "group_m": group_m,
            "stream_k": stream_k,
            "cluster": cluster,
            "epilogue_tokens": epilogue_tokens,
            "epilogue_buffers": epilogue_buffers,
        }

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_group_scale: torch.Tensor,
        weight_channel_scale: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        compiled = self.kernel(**self.config)
        return compiled(
            activation,
            packed_weight.view(_weight_view(self.n, self.k)),
            weight_group_scale,
            weight_channel_scale,
            activation_scale,
            self._ws,
            self._counters,
            self._segments,
            self._contributors,
        )
