"""Warp-specialized paged MHA decode for SM90.

Each query row is a matrix-vector product: with a few query rows the M axis of
an MMA is mostly padding, so the contractions run on the CUDA cores. A causal
row ``i`` of ``S_q`` sees the cache minus its last ``S_q - 1 - i`` keys. The
time is decided by how few launches reach the device and how early the KV tiles
are in flight.

The pipeline is written out rather than scheduled:

* one producer warp group and one consumer warp group, split by ``T.ws`` and
  placed in the two arms of a single ``If`` (two sequential ``T.ws`` regions let
  the thread-sync pass emit a block-wide barrier the producer never reaches);
* ``T.tma_copy`` moves each K and V tile into a per-stage ring, announced on
  ``T.alloc_barrier`` mbarriers with the phase parity carried by hand;
* ``T.annotate_layout`` states the swizzle the TMA destinations use;
* a block serves ``group`` query rows, which share every tile it loads; and
* ``lanes_per_row`` lanes score one key row, each over a span of the head dim.

Each consumer warp is an online-softmax accumulator over the rows it owns; the
four warp partials merge through shared memory, and the last split block to
finish for an output merges the per-split partials.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops.kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    AttentionCall,
    MHAPagedDecodeFwdInterface,
)
from tileops.kernels.constants import LOG2E, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import WARP_LANES

__all__ = ["MHADecodePagedWSKernel"]

# Warps in the consumer group. One warp group of each role, 256 threads: two
# consumer groups deadlock the block-wide sync the layout pass inserts.
_CONS_WARPS = 4
_CONS = _CONS_WARPS * WARP_LANES
_PROD = 128
# Named barrier the consumer group uses on its own, never block-wide.
_MERGE_BARRIER = 1
# Finite stand-in for -inf in the running max, so a fully masked tile rescales
# by exactly one instead of evaluating exp2(-inf - -inf).
_NEG_FLOOR = -1.0e38
# What an empty split publishes as its log-sum-exp, beside a zero partial: finite, so
# the merge weighs it exp2(_EMPTY_LSE - peak), zero beside a split that saw a key and
# one when none did, never 0 * NaN.
_EMPTY_LSE = -1.0e30


@functools.lru_cache(maxsize=32)
def _mha_decode_paged_ws_kernel(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: str,
    max_pages_per_req: int,
):
    """Build the JIT'd decode kernel for one shape specialization."""
    scale = dim**-0.5 * LOG2E
    accum = "float"
    # Output elements a lane accumulates: the warp spans the head dim once.
    vec = dim // WARP_LANES
    # Key elements a lane reads per shared-memory load, for 16-bit keys.
    kchunk = VECTOR_ACCESS_BYTES // 2

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_N: int, num_split: int, stages: int, group: int):
        rows_per_warp = block_N // _CONS_WARPS
        # Lanes that share one key row's score, so every lane works at any tile height.
        lanes_per_row = WARP_LANES // rows_per_warp
        span = dim // lanes_per_row
        chunk = min(kchunk, span)
        threads = _CONS + _PROD
        n_groups = seqlen_q // group

        @T.prim_func
        def mha_decode_paged_ws(
            Q: T.Tensor([batch, seqlen_q, heads, dim], dtype),
            K: T.Tensor([seqlen_kv, heads, dim], dtype),
            V: T.Tensor([seqlen_kv, heads, dim], dtype),
            real_seqlen_kv: T.Tensor([batch], "int32"),
            block_table: T.Tensor([batch, max_pages_per_req], "int32"),
            glse: T.Tensor([batch, seqlen_q, heads, num_split], accum),
            O_partial: T.Tensor([batch, seqlen_q, heads, num_split, dim], accum),
            Arrived: T.Tensor([batch * seqlen_q * heads], "int32"),
            Output: T.Tensor([batch, seqlen_q, heads, dim], dtype),
        ):
            with T.Kernel(num_split, heads, batch * n_groups, threads=threads) as (bs, bh, bg):
                bb = bg // n_groups
                # The block's query rows share every key and value tile it loads.
                q0 = bg % n_groups * group
                # Every ring and barrier is declared at kernel-body level:
                # an allocation made inside a guarded stage is not visible to
                # the other arm.
                Ks = T.alloc_shared([stages, block_N, dim], dtype)
                Vs = T.alloc_shared([stages, block_N, dim], dtype)
                warp_m = T.alloc_shared([group, _CONS_WARPS], accum, scope="shared")
                warp_l = T.alloc_shared([group, _CONS_WARPS], accum, scope="shared")
                warp_o = T.alloc_shared([group, _CONS_WARPS, dim], accum, scope="shared")
                q_s = T.alloc_shared([group, dim], accum, scope="shared")
                arrived_before = T.alloc_shared([1], "int32", scope="shared")
                T.annotate_layout(
                    {
                        Ks: make_swizzled_layout(Ks),
                        Vs: make_swizzled_layout(Vs),
                    }
                )

                # A ready barrier is completed by whoever announces the tile, an
                # empty barrier by every consumer thread that finished reading
                # it. The counts are the two group sizes; getting them wrong
                # hangs the block and takes the context with it.
                k_ready = T.alloc_barrier([_PROD] * stages)
                k_free = T.alloc_barrier([_CONS] * stages)
                v_ready = T.alloc_barrier([_PROD] * stages)
                v_free = T.alloc_barrier([_CONS] * stages)

                tx = T.get_thread_binding()
                kv_len = real_seqlen_kv[bb]
                # Causal row q sees the cache minus its last seqlen_q - 1 - q keys; the
                # block loads the tiles its last row sees.
                row_drop = seqlen_q - q0 - group if is_causal else 0
                kv_len_block = T.max(kv_len - row_drop, 0)
                # Tiles are page-aligned and block-aligned, so one tile never
                # straddles two pages and the block table is read once per tile.
                tiles_total = T.ceildiv(kv_len_block, block_N)
                tiles_per_split = T.ceildiv(tiles_total, num_split)
                tile_begin = bs * tiles_per_split
                n_tiles = T.max(T.min(tile_begin + tiles_per_split, tiles_total) - tile_begin, 0)

                if tx >= _CONS:
                    with T.ws(1):
                        for t in T.serial(n_tiles):
                            st = t % stages
                            free_parity = ((t // stages) % 2) ^ 1
                            row0 = (tile_begin + t) * block_N
                            base = block_table[bb, row0 // page_size] * page_size + row0 % page_size
                            T.mbarrier_wait_parity(k_free[st], free_parity)
                            T.tma_copy(
                                K[base : base + block_N, bh, :], Ks[st, :, :], barrier=k_ready[st]
                            )
                            T.mbarrier_arrive(k_ready[st])
                            T.mbarrier_wait_parity(v_free[st], free_parity)
                            T.tma_copy(
                                V[base : base + block_N, bh, :], Vs[st, :, :], barrier=v_ready[st]
                            )
                            T.mbarrier_arrive(v_ready[st])
                else:
                    with T.ws(0):
                        warp = tx // WARP_LANES
                        lane = tx % WARP_LANES
                        d0 = lane * vec
                        # The key row this lane scores, and where its span of it starts.
                        j_own = warp * rows_per_warp + lane // lanes_per_row
                        span0 = lane % lanes_per_row * span

                        kbuf = T.alloc_local([chunk], accum)
                        qbuf = T.alloc_local([chunk], accum)
                        dot = T.alloc_local([group], accum)
                        prob = T.alloc_local([group], accum)
                        acc_o = T.alloc_local([group, vec], accum)
                        row_len = T.alloc_local([group], "int32")
                        m_run = T.alloc_local([group], accum)
                        l_run = T.alloc_local([group], accum)
                        m_new = T.alloc_local([1], accum)
                        resc = T.alloc_local([1], accum)
                        pj = T.alloc_local([1], accum)
                        # The cross-split merge's own registers.
                        lse = T.alloc_local([num_split], accum)
                        acc = T.alloc_local([vec], accum)
                        peak = T.alloc_local([1], accum)
                        total = T.alloc_local([1], accum)
                        weight = T.alloc_local([1], accum)

                        for i in T.serial(T.ceildiv(group * dim, _CONS)):
                            idx = i * _CONS + tx
                            if idx < group * dim:
                                q_s[idx // dim, idx % dim] = T.cast(
                                    Q[bb, q0 + idx // dim, bh, idx % dim], accum
                                )
                        for g in T.unroll(group):
                            row_len[g] = kv_len_block
                            if is_causal:
                                row_len[g] = T.max(kv_len_block - (group - 1 - g), 0)
                            for c in T.serial(vec):
                                acc_o[g, c] = 0
                            # A finite floor, not -inf. A warp whose rows are all
                            # masked, or a split past the end of a short cache, would
                            # otherwise reach exp2(-inf - -inf) = NaN and poison the
                            # merge; from a floor the same arithmetic yields a
                            # rescale of exactly 1 over a zero accumulator.
                            m_run[g] = _NEG_FLOOR
                            # This lane's share of the row sum.
                            l_run[g] = 0
                        T.sync_threads(barrier_id=_MERGE_BARRIER, arrive_count=_CONS)

                        for t in T.serial(n_tiles):
                            st = t % stages
                            ready_parity = (t // stages) % 2
                            row0 = (tile_begin + t) * block_N

                            T.mbarrier_wait_parity(k_ready[st], ready_parity)
                            for g in T.unroll(group):
                                dot[g] = 0
                            for cc in T.serial(span // chunk):
                                for v in T.vectorized(chunk):
                                    kbuf[v] = T.cast(Ks[st, j_own, span0 + cc * chunk + v], accum)
                                for g in T.unroll(group):
                                    for v in T.vectorized(chunk):
                                        qbuf[v] = q_s[g, span0 + cc * chunk + v]
                                    for v in T.serial(chunk):
                                        dot[g] += qbuf[v] * kbuf[v]
                            T.mbarrier_arrive(k_free[st])

                            # One max per query row and tile, across the warp's rows.
                            for g in T.unroll(group):
                                for r in T.unroll(lanes_per_row.bit_length() - 1):
                                    dot[g] += T.shfl_xor(
                                        dot[g], T.shift_right(lanes_per_row // 2, r)
                                    )
                                dot[g] = T.if_then_else(
                                    row0 + j_own < row_len[g], dot[g] * scale, -T.infinity(accum)
                                )
                                m_new[0] = T.max(m_run[g], T.warp_reduce_max(dot[g]))
                                resc[0] = T.exp2(m_run[g] - m_new[0])
                                m_run[g] = m_new[0]
                                prob[g] = T.exp2(dot[g] - m_new[0])
                                # Every lane of a row holds its weight; one counts it.
                                l_run[g] = l_run[g] * resc[0] + T.if_then_else(
                                    lane % lanes_per_row == 0, prob[g], 0.0
                                )
                                for c in T.serial(vec):
                                    acc_o[g, c] *= resc[0]

                            T.mbarrier_wait_parity(v_ready[st], ready_parity)
                            # A tile not wholly within a row's length skips the keys past it,
                            # since a non-finite value survives a zero weight.
                            for g in T.unroll(group):
                                if row0 + block_N <= row_len[g]:
                                    for jj in T.serial(rows_per_warp):
                                        j = warp * rows_per_warp + jj
                                        pj[0] = T.shfl_sync(prob[g], jj * lanes_per_row)
                                        for c in T.serial(vec):
                                            acc_o[g, c] += pj[0] * T.cast(Vs[st, j, d0 + c], accum)
                                else:
                                    for jj in T.serial(rows_per_warp):
                                        j = warp * rows_per_warp + jj
                                        pj[0] = T.shfl_sync(prob[g], jj * lanes_per_row)
                                        if row0 + j < row_len[g]:
                                            for c in T.serial(vec):
                                                acc_o[g, c] += pj[0] * T.cast(
                                                    Vs[st, j, d0 + c], accum
                                                )
                            T.mbarrier_arrive(v_free[st])

                        # Merge the warp partials into one partial per split.
                        for g in T.unroll(group):
                            l_run[g] = T.warp_reduce_sum(l_run[g])
                            for c in T.serial(vec):
                                warp_o[g, warp, d0 + c] = acc_o[g, c]
                            if lane == 0:
                                warp_m[g, warp] = m_run[g]
                                warp_l[g, warp] = l_run[g]
                        T.sync_threads(barrier_id=_MERGE_BARRIER, arrive_count=_CONS)

                        # Warp w merges rows w, w + _CONS_WARPS, ...
                        for g in T.unroll(group):
                            if warp == g % _CONS_WARPS:
                                m_new[0] = _NEG_FLOOR
                                for u in T.serial(_CONS_WARPS):
                                    m_new[0] = T.max(m_new[0], warp_m[g, u])
                                l_run[g] = 0
                                for c in T.serial(vec):
                                    acc_o[g, c] = 0
                                for u in T.serial(_CONS_WARPS):
                                    resc[0] = T.exp2(warp_m[g, u] - m_new[0])
                                    l_run[g] += warp_l[g, u] * resc[0]
                                    for c in T.serial(vec):
                                        acc_o[g, c] += warp_o[g, u, d0 + c] * resc[0]
                                # A split with no rows publishes a zero partial and a
                                # finite floor, so the cross-split merge weights it
                                # out exactly instead of multiplying zero by a NaN.
                                resc[0] = T.if_then_else(l_run[g] > 0, 1.0 / l_run[g], 0.0)
                                if num_split == 1:
                                    for c in T.serial(vec):
                                        Output[bb, q0 + g, bh, d0 + c] = T.cast(
                                            acc_o[g, c] * resc[0], dtype
                                        )
                                else:
                                    for c in T.serial(vec):
                                        O_partial[bb, q0 + g, bh, bs, d0 + c] = (
                                            acc_o[g, c] * resc[0]
                                        )
                                    if lane == 0:
                                        glse[bb, q0 + g, bh, bs] = T.if_then_else(
                                            l_run[g] > 0, T.log2(l_run[g]) + m_new[0], _EMPTY_LSE
                                        )

                        if num_split > 1:
                            # The last split to arrive merges every split. The release on
                            # the count publishes this block's partials, which the barrier
                            # orders before it; the acquire makes the others' visible.
                            T.sync_threads(barrier_id=_MERGE_BARRIER, arrive_count=_CONS)
                            if tx == 0:
                                arrived_before[0] = T.atomic_add(
                                    Arrived[bg * heads + bh],
                                    1,
                                    memory_order="acq_rel",
                                    return_prev=True,
                                )
                            T.sync_threads(barrier_id=_MERGE_BARRIER, arrive_count=_CONS)
                            if arrived_before[0] == num_split - 1:
                                for g in T.unroll(group):
                                    if warp == g % _CONS_WARPS:
                                        # Every partial is read before any is used:
                                        # folding into the read loop chains the loads.
                                        peak[0] = _EMPTY_LSE
                                        for s in T.serial(num_split):
                                            lse[s] = glse[bb, q0 + g, bh, s]
                                        for s in T.serial(num_split):
                                            peak[0] = T.max(peak[0], lse[s])
                                        total[0] = 0
                                        for c in T.serial(vec):
                                            acc[c] = 0
                                        for s in T.serial(num_split):
                                            weight[0] = T.exp2(lse[s] - peak[0])
                                            total[0] += weight[0]
                                            for c in T.serial(vec):
                                                acc[c] += (
                                                    O_partial[bb, q0 + g, bh, s, d0 + c] * weight[0]
                                                )
                                        total[0] = 1.0 / total[0]
                                        for c in T.serial(vec):
                                            Output[bb, q0 + g, bh, d0 + c] = T.cast(
                                                acc[c] * total[0], dtype
                                            )
                                # Leave the count zeroed for the next launch.
                                if tx == 0:
                                    Arrived[bg * heads + bh] = 0

        return mha_decode_paged_ws

    return _func


class MHADecodePagedWSKernel(Kernel, MHAPagedDecodeFwdInterface):
    """SM90 paged MHA decode: hand-written warp specialization, no MMA."""

    supported_archs: list[int] = [90]
    # Tile heights the kernel picks from.
    _TILE_HEIGHTS = (16, 32, 64, 128)
    # Head dims up to which a lane's output slice, dim / 32 registers per row, fits.
    _MAX_DIM = 256
    # Per-lane registers the per-row state of a block's query rows may take, and the
    # scalars of that state beside the output slice: score, weight, max, sum, length.
    _ROW_STATE_REGS = 128
    _ROW_SCALARS = 5
    # Pipeline depths of the K/V ring autotune weighs; the default takes the shallowest.
    _STAGE_CHOICES = (2, 3)
    # Splits per output, and the grid size past which fewer are taken. Fitted on H200:
    # 8 splits measured fastest on every manifest row; re-measure the rows to move it.
    _SPLITS = 8
    _MAX_BLOCKS = 1024
    # Multiply-adds up to which the kernel serves several query rows, counted as
    # ``batch * heads * seqlen_q * seqlen_kv * dim`` with the pool size bounding every
    # request. Fitted on H200: the CUDA-core contraction lost to the tensor-core kernel
    # from 2**28 up; re-measure both kernels near the bound to move it.
    _MAX_MULTI_QUERY_MACS = 2**27

    @classmethod
    def applies(cls, call: AttentionCall) -> bool:
        """A 16-bit call without softcap or window, a head dim a warp spans, a page some
        tile height divides, and several query rows only below the work bound."""
        macs = call.batch * call.heads * call.max_seqlen_q * call.seqlen_kv * call.dim
        return (
            call.max_seqlen_q >= 1
            and (call.max_seqlen_q == 1 or macs <= cls._MAX_MULTI_QUERY_MACS)
            and call.softcap == 0.0
            and call.dtype in ATTENTION_DTYPES
            and not call.is_fp8
            and not call.uses_sliding_window
            and call.dim % WARP_LANES == 0
            and 0 < call.dim <= cls._MAX_DIM
            and bool(cls._tile_heights(call.page_size, call.seqlen_kv))
        )

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        # The device index is in the identity: the kernel is compiled for its architecture.
        index = call.device.index if call.device is not None else None
        args = (
            call.batch,
            call.heads,
            call.max_seqlen_q,
            call.seqlen_kv,
            call.dim,
            call.page_size,
            call.is_causal,
            call.dtype,
        )
        return (*args, call.max_pages_per_req, index), lambda: cls(
            *args, device_index=index, max_pages_per_req=call.max_pages_per_req
        )

    def __init__(
        self,
        batch: int,
        heads: int,
        seqlen_q: int,
        seqlen_kv: int,
        dim: int,
        page_size: int,
        is_causal: bool = False,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
        max_pages_per_req: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.max_pages_per_req = (
            max_pages_per_req
            if max_pages_per_req is not None
            else (seqlen_kv + page_size - 1) // page_size
        )
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _mha_decode_paged_ws_kernel(
            self.batch,
            self.heads,
            self.seqlen_q,
            self.seqlen_kv,
            self.dim,
            self.page_size,
            self.is_causal,
            self.dtype_str,
            self.max_pages_per_req,
        )
        self._arrived: dict[int, torch.Tensor] = {}
        self._supply_prog = self._make_supply_prog()
        self.init_config(config, tune)

    # -- configuration ----------------------------------------------------

    @classmethod
    def _tile_heights(cls, page_size: int, seqlen_kv: int) -> list[int]:
        """Tile heights that divide the page size, so a tile reads one block-table entry."""
        return [
            n for n in cls._TILE_HEIGHTS if n <= page_size and page_size % n == 0 and n <= seqlen_kv
        ]

    def _group_choices(self) -> list[int]:
        """Query rows per block: divisors of ``seqlen_q`` whose per-lane state fits.

        Per row a lane holds its output slice plus ``_ROW_SCALARS`` scalars.
        """
        per_row = self.dim // WARP_LANES + self._ROW_SCALARS
        return [
            g
            for g in range(1, self.seqlen_q + 1)
            if self.seqlen_q % g == 0 and g * per_row <= self._ROW_STATE_REGS
        ]

    def _num_split(self, group: int) -> int:
        """``_SPLITS`` splits, halved while the grid would exceed ``_MAX_BLOCKS`` blocks.

        Set by rule: the tuner ranks these few-microsecond candidates with a warm L2
        and picks split counts the cold-cache benchmark measures slower. It still
        weighs one split, which wins once the grid fills the device without splitting.
        """
        work_items = self.batch * (self.seqlen_q // group) * self.heads
        num_split = self._SPLITS
        while num_split > 1 and (
            num_split * work_items > self._MAX_BLOCKS or num_split > self.seqlen_kv
        ):
            num_split //= 2
        return num_split

    @property
    def default_config(self) -> dict:
        """The largest admissible row group, the rule's split count, the tallest tile."""
        group = max(self._group_choices())
        num_split = self._num_split(group)
        rows_per_split = -(-self.seqlen_kv // num_split)
        block_N = max(
            (n for n in self._tile_heights(self.page_size, self.seqlen_kv) if n <= rows_per_split),
            default=min(self._tile_heights(self.page_size, self.seqlen_kv)),
        )
        return {
            "block_N": block_N,
            "num_split": num_split,
            "stages": min(self._STAGE_CHOICES),
            "group": group,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_N": block_N, "num_split": num_split, "stages": stages, "group": group}
            for block_N in self._tile_heights(self.page_size, self.seqlen_kv)
            for group in self._group_choices()
            for num_split in sorted({1, self._num_split(group)})
            for stages in self._STAGE_CHOICES
        ]

    # -- autotuning inputs ------------------------------------------------

    def _make_supply_prog(self):
        """Supply in-range paging metadata and zeroed split counts to the autotuner.

        The int32 inputs are a length, a page table and the split counts, not data:
        random values would index outside the cache or skip the merge, so the
        candidates are fed values the kernel can legally follow. They are matched by
        position, since the counts can have the shape of the lengths.
        """
        from tilelang.utils.tensor import get_tensor_supply as _get_tensor_supply

        default_supply = _get_tensor_supply(tilelang.TensorSupplyType.Auto)
        batch, seqlen_kv, page_size = self.batch, self.seqlen_kv, self.page_size
        num_pages = self.max_pages_per_req
        counts = self.batch * self.seqlen_q * self.heads
        # Positions of real_seqlen_kv, block_table and Arrived in the kernel signature.
        lengths_arg, table_arg, arrived_arg = 3, 4, 7

        def supply_prog(params):
            table = torch.arange(num_pages, dtype=torch.int32, device="cuda")
            given = {
                lengths_arg: torch.full(
                    (batch,),
                    min(seqlen_kv, num_pages * page_size),
                    dtype=torch.int32,
                    device="cuda",
                ),
                table_arg: table.unsqueeze(0).expand(batch, -1).contiguous(),
                arrived_arg: torch.zeros(counts, dtype=torch.int32, device="cuda"),
            }
            return [given[i] if i in given else default_supply(p) for i, p in enumerate(params)]

        return supply_prog

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    # -- execution --------------------------------------------------------

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        num_split = self.config["num_split"]
        # torch.empty, never zeros: a fill would be one more launch in front of
        # a kernel whose whole cost is launches, and both buffers are written in
        # full before they are read.
        glse = torch.empty(
            (self.batch, self.seqlen_q, self.heads, num_split), dtype=torch.float32, device=Q.device
        )
        O_partial = torch.empty(
            (self.batch, self.seqlen_q, self.heads, num_split, self.dim),
            dtype=torch.float32,
            device=Q.device,
        )
        return self.kernel(
            self.config["block_N"], num_split, self.config["stages"], self.config["group"]
        )(Q, K, V, real_seqlen_kv, block_table, glse, O_partial, self._arrival_counts(Q.device))

    def _arrival_counts(self, device: torch.device) -> torch.Tensor:
        """The zeroed per-output split counts; the kernel leaves them zeroed again.

        One buffer per stream, since two launches on different streams would share
        the counts; a captured graph owns one from its private pool.
        """
        size = self.batch * self.seqlen_q * self.heads
        if torch.cuda.is_current_stream_capturing():
            return torch.zeros(size, dtype=torch.int32, device=device)
        stream = torch.cuda.current_stream(device).cuda_stream
        counts = self._arrived.get(stream)
        if counts is None:
            counts = torch.zeros(size, dtype=torch.int32, device=device)
            self._arrived[stream] = counts
        return counts
