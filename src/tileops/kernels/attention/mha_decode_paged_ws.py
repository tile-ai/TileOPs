"""Warp-specialized paged MHA decode for Hopper.

``seqlen_q`` is 1, so the score computation is a matrix-vector product: the M
axis of an MMA would be 94-98% padding, and the tensor cores buy nothing on a
kernel whose whole KV cache is a few megabytes. This kernel therefore keeps the
contractions on the CUDA cores and spends its budget on the two things that do
decide the time -- how few launches reach the device, and how early the KV tiles
are in flight.

The pipeline is written out rather than scheduled:

* one producer warp group and one consumer warp group, split by ``T.ws`` and
  placed in the two arms of a single ``If`` (two sequential ``T.ws`` regions let
  the thread-sync pass emit a block-wide barrier the producer never reaches);
* ``T.tma_copy`` moves each K and V tile into a per-stage ring, announced on
  ``T.alloc_barrier`` mbarriers with the phase parity carried by hand;
* ``T.set_max_nreg`` hands the producer's registers to the consumer;
* ``T.annotate_layout`` states the swizzle the TMA destinations use; and
* every reduction is a ``T.shfl_xor`` chain over thread-indexed registers, so a
  consumer warp owns its rows outright and needs no block-wide reduction.

Each consumer warp is its own online-softmax accumulator over the rows it owns;
the four warp partials merge through shared memory, and the per-split partials
merge across the grid in a second launch.
"""

import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.layout import make_swizzled_layout

from tileops.kernels.kernel_base import Kernel

from .call_spec import paged_decode_ws_region

__all__ = ["MHADecodePagedWsKernel"]

#: log2(e): the softmax runs on exp2, which is one instruction.
LOG2E = 1.44269504

WARP = 32
#: Warps in the consumer group. One warp group of each role, 256 threads: two
#: consumer groups deadlock the block-wide sync the layout pass inserts.
CONS_WARPS = 4
CONS = CONS_WARPS * WARP
PROD = 128
#: Named barrier the consumer group uses on its own, never block-wide.
_MERGE_BARRIER = 1
#: Blocks to aim the split count at, so one wave covers the device.
_TARGET_BLOCKS = 128
#: Finite stand-in for -inf in the running max, so a fully masked tile rescales
#: by exactly one instead of evaluating exp2(-inf - -inf).
_NEG_FLOOR = -1.0e38
#: What an empty split publishes as its log-sum-exp: finite, so the cross-split
#: merge gives it weight exp2(_EMPTY_LSE - peak) = 0 rather than 0 * NaN.
_EMPTY_LSE = -1.0e30


def _jit_kwargs() -> dict:
    return dict(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )


def _mha_decode_paged_ws_kernel(batch: int, heads: int, seqlen_kv: int, dim: int,
                                page_size: int, dtype: str):
    """Build the JIT'd (split, combine) pair for one shape specialization."""
    scale = dim**-0.5 * LOG2E
    accum = "float"
    num_pages = (seqlen_kv + page_size - 1) // page_size
    #: Head-vector elements a lane owns. The dot product is a whole warp wide,
    #: so this is what makes the shuffle chain exactly five steps.
    vec = dim // WARP

    @tilelang.jit(**_jit_kwargs())
    def _func(block_N: int, num_split: int, stages: int):
        rows_per_warp = block_N // CONS_WARPS
        threads = CONS + PROD

        @T.macro
        def split(
            Q: T.Tensor([batch, 1, heads, dim], dtype),
            K: T.Tensor([seqlen_kv, heads, dim], dtype),
            V: T.Tensor([seqlen_kv, heads, dim], dtype),
            real_seqlen_kv: T.Tensor([batch], "int32"),
            block_table: T.Tensor([batch, num_pages], "int32"),
            glse: T.Tensor([batch, heads, num_split], accum),
            O_partial: T.Tensor([batch, heads, num_split, dim], accum),
        ):
            with T.Kernel(num_split, heads, batch, threads=threads) as (bs, bh, bb):
                # Every ring and barrier is declared at kernel-body level:
                # an allocation made inside a guarded stage is not visible to
                # the other arm.
                Ks = T.alloc_shared([stages, block_N, dim], dtype)
                Vs = T.alloc_shared([stages, block_N, dim], dtype)
                warp_m = T.alloc_shared([CONS_WARPS], accum, scope="shared")
                warp_l = T.alloc_shared([CONS_WARPS], accum, scope="shared")
                warp_o = T.alloc_shared([CONS_WARPS, dim], accum, scope="shared")
                T.annotate_layout({
                    Ks: make_swizzled_layout(Ks),
                    Vs: make_swizzled_layout(Vs),
                })

                # A ready barrier is completed by whoever announces the tile, an
                # empty barrier by every consumer thread that finished reading
                # it. The counts are the two group sizes; getting them wrong
                # hangs the block and takes the context with it.
                k_ready = T.alloc_barrier([PROD] * stages)
                k_free = T.alloc_barrier([CONS] * stages)
                v_ready = T.alloc_barrier([PROD] * stages)
                v_free = T.alloc_barrier([CONS] * stages)

                tx = T.get_thread_binding()
                kv_len = real_seqlen_kv[bb]
                # Tiles are page-aligned and block-aligned, so one tile never
                # straddles two pages and the block table is read once per tile.
                tiles_total = T.ceildiv(kv_len, block_N)
                tiles_per_split = T.ceildiv(tiles_total, num_split)
                tile_begin = bs * tiles_per_split
                n_tiles = T.max(
                    T.min(tile_begin + tiles_per_split, tiles_total) - tile_begin, 0)

                if tx >= CONS:
                    with T.ws(1):
                        T.set_max_nreg(24, 0)
                        for t in T.serial(n_tiles):
                            st = t % stages
                            free_parity = ((t // stages) % 2) ^ 1
                            row0 = (tile_begin + t) * block_N
                            base = (block_table[bb, row0 // page_size] * page_size
                                    + row0 % page_size)
                            T.mbarrier_wait_parity(k_free[st], free_parity)
                            T.tma_copy(K[base:base + block_N, bh, :], Ks[st, :, :],
                                       barrier=k_ready[st])
                            T.mbarrier_arrive(k_ready[st])
                            T.mbarrier_wait_parity(v_free[st], free_parity)
                            T.tma_copy(V[base:base + block_N, bh, :], Vs[st, :, :],
                                       barrier=v_ready[st])
                            T.mbarrier_arrive(v_ready[st])
                else:
                    with T.ws(0):
                        T.set_max_nreg(240, 1)
                        warp = tx // WARP
                        lane = tx % WARP
                        d0 = lane * vec

                        q_reg = T.alloc_local([vec], accum)
                        acc_o = T.alloc_local([vec], accum)
                        scores = T.alloc_local([rows_per_warp], accum)
                        dot = T.alloc_local([1], accum)
                        m_run = T.alloc_local([1], accum)
                        m_new = T.alloc_local([1], accum)
                        l_run = T.alloc_local([1], accum)
                        resc = T.alloc_local([1], accum)
                        prob = T.alloc_local([1], accum)

                        for c in T.serial(vec):
                            q_reg[c] = T.cast(Q[bb, 0, bh, d0 + c], accum)
                            acc_o[c] = 0
                        # A finite floor, not -inf. A warp whose rows are all
                        # masked, or a split past the end of a short cache, would
                        # otherwise reach exp2(-inf - -inf) = NaN and poison the
                        # merge; from a floor the same arithmetic yields a
                        # rescale of exactly 1 over a zero accumulator.
                        m_run[0] = _NEG_FLOOR
                        l_run[0] = 0

                        for t in T.serial(n_tiles):
                            st = t % stages
                            ready_parity = (t // stages) % 2
                            row0 = (tile_begin + t) * block_N

                            T.mbarrier_wait_parity(k_ready[st], ready_parity)
                            for jj in T.serial(rows_per_warp):
                                j = warp * rows_per_warp + jj
                                dot[0] = 0
                                for c in T.serial(vec):
                                    dot[0] += q_reg[c] * T.cast(Ks[st, j, d0 + c], accum)
                                # Written out: the eager builder rewrites
                                # assignments only inside the traced function, so
                                # a helper cannot hold the shuffle chain.
                                dot[0] += T.shfl_xor(dot[0], 16)
                                dot[0] += T.shfl_xor(dot[0], 8)
                                dot[0] += T.shfl_xor(dot[0], 4)
                                dot[0] += T.shfl_xor(dot[0], 2)
                                dot[0] += T.shfl_xor(dot[0], 1)
                                scores[jj] = T.if_then_else(row0 + j < kv_len,
                                                            dot[0] * scale,
                                                            -T.infinity(accum))
                            T.mbarrier_arrive(k_free[st])

                            # The whole online softmax stays inside this warp:
                            # after the shuffle chain every lane already holds
                            # every score the warp owns.
                            m_new[0] = m_run[0]
                            for jj in T.serial(rows_per_warp):
                                m_new[0] = T.max(m_new[0], scores[jj])
                            resc[0] = T.exp2(m_run[0] - m_new[0])
                            m_run[0] = m_new[0]
                            l_run[0] *= resc[0]
                            for c in T.serial(vec):
                                acc_o[c] *= resc[0]

                            T.mbarrier_wait_parity(v_ready[st], ready_parity)
                            for jj in T.serial(rows_per_warp):
                                j = warp * rows_per_warp + jj
                                prob[0] = T.exp2(scores[jj] - m_run[0])
                                l_run[0] += prob[0]
                                for c in T.serial(vec):
                                    acc_o[c] += prob[0] * T.cast(Vs[st, j, d0 + c], accum)
                            T.mbarrier_arrive(v_free[st])

                        # Merge the warp partials, then publish one partial per
                        # split for the second launch to merge across the grid.
                        for c in T.serial(vec):
                            warp_o[warp, d0 + c] = acc_o[c]
                        if lane == 0:
                            warp_m[warp] = m_run[0]
                            warp_l[warp] = l_run[0]
                        T.sync_threads(barrier_id=_MERGE_BARRIER, arrive_count=CONS)

                        if warp == 0:
                            m_new[0] = _NEG_FLOOR
                            for u in T.serial(CONS_WARPS):
                                m_new[0] = T.max(m_new[0], warp_m[u])
                            l_run[0] = 0
                            for c in T.serial(vec):
                                acc_o[c] = 0
                            for u in T.serial(CONS_WARPS):
                                resc[0] = T.exp2(warp_m[u] - m_new[0])
                                l_run[0] += warp_l[u] * resc[0]
                                for c in T.serial(vec):
                                    acc_o[c] += warp_o[u, d0 + c] * resc[0]
                            # A split with no rows publishes a zero partial and a
                            # finite floor, so the cross-split merge weights it
                            # out exactly instead of multiplying zero by a NaN.
                            resc[0] = T.if_then_else(l_run[0] > 0, 1.0 / l_run[0], 0.0)
                            for c in T.serial(vec):
                                O_partial[bb, bh, bs, d0 + c] = acc_o[c] * resc[0]
                            if lane == 0:
                                glse[bb, bh, bs] = T.if_then_else(
                                    l_run[0] > 0, T.log2(l_run[0]) + m_new[0], _EMPTY_LSE)

        @T.macro
        def combine(
            glse: T.Tensor([batch, heads, num_split], accum),
            O_partial: T.Tensor([batch, heads, num_split, dim], accum),
            Output: T.Tensor([batch, 1, heads, dim], dtype),
        ):
            with T.Kernel(heads, batch, threads=WARP) as (bh, bb):
                lane = T.get_thread_binding()
                d0 = lane * vec
                lse = T.alloc_local([num_split], accum)
                part = T.alloc_local([num_split, vec], accum)
                acc = T.alloc_local([vec], accum)
                peak = T.alloc_local([1], accum)
                total = T.alloc_local([1], accum)
                weight = T.alloc_local([1], accum)

                # Read every partial before consuming any: an accumulate inside
                # the read loop turns num_split independent loads into a
                # dependent chain, and this launch is otherwise launch-bound.
                peak[0] = _EMPTY_LSE
                for s in T.serial(num_split):
                    lse[s] = glse[bb, bh, s]
                    for c in T.serial(vec):
                        part[s, c] = O_partial[bb, bh, s, d0 + c]
                for s in T.serial(num_split):
                    peak[0] = T.max(peak[0], lse[s])

                total[0] = 0
                for c in T.serial(vec):
                    acc[c] = 0
                for s in T.serial(num_split):
                    weight[0] = T.exp2(lse[s] - peak[0])
                    total[0] += weight[0]
                    for c in T.serial(vec):
                        acc[c] += part[s, c] * weight[0]
                total[0] = 1.0 / total[0]
                for c in T.serial(vec):
                    Output[bb, 0, bh, d0 + c] = T.cast(acc[c] * total[0], dtype)

        @T.prim_func
        def mha_decode_paged_ws(
            Q: T.Tensor([batch, 1, heads, dim], dtype),
            K: T.Tensor([seqlen_kv, heads, dim], dtype),
            V: T.Tensor([seqlen_kv, heads, dim], dtype),
            real_seqlen_kv: T.Tensor([batch], "int32"),
            block_table: T.Tensor([batch, num_pages], "int32"),
            glse: T.Tensor([batch, heads, num_split], accum),
            O_partial: T.Tensor([batch, heads, num_split, dim], accum),
            Output: T.Tensor([batch, 1, heads, dim], dtype),
        ):
            split(Q, K, V, real_seqlen_kv, block_table, glse, O_partial)
            combine(glse, O_partial, Output)

        return mha_decode_paged_ws

    return _func


# Custom op (torch.compile compatible wrapper)


@torch.library.custom_op("top::mha_decode_paged_ws_op", mutates_args=())
def _mha_decode_paged_ws_op(batch: int, heads: int, seqlen_kv: int, dim: int, page_size: int,
                            dtype: str, block_N: int, num_split: int, stages: int,
                            Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                            real_seqlen_kv: torch.Tensor, block_table: torch.Tensor,
                            glse: torch.Tensor,
                            O_partial: torch.Tensor) -> torch.Tensor:
    kernel = _mha_decode_paged_ws_kernel(batch, heads, seqlen_kv, dim, page_size, dtype)
    return kernel(block_N, num_split, stages)(Q, K, V, real_seqlen_kv, block_table, glse,
                                              O_partial)


@_mha_decode_paged_ws_op.register_fake
def _(batch: int, heads: int, seqlen_kv: int, dim: int, page_size: int, dtype: str,
      block_N: int, num_split: int, stages: int, Q: torch.Tensor, K: torch.Tensor,
      V: torch.Tensor, real_seqlen_kv: torch.Tensor, block_table: torch.Tensor,
      glse: torch.Tensor, O_partial: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(Q)


# Kernel class


class MHADecodePagedWsKernel(Kernel):
    """Hopper paged MHA decode: hand-written warp specialization, no MMA."""

    supported_archs: list[int] = [90]

    @classmethod
    def applies(cls, call) -> bool:
        return paged_decode_ws_region(call)

    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen_q: int,
                 seqlen_kv: int,
                 dim: int,
                 page_size: int,
                 is_causal: bool = False,
                 dtype: torch.dtype = torch.float16,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _mha_decode_paged_ws_kernel(self.batch, self.heads, self.seqlen_kv,
                                                  self.dim, self.page_size, self.dtype_str)
        self._supply_prog = self._make_supply_prog()
        self.init_config(config, tune)

    # -- configuration ----------------------------------------------------

    def _block_n_choices(self) -> list[int]:
        """Tile heights that keep one tile inside one page.

        A tile that straddles a page boundary would need a second block-table
        read and a second TMA base, so the tile height divides the page size.
        It also splits evenly across the consumer warps.
        """
        return [
            n for n in (16, 32, 64, 128)
            if n % CONS_WARPS == 0 and n <= self.page_size and self.page_size % n == 0
            and n <= self.seqlen_kv
        ]

    def _split_choices(self) -> list[int]:
        work_items = self.batch * self.heads
        return sorted({
            s for s in (1, 2, 4, 8, 16, 32, 64)
            if s <= self.seqlen_kv and s * work_items <= 8 * _TARGET_BLOCKS
        })

    @property
    def default_config(self) -> dict:
        """Aim the grid at one wave, then take the tallest tile that fits.

        Measured on H200: the split count that puts roughly ``_TARGET_BLOCKS``
        blocks on the device wins across all four manifest shapes, and among the
        tile heights that then cover a split, the tallest is at worst within
        noise of the best.
        """
        work_items = max(1, self.batch * self.heads)
        num_split = max(1, min(_TARGET_BLOCKS // work_items, self.seqlen_kv))
        num_split = max((s for s in self._split_choices() if s <= num_split),
                        default=1)
        rows_per_split = -(-self.seqlen_kv // num_split)
        block_N = max((n for n in self._block_n_choices() if n <= rows_per_split),
                      default=min(self._block_n_choices()))
        return {"block_N": block_N, "num_split": num_split, "stages": 2}

    @property
    def autotune_configs(self) -> list[dict]:
        combos = itertools.product(self._block_n_choices(), self._split_choices(), (2, 3))
        return [{
            "block_N": block_N,
            "num_split": num_split,
            "stages": stages,
        } for block_N, num_split, stages in combos]

    # -- autotuning inputs ------------------------------------------------

    def _make_supply_prog(self):
        """Supply in-range paging metadata to the autotuner.

        The int32 inputs are a length and a page table, not data: random values
        would index outside the cache, so the candidates have to be fed a table
        the kernel can legally follow.
        """
        from tilelang.utils.tensor import get_tensor_supply as _get_tensor_supply

        default_supply = _get_tensor_supply(tilelang.TensorSupplyType.Auto)
        batch, seqlen_kv, page_size = self.batch, self.seqlen_kv, self.page_size
        num_pages = (seqlen_kv + page_size - 1) // page_size

        def supply_prog(params):
            inputs = []
            for param in params:
                shape = tuple(int(s) for s in param.shape)
                if str(param.dtype) == "int32" and shape == (batch,):
                    inputs.append(
                        torch.full((batch,), seqlen_kv, dtype=torch.int32, device="cuda"))
                elif str(param.dtype) == "int32" and shape == (batch, num_pages):
                    table = torch.arange(num_pages, dtype=torch.int32, device="cuda")
                    inputs.append(table.unsqueeze(0).expand(batch, -1).contiguous())
                else:
                    inputs.append(default_supply(param))
            return inputs

        return supply_prog

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    # -- execution --------------------------------------------------------

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        num_split = self.config["num_split"]
        # torch.empty, never zeros: a fill would be one more launch in front of
        # a kernel whose whole cost is launches, and both buffers are written in
        # full before they are read.
        glse = torch.empty((self.batch, self.heads, num_split),
                           dtype=torch.float32, device=Q.device)
        O_partial = torch.empty((self.batch, self.heads, num_split, self.dim),
                                dtype=torch.float32, device=Q.device)
        return _mha_decode_paged_ws_op(
            self.batch, self.heads, self.seqlen_kv, self.dim, self.page_size,
            self.dtype_str, self.config["block_N"], num_split, self.config["stages"],
            Q, K, V, real_seqlen_kv, block_table, glse, O_partial)
