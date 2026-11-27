"""Seeded categorical draw: one inverse-CDF search per row, over one read of the row."""

import operator
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import (
    BLOCK_SHARED_BYTES_OPT_IN,
    SHARED_BUFFER_ALIGN_BYTES,
)
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sampling.call_spec import SamplingCall, SamplingFromProbsFwdInterface
from tileops._kernels.sampling.philox import draw, uniform_of
from tileops._kernels.sampling.row_tiles import (
    gather_part_totals,
    load_vector,
    pick_entry,
    row_split,
    vector_width,
)
from tileops._kernels.warp_reduce import warp_reduce
from tileops._utils import WARP_LANES

__all__ = ["SamplingFromProbsFwdKernel"]


@target_cache(maxsize=32)
def _sampling_from_probs_kernel(
    batch: int, vocab: int, vec: int, threads: int, parts: int, pace: int
):
    """Build the draw over ``batch`` rows of ``vocab`` weights, ``parts`` blocks to a row.

    A thread folds one 16-byte vector a slot and a warp its 32 threads, leaving one ``seg`` entry
    per (slot, warp); the search descends the row's blocks, then ``seg``, then the winning warp's
    lanes, then that lane's weights, re-reading 32 vectors. ``pace`` vectors are in flight before
    any is folded. Warp folds are float32 over groups ``vocab`` and ``threads`` fix; every sum
    above them is float64.
    """
    full = vocab // vec
    row_tiles = -(-full // threads)
    chunk = -(-row_tiles // parts)
    warps = threads // WARP_LANES
    grid = batch * parts
    # Entries of ``seg``, and how many of them one lane of a descent walks.
    leaves = chunk * warps
    leaf_span = -(-leaves // WARP_LANES)
    rounds = -(-chunk // pace)
    part_span = -(-parts // WARP_LANES)

    @tilelang.jit(compile_flags=csrc_include("streaming_load.h"))
    def _sampling_from_probs_func():
        @T.macro
        def take(dst, slot, src, spot, inside):
            """Read the ``vec`` weights of ``src`` at element offset ``spot`` into
            ``dst[slot]``, evict-first, and zero where that vector is past the row."""
            for c in T.unroll(vec):
                dst[slot, c] = T.float32(0)
            if inside:
                load_vector(dst, slot, src, spot, vec, True)

        @T.macro
        def fold(dst, src, slot):
            """Leave in ``dst[0]`` the float32 sum of ``src[slot]``, in index order."""
            dst[0] = src[slot, 0]
            for c in T.serial(1, vec):
                dst[0] = dst[0] + src[slot, c]

        @T.macro
        def locate(src, n, span, frac, base, lane, pick, edge, acc, idx):
            """Leave in ``pick[0]`` the entry of ``src[0:n]`` the draw lands on and in ``edge[0]`` how
            much of the draw point is left inside it: the first entry whose inclusive prefix passes
            ``frac * total + base``, else the last entry carrying weight (a point rounded past the
            total). Lane ``l`` of warp 0 sums ``src[l * span : (l + 1) * span)``.
            """
            tx = T.get_thread_binding()
            if tx < WARP_LANES:
                acc[0] = T.cast(0, "float64")
                for i in T.serial(span):
                    if tx * span + i < n:
                        acc[0] = acc[0] + src[tx * span + i]
                lane[tx] = acc[0]
            T.sync_threads()
            if tx == 0:
                acc[0] = T.cast(0, "float64")
                for l in T.serial(WARP_LANES):
                    edge[l + 1] = acc[0]
                    acc[0] = acc[0] + lane[l]
                edge[0] = acc[0]
            T.sync_threads()
            if tx < WARP_LANES:
                acc[0] = edge[tx + 1]
                acc[1] = frac * edge[0] + base
                idx[0] = n
                idx[1] = -1
                for i in T.serial(span):
                    if tx * span + i < n:
                        if (idx[0] == n) & (acc[0] + src[tx * span + i] > acc[1]):
                            idx[0] = tx * span + i
                        if src[tx * span + i] > T.cast(0, "float64"):
                            idx[1] = tx * span + i
                        acc[0] = acc[0] + src[tx * span + i]
                pick_entry(idx, pick, n)
            T.sync_threads()
            if tx == 0:
                acc[1] = frac * edge[0] + base
                acc[0] = edge[pick[0] // span + 1]
                for i in T.serial(span):
                    if (pick[0] // span) * span + i < pick[0]:
                        acc[0] = acc[0] + src[(pick[0] // span) * span + i]
                edge[0] = acc[1] - acc[0]
            T.sync_threads()

        @T.prim_func
        def _sampling_from_probs_main(
            probs: T.Tensor((batch * vocab,), "float32"),
            seed: T.Tensor((1,), "int64"),
            offset: T.Tensor((1,), "int64"),
            partial: T.Tensor((grid,), "float64"),
            samples: T.Tensor((batch,), "int32"),
        ):
            with T.Kernel(grid, threads=threads) as bx:
                tx = T.get_thread_binding()
                seg = T.alloc_shared((leaves,), "float64")
                chunks = T.alloc_shared((parts,), "float64")
                tail = T.alloc_shared((WARP_LANES,), "float64")
                lane = T.alloc_shared((WARP_LANES,), "float64")
                # 0 the total of the entries a descent walks, then what is left of the draw
                # point inside the entry it chose; 1 on, one exclusive prefix per lane.
                edge = T.alloc_shared((WARP_LANES + 1,), "float64")
                pick = T.alloc_shared((1,), "int32")
                # The block of the row that owns the draw, and the element the winning warp
                # starts at, both read after ``pick`` has moved on to the next level.
                owner = T.alloc_shared((2,), "int32")
                cur = T.alloc_local((pace, vec), "float32")
                warp = T.alloc_local((1,), "float32")
                acc = T.alloc_local((2,), "float64")
                idx = T.alloc_local((2,), "int32")
                counter = T.alloc_local((4,), "uint32")
                bump = T.alloc_local((2,), "uint32")
                key = T.alloc_local((2,), "uint32")
                spot = T.alloc_local((1,), "int32")
                # 0 what is left of the draw point inside the entry the level above chose,
                # 1 the row's uniform while the first level still has to scale it by a total.
                aim = T.alloc_local((2,), "float64")

                line = bx // parts
                row = line * vocab
                head = bx % parts * chunk

                for r in T.serial(rounds):
                    for j in T.unroll(pace):
                        if r * pace + j < chunk:
                            spot[0] = ((head + r * pace + j) * threads + tx) * vec
                            take(cur, j, probs, row + spot[0], spot[0] < full * vec)
                    for j in T.unroll(pace):
                        if r * pace + j < chunk:
                            fold(warp, cur, j)
                            warp_reduce(warp, operator.add, high_first=True)
                            if tx % WARP_LANES == 0:
                                seg[(r * pace + j) * warps + tx // WARP_LANES] = T.cast(
                                    warp[0], "float64"
                                )
                T.sync_threads()

                # The row's uniform, counted by the row alone, so it is the same value however
                # the launch splits the row.
                draw(counter, key, bump, seed, offset, T.uint32(0), line)
                aim[1] = T.cast(uniform_of(counter[0]), "float64")

                if parts == 1:
                    if tx == 0:
                        owner[0] = 0
                    aim[0] = T.cast(0, "float64")
                else:
                    gather_part_totals(
                        seg, lane, partial, chunks, acc, bx, line, leaves, leaf_span, parts, threads
                    )
                    locate(
                        chunks,
                        parts,
                        part_span,
                        aim[1],
                        T.cast(0, "float64"),
                        lane,
                        pick,
                        edge,
                        acc,
                        idx,
                    )
                    if tx == 0:
                        owner[0] = pick[0]
                    aim[0] = edge[0]
                    aim[1] = T.cast(0, "float64")
                T.sync_threads()

                if owner[0] == bx % parts:
                    locate(seg, leaves, leaf_span, aim[1], aim[0], lane, pick, edge, acc, idx)
                    aim[0] = edge[0]
                    if tx == 0:
                        owner[1] = (
                            row
                            + ((head + pick[0] // warps) * threads + pick[0] % warps * WARP_LANES)
                            * vec
                        )
                    T.sync_threads()
                    if tx < WARP_LANES:
                        spot[0] = owner[1] + tx * vec
                        take(cur, 0, probs, spot[0], spot[0] - row < full * vec)
                        fold(warp, cur, 0)
                        tail[tx] = T.cast(warp[0], "float64")
                    T.sync_threads()
                    locate(
                        tail,
                        WARP_LANES,
                        1,
                        T.cast(0, "float64"),
                        aim[0],
                        lane,
                        pick,
                        edge,
                        acc,
                        idx,
                    )
                    if tx == 0:
                        # The chosen lane's weights, walked in index order: the token is the
                        # first whose weight passes what is left of the draw point.
                        spot[0] = owner[1] + pick[0] * vec
                        acc[0] = edge[0]
                        idx[0] = -1
                        idx[1] = 0
                        for c in T.serial(vec):
                            if spot[0] - row + c < full * vec:
                                acc[1] = T.cast(probs[spot[0] + c], "float64")
                                if acc[1] > T.cast(0, "float64"):
                                    idx[1] = c
                                    if (idx[0] < 0) & (acc[1] > acc[0]):
                                        idx[0] = c
                                acc[0] = acc[0] - acc[1]
                        samples[line] = spot[0] - row + T.if_then_else(idx[0] >= 0, idx[0], idx[1])

        return _sampling_from_probs_main

    return _sampling_from_probs_func


class SamplingFromProbsFwdKernel(Kernel, SamplingFromProbsFwdInterface):
    """Draw one token per row of probabilities: the first index whose inclusive prefix sum passes
    ``u * total``, ``u`` the row's Philox uniform of seed, offset and row index.

    One pass folds the row into per-(slot, warp) sums kept on chip; the search descends those and
    re-reads only the winning warp's 32 vectors. A batch that leaves blocks idle splits each row
    across blocks meeting at one grid barrier. Sums are float32 over groups the vocabulary fixes
    and float64 above, so the token moves with the launch only where the draw point sits within a
    float64 rounding of a token boundary.

    Args:
        call: The call's shape, dtype and device facts.
        config: Unused; the launch follows from the call.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general: ClassVar[bool] = True

    # Keep reduction order fixed so launch tuning cannot change seeded draws.
    _THREADS: ClassVar[int] = 512
    # Vector loads in flight before folding.
    _PACE: ClassVar[int] = 8

    @classmethod
    def _plan(cls, vocab: int, itemsize: int, batch: int, sm_count: int) -> tuple[int, int, int]:
        """The vector width of a row, how many blocks share one, and a block's ``seg`` entries.

        The block size stays fixed whatever the batch: a different one would move the token a
        seed draws.
        """
        vec = vector_width(vocab, itemsize)
        row_tiles = max(1, -(-(vocab // vec) // cls._THREADS))
        parts = row_split(row_tiles, batch, sm_count)
        return vec, parts, -(-row_tiles // parts) * (cls._THREADS // WARP_LANES)

    @classmethod
    def refusal(cls, call: SamplingCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.batch * call.vocab > 2**31 - 1:
            return f"indexes elements with int32, and B * V = {call.batch * call.vocab}"
        _vec, parts, leaves = cls._plan(call.vocab, call.dtype.itemsize, call.batch, call.sm_count)
        # ``seg``, one float64 entry per (slot, warp) of a block's chunk, against the budget
        # less one alignment for each of the descent's six other buffers.
        budget = BLOCK_SHARED_BYTES_OPT_IN[call.arch] - 6 * SHARED_BUFFER_ALIGN_BYTES
        held = -(-8 * leaves // SHARED_BUFFER_ALIGN_BYTES) * SHARED_BUFFER_ALIGN_BYTES
        if held > budget:
            return (
                f"folds a row into {leaves} shared float64 entries, which take {held} bytes "
                f"of the {budget} a block has left"
            )
        return None

    def __init__(self, call: SamplingCall, config: dict | None = None) -> None:
        super().__init__(device_index=call.device_index)
        self.call = call
        self.dtype = call.dtype
        vec, self._parts, _leaves = self._plan(
            call.vocab, call.dtype.itemsize, call.batch, call.sm_count
        )
        self.kernel = _sampling_from_probs_kernel(
            call.batch, call.vocab, vec, self._THREADS, self._parts, self._PACE
        )
        self.init_config(config)

    def forward(
        self, probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        self._require_devices(probs=probs, seed=seed, offset=offset)
        partial = torch.empty(
            self.call.batch * self._parts, dtype=torch.float64, device=probs.device
        )
        samples = torch.empty(self.call.batch, dtype=torch.int32, device=probs.device)
        self.program(self._compile)(probs.view(-1), seed, offset, partial, samples)
        return samples

    def _compile(self, cfg: dict):
        """The program, which takes no launch parameter from *cfg*."""
        return self.kernel()
