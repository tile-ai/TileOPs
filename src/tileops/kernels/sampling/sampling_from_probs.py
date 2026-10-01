"""Seeded categorical draw: one inverse-CDF search per row, over one read of the row."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import (
    BLOCK_SHARED_BYTES_OPT_IN,
    SHARED_BUFFER_ALIGN_BYTES,
    VECTOR_ACCESS_BYTES,
)
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import SamplingCall, SamplingFromProbsFwdInterface
from tileops.kernels.sampling.row_tiles import load_vector, row_split, vector_width
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["SamplingFromProbsFwdKernel"]


@functools.lru_cache(maxsize=32)
def _sampling_from_probs_kernel(
    batch: int, vocab: int, vec: int, threads: int, parts: int, pace: int
):
    """Build the draw over ``batch`` rows of ``vocab`` weights, ``parts`` blocks to a row.

    A thread folds the ``vec`` weights of one 16-byte vector and a warp folds its 32 threads,
    leaving one entry of ``seg`` per (slot, warp), a slot being one vector per thread. Those
    entries are the leaves the search descends: the row's blocks, then the block's ``seg``
    entries, then the winning warp's lanes, then that lane's weights. Only the last of those
    re-reads the row, and it re-reads 32 vectors of it. ``pace`` vectors are in flight before
    any of them is folded, so a thread's loads overlap rather than wait on the fold between.

    The two folds a warp does are float32 over groups that ``vocab`` and ``threads`` fix
    between them; every sum above them is float64. A launch that splits the row differently
    therefore regroups float64 sums only, and the token a seed draws moves only where the
    draw point sits within a float64 rounding of a token boundary.
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
    # Philox4x32-10 (Salmon et al., SC'11): round multipliers, Weyl key increments, and the
    # top bits of one 32-bit output word that make a uniform, as ``workloads/sampling.py``
    # takes them.
    even_mul, odd_mul = 0xD2511F53, 0xCD9E8D57
    even_weyl, odd_weyl = 0x9E3779B9, 0xBB67AE85
    philox_rounds = 10
    uniform_bits = 24

    @tilelang.jit(compile_flags=["-include", csrc_path("streaming_load.h")])
    def _sampling_from_probs_func():
        @T.macro
        def take(dst, slot, src, spot, inside):
            """Read the ``vec`` weights of ``src`` at element offset ``spot`` into
            ``dst[slot]``, and zero where that vector is past the row.

            Nothing reads a vector from memory twice but the 32 the last descent walks, so
            every read is evict-first.
            """
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
            """Leave in ``pick[0]`` the entry of ``src[0:n]`` the draw lands on and in
            ``edge[0]`` how much of the draw point is left inside that entry.

            Lane ``l`` of warp 0 owns ``src[l * span : (l + 1) * span)`` and sums it into
            ``lane[l]``; lane 0 turns those sums into the exclusive prefixes ``edge[1:]``;
            every lane then walks its own entries against a draw point of
            ``frac * total + base``. The entry taken is the first whose inclusive prefix
            passes that point, and the last entry carrying weight when none does.

            A zero-weight entry is never the first to pass: it leaves the inclusive prefix
            where the entry before it left it, so whenever it would pass, that one already
            did. The first-exceeding test therefore needs no weight test of its own; the
            fallback does, or it would land on a trailing zero.

            The point handed to the level below is never negative, and needs no clamp. The
            entry taken either passes the point, and then its own exclusive prefix does not;
            or it is the fallback, reached only when no entry passes the point, and then
            every exclusive prefix is below it.

            What the fallback is for is the other end. A level's entries are re-derived
            where the level above folded them: the warp sums ``seg`` holds are float32
            butterflies, while the 32 lanes below one of them are summed in float64. The
            point can therefore land past a level's total by a float32 rounding of that
            entry, and the last entry carrying weight is where it belongs.
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
                for stage in T.unroll(WARP_SHUFFLE_STAGES):
                    reach = T.int32(WARP_LANES // 2) >> stage
                    idx[0] = T.min(idx[0], T.shfl_xor(idx[0], reach, width=WARP_LANES))
                    idx[1] = T.max(idx[1], T.shfl_xor(idx[1], reach, width=WARP_LANES))
                if tx == 0:
                    pick[0] = T.if_then_else(idx[0] < n, idx[0], T.max(idx[1], 0))
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
                            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                                reach = T.int32(WARP_LANES // 2) >> stage
                                warp[0] = warp[0] + T.shfl_xor(warp[0], reach, width=WARP_LANES)
                            if tx % WARP_LANES == 0:
                                seg[(r * pace + j) * warps + tx // WARP_LANES] = T.cast(
                                    warp[0], "float64"
                                )
                T.sync_threads()

                # The row's uniform: Philox4x32-10 keyed by the seed and counted by the row
                # alone, so it is the same value however the launch splits the row.
                key[0] = T.cast(seed[0] & T.int64(0xFFFFFFFF), "uint32")
                key[1] = T.cast((seed[0] >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32")
                counter[0] = T.uint32(0)
                counter[1] = T.cast(line, "uint32")
                counter[2] = T.cast(offset[0] & T.int64(0xFFFFFFFF), "uint32")
                counter[3] = T.cast((offset[0] >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32")
                for step in T.serial(philox_rounds):
                    if step > 0:
                        key[0] = key[0] + T.uint32(even_weyl)
                        key[1] = key[1] + T.uint32(odd_weyl)
                    bump[0] = (
                        T.call_extern("uint32", "__umulhi", counter[2], T.uint32(odd_mul))
                        ^ counter[1]
                        ^ key[0]
                    )
                    bump[1] = (
                        T.call_extern("uint32", "__umulhi", counter[0], T.uint32(even_mul))
                        ^ counter[3]
                        ^ key[1]
                    )
                    counter[1] = counter[2] * T.uint32(odd_mul)
                    counter[3] = counter[0] * T.uint32(even_mul)
                    counter[0] = bump[0]
                    counter[2] = bump[1]
                aim[1] = T.cast(
                    T.cast(counter[0] >> T.uint32(32 - uniform_bits), "float32")
                    * T.float32(2.0**-uniform_bits),
                    "float64",
                )

                if parts == 1:
                    if tx == 0:
                        owner[0] = 0
                    aim[0] = T.cast(0, "float64")
                else:
                    # The chunk total, summed exactly as ``locate`` sums the same entries,
                    # so the prefix a block takes from the barrier and the prefix it then
                    # descends agree entry for entry.
                    if tx < WARP_LANES:
                        acc[0] = T.cast(0, "float64")
                        for i in T.serial(leaf_span):
                            if tx * leaf_span + i < leaves:
                                acc[0] = acc[0] + seg[tx * leaf_span + i]
                        lane[tx] = acc[0]
                    T.sync_threads()
                    if tx == 0:
                        acc[0] = T.cast(0, "float64")
                        for l in T.serial(WARP_LANES):
                            acc[0] = acc[0] + lane[l]
                        partial[bx] = acc[0]
                    T.sync_grid()
                    for i in T.serial(-(-parts // threads)):
                        if i * threads + tx < parts:
                            chunks[i * threads + tx] = partial[line * parts + i * threads + tx]
                    T.sync_threads()
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
    """Draw one token per row of probabilities, reading the row once and 32 vectors twice.

    The token is the first index whose inclusive prefix sum passes ``u * total``, with ``u``
    the row's Philox uniform. One streaming pass folds the row into per-(slot, warp) sums and
    keeps them on chip, so the search that follows descends those sums instead of the row and
    re-reads only the 32 vectors the winning warp holds. A batch that leaves blocks idle
    splits each row across several of them, whose totals meet across one grid barrier.

    ``u`` is a function of the seed, the offset and the row index alone, so it does not move
    when the launch does; the sums the search descends are float32 over groups that the
    vocabulary fixes and float64 above them, so the token does not move either, except where
    the draw point sits within a float64 rounding of a token boundary.

    Args:
        call: The call's shape, dtype and device facts.
        config: Unused; the launch follows from the call.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [90]
    general: ClassVar[bool] = True

    # Threads of a block. Fixed rather than fitted: it is one of the two group sizes the
    # float32 folds run over, so changing it with the shape would change which token a seed
    # draws. Re-fit it only together with that guarantee, by timing the manifest rows at 256,
    # 512 and 1024 and taking the one value that serves every row.
    _THREADS: ClassVar[int] = 512
    # Vector loads a thread keeps in flight before it folds any of them. Re-fit by timing
    # the manifest rows at 4, 8 and 16.
    _PACE: ClassVar[int] = 8

    @classmethod
    def _plan(cls, vocab: int, itemsize: int, batch: int, sm_count: int) -> tuple[int, int, int]:
        """The vector width of a row, how many blocks share one, and a block's ``seg`` entries.

        The block size is fixed, so the tile count a row's vectors make is fixed with it;
        the two logit filters that take the same split reach it through a thread count that
        the batch lowers, which this cannot do without moving the token a seed draws.
        """
        vec = vector_width(vocab, itemsize)
        row_tiles = max(1, -(-(vocab // vec) // cls._THREADS))
        parts = row_split(row_tiles, batch, sm_count)
        return vec, parts, -(-row_tiles // parts) * (cls._THREADS // WARP_LANES)

    @classmethod
    def refusal(cls, call: SamplingCall) -> Optional[str]:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.batch * call.vocab > 2**31 - 1:
            return f"indexes elements with int32, and B * V = {call.batch * call.vocab}"
        _vec, parts, leaves = cls._plan(call.vocab, call.dtype.itemsize, call.batch, call.sm_count)
        # ``seg`` holds one float64 entry per (slot, warp) of a block's chunk. Every shared
        # buffer costs its size rounded up to an alignment, so the six the descent takes
        # cost one alignment each whatever ``parts`` is, and ``seg`` is read against what
        # the budget has left.
        budget = BLOCK_SHARED_BYTES_OPT_IN[call.arch] - 6 * SHARED_BUFFER_ALIGN_BYTES
        held = -(-8 * leaves // SHARED_BUFFER_ALIGN_BYTES) * SHARED_BUFFER_ALIGN_BYTES
        if held > budget:
            return (
                f"folds a row into {leaves} shared float64 entries, which take {held} bytes "
                f"of the {budget} a block has left"
            )
        return None

    def __init__(
        self, call: SamplingCall, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self._vec, self._parts, _leaves = self._plan(
            call.vocab, call.dtype.itemsize, call.batch, call.sm_count
        )
        self.kernel = _sampling_from_probs_kernel(
            call.batch, call.vocab, self._vec, self._THREADS, self._parts, self._PACE
        )
        self.init_config(config, tune)

    def forward(
        self, probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        self._require_cuda(probs=probs, seed=seed, offset=offset)
        # A row folded a vector at a time is read from the start of the storage; a row
        # folded weight by weight needs no alignment.
        if self._vec > 1 and probs.data_ptr() % VECTOR_ACCESS_BYTES:
            probs = probs.clone()
        partial = torch.empty(
            self.call.batch * self._parts, dtype=torch.float64, device=probs.device
        )
        samples = torch.empty(self.call.batch, dtype=torch.int32, device=probs.device)
        self.kernel()(probs.view(-1), seed, offset, partial, samples)
        return samples
