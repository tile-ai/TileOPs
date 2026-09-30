"""Chain speculative sampling: verify a draft chain, then draw once from the residual row."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import BLOCK_SHARED_BYTES_OPT_IN, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import ChainSpeculativeSamplingFwdInterface, SamplingCall
from tileops.kernels.sampling.row_tiles import load_vector, vector_width
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["ChainSpeculativeSamplingFwdKernel"]


@functools.lru_cache(maxsize=32)
def _chain_speculative_sampling_kernel(
    batch: int, num_draft: int, vocab: int, vec: int, threads: int, parts: int, pace: int
):
    """Build the verification of ``batch`` chains of ``num_draft`` drafts over ``vocab`` tokens.

    The chain test reads ``2 * num_draft`` scalars, so the whole launch moves the one
    residual row ``max(0, target - draft)`` the accepted prefix stops at. A thread folds the
    ``vec`` weights of one 16-byte vector and a warp folds its 32 threads, leaving one entry
    of ``seg`` per (slot, warp), a slot being one vector per thread. Those entries are the
    leaves the draw descends: the row's blocks, then the block's ``seg`` entries, then the
    winning warp's lanes, then that lane's weights. Only the last of those re-reads the row,
    and it re-reads 32 vectors of it.

    The two folds a warp does are float32 over groups that ``vocab`` and ``threads`` fix
    between them; every sum above them is float64. A launch that splits the row differently
    therefore regroups float64 sums only.
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
    draw_rounds = -(-(num_draft + 1) // threads)
    part_span = -(-parts // WARP_LANES)
    # Philox4x32-10 (Salmon et al., SC'11): round multipliers, Weyl key increments, and the
    # top bits of one 32-bit output word that make a uniform, as ``workloads/sampling.py``
    # takes them.
    even_mul, odd_mul = 0xD2511F53, 0xCD9E8D57
    even_weyl, odd_weyl = 0x9E3779B9, 0xBB67AE85
    philox_rounds = 10
    uniform_bits = 24

    @tilelang.jit(compile_flags=["-include", csrc_path("streaming_load.h")])
    def _chain_speculative_sampling_func():
        @T.macro
        def stage(held, other, target_probs, draft_probs, slot, at_t, at_d, inside, drafted):
            """Read the ``vec`` target and draft weights at element offset ``at_t`` of the
            rows into ``held[slot]`` and ``other[slot]``, zero where the vector is past the row.

            ``drafted`` false reads no draft row, which is the bonus position, where the whole
            chain was accepted.
            """
            for c in T.unroll(vec):
                held[slot, c] = T.float32(0)
                other[slot, c] = T.float32(0)
            if inside:
                load_vector(held, slot, target_probs, at_t, vec, True)
                if drafted:
                    load_vector(other, slot, draft_probs, at_d, vec, True)

        @T.macro
        def fold(dst, held, other, slot):
            """Leave in ``dst[0]`` the float32 sum of the residual ``max(0, target - draft)``
            over the ``vec`` weights ``stage`` read into slot ``slot``."""
            for c in T.unroll(vec):
                held[slot, c] = T.max(held[slot, c] - other[slot, c], T.float32(0))
            for c in T.serial(1, vec):
                held[slot, 0] = held[slot, 0] + held[slot, c]
            dst[0] = held[slot, 0]

        @T.macro
        def locate(src, n, span, frac, base, lane, pick, edge, acc, idx):
            """Leave in ``pick[0]`` the entry of ``src[0:n]`` the draw lands on and in
            ``edge[0]`` how much of the draw point is left inside that entry.

            Lane ``l`` of warp 0 owns ``src[l * span : (l + 1) * span)`` and sums it into
            ``lane[l]``; lane 0 turns those sums into the exclusive prefixes ``edge[1:]``;
            every lane then walks its own entries against a draw point of
            ``frac * total + base``. The entry taken is the first whose inclusive prefix
            passes that point and whose own value is positive, and the last positive entry
            when none does. A zero-weight entry is therefore never taken, and a point that
            rounding leaves past the total still lands on weight the row carries.
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
                # The point never sits below zero. The prefix that placed this level was
                # summed in another order than the entries here, so subtracting it can leave
                # a negative remainder, and every entry would then pass the point, a
                # zero-weight one first.
                acc[1] = T.max(frac * edge[0] + base, T.cast(0, "float64"))
                idx[0] = n
                idx[1] = -1
                for i in T.serial(span):
                    if tx * span + i < n:
                        if src[tx * span + i] > T.cast(0, "float64"):
                            idx[1] = tx * span + i
                            if (idx[0] == n) & (acc[0] + src[tx * span + i] > acc[1]):
                                idx[0] = tx * span + i
                        acc[0] = acc[0] + src[tx * span + i]
                for stage in T.unroll(WARP_SHUFFLE_STAGES):
                    reach = T.int32(WARP_LANES // 2) >> stage
                    idx[0] = T.min(idx[0], T.shfl_xor(idx[0], reach, width=WARP_LANES))
                    idx[1] = T.max(idx[1], T.shfl_xor(idx[1], reach, width=WARP_LANES))
                if tx == 0:
                    pick[0] = T.if_then_else(idx[0] < n, idx[0], T.max(idx[1], 0))
            T.sync_threads()
            if tx == 0:
                acc[1] = T.max(frac * edge[0] + base, T.cast(0, "float64"))
                acc[0] = edge[pick[0] // span + 1]
                for i in T.serial(span):
                    if (pick[0] // span) * span + i < pick[0]:
                        acc[0] = acc[0] + src[(pick[0] // span) * span + i]
                edge[0] = acc[1] - acc[0]
            T.sync_threads()

        @T.prim_func
        def _chain_speculative_sampling_main(
            draft_probs: T.Tensor((batch * num_draft * vocab,), "float32"),
            draft_token_ids: T.Tensor((batch * num_draft,), "int32"),
            target_probs: T.Tensor((batch * (num_draft + 1) * vocab,), "float32"),
            seed: T.Tensor((1,), "int64"),
            offset: T.Tensor((1,), "int64"),
            partial: T.Tensor((grid,), "float64"),
            output_token_ids: T.Tensor((batch * (num_draft + 1),), "int32"),
            num_accepted: T.Tensor((batch,), "int32"),
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
                # Per draft, the chain position it rejects at, or ``num_draft`` when it is
                # accepted; the least of them is the accepted prefix length.
                verdict = T.alloc_shared((max(num_draft, 1),), "int32")
                # The row's own uniform, drawn after the chain's, broadcast to every thread.
                shared_draw = T.alloc_shared((1,), "float32")
                # The block of the row that owns the draw, and the element the winning warp
                # starts at, both read after ``pick`` has moved on to the next level.
                owner = T.alloc_shared((2,), "int32")
                held = T.alloc_local((pace, vec), "float32")
                other = T.alloc_local((pace, vec), "float32")
                warp = T.alloc_local((1,), "float32")
                acc = T.alloc_local((2,), "float64")
                idx = T.alloc_local((2,), "int32")
                counter = T.alloc_local((4,), "uint32")
                bump = T.alloc_local((2,), "uint32")
                key = T.alloc_local((2,), "uint32")
                spot = T.alloc_local((2,), "int32")
                # 0 what is left of the draw point inside the entry the level above chose,
                # 1 the row's uniform while the first level still has to scale it by a total.
                aim = T.alloc_local((2,), "float64")
                # The accepted prefix length, and the base element of the target and draft
                # rows the residual is formed from.
                stop = T.alloc_local((3,), "int32")

                line = bx // parts
                head = bx % parts * chunk

                # Draw the row's ``num_draft + 1`` uniforms, one per thread and a chain
                # longer than the block over several rounds: draw ``j`` tests draft ``j`` and
                # the last places the token. Philox4x32-10 keyed by the seed and counted by
                # the draw and the row alone, so a uniform is the same value however the
                # launch splits the row.
                for d in T.serial(draw_rounds):
                    spot[0] = d * threads + tx
                    if spot[0] < num_draft + 1:
                        key[0] = T.cast(seed[0] & T.int64(0xFFFFFFFF), "uint32")
                        key[1] = T.cast((seed[0] >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32")
                        counter[0] = T.cast(spot[0], "uint32")
                        counter[1] = T.cast(line, "uint32")
                        counter[2] = T.cast(offset[0] & T.int64(0xFFFFFFFF), "uint32")
                        counter[3] = T.cast(
                            (offset[0] >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32"
                        )
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
                        warp[0] = T.cast(
                            counter[0] >> T.uint32(32 - uniform_bits), "float32"
                        ) * T.float32(2.0**-uniform_bits)
                        if spot[0] == num_draft:
                            shared_draw[0] = warp[0]
                        else:
                            # Draft ``spot[0]`` is accepted while ``u * draft < target`` at its id.
                            idx[0] = draft_token_ids[line * num_draft + spot[0]]
                            verdict[spot[0]] = T.if_then_else(
                                warp[0] * draft_probs[(line * num_draft + spot[0]) * vocab + idx[0]]
                                < target_probs[(line * (num_draft + 1) + spot[0]) * vocab + idx[0]],
                                num_draft,
                                spot[0],
                            )
                T.sync_threads()

                stop[0] = num_draft
                for j in T.serial(num_draft):
                    stop[0] = T.min(stop[0], verdict[j])
                # The bonus position has no draft row, so its residual is the target row.
                stop[1] = (line * (num_draft + 1) + stop[0]) * vocab
                stop[2] = (line * num_draft + T.min(stop[0], num_draft - 1)) * vocab
                aim[1] = T.cast(shared_draw[0], "float64")

                # A round stages ``pace`` vectors of each row before folding any of them, so
                # that many loads are in flight while the folds and their warp reductions run.
                for r in T.serial(rounds):
                    for j in T.unroll(pace):
                        spot[0] = ((head + r * pace + j) * threads + tx) * vec
                        stage(
                            held,
                            other,
                            target_probs,
                            draft_probs,
                            j,
                            stop[1] + spot[0],
                            stop[2] + spot[0],
                            (r * pace + j < chunk) & (spot[0] < full * vec),
                            stop[0] < num_draft,
                        )
                    for j in T.unroll(pace):
                        fold(warp, held, other, j)
                        for step in T.unroll(WARP_SHUFFLE_STAGES):
                            reach = T.int32(WARP_LANES // 2) >> step
                            warp[0] = warp[0] + T.shfl_xor(warp[0], reach, width=WARP_LANES)
                        if (tx % WARP_LANES == 0) & (r * pace + j < chunk):
                            seg[(r * pace + j) * warps + tx // WARP_LANES] = T.cast(
                                warp[0], "float64"
                            )
                T.sync_threads()

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
                            (head + pick[0] // warps) * threads + pick[0] % warps * WARP_LANES
                        ) * vec
                    T.sync_threads()
                    if tx < WARP_LANES:
                        spot[0] = owner[1] + tx * vec
                        stage(
                            held,
                            other,
                            target_probs,
                            draft_probs,
                            0,
                            stop[1] + spot[0],
                            stop[2] + spot[0],
                            spot[0] < full * vec,
                            stop[0] < num_draft,
                        )
                        fold(warp, held, other, 0)
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
                            if spot[0] + c < full * vec:
                                acc[1] = T.cast(
                                    T.max(
                                        target_probs[stop[1] + spot[0] + c]
                                        - T.if_then_else(
                                            stop[0] < num_draft,
                                            draft_probs[stop[2] + spot[0] + c],
                                            T.float32(0),
                                        ),
                                        T.float32(0),
                                    ),
                                    "float64",
                                )
                                if acc[1] > T.cast(0, "float64"):
                                    idx[1] = c
                                    if (idx[0] < 0) & (acc[1] > acc[0]):
                                        idx[0] = c
                                acc[0] = acc[0] - acc[1]
                        spot[1] = spot[0] + T.if_then_else(idx[0] >= 0, idx[0], idx[1])
                        # The accepted drafts, the drawn token, then -1.
                        for j in T.serial(num_draft + 1):
                            output_token_ids[line * (num_draft + 1) + j] = T.if_then_else(
                                j < stop[0],
                                draft_token_ids[line * num_draft + T.min(j, num_draft - 1)],
                                T.if_then_else(j == stop[0], spot[1], -1),
                            )
                        num_accepted[line] = stop[0]

        return _chain_speculative_sampling_main

    return _chain_speculative_sampling_func


class ChainSpeculativeSamplingFwdKernel(Kernel, ChainSpeculativeSamplingFwdInterface):
    """Verify each request's draft chain and draw the token after it, over one read of one row.

    The chain test is ``2 N`` scalars a request, so the launch moves one row: the residual
    ``max(0, target - draft)`` at the position the accepted prefix stops at, the draft read
    as zero at the bonus position. One streaming pass folds that row into per-(slot, warp)
    sums and keeps them on chip, so the inverse-CDF search that follows descends those sums
    instead of the row and re-reads only the 32 vectors the winning warp holds. A batch that
    leaves blocks idle splits each row across several of them, whose totals meet across one
    grid barrier.

    The uniforms are a function of the seed, the offset, the draw index and the row index
    alone, and the acceptance test is the reference's float32 ``u * draft < target``, so
    ``num_accepted`` and the accepted prefix are the reference's exactly. The token after the
    prefix follows the same distribution but need not be the reference's index: the
    inverse-CDF prefix is accumulated over the order the launch folds the row in.

    Args:
        call: The call's shape, dtype and device facts.
        config: Unused; the launch follows from the call.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [90]
    general: ClassVar[bool] = True
    # ``draft_token_ids`` indexes a row, so a value outside ``[0, V)`` reads out of bounds.
    autotune_accepts_random_int_inputs: bool = False

    # Threads of a block, and the narrower block a crowded batch takes. A thread stages
    # ``_PACE`` vectors of each row in registers, and a 512-thread block of them leaves an SM
    # room for one resident block; a batch that already gives the device more blocks than it
    # has SMs wants a second block resident instead, and gets it from the narrower one:
    # llama2-7b-b256-n1 runs 19.1 us at 256 threads against 22.0 at 512 and 25.6 at 128,
    # where every other row is slower at 256. Re-fit by timing the manifest rows at 128, 256,
    # 512 and 1024.
    _THREADS: ClassVar[int] = 512
    _CROWDED_THREADS: ClassVar[int] = 256
    # Vectors of each row a thread stages before folding any of them. Re-fit by timing the
    # manifest rows at 1, 2, 4, 8 and 16.
    _PACE: ClassVar[int] = 4

    @classmethod
    def _plan(cls, vocab: int, batch: int, sm_count: int) -> tuple[int, int, int, int]:
        """A row's vector width, a block's threads, how many blocks share a row, and its leaves.

        A row whose bytes are a whole number of 16-byte vectors is folded a vector at a time;
        any other row is folded weight by weight. A row is split only while the batch leaves
        blocks idle, and never past its own vectors: the grid barrier a split takes needs the
        whole grid resident. The leaves are the ``seg`` entries a block's chunk folds into,
        one per (slot, warp).
        """
        vec = vector_width(vocab, torch.float32.itemsize)
        threads = cls._CROWDED_THREADS if batch >= sm_count else cls._THREADS
        row_tiles = max(1, -(-(vocab // vec) // threads))
        parts = max(1, min(row_tiles, sm_count // max(batch, 1)))
        return vec, threads, parts, -(-row_tiles // parts) * (threads // WARP_LANES)

    @classmethod
    def refusal(cls, call: SamplingCall) -> Optional[str]:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        elements = call.batch * (call.num_draft + 1) * call.vocab
        if elements > 2**31 - 1:
            return f"indexes elements with int32, and B * (N + 1) * V = {elements}"
        _vec, _threads, parts, leaves = cls._plan(call.vocab, call.batch, call.sm_count)
        # One float64 entry per (slot, warp) of a block's chunk, against the shared budget
        # less the scratch the descent takes.
        budget = (
            BLOCK_SHARED_BYTES_OPT_IN[call.arch]
            - 8 * (parts + 3 * WARP_LANES + 2)
            - 4 * call.num_draft
        )
        if 8 * leaves > budget:
            return f"folds a row into {leaves} shared float64 entries, past {budget // 8}"
        return None

    def __init__(
        self, call: SamplingCall, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        vec, threads, self._parts, _leaves = self._plan(call.vocab, call.batch, call.sm_count)
        self.kernel = _chain_speculative_sampling_kernel(
            call.batch, call.num_draft, call.vocab, vec, threads, self._parts, self._PACE
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {}

    def forward(
        self,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probs: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_cuda(
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            target_probs=target_probs,
            seed=seed,
            offset=offset,
        )
        # The fold reads 16-byte vectors from the start of each row.
        if target_probs.data_ptr() % VECTOR_ACCESS_BYTES:
            target_probs = target_probs.clone()
        if draft_probs.data_ptr() % VECTOR_ACCESS_BYTES:
            draft_probs = draft_probs.clone()
        device = target_probs.device
        partial = torch.empty(self.call.batch * self._parts, dtype=torch.float64, device=device)
        output_token_ids = torch.empty(
            self.call.batch, self.call.num_draft + 1, dtype=torch.int32, device=device
        )
        num_accepted = torch.empty(self.call.batch, dtype=torch.int32, device=device)
        self.kernel()(
            draft_probs.view(-1),
            draft_token_ids.view(-1),
            target_probs.view(-1),
            seed,
            offset,
            partial,
            output_token_ids.view(-1),
            num_accepted,
        )
        return output_token_ids, num_accepted
