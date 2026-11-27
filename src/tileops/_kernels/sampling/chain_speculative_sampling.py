"""Chain speculative sampling: verify a draft chain, then draw once from the residual row."""

import operator
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import BLOCK_SHARED_BYTES_OPT_IN
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sampling.call_spec import ChainSpeculativeSamplingFwdInterface, SamplingCall
from tileops._kernels.sampling.philox import draw, uniform_of
from tileops._kernels.sampling.row_tiles import (
    gather_part_totals,
    load_vector,
    pick_entry,
    row_split,
    vector_width,
)
from tileops._kernels.tiling import align_up
from tileops._kernels.warp_reduce import warp_reduce
from tileops._utils import WARP_LANES

__all__ = ["ChainSpeculativeSamplingFwdKernel"]


@target_cache(maxsize=32)
def _chain_speculative_sampling_kernel(
    batch: int, num_draft: int, vocab: int, vec: int, threads: int, parts: int, pace: int
):
    """Build the verification of ``batch`` chains of ``num_draft`` drafts over ``vocab`` tokens.

    The launch moves the one residual row ``max(0, target - draft)`` the accepted prefix stops
    at, folded and searched as in `_sampling_from_probs_kernel`: float32 warp folds over groups
    ``vocab`` and ``threads`` fix, float64 above them.
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

    @tilelang.jit(compile_flags=csrc_include("streaming_load.h"))
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
        def locate(src, n, span, frac, base, lane, pick, prefix, total, left, acc, idx):
            """Leave in ``pick[0]`` the entry of ``src[0:n]`` the draw lands on and in ``left[0]`` how
            much of the draw point is left inside it: the first entry with positive weight whose
            inclusive prefix passes ``frac * total + base``, else the last positive entry, so rounding
            past the total still lands on weight the row carries.
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
                    prefix[l] = acc[0]
                    acc[0] = acc[0] + lane[l]
                total[0] = acc[0]
            T.sync_threads()
            if tx < WARP_LANES:
                acc[0] = prefix[tx]
                # Clamped at zero: the level above summed in another order, so the remainder can
                # be negative, and only positive entries may pass.
                acc[1] = T.max(frac * total[0] + base, T.cast(0, "float64"))
                idx[0] = n
                idx[1] = -1
                for i in T.serial(span):
                    if tx * span + i < n:
                        if src[tx * span + i] > T.cast(0, "float64"):
                            idx[1] = tx * span + i
                            if (idx[0] == n) & (acc[0] + src[tx * span + i] > acc[1]):
                                idx[0] = tx * span + i
                        acc[0] = acc[0] + src[tx * span + i]
                pick_entry(idx, pick, n)
            T.sync_threads()
            if tx == 0:
                acc[1] = T.max(frac * total[0] + base, T.cast(0, "float64"))
                acc[0] = prefix[pick[0] // span]
                for i in T.serial(span):
                    if (pick[0] // span) * span + i < pick[0]:
                        acc[0] = acc[0] + src[(pick[0] // span) * span + i]
                left[0] = acc[1] - acc[0]
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
                # The exclusive prefix of each lane's entries, read by a computed lane, so
                # it stays one buffer.
                prefix = T.alloc_shared((WARP_LANES,), "float64")
                total = T.alloc_shared((1,), "float64")
                left = T.alloc_shared((1,), "float64")
                pick = T.alloc_shared((1,), "int32")
                # Per draft, the chain position it rejects at, or ``num_draft`` when accepted.
                verdict = T.alloc_shared((max(num_draft, 1),), "int32")
                token_uniform = T.alloc_shared((1,), "float32")
                owner_part = T.alloc_shared((1,), "int32")
                warp_at = T.alloc_shared((1,), "int32")
                held = T.alloc_local((pace, vec), "float32")
                other = T.alloc_local((pace, vec), "float32")
                warp = T.alloc_local((1,), "float32")
                acc = T.alloc_local((2,), "float64")
                idx = T.alloc_local((2,), "int32")
                counter = T.alloc_local((4,), "uint32")
                bump = T.alloc_local((2,), "uint32")
                key = T.alloc_local((2,), "uint32")
                draw_index = T.alloc_local((1,), "int32")
                element = T.alloc_local((1,), "int32")
                token = T.alloc_local((1,), "int32")
                remainder = T.alloc_local((1,), "float64")
                uniform = T.alloc_local((1,), "float64")
                accepted = T.alloc_local((1,), "int32")
                target_at = T.alloc_local((1,), "int32")
                draft_at = T.alloc_local((1,), "int32")

                line = bx // parts
                head = bx % parts * chunk

                # Counted by the draw and the row alone, so a uniform is the same value
                # however the launch splits the row.
                for d in T.serial(draw_rounds):
                    draw_index[0] = d * threads + tx
                    if draw_index[0] < num_draft + 1:
                        draw(counter, key, bump, seed, offset, draw_index[0], line)
                        warp[0] = uniform_of(counter[0])
                        if draw_index[0] == num_draft:
                            token_uniform[0] = warp[0]
                        else:
                            # A draft is accepted while ``u * draft < target`` at its id.
                            idx[0] = draft_token_ids[line * num_draft + draw_index[0]]
                            verdict[draw_index[0]] = T.if_then_else(
                                warp[0]
                                * draft_probs[(line * num_draft + draw_index[0]) * vocab + idx[0]]
                                < target_probs[
                                    (line * (num_draft + 1) + draw_index[0]) * vocab + idx[0]
                                ],
                                num_draft,
                                draw_index[0],
                            )
                T.sync_threads()

                accepted[0] = num_draft
                for j in T.serial(num_draft):
                    accepted[0] = T.min(accepted[0], verdict[j])
                # The bonus position has no draft row, so its residual is the target row.
                target_at[0] = (line * (num_draft + 1) + accepted[0]) * vocab
                draft_at[0] = (line * num_draft + T.min(accepted[0], num_draft - 1)) * vocab
                uniform[0] = T.cast(token_uniform[0], "float64")

                for r in T.serial(rounds):
                    for j in T.unroll(pace):
                        element[0] = ((head + r * pace + j) * threads + tx) * vec
                        stage(
                            held,
                            other,
                            target_probs,
                            draft_probs,
                            j,
                            target_at[0] + element[0],
                            draft_at[0] + element[0],
                            (r * pace + j < chunk) & (element[0] < full * vec),
                            accepted[0] < num_draft,
                        )
                    for j in T.unroll(pace):
                        fold(warp, held, other, j)
                        warp_reduce(warp, operator.add, high_first=True)
                        if (tx % WARP_LANES == 0) & (r * pace + j < chunk):
                            seg[(r * pace + j) * warps + tx // WARP_LANES] = T.cast(
                                warp[0], "float64"
                            )
                T.sync_threads()

                if parts == 1:
                    if tx == 0:
                        owner_part[0] = 0
                    remainder[0] = T.cast(0, "float64")
                else:
                    gather_part_totals(
                        seg, lane, partial, chunks, acc, bx, line, leaves, leaf_span, parts, threads
                    )
                    locate(
                        chunks,
                        parts,
                        part_span,
                        uniform[0],
                        T.cast(0, "float64"),
                        lane,
                        pick,
                        prefix,
                        total,
                        left,
                        acc,
                        idx,
                    )
                    if tx == 0:
                        owner_part[0] = pick[0]
                    remainder[0] = left[0]
                    uniform[0] = T.cast(0, "float64")
                T.sync_threads()

                if owner_part[0] == bx % parts:
                    locate(
                        seg,
                        leaves,
                        leaf_span,
                        uniform[0],
                        remainder[0],
                        lane,
                        pick,
                        prefix,
                        total,
                        left,
                        acc,
                        idx,
                    )
                    remainder[0] = left[0]
                    if tx == 0:
                        warp_at[0] = (
                            (head + pick[0] // warps) * threads + pick[0] % warps * WARP_LANES
                        ) * vec
                    T.sync_threads()
                    if tx < WARP_LANES:
                        element[0] = warp_at[0] + tx * vec
                        stage(
                            held,
                            other,
                            target_probs,
                            draft_probs,
                            0,
                            target_at[0] + element[0],
                            draft_at[0] + element[0],
                            element[0] < full * vec,
                            accepted[0] < num_draft,
                        )
                        fold(warp, held, other, 0)
                        tail[tx] = T.cast(warp[0], "float64")
                    T.sync_threads()
                    locate(
                        tail,
                        WARP_LANES,
                        1,
                        T.cast(0, "float64"),
                        remainder[0],
                        lane,
                        pick,
                        prefix,
                        total,
                        left,
                        acc,
                        idx,
                    )
                    if tx == 0:
                        # The chosen lane's weights, walked in index order: the token is the
                        # first whose weight passes what is left of the draw point.
                        element[0] = warp_at[0] + pick[0] * vec
                        acc[0] = left[0]
                        idx[0] = -1
                        idx[1] = 0
                        for c in T.serial(vec):
                            if element[0] + c < full * vec:
                                acc[1] = T.cast(
                                    T.max(
                                        target_probs[target_at[0] + element[0] + c]
                                        - T.if_then_else(
                                            accepted[0] < num_draft,
                                            draft_probs[draft_at[0] + element[0] + c],
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
                        token[0] = element[0] + T.if_then_else(idx[0] >= 0, idx[0], idx[1])
                        for j in T.serial(num_draft + 1):
                            output_token_ids[line * (num_draft + 1) + j] = T.if_then_else(
                                j < accepted[0],
                                draft_token_ids[line * num_draft + T.min(j, num_draft - 1)],
                                T.if_then_else(j == accepted[0], token[0], -1),
                            )
                        num_accepted[line] = accepted[0]

        return _chain_speculative_sampling_main

    return _chain_speculative_sampling_func


class ChainSpeculativeSamplingFwdKernel(Kernel, ChainSpeculativeSamplingFwdInterface):
    """Verify each request's draft chain and draw the token after it, over one read of one row: the
    residual ``max(0, target - draft)`` where the accepted prefix stops (draft zero at the bonus
    position), searched as `SamplingFromProbsFwdKernel` searches a row.

    The uniforms are a function of seed, offset, draw index and row index alone, and acceptance
    is the reference's float32 ``u * draft < target``, so ``num_accepted`` and the prefix are the
    reference's exactly; the token after it follows the same distribution but need not be the
    reference's index.

    Args:
        call: The call's shape, dtype and device facts.
        config: Unused; the launch follows from the call.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general: ClassVar[bool] = True
    # ``draft_token_ids`` indexes a row, so a value outside ``[0, V)`` reads out of bounds.
    autotune_accepts_random_int_inputs: bool = False

    # Threads of a block, and the narrower block a batch of at least one block an SM takes, so
    # two blocks stay resident on an SM.
    _THREADS: ClassVar[int] = 512
    _CROWDED_THREADS: ClassVar[int] = 256
    # Vectors of each row a thread stages before folding any of them.
    _PACE: ClassVar[int] = 4

    @classmethod
    def _plan(cls, vocab: int, batch: int, sm_count: int) -> tuple[int, int, int, int]:
        """A row's vector width, a block's threads, how many blocks share a row, and its leaves.

        The leaves are the ``seg`` entries a block's chunk folds into, one per (slot, warp).
        """
        vec = vector_width(vocab, torch.float32.itemsize)
        threads = cls._CROWDED_THREADS if batch >= sm_count else cls._THREADS
        row_tiles = max(1, -(-(vocab // vec) // threads))
        parts = row_split(row_tiles, batch, sm_count)
        return vec, threads, parts, -(-row_tiles // parts) * (threads // WARP_LANES)

    @classmethod
    def refusal(cls, call: SamplingCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        elements = call.batch * (call.num_draft + 1) * call.vocab
        if elements > 2**31 - 1:
            return f"indexes elements with int32, and B * (N + 1) * V = {elements}"
        _vec, _threads, parts, leaves = cls._plan(call.vocab, call.batch, call.sm_count)
        # ``seg``, one float64 entry per (slot, warp) of a block's chunk, against the budget
        # less the descent's scratch, each buffer on a 16-byte boundary.
        budget = BLOCK_SHARED_BYTES_OPT_IN[call.arch] - (
            align_up(8 * parts, 16)
            + 3 * 8 * WARP_LANES  # tail, lane, prefix
            + 2 * 16  # total, left
            + align_up(4 * max(call.num_draft, 1), 16)  # verdict
            + 4 * 16  # pick, token_uniform, owner_part, warp_at
        )
        if align_up(8 * leaves, 16) > budget:
            return f"folds a row into {leaves} shared float64 entries, past {budget // 8}"
        return None

    def __init__(self, call: SamplingCall, config: dict | None = None) -> None:
        super().__init__(device_index=call.device_index)
        self.call = call
        self.dtype = call.dtype
        vec, threads, self._parts, _leaves = self._plan(call.vocab, call.batch, call.sm_count)
        self.kernel = _chain_speculative_sampling_kernel(
            call.batch, call.num_draft, call.vocab, vec, threads, self._parts, self._PACE
        )
        self.init_config(config)

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
        self._require_devices(
            draft_probs=draft_probs,
            draft_token_ids=draft_token_ids,
            target_probs=target_probs,
            seed=seed,
            offset=offset,
        )
        device = target_probs.device
        partial = torch.empty(self.call.batch * self._parts, dtype=torch.float64, device=device)
        output_token_ids = torch.empty(
            self.call.batch, self.call.num_draft + 1, dtype=torch.int32, device=device
        )
        num_accepted = torch.empty(self.call.batch, dtype=torch.int32, device=device)
        self.program(self._compile)(
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

    def _compile(self, cfg: dict):
        """The program, which takes no launch parameter from *cfg*."""
        return self.kernel()
