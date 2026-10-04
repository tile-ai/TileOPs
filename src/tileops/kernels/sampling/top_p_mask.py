"""Per-row top-p logit mask: a radix search over the row's values, run on chip."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import (
    LOG2E,
    MAX_BLOCK_THREADS,
    SHARED_BUFFER_ALIGN_BYTES,
    VECTOR_ACCESS_BYTES,
)
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling.call_spec import SamplingCall, TopPMaskFwdInterface
from tileops.kernels.sampling.row_tiles import (
    INF_BITS,
    MAGNITUDE_BITS,
    QUIET_NAN,
    block_extremes,
    fold_extremes,
    load_vector,
    row_split,
    store_masked,
    vector_width,
)
from tileops.utils import WARP_LANES

__all__ = ["TopPMaskFwdKernel"]

# Bins one search pass splits its bracket of keys into. A pass costs one shared-memory add
# an element whatever this is, so the figure to keep down is the passes, and 256 bins spend
# a bfloat16 key in two of them. It is also the widest split whose per-lane bins fit the
# shared budget, at 256 * 32 * 4 = 32 KB. Re-fit by timing a manifest row at 128 and 256.
_SEARCH_BINS: int = 256


@functools.lru_cache(maxsize=32)
def _top_p_mask_kernel(
    batch: int, vocab: int, dtype: str, vec: int, threads: int, parts: int, passes: int
):
    """Build the mask of ``batch`` rows of ``vocab`` logits, ``parts`` blocks to a row.

    A tile is one ``vec``-element vector per thread; ``reg_tiles`` of them stay in registers
    and ``smem_tiles`` in shared memory, and the rest are read again per pass. ``parts > 1``
    takes one grid barrier per pass to meet the blocks of a row in ``part_top`` and
    ``part_bins``; ``parts == 1`` takes none.
    """
    # Vectors of a row; a row divides into whole ones, since ``vec`` falls back to 1.
    full = vocab // vec
    row_tiles = -(-full // threads)
    chunk = -(-row_tiles // parts)
    grid = batch * parts
    warps = threads // WARP_LANES
    digit_bits = (_SEARCH_BINS - 1).bit_length()
    # One thread per bin folds and walks the bins.
    assert threads >= _SEARCH_BINS
    key_bits = 32
    # A float32's order-preserving unsigned key flips the sign bit of a non-negative value
    # and every bit of a negative one, so the keys sort the way the values do.
    sign_bit = 31

    @tilelang.jit(compile_flags=["-include", csrc_path("streaming_load.h")])
    def _top_p_mask_func(reg_tiles: int, smem_tiles: int, pace: int):
        kept = min(chunk, reg_tiles)
        parked = min(chunk - kept, smem_tiles)
        streamed = chunk - kept - parked
        parked_rounds = -(-parked // pace)
        streamed_rounds = -(-streamed // pace)

        @T.macro
        def tally(bins, src, slot, row_max, base, reach, step, lane):
            """Add each in-bracket value's probability weight to its bin of this pass."""
            for c in T.unroll(vec):
                value = T.cast(src[slot, c], "float32")
                raw = T.reinterpret(value, "uint32")
                key = raw ^ T.if_then_else(
                    raw >> sign_bit != T.uint32(0), ~T.uint32(0), T.uint32(1) << sign_bit
                )
                if key >= base[0] and key - base[0] <= reach[0]:
                    T.atomic_add(
                        bins[T.cast((key - base[0]) >> step[0], "int32"), lane],
                        T.exp2((value - row_max[0]) * T.float32(LOG2E)),
                    )

        @T.prim_func
        def _top_p_mask_main(
            logits: T.Tensor((batch * vocab,), dtype),
            p: T.Tensor((batch,), "float32"),
            part_top: T.Tensor((grid,), "float32"),
            part_bins: T.Tensor((grid * _SEARCH_BINS,), "float32"),
            masked: T.Tensor((batch * vocab,), dtype),
        ):
            with T.Kernel(grid, threads=threads) as bx:
                tx = T.get_thread_binding()
                in_regs = T.alloc_local((max(kept, 1), vec), dtype)
                in_smem = T.alloc_shared((max(parked, 1), threads * vec), dtype)
                cur = T.alloc_local((pace, vec), dtype)
                top = T.alloc_local((1,), "float32")
                seen = T.alloc_local((1,), "uint32")
                warp_top = T.alloc_shared((warps,), "float32")
                warp_seen = T.alloc_shared((warps,), "uint32")
                # A lane's copy of the bins is its own shared-memory bank, so no two lanes
                # of one atomic instruction meet whatever bins their elements fall in.
                bins = T.alloc_shared((_SEARCH_BINS, WARP_LANES), "float32")
                suffix = T.alloc_shared((_SEARCH_BINS + 1,), "float32")
                row_top = T.alloc_shared((1,), "float32")
                target = T.alloc_shared((1,), "float32")
                carried = T.alloc_shared((1,), "float32")
                base = T.alloc_local((1,), "uint32")
                reach = T.alloc_local((1,), "uint32")
                step = T.alloc_local((1,), "int32")
                bracket = T.alloc_shared((1,), "uint32")
                picked = T.alloc_shared((1,), "int32")
                row_max = T.alloc_local((1,), "float32")
                fold_bins = T.alloc_local((1,), "float32")
                cut = T.alloc_local((1,), "float32")

                line = bx // parts
                start = line * vocab
                head = bx % parts * chunk * threads + tx
                lane = tx % WARP_LANES
                top[0] = -T.infinity("float32")
                seen[0] = T.uint32(0)
                # A tile the block holds is read evict-first, since nothing reads it from
                # memory again; a tile it will read again stays cacheable for that read.
                for j in T.unroll(kept):
                    if head + j * threads < full:
                        load_vector(
                            in_regs, j, logits, start + (head + j * threads) * vec, vec, True
                        )
                for j in T.unroll(kept):
                    if head + j * threads < full:
                        fold_extremes(top, seen, in_regs, j, vec)
                for r in T.serial(parked_rounds):
                    for j in T.unroll(pace):
                        t = kept + r * pace + j
                        if t < kept + parked and head + t * threads < full:
                            load_vector(
                                cur, j, logits, start + (head + t * threads) * vec, vec, True
                            )
                    for j in T.unroll(pace):
                        t = kept + r * pace + j
                        if t < kept + parked and head + t * threads < full:
                            fold_extremes(top, seen, cur, j, vec)
                            for c in T.vectorized(vec):
                                in_smem[t - kept, tx * vec + c] = cur[j, c]
                for r in T.serial(streamed_rounds):
                    for j in T.unroll(pace):
                        t = kept + parked + r * pace + j
                        if t < chunk and head + t * threads < full:
                            load_vector(
                                cur, j, logits, start + (head + t * threads) * vec, vec, False
                            )
                    for j in T.unroll(pace):
                        t = kept + parked + r * pace + j
                        if t < chunk and head + t * threads < full:
                            fold_extremes(top, seen, cur, j, vec)
                block_extremes(top, seen, warp_top, warp_seen, warps)

                if parts > 1:
                    if tx == 0:
                        part_top[bx] = T.if_then_else(
                            seen[0] > T.uint32(INF_BITS),
                            T.reinterpret(T.uint32(QUIET_NAN), "float32"),
                            top[0],
                        )
                    T.sync_grid()
                    # One chunk maximum per thread, then one block reduction: every thread
                    # reading all of them instead serializes that many loads.
                    top[0] = -T.infinity("float32")
                    seen[0] = T.uint32(0)
                    for c in T.serial(-(-parts // threads)):
                        if c * threads + tx < parts:
                            value = part_top[line * parts + c * threads + tx]
                            top[0] = T.max(top[0], value)
                            seen[0] = T.max(
                                seen[0], T.reinterpret(value, "uint32") & T.uint32(MAGNITUDE_BITS)
                            )
                    block_extremes(top, seen, warp_top, warp_seen, warps)
                if tx == 0:
                    row_top[0] = top[0]
                    carried[0] = T.float32(0)
                    bracket[0] = T.uint32(0)
                T.sync_threads()
                row_max[0] = row_top[0]

                # The first pass covers every key, so its bins total Z.
                for level in T.serial(passes):
                    base[0] = bracket[0]
                    step[0] = key_bits - digit_bits * (level + 1)
                    reach[0] = ~T.uint32(0) >> (digit_bits * level)
                    for b in T.serial(-(-WARP_LANES * _SEARCH_BINS // threads)):
                        if b * threads + tx < WARP_LANES * _SEARCH_BINS:
                            bins[
                                (b * threads + tx) // WARP_LANES, (b * threads + tx) % WARP_LANES
                            ] = T.float32(0)
                    T.sync_threads()
                    for j in T.unroll(kept):
                        if head + j * threads < full:
                            tally(bins, in_regs, j, row_max, base, reach, step, lane)
                    for t in T.serial(kept, kept + parked):
                        if head + t * threads < full:
                            for c in T.vectorized(vec):
                                cur[0, c] = in_smem[t - kept, tx * vec + c]
                            tally(bins, cur, 0, row_max, base, reach, step, lane)
                    for r in T.serial(streamed_rounds):
                        for j in T.unroll(pace):
                            t = kept + parked + r * pace + j
                            if t < chunk and head + t * threads < full:
                                load_vector(
                                    cur, j, logits, start + (head + t * threads) * vec, vec, False
                                )
                        for j in T.unroll(pace):
                            t = kept + parked + r * pace + j
                            if t < chunk and head + t * threads < full:
                                tally(bins, cur, j, row_max, base, reach, step, lane)
                    T.sync_threads()
                    if tx < _SEARCH_BINS:
                        fold_bins[0] = bins[tx, 0]
                        for w in T.serial(1, WARP_LANES):
                            fold_bins[0] += bins[tx, w]
                        suffix[tx] = fold_bins[0]
                    if tx == 0:
                        suffix[_SEARCH_BINS] = T.float32(0)
                        picked[0] = 0
                    T.sync_threads()

                    if parts > 1:
                        if tx < _SEARCH_BINS:
                            part_bins[bx * _SEARCH_BINS + tx] = suffix[tx]
                        T.sync_grid()
                        if tx < _SEARCH_BINS:
                            fold_bins[0] = T.float32(0)
                            for c in T.serial(parts):
                                fold_bins[0] += part_bins[(line * parts + c) * _SEARCH_BINS + tx]
                            suffix[tx] = fold_bins[0]
                        T.sync_threads()
                        # Every block has read this pass's slice before the next pass
                        # overwrites the block's own.
                        T.sync_grid()

                    # Suffix sums, so one read gives the weight at or above any bin.
                    for stage in T.serial(digit_bits):
                        fold_bins[0] = T.float32(0)
                        if tx + (1 << stage) < _SEARCH_BINS:
                            fold_bins[0] = suffix[tx + (1 << stage)]
                        T.sync_threads()
                        if tx < _SEARCH_BINS:
                            suffix[tx] += fold_bins[0]
                        T.sync_threads()

                    if tx == 0 and level == 0:
                        target[0] = suffix[0] * p[line]
                    T.sync_threads()
                    # The highest bin the weight still reaches over: the top bin when every
                    # bin does, which is what p = 0 asks for, and bin 0 when none does, which a
                    # later pass's re-tally can round below the first pass's total. The scan
                    # sums overlapping ranges under different association trees, so two bins
                    # can qualify by a rounding and the highest wins rather than the race.
                    if tx < _SEARCH_BINS and carried[0] + suffix[tx] >= target[0]:
                        T.atomic_max(picked[0], tx)
                    T.sync_threads()
                    if tx == 0:
                        bracket[0] = base[0] + (T.cast(picked[0], "uint32") << step[0])
                        carried[0] = carried[0] + suffix[picked[0] + 1]
                    T.sync_threads()

                # A cut above every finite key decodes to a NaN, so p = 0 takes the infinity
                # that masks the row whole instead. A row whose float32 softmax is all NaN is
                # masked nowhere, which a NaN cut leaves every comparison against it false.
                cut[0] = T.if_then_else(
                    seen[0] > T.uint32(INF_BITS) or T.abs(row_max[0]) >= T.infinity("float32"),
                    T.reinterpret(T.uint32(QUIET_NAN), "float32"),
                    T.if_then_else(
                        bracket[0] > (T.uint32(1) << sign_bit) | T.uint32(INF_BITS),
                        T.infinity("float32"),
                        T.reinterpret(
                            T.if_then_else(
                                bracket[0] >> sign_bit != T.uint32(0),
                                bracket[0] ^ (T.uint32(1) << sign_bit),
                                ~bracket[0],
                            ),
                            "float32",
                        ),
                    ),
                )

                # The tiles read a second time go first, while L2 still holds them.
                for r in T.serial(streamed_rounds):
                    for j in T.unroll(pace):
                        t = kept + parked + r * pace + j
                        if t < chunk and head + t * threads < full:
                            load_vector(
                                cur, j, logits, start + (head + t * threads) * vec, vec, True
                            )
                    for j in T.unroll(pace):
                        t = kept + parked + r * pace + j
                        if t < chunk and head + t * threads < full:
                            store_masked(
                                masked,
                                cur,
                                j,
                                cut[0],
                                start + (head + t * threads) * vec,
                                vec,
                                dtype,
                            )
                for t in T.serial(kept, kept + parked):
                    if head + t * threads < full:
                        for c in T.vectorized(vec):
                            cur[0, c] = in_smem[t - kept, tx * vec + c]
                        store_masked(
                            masked, cur, 0, cut[0], start + (head + t * threads) * vec, vec, dtype
                        )
                for j in T.unroll(kept):
                    if head + j * threads < full:
                        store_masked(
                            masked,
                            in_regs,
                            j,
                            cut[0],
                            start + (head + j * threads) * vec,
                            vec,
                            dtype,
                        )

        return _top_p_mask_main

    return _top_p_mask_func


class TopPMaskFwdKernel(Kernel, TopPMaskFwdInterface):
    """Mask every logit the row's nucleus leaves out, reading the row about once.

    The cut is the lowest value whose strictly-above probability weight is below
    ``p[b] * Z``. A radix search over the values' order-preserving unsigned keys finds that
    key exactly, one digit a pass, and the chunk stays on chip across the passes. A batch
    that leaves blocks idle splits each row across several of them.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``reg_tiles``, ``smem_tiles`` and ``pace``.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general: ClassVar[bool] = True

    # Threads of a block, one block per SM so that a split row's barriers have the grid
    # resident. A batch whose rows hold too few tiles to fill the grid halves it, down to
    # _MIN_THREADS. The search reads the held chunk once a pass, so the passes are hidden by
    # the threads there are rather than by what the block holds: llama-8b-b256 bfloat16 runs
    # 198.5 us at 1024 threads against 263.8 at 512 and 381 at 256. Re-fit by timing the
    # manifest rows at 256, 512 and 1024.
    _THREADS: ClassVar[int] = MAX_BLOCK_THREADS
    _MIN_THREADS: ClassVar[int] = 512
    # Most tiles of one item a thread keeps in registers; shared memory holds the rest, and
    # what neither holds is read again per search pass. Registers cost occupancy that the
    # passes need more than they need the row on chip: llama-8b-b256 bfloat16 runs 198.5 us
    # at 1 tile against 212.8 at 2 and 245.5 at 4. Re-fit by timing that row at 0, 1, 2, 4.
    _REG_TILES: ClassVar[int] = 1
    # Vector loads a thread keeps in flight. Re-fit by timing 4, 8 and 16.
    _PACE: ClassVar[int] = 8
    # Bits a dtype's float32 key varies in. Widening bfloat16 to float32 leaves 16 zero low
    # bits and float16 leaves 13, so a search over those dtypes ends in fewer passes.
    _KEY_BITS: ClassVar[dict] = {torch.bfloat16: 16, torch.float16: 19, torch.float32: 32}

    @classmethod
    def refusal(cls, call: SamplingCall) -> Optional[str]:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.dtype not in cls._KEY_BITS:
            return f"searches float16, bfloat16 and float32 keys, not {call.dtype}"
        # Element offsets are int32 and run up to one tile past B * V. A tile holds
        # _THREADS vectors, and a vector is at most VECTOR_ACCESS_BYTES elements.
        largest_tile = cls._THREADS * VECTOR_ACCESS_BYTES
        if call.batch * call.vocab > 2**31 - 1 - largest_tile:
            return f"indexes elements with int32, and B * V = {call.batch * call.vocab}"
        return None

    def __init__(
        self, call: SamplingCall, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self._vec = vector_width(call.vocab, call.dtype.itemsize)
        vectors = call.vocab // self._vec
        threads = self._THREADS
        while threads > self._MIN_THREADS and call.batch * -(-vectors // threads) < call.sm_count:
            threads //= 2
        row_tiles = max(1, -(-vectors // threads))
        self._parts = row_split(row_tiles, call.batch, call.sm_count)

        self._passes = -(-self._KEY_BITS[call.dtype] // (_SEARCH_BINS - 1).bit_length())
        # The per-lane bins and their scan, plus at most one alignment each for the ten
        # buffers the search and the reductions take beside the row.
        reserve = (
            WARP_LANES * _SEARCH_BINS + _SEARCH_BINS + 1
        ) * 4 + 10 * SHARED_BUFFER_ALIGN_BYTES
        self._smem_tiles = (call.smem_budget - reserve) // (
            threads * self._vec * call.dtype.itemsize
        )
        chunk = -(-row_tiles // self._parts)
        self._reg_tiles = min(self._REG_TILES, max(1, chunk // 2))
        self.kernel = _top_p_mask_kernel(
            call.batch,
            call.vocab,
            self.dtype_str,
            self._vec,
            threads,
            self._parts,
            self._passes,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"reg_tiles": self._reg_tiles, "smem_tiles": self._smem_tiles, "pace": self._PACE}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        self._require_cuda(logits=logits, p=p)
        # A row of whole 16-byte vectors is read as vectors, from the start of the row.
        if self._vec > 1 and logits.data_ptr() % VECTOR_ACCESS_BYTES:
            logits = logits.clone()
        blocks = self.call.batch * self._parts
        part_top = torch.empty(blocks, dtype=torch.float32, device=logits.device)
        part_bins = torch.empty(blocks * _SEARCH_BINS, dtype=torch.float32, device=logits.device)
        masked = torch.empty_like(logits)
        self.kernel(self.config["reg_tiles"], self.config["smem_tiles"], self.config["pace"])(
            logits.view(-1), p, part_top, part_bins, masked.view(-1)
        )
        return masked
