"""Per-row top-p logit mask: a radix search over the row's values, run on chip."""

from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import (
    LOG2E,
    MAX_BLOCK_THREADS,
    SHARED_BUFFER_ALIGN_BYTES,
)
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sampling.call_spec import SamplingCall, TopPMaskFwdInterface
from tileops._kernels.sampling.row_tiles import (
    INF_BITS,
    QUIET_NAN,
    load_vector,
    mask_plan,
    mask_tiles,
    offset_refusal,
    vector_width,
)
from tileops._kernels.warp_reduce import warp_reduce
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["TopPMaskFwdKernel"]


@target_cache(maxsize=32)
def _top_p_mask_kernel(
    batch: int,
    vocab: int,
    dtype: str,
    vec: int,
    threads: int,
    parts: int,
    passes: int,
    search_bins: int,
):
    """Build the mask of ``batch`` rows of ``vocab`` logits, ``parts`` blocks to a row.

    ``parts > 1`` meets the blocks of a row across one grid barrier for the maximum, in
    ``part_top``, and one a pass, adding the blocks' bins into the pass's own slice of
    ``part_bins``; every warp then walks the row's bins itself. ``parts == 1`` takes no grid
    barrier, and the block walks its bins together.
    """
    grid = batch * parts
    digit_bits = (search_bins - 1).bit_length()
    # One thread per bin folds the bins.
    assert threads >= search_bins
    lane_bins = search_bins // WARP_LANES
    # A split row's bins of every pass; a row one block holds keeps none.
    part_bin_count = batch * passes * search_bins if parts > 1 else 1
    key_bits = 32
    # A float32's order-preserving unsigned key flips the sign bit of a non-negative value
    # and every bit of a negative one, so the keys sort the way the values do.
    sign_bit = 31

    @tilelang.jit(compile_flags=csrc_include("streaming_load.h"))
    def _top_p_mask_func(reg_tiles: int, smem_tiles: int, pace: int):
        tiles = mask_tiles(vocab, dtype, vec, threads, parts, reg_tiles, smem_tiles, pace)
        full, chunk, kept, parked = tiles.full, tiles.chunk, tiles.kept, tiles.parked
        streamed_rounds = tiles.streamed_rounds

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

        @T.macro
        def clear_bins(bins):
            """Zero every lane's copy of the bins, for the whole block to add into."""
            tx = T.get_thread_binding()
            for b in T.serial(-(-WARP_LANES * search_bins // threads)):
                if b * threads + tx < WARP_LANES * search_bins:
                    bins[(b * threads + tx) // WARP_LANES, (b * threads + tx) % WARP_LANES] = (
                        T.float32(0)
                    )
            T.sync_threads()

        @T.prim_func
        def _top_p_mask_main(
            logits: T.Tensor((batch * vocab,), dtype),
            p: T.Tensor((batch,), "float32"),
            part_top: T.Tensor((grid,), "float32"),
            part_bins: T.Tensor((part_bin_count,), "float32"),
            masked: T.Tensor((batch * vocab,), dtype),
        ):
            with T.Kernel(grid, threads=threads) as bx:
                tx = T.get_thread_binding()
                in_regs = T.alloc_local((max(kept, 1), vec), dtype)
                in_smem = T.alloc_shared((max(parked, 1), threads * vec), dtype)
                cur = T.alloc_local((pace, vec), dtype)
                top = T.alloc_local((1,), "float32")
                seen = T.alloc_local((1,), "uint32")
                warp_top = T.alloc_shared((tiles.warps,), "float32")
                warp_seen = T.alloc_shared((tiles.warps,), "uint32")
                # A lane's copy of the bins is its own shared-memory bank, so no two lanes
                # of one atomic instruction meet whatever bins their elements fall in.
                bins = T.alloc_shared((search_bins, WARP_LANES), "float32")
                base = T.alloc_local((1,), "uint32")
                reach = T.alloc_local((1,), "uint32")
                step = T.alloc_local((1,), "int32")
                if parts > 1:
                    tallied = T.alloc_shared((search_bins,), "float32")
                    seg = T.alloc_local((lane_bins,), "float32")
                    bracket = T.alloc_local((1,), "uint32")
                    picked = T.alloc_local((1,), "int32")
                    target = T.alloc_local((1,), "float32")
                    carried = T.alloc_local((1,), "float32")
                    after = T.alloc_local((1,), "float32")
                else:
                    suffix = T.alloc_shared((search_bins + 1,), "float32")
                    row_top = T.alloc_shared((1,), "float32")
                    target = T.alloc_shared((1,), "float32")
                    carried = T.alloc_shared((1,), "float32")
                    bracket = T.alloc_shared((1,), "uint32")
                    picked = T.alloc_shared((1,), "int32")
                row_max = T.alloc_local((1,), "float32")
                fold_bins = T.alloc_local((1,), "float32")
                cut = T.alloc_local((1,), "float32")

                line = bx // parts
                start = line * vocab
                head = bx % parts * chunk * threads + tx
                lane = tx % WARP_LANES
                # The row's first block clears the row's bins of every pass, which the
                # maximum's grid barrier orders before any block adds to them.
                if parts > 1 and bx % parts == 0:
                    for i in T.serial(-(-passes * search_bins // threads)):
                        if i * threads + tx < passes * search_bins:
                            part_bins[line * passes * search_bins + i * threads + tx] = T.float32(0)
                tiles.row_extremes(
                    logits,
                    start,
                    head,
                    line,
                    bx,
                    in_regs,
                    in_smem,
                    cur,
                    top,
                    seen,
                    warp_top,
                    warp_seen,
                    part_top,
                )
                if parts > 1:
                    clear_bins(bins)
                    row_max[0] = top[0]
                    carried[0] = T.float32(0)
                    bracket[0] = T.uint32(0)
                else:
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
                    if parts == 1:
                        clear_bins(bins)
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
                    if parts > 1:
                        # A bin's thread folds its lane copies starting at its own bank, so the
                        # 32 threads of a warp read 32 banks, and clears them for the next pass.
                        if tx < search_bins:
                            fold_bins[0] = T.float32(0)
                            for w in T.unroll(WARP_LANES):
                                fold_bins[0] += bins[tx, (tx + w) % WARP_LANES]
                                bins[tx, (tx + w) % WARP_LANES] = T.float32(0)
                            if fold_bins[0] != T.float32(0):
                                T.atomic_add(
                                    part_bins[(line * passes + level) * search_bins + tx],
                                    fold_bins[0],
                                )
                        T.sync_grid()
                        if tx < search_bins:
                            tallied[tx] = part_bins[(line * passes + level) * search_bins + tx]
                        T.sync_threads()
                        # Every warp walks the bins itself, ``lane_bins`` a lane, and so reaches
                        # the same pick as every other warp of the row with no barrier after it.
                        for i in T.unroll(lane_bins):
                            seg[i] = tallied[lane * lane_bins + i]
                        # Suffix sums: the lane's own, then the lanes above it.
                        for i in T.unroll(lane_bins - 1):
                            seg[lane_bins - 2 - i] += seg[lane_bins - 1 - i]
                        after[0] = seg[0]
                        for stage in T.unroll(WARP_SHUFFLE_STAGES):
                            fold_bins[0] = T.shfl_down(after[0], 1 << stage)
                            if lane + (1 << stage) < WARP_LANES:
                                after[0] += fold_bins[0]
                        fold_bins[0] = T.shfl_down(after[0], 1)
                        if lane + 1 < WARP_LANES:
                            for i in T.unroll(lane_bins):
                                seg[i] += fold_bins[0]
                        if level == 0:
                            target[0] = T.shfl_sync(seg[0], 0) * p[line]
                        # The highest bin the weight still reaches over, as the block's walk
                        # below picks it.
                        picked[0] = 0
                        for i in T.unroll(lane_bins):
                            if carried[0] + seg[i] >= target[0]:
                                picked[0] = lane * lane_bins + i
                        warp_reduce(picked, T.max, high_first=True)
                        # The weight above the picked bin, from the lane that holds the bin
                        # after it; none is above the top bin.
                        fold_bins[0] = T.float32(0)
                        for i in T.unroll(lane_bins):
                            if lane * lane_bins + i == picked[0] + 1:
                                fold_bins[0] = seg[i]
                        fold_bins[0] = T.shfl_sync(
                            fold_bins[0], T.min((picked[0] + 1) // lane_bins, WARP_LANES - 1)
                        )
                        bracket[0] = base[0] + (T.cast(picked[0], "uint32") << step[0])
                        carried[0] = carried[0] + fold_bins[0]
                    else:
                        if tx < search_bins:
                            fold_bins[0] = bins[tx, 0]
                            for w in T.serial(1, WARP_LANES):
                                fold_bins[0] += bins[tx, w]
                            suffix[tx] = fold_bins[0]
                        if tx == 0:
                            suffix[search_bins] = T.float32(0)
                            picked[0] = 0
                        T.sync_threads()

                        # Suffix sums, so one read gives the weight at or above any bin.
                        for stage in T.serial(digit_bits):
                            fold_bins[0] = T.float32(0)
                            if tx + (1 << stage) < search_bins:
                                fold_bins[0] = suffix[tx + (1 << stage)]
                            T.sync_threads()
                            if tx < search_bins:
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
                        if tx < search_bins and carried[0] + suffix[tx] >= target[0]:
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

                tiles.write_masked(masked, logits, start, head, in_regs, in_smem, cur, cut)

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
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general: ClassVar[bool] = True

    # Keep the grid resident for split-row barriers; shrink blocks when tiles underfill it.
    _THREADS: ClassVar[int] = MAX_BLOCK_THREADS
    _MIN_THREADS: ClassVar[int] = 512
    # Cap register-held tiles to preserve occupancy; shared memory holds the rest.
    _REG_TILES: ClassVar[int] = 1
    # Vector loads a thread keeps in flight.
    _PACE: ClassVar[int] = 8
    # Bits a dtype's float32 key varies in. Widening bfloat16 to float32 leaves 16 zero low
    # bits and float16 leaves 13, so a search over those dtypes ends in fewer passes.
    _KEY_BITS: ClassVar[dict] = {torch.bfloat16: 16, torch.float16: 19, torch.float32: 32}
    # Bins one search pass splits its bracket of keys into. A pass costs one shared-memory add
    # an element whatever this is, so the figure to keep down is the passes, and 256 bins spend
    # a bfloat16 key in two of them. It is also the widest split whose per-lane bins fit the
    # shared budget, at 256 * 32 * 4 = 32 KB.
    _SEARCH_BINS: ClassVar[int] = 256

    @classmethod
    def refusal(cls, call: SamplingCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.dtype not in cls._KEY_BITS:
            return f"searches float16, bfloat16 and float32 keys, not {call.dtype}"
        return offset_refusal(call, cls._THREADS)

    def __init__(self, call: SamplingCall, config: dict | None = None) -> None:
        super().__init__(device_index=call.device_index)
        self.call = call
        self.dtype = call.dtype
        vec = vector_width(call.vocab, call.dtype.itemsize)
        threads, self._parts, chunk = mask_plan(
            call.vocab // vec, call.batch, call.sm_count, self._THREADS, self._MIN_THREADS
        )

        self._passes = -(-self._KEY_BITS[call.dtype] // (self._SEARCH_BINS - 1).bit_length())
        # The per-lane bins and their scan, plus at most one alignment each for the ten
        # buffers the search and the reductions take beside the row.
        reserve = (
            WARP_LANES * self._SEARCH_BINS + self._SEARCH_BINS + 1
        ) * 4 + 10 * SHARED_BUFFER_ALIGN_BYTES
        self._smem_tiles = (call.smem_budget - reserve) // (threads * vec * call.dtype.itemsize)
        self._reg_tiles = min(self._REG_TILES, max(1, chunk // 2))
        self.kernel = _top_p_mask_kernel(
            call.batch,
            call.vocab,
            self.dtype_str,
            vec,
            threads,
            self._parts,
            self._passes,
            self._SEARCH_BINS,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"reg_tiles": self._reg_tiles, "smem_tiles": self._smem_tiles, "pace": self._PACE}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        self._require_devices(logits=logits, p=p)
        blocks = self.call.batch * self._parts
        part_top = torch.empty(blocks, dtype=torch.float32, device=logits.device)
        part_bins = torch.empty(
            self.call.batch * self._passes * self._SEARCH_BINS if self._parts > 1 else 1,
            dtype=torch.float32,
            device=logits.device,
        )
        masked = torch.empty_like(logits)
        self.program(self._compile)(logits.view(-1), p, part_top, part_bins, masked.view(-1))
        return masked

    def _compile(self, cfg: dict):
        """The program of config *cfg*."""
        return self.kernel(cfg["reg_tiles"], cfg["smem_tiles"], cfg["pace"])
