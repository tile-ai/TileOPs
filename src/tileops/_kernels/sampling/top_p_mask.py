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
    VECTOR_ACCESS_BYTES,
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
    widths: tuple[int, ...],
):
    """Build the mask of ``batch`` rows of ``vocab`` logits, ``parts`` blocks to a row.

    ``parts > 1`` meets the blocks of a row across one grid barrier for the maximum, in
    ``part_top``, and one a pass, adding the blocks' bins into the pass's own slice of
    ``part_bins``; every warp then walks the row's bins itself. Where ``vocab`` leaves room,
    its bins add weights as fixed-point integers, so their totals do not depend on the order
    threads and blocks add in. ``parts == 1`` takes no grid barrier, and the block walks its
    bins together.
    """
    grid = batch * parts
    digit_bits = (search_bins - 1).bit_length()
    # Every bin has a thread to fold it.
    assert threads >= search_bins
    lane_bins = search_bins // WARP_LANES
    # Lane copies of a bin one 16-byte shared-memory access reaches, and such groups a bin has.
    quad = VECTOR_ACCESS_BYTES // 4
    quads = WARP_LANES // quad
    # Groups of one bin's lane copies a thread of a split row folds; the groups split evenly.
    assert quads * search_bins % threads == 0
    fold_quads = quads * search_bins // threads
    # Adjacent lanes that fold one bin, and the shuffle stages that sum them into one add.
    fold_lanes = quads // fold_quads
    fold_stages = fold_lanes.bit_length() - 1
    # A split row's bins of every pass; a row one block holds keeps none.
    part_bin_count = batch * passes * search_bins if parts > 1 else 1
    # A split row's fixed-point pieces, from ``TopPMaskFwdKernel``; none adds float32 weights.
    # Every piece but the first and the last is ``middle_width`` bits.
    fixed = bool(widths)
    pieces = len(widths)
    fraction_bits = sum(widths)
    first_width = widths[0] if fixed else 0
    middle_width = widths[min(1, pieces - 1)] if fixed else 0
    last_width = widths[-1] if fixed else 0
    key_bits = 32
    # A float32's order-preserving unsigned key flips the sign bit of a non-negative value
    # and every bit of a negative one, so the keys sort the way the values do.
    sign_bit = 31

    headers = ("streaming_load.h", "fixed_add.h") if fixed else ("streaming_load.h",)

    @tilelang.jit(compile_flags=csrc_include(*headers))
    def _top_p_mask_func(reg_tiles: int, smem_tiles: int, pace: int):
        tiles = mask_tiles(vocab, dtype, vec, threads, parts, reg_tiles, smem_tiles, pace)
        full, chunk, kept, parked = tiles.full, tiles.chunk, tiles.kept, tiles.parked
        streamed_rounds = tiles.streamed_rounds

        @T.macro
        def tally(bins, src, slot, row_max, base, reach, step, lane, scaled):
            """Add each in-bracket value's probability weight to its bin of this pass."""
            for c in T.unroll(vec):
                value = T.cast(src[slot, c], "float32")
                raw = T.reinterpret(value, "uint32")
                key = raw ^ T.if_then_else(
                    raw >> sign_bit != T.uint32(0), ~T.uint32(0), T.uint32(1) << sign_bit
                )
                if key >= base[0] and key - base[0] <= reach[0]:
                    if fixed:
                        # Each piece takes the next bits of the scaled weight: its whole part,
                        # then the exact remainder rescaled by the next piece's width.
                        scaled[0] = T.exp2((value - row_max[0]) * T.float32(LOG2E)) * T.float32(
                            2.0**first_width
                        )
                        for k in T.unroll(pieces):
                            if k > 0:
                                scaled[0] = (
                                    scaled[0] - T.cast(T.cast(scaled[0], "uint32"), "float32")
                                ) * T.if_then_else(
                                    k == pieces - 1,
                                    T.float32(2.0**last_width),
                                    T.float32(2.0**middle_width),
                                )
                            T.atomic_add(
                                bins[k, T.cast((key - base[0]) >> step[0], "int32"), lane],
                                T.cast(scaled[0], "uint32"),
                            )
                    else:
                        T.atomic_add(
                            bins[T.cast((key - base[0]) >> step[0], "int32"), lane],
                            T.exp2((value - row_max[0]) * T.float32(LOG2E)),
                        )

        @T.macro
        def clear_bins(bins, width: int):
            """Zero every lane's copy of the bins, ``width`` lanes a store.

            A block barrier must follow before any add.
            """
            tx = T.get_thread_binding()
            groups = WARP_LANES // width
            for b in T.serial(-(-groups * search_bins // threads)):
                if b * threads + tx < groups * search_bins:
                    if fixed:
                        for k in T.unroll(pieces):
                            for c in T.vectorized(width):
                                bins[
                                    k,
                                    (b * threads + tx) // groups,
                                    (b * threads + tx) % groups * width + c,
                                ] = T.uint32(0)
                    else:
                        for c in T.vectorized(width):
                            bins[
                                (b * threads + tx) // groups,
                                (b * threads + tx) % groups * width + c,
                            ] = T.float32(0)

        @T.prim_func
        def _top_p_mask_main(
            logits: T.Tensor((batch * vocab,), dtype),
            p: T.Tensor((batch,), "float32"),
            part_top: T.Tensor((grid,), "float32"),
            part_bins: T.Tensor((part_bin_count,), "uint64" if fixed else "float32"),
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
                if fixed:
                    bins = T.alloc_shared((pieces, search_bins, WARP_LANES), "uint32")
                else:
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
                    fold_fixed = T.alloc_local((1,), "uint64")
                    row_p = T.alloc_local((1,), "float32")
                    reached = T.alloc_local((1,), "uint64")
                    if fixed:
                        held = T.alloc_local((quad,), "uint32")
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
                            part_bins[line * passes * search_bins + i * threads + tx] = T.cast(
                                0, "uint64" if fixed else "float32"
                            )
                if parts > 1:
                    # Read before the row so the walk of the first pass does not wait on it, and
                    # clear the bins before the row rather than after the maximum's grid barrier:
                    # the barriers of the maximum order the clear before the first add.
                    row_p[0] = p[line]
                    clear_bins(bins, quad)
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
                        clear_bins(bins, 1)
                        T.sync_threads()
                    for j in T.unroll(kept):
                        if head + j * threads < full:
                            tally(bins, in_regs, j, row_max, base, reach, step, lane, fold_bins)
                    for t in T.serial(kept, kept + parked):
                        if head + t * threads < full:
                            for c in T.vectorized(vec):
                                cur[0, c] = in_smem[t - kept, tx * vec + c]
                            tally(bins, cur, 0, row_max, base, reach, step, lane, fold_bins)
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
                                tally(bins, cur, j, row_max, base, reach, step, lane, fold_bins)
                    T.sync_threads()
                    if parts > 1:
                        if fixed:
                            # A bin's ``fold_lanes`` adjacent lanes fold its lane copies a group
                            # at a time, from a start that differs per bin and per lane, so the 8
                            # threads of a 16-byte access phase read 8 groups of banks. Fixed-point
                            # totals are exact in any order; the lanes' sums meet in shuffles so a
                            # bin takes one global add.
                            fold_fixed[0] = T.uint64(0)
                            for w in T.unroll(fold_quads):
                                for k in T.unroll(pieces):
                                    for c in T.vectorized(quad):
                                        held[c] = bins[
                                            k,
                                            tx // fold_lanes,
                                            (tx // fold_lanes + tx % fold_lanes * fold_quads + w)
                                            % quads
                                            * quad
                                            + c,
                                        ]
                                    for c in T.unroll(quad):
                                        fold_fixed[0] += T.cast(
                                            held[c], "uint64"
                                        ) << T.if_then_else(
                                            k == pieces - 1,
                                            0,
                                            fraction_bits - first_width - k * middle_width,
                                        )
                            for stage in T.unroll(fold_stages):
                                fold_fixed[0] += T.shfl_xor(fold_fixed[0], 1 << stage)
                            if tx % fold_lanes == 0 and fold_fixed[0] != T.uint64(0):
                                T.call_extern(
                                    "handle",
                                    "tileops::atomic_add_u64",
                                    T.address_of(
                                        part_bins[
                                            (line * passes + level) * search_bins + tx // fold_lanes
                                        ]
                                    ),
                                    fold_fixed[0],
                                )
                        else:
                            # One thread a bin adds its float32 copies in one fixed order,
                            # starting at its own bank: float adds do not associate, so another
                            # order would change the totals.
                            if tx < search_bins:
                                fold_bins[0] = T.float32(0)
                                for w in T.unroll(WARP_LANES):
                                    fold_bins[0] += bins[tx, (tx + w) % WARP_LANES]
                                if fold_bins[0] != T.float32(0):
                                    T.atomic_add(
                                        part_bins[(line * passes + level) * search_bins + tx],
                                        fold_bins[0],
                                    )
                        T.sync_grid()
                        # The row's totals load while the block clears its copies for the next
                        # pass; the barrier after them orders the clear before the next add.
                        if tx < search_bins:
                            if fixed:
                                fold_fixed[0] = part_bins[
                                    (line * passes + level) * search_bins + tx
                                ]
                            else:
                                fold_bins[0] = part_bins[(line * passes + level) * search_bins + tx]
                        if level + 1 < passes:
                            clear_bins(bins, quad)
                        # An exact total rounds once to float32, so the walk below sums and
                        # compares in float32 as the block's walk does.
                        if tx < search_bins:
                            if fixed:
                                tallied[tx] = T.cast(fold_fixed[0], "float32") * T.float32(
                                    2.0**-fraction_bits
                                )
                            else:
                                tallied[tx] = fold_bins[0]
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
                            target[0] = T.shfl_sync(seg[0], 0) * row_p[0]
                        # The highest bin the weight still reaches over, as the block's walk
                        # below picks it.
                        picked[0] = -1
                        for i in T.unroll(lane_bins):
                            if carried[0] + seg[i] >= target[0]:
                                picked[0] = lane * lane_bins + i
                        # The highest lane holding such a bin holds the highest; none picks bin 0.
                        reached[0] = T.ballot(T.if_then_else(picked[0] >= 0, 1, 0))
                        picked[0] = T.max(
                            T.shfl_sync(
                                picked[0],
                                T.if_then_else(
                                    reached[0] == T.uint64(0),
                                    0,
                                    63 - T.cast(T.clz(reached[0]), "int32"),
                                ),
                            ),
                            0,
                        )
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
    # a bfloat16 key in two of them. It is also the widest split whose per-lane float32 bins
    # fit the shared budget, at 256 * 32 * 4 = 32 KB; a split row's fixed-point pieces take that
    # once each.
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
        # A split row adds each weight, at most 1, as ``floor(weight * 2^fraction_bits)``, so
        # its truncations total below ``vocab * 2^-fraction_bits``, under the float32 rounding
        # (2^-24 of the significand) of a total of at least 1, and its exact total, below
        # ``vocab * 2^fraction_bits``, fits 64 bits. The weight goes into 32-bit pieces of the
        # bins, highest bits first. A lane's copy of a piece takes one weight a tile from each
        # warp, ``lane_adds`` a pass: the first piece, at most ``2^widths[0]`` a weight, and
        # each later one, below ``2^width``, keep its total in 32 bits.
        fraction_bits = call.vocab.bit_length() + 24
        lane_adds = threads // WARP_LANES * chunk * vec
        widths = [min(((2**32 - 1) // lane_adds).bit_length() - 1, fraction_bits)]
        while sum(widths) < fraction_bits:
            later = ((2**32 - 1) // lane_adds + 1).bit_length() - 1
            widths.append(min(later, fraction_bits - sum(widths)))

        # The per-lane bins, one 4-byte copy a piece, and their scan, plus at most one alignment
        # each for the ten buffers the search and the reductions take beside the row. A row one
        # block holds, or a split row whose total or bins do not fit, adds float32 weights in
        # one copy.
        def reserve(copies: int) -> int:
            return (
                (WARP_LANES * copies + 1) * self._SEARCH_BINS * 4
                + 4
                + (10 * SHARED_BUFFER_ALIGN_BYTES)
            )

        if (
            self._parts == 1
            or call.vocab.bit_length() + fraction_bits > 64
            or reserve(len(widths)) > call.smem_budget
        ):
            widths = []
        self._fixed = bool(widths)
        self._smem_tiles = (call.smem_budget - reserve(max(len(widths), 1))) // (
            threads * vec * call.dtype.itemsize
        )
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
            tuple(widths),
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
            dtype=torch.uint64 if self._fixed else torch.float32,
            device=logits.device,
        )
        masked = torch.empty_like(logits)
        self.program(self._compile)(logits.view(-1), p, part_top, part_bins, masked.view(-1))
        return masked

    def _compile(self, cfg: dict):
        """The program of config *cfg*."""
        return self.kernel(cfg["reg_tiles"], cfg["smem_tiles"], cfg["pace"])
