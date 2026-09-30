"""Per-row min-p logit mask: a persistent grid reduces each row, then masks it on chip."""

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
from tileops.kernels.sampling.call_spec import MinPMaskFwdInterface, SamplingCall
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["MinPMaskFwdKernel"]


def _vector_width(vocab: int, itemsize: int) -> int:
    """Elements of a 16-byte vector, or 1 where a row's bytes are not a whole number of them.

    Every row starts on a vector only when the row's bytes are; a row that does not is read
    and written element by element.
    """
    return VECTOR_ACCESS_BYTES // itemsize if vocab * itemsize % VECTOR_ACCESS_BYTES == 0 else 1


@functools.lru_cache(maxsize=32)
def _min_p_mask_kernel(batch: int, vocab: int, dtype: str, threads: int, parts: int):
    """Build the mask of ``batch`` rows of ``vocab`` logits, ``parts`` blocks to a row.

    One block owns one contiguous chunk of one row. A tile of a chunk is one 16-byte vector
    per thread; of them the first ``reg_tiles`` stay in registers and the next
    ``smem_tiles`` in shared memory, and the rest are read a second time. ``parts > 1``
    splits a row across blocks, so those maxima meet in ``partial`` across one grid barrier;
    ``parts == 1`` leaves a row's maximum in the block that read it and takes no barrier.
    """
    itemsize = torch.empty((), dtype=getattr(torch, dtype)).element_size()
    vec = _vector_width(vocab, itemsize)
    # Vectors of a row; a row divides into whole ones, since ``vec`` falls back to 1.
    full = vocab // vec
    row_tiles = -(-full // threads)
    chunk = -(-row_tiles // parts)
    grid = batch * parts
    warps = threads // WARP_LANES
    # A float32's magnitude bits, and the largest of them that is not a NaN.
    magnitude_bits = 0x7FFFFFFF
    inf_bits = 0x7F800000
    quiet_nan = 0x7FC00000

    @tilelang.jit(compile_flags=["-include", csrc_path("streaming_load.h")])
    def _min_p_mask_func(reg_tiles: int, smem_tiles: int, pace: int):
        # Tiles of one item held in registers, then in shared memory, then read again.
        kept = min(chunk, reg_tiles)
        parked = min(chunk - kept, smem_tiles)
        streamed = chunk - kept - parked
        parked_rounds = -(-parked // pace)
        streamed_rounds = -(-streamed // pace)

        @T.macro
        def load(dst, slot, src, at, evict_first: bool):
            if vec == 1:
                dst[slot, 0] = src[at]
            elif evict_first:
                T.call_extern(
                    "handle",
                    "tl::tileops_load16_evict_first",
                    T.address_of(dst[slot, 0]),
                    T.address_of(src[at]),
                )
            else:
                T.call_extern(
                    "handle",
                    "tl::tileops_load16",
                    T.address_of(dst[slot, 0]),
                    T.address_of(src[at]),
                )

        @T.macro
        def fold(top, seen, src, slot):
            for c in T.unroll(vec):
                value = T.cast(src[slot, c], "float32")
                top[0] = T.max(top[0], value)
                seen[0] = T.max(seen[0], T.reinterpret(value, "uint32") & T.uint32(magnitude_bits))

        @T.macro
        def store(dst, src, slot, cut, at):
            for c in T.unroll(vec):
                value = T.cast(src[slot, c], "float32")
                src[slot, c] = T.if_then_else(
                    value < cut, T.cast(-T.infinity("float32"), dtype), src[slot, c]
                )
            if vec == 1:
                dst[at] = src[slot, 0]
            else:
                for c in T.vectorized(vec):
                    dst[at + c] = src[slot, c]

        @T.macro
        def block_max(top, seen, warp_top, warp_seen):
            """Leave the block's maxima in ``top[0]`` and ``seen[0]`` of every thread."""
            tx = T.get_thread_binding()
            for stage in T.serial(WARP_SHUFFLE_STAGES):
                reach = T.int32(WARP_LANES // 2) >> stage
                top[0] = T.max(top[0], T.shfl_xor(top[0], reach, width=WARP_LANES))
                seen[0] = T.max(seen[0], T.shfl_xor(seen[0], reach, width=WARP_LANES))
            if tx % WARP_LANES == 0:
                warp_top[tx // WARP_LANES] = top[0]
                warp_seen[tx // WARP_LANES] = seen[0]
            T.sync_threads()
            top[0] = warp_top[0]
            seen[0] = warp_seen[0]
            for w in T.serial(1, warps):
                top[0] = T.max(top[0], warp_top[w])
                seen[0] = T.max(seen[0], warp_seen[w])

        @T.prim_func
        def _min_p_mask_main(
            logits: T.Tensor((batch * vocab,), dtype),
            min_p: T.Tensor((batch,), "float32"),
            partial: T.Tensor((grid,), "float32"),
            masked: T.Tensor((batch * vocab,), dtype),
        ):
            with T.Kernel(grid, threads=threads) as bx:
                tx = T.get_thread_binding()
                in_regs = T.alloc_local((max(kept, 1), vec), dtype)
                in_smem = T.alloc_shared((max(parked, 1), threads * vec), dtype)
                cur = T.alloc_local((pace, vec), dtype)
                top = T.alloc_local((1,), "float32")
                seen = T.alloc_local((1,), "uint32")
                bound = T.alloc_local((1,), "float32")
                warp_top = T.alloc_shared((warps,), "float32")
                warp_seen = T.alloc_shared((warps,), "uint32")

                line = bx // parts
                row = line * vocab
                head = bx % parts * chunk * threads + tx
                top[0] = -T.infinity("float32")
                seen[0] = T.uint32(0)
                # A tile the block holds is read evict-first, since nothing reads it from
                # memory again; a tile it will read again stays cacheable for that read.
                for j in T.unroll(kept):
                    if head + j * threads < full:
                        load(in_regs, j, logits, row + (head + j * threads) * vec, True)
                for j in T.unroll(kept):
                    if head + j * threads < full:
                        fold(top, seen, in_regs, j)
                for r in T.serial(parked_rounds):
                    for j in T.unroll(pace):
                        t = kept + r * pace + j
                        if t < kept + parked and head + t * threads < full:
                            load(cur, j, logits, row + (head + t * threads) * vec, True)
                    for j in T.unroll(pace):
                        t = kept + r * pace + j
                        if t < kept + parked and head + t * threads < full:
                            fold(top, seen, cur, j)
                            for c in T.vectorized(vec):
                                in_smem[t - kept, tx * vec + c] = cur[j, c]
                for r in T.serial(streamed_rounds):
                    for j in T.unroll(pace):
                        t = kept + parked + r * pace + j
                        if t < chunk and head + t * threads < full:
                            load(cur, j, logits, row + (head + t * threads) * vec, False)
                    for j in T.unroll(pace):
                        t = kept + parked + r * pace + j
                        if t < chunk and head + t * threads < full:
                            fold(top, seen, cur, j)
                block_max(top, seen, warp_top, warp_seen)

                if parts > 1:
                    if tx == 0:
                        partial[bx] = T.if_then_else(
                            seen[0] > T.uint32(inf_bits),
                            T.reinterpret(T.uint32(quiet_nan), "float32"),
                            top[0],
                        )
                    T.sync_grid()
                    # One chunk maximum per thread, then one block reduction: every thread
                    # reading all of them instead serializes that many loads.
                    top[0] = -T.infinity("float32")
                    seen[0] = T.uint32(0)
                    for c in T.serial(-(-parts // threads)):
                        if c * threads + tx < parts:
                            value = partial[line * parts + c * threads + tx]
                            top[0] = T.max(top[0], value)
                            seen[0] = T.max(
                                seen[0], T.reinterpret(value, "uint32") & T.uint32(magnitude_bits)
                            )
                    block_max(top, seen, warp_top, warp_seen)
                # The row's threshold, NaN where the row holds one, as torch's amax leaves it.
                bound[0] = T.if_then_else(
                    seen[0] > T.uint32(inf_bits),
                    T.reinterpret(T.uint32(quiet_nan), "float32"),
                    top[0] + T.log(min_p[line]),
                )

                # The tiles read a second time go first, while L2 still holds them.
                for r in T.serial(streamed_rounds):
                    for j in T.unroll(pace):
                        t = kept + parked + r * pace + j
                        if t < chunk and head + t * threads < full:
                            load(cur, j, logits, row + (head + t * threads) * vec, True)
                    for j in T.unroll(pace):
                        t = kept + parked + r * pace + j
                        if t < chunk and head + t * threads < full:
                            store(masked, cur, j, bound[0], row + (head + t * threads) * vec)
                for t in T.serial(kept, kept + parked):
                    if head + t * threads < full:
                        for c in T.vectorized(vec):
                            cur[0, c] = in_smem[t - kept, tx * vec + c]
                        store(masked, cur, 0, bound[0], row + (head + t * threads) * vec)
                for j in T.unroll(kept):
                    if head + j * threads < full:
                        store(masked, in_regs, j, bound[0], row + (head + j * threads) * vec)

        return _min_p_mask_main

    return _min_p_mask_func


class MinPMaskFwdKernel(Kernel, MinPMaskFwdInterface):
    """Mask every logit of a row below its max plus ``log(min_p)``, reading it about once.

    A row's maximum has to settle before any of its logits is masked, so the launch holds
    as much of the row on chip as registers and shared memory take and reads only the rest
    a second time. A batch that leaves blocks idle splits each row across several of them,
    whose maxima meet across one grid barrier. The mask is the reference's exactly: the
    threshold is one float32 add over torch's own row max and ``log(min_p)``, a kept logit
    is passed through bit for bit, and a row holding a NaN passes through whole.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``reg_tiles``, ``smem_tiles`` and ``pace``.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [90]
    general: ClassVar[bool] = True

    # Threads of a block, one block per SM so that a split row's barrier has the grid
    # resident. A batch whose rows hold too few tiles to fill the grid halves it, down to
    # _MIN_THREADS: llama-8b-b1 runs 3.26 us at 256 threads and 3.36 at 512, where only 32
    # blocks have work. Re-fit by timing the manifest rows at 128, 256, 512 and 1024.
    _THREADS: ClassVar[int] = 512
    _MIN_THREADS: ClassVar[int] = 256
    # Most tiles of one item a thread keeps in registers; shared memory holds the rest.
    # Registers never take more than half an item, which is what the llama2-7b-b100 row
    # measures: 4 register tiles and 4 shared ones run 6.02 us where 8 register tiles and no
    # shared one run 6.50. Re-fit by timing that row and llama-8b-b256 at 2, 4, 8 and 16.
    _REG_TILES: ClassVar[int] = 8
    # Vector loads a thread keeps in flight. Re-fit by timing 4, 8 and 16.
    _PACE: ClassVar[int] = 8

    @classmethod
    def refusal(cls, call: SamplingCall) -> Optional[str]:
        reason = super().refusal(call)
        # Element offsets are int32 and run up to one tile past B * V. A tile holds
        # _THREADS vectors, and a vector is at most VECTOR_ACCESS_BYTES elements.
        largest_tile = cls._THREADS * VECTOR_ACCESS_BYTES
        if reason is None and call.batch * call.vocab > 2**31 - 1 - largest_tile:
            return f"indexes elements with int32, and B * V = {call.batch * call.vocab}"
        return reason

    def __init__(
        self, call: SamplingCall, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        vectors = call.vocab // _vector_width(call.vocab, call.dtype.itemsize)
        threads = self._THREADS
        while threads > self._MIN_THREADS and call.batch * -(-vectors // threads) < call.sm_count:
            threads //= 2
        row_tiles = max(1, -(-vectors // threads))
        # Split a row only while the batch leaves blocks idle, and never past its tiles.
        # The grid barrier a split takes needs the grid resident, which this keeps it.
        self._parts = max(1, min(row_tiles, call.sm_count // max(call.batch, 1)))
        # The two reduction scratch buffers take one alignment of the budget each.
        smem_bytes = BLOCK_SHARED_BYTES_OPT_IN[call.arch] - 2 * SHARED_BUFFER_ALIGN_BYTES
        self._smem_tiles = smem_bytes // (threads * VECTOR_ACCESS_BYTES)
        chunk = -(-row_tiles // self._parts)
        self._reg_tiles = min(self._REG_TILES, max(1, chunk // 2))
        self.kernel = _min_p_mask_kernel(
            call.batch, call.vocab, self.dtype_str, threads, self._parts
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"reg_tiles": self._reg_tiles, "smem_tiles": self._smem_tiles, "pace": self._PACE}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        self._require_cuda(logits=logits, min_p=min_p)
        # The kernel reads 16-byte vectors from the start of each row.
        if logits.data_ptr() % VECTOR_ACCESS_BYTES:
            logits = logits.clone()
        partial = torch.empty(
            self.call.batch * self._parts, dtype=torch.float32, device=logits.device
        )
        masked = torch.empty_like(logits)
        self.kernel(self.config["reg_tiles"], self.config["smem_tiles"], self.config["pace"])(
            logits.view(-1), min_p, partial, masked.view(-1)
        )
        return masked
