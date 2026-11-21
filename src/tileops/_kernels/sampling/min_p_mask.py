"""Per-row min-p logit mask: a persistent grid reduces each row, then masks it on chip."""

from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import (
    BLOCK_SHARED_BYTES_OPT_IN,
    SHARED_BUFFER_ALIGN_BYTES,
    VECTOR_ACCESS_BYTES,
)
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.sampling.call_spec import MinPMaskFwdInterface, SamplingCall
from tileops._kernels.sampling.row_tiles import (
    INF_BITS,
    QUIET_NAN,
    mask_plan,
    mask_tiles,
    offset_refusal,
    vector_width,
)

__all__ = ["MinPMaskFwdKernel"]


@target_cache(maxsize=32)
def _min_p_mask_kernel(batch: int, vocab: int, dtype: str, threads: int, parts: int):
    """Build the mask of ``batch`` rows of ``vocab`` logits, ``parts`` blocks to a row.

    ``parts == 1`` leaves a row's maximum in the block that read it and takes no barrier.
    """
    itemsize = torch.empty((), dtype=getattr(torch, dtype)).element_size()
    vec = vector_width(vocab, itemsize)
    grid = batch * parts

    @tilelang.jit(compile_flags=csrc_include("streaming_load.h"))
    def _min_p_mask_func(reg_tiles: int, smem_tiles: int, pace: int):
        tiles = mask_tiles(vocab, dtype, vec, threads, parts, reg_tiles, smem_tiles, pace)

        @T.prim_func
        def _min_p_mask_main(
            logits: T.Tensor((batch * vocab,), dtype),
            min_p: T.Tensor((batch,), "float32"),
            partial: T.Tensor((grid,), "float32"),
            masked: T.Tensor((batch * vocab,), dtype),
        ):
            with T.Kernel(grid, threads=threads) as bx:
                tx = T.get_thread_binding()
                in_regs = T.alloc_local((max(tiles.kept, 1), vec), dtype)
                in_smem = T.alloc_shared((max(tiles.parked, 1), threads * vec), dtype)
                cur = T.alloc_local((pace, vec), dtype)
                top = T.alloc_local((1,), "float32")
                seen = T.alloc_local((1,), "uint32")
                bound = T.alloc_local((1,), "float32")
                warp_top = T.alloc_shared((tiles.warps,), "float32")
                warp_seen = T.alloc_shared((tiles.warps,), "uint32")

                line = bx // parts
                row = line * vocab
                head = bx % parts * tiles.chunk * threads + tx
                tiles.row_extremes(
                    logits,
                    row,
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
                    partial,
                )
                # The row's threshold, NaN where the row holds one, as torch's amax leaves it.
                bound[0] = T.if_then_else(
                    seen[0] > T.uint32(INF_BITS),
                    T.reinterpret(T.uint32(QUIET_NAN), "float32"),
                    top[0] + T.log(min_p[line]),
                )
                tiles.write_masked(masked, logits, row, head, in_regs, in_smem, cur, bound)

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
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general: ClassVar[bool] = True

    # Keep the grid resident for split-row barriers; shrink blocks when tiles underfill it.
    _THREADS: ClassVar[int] = 512
    _MIN_THREADS: ClassVar[int] = 256
    # Cap register-held tiles to preserve occupancy; shared memory holds the rest.
    _REG_TILES: ClassVar[int] = 8
    # Vector loads a thread keeps in flight.
    _PACE: ClassVar[int] = 8

    @classmethod
    def refusal(cls, call: SamplingCall) -> str | None:
        reason = super().refusal(call)
        if reason is not None:
            return reason
        return offset_refusal(call, cls._THREADS)

    def __init__(self, call: SamplingCall, config: dict | None = None) -> None:
        super().__init__(device_index=call.device_index)
        self.call = call
        self.dtype = call.dtype
        vectors = call.vocab // vector_width(call.vocab, call.dtype.itemsize)
        threads, self._parts, chunk = mask_plan(
            vectors, call.batch, call.sm_count, self._THREADS, self._MIN_THREADS
        )
        # The two reduction scratch buffers take one alignment of the budget each.
        smem_bytes = BLOCK_SHARED_BYTES_OPT_IN[call.arch] - 2 * SHARED_BUFFER_ALIGN_BYTES
        self._smem_tiles = smem_bytes // (threads * VECTOR_ACCESS_BYTES)
        self._reg_tiles = min(self._REG_TILES, max(1, chunk // 2))
        self.kernel = _min_p_mask_kernel(
            call.batch, call.vocab, self.dtype_str, threads, self._parts
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"reg_tiles": self._reg_tiles, "smem_tiles": self._smem_tiles, "pace": self._PACE}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        self._require_devices(logits=logits, min_p=min_p)
        partial = torch.empty(
            self.call.batch * self._parts, dtype=torch.float32, device=logits.device
        )
        masked = torch.empty_like(logits)
        self.program(self._compile)(logits.view(-1), min_p, partial, masked.view(-1))
        return masked

    def _compile(self, cfg: dict):
        """The program of config *cfg*."""
        return self.kernel(cfg["reg_tiles"], cfg["smem_tiles"], cfg["pace"])
