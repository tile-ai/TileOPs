"""Masked scatter: the true positions of a mask take consecutive source elements, in order."""

import dataclasses
import math
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.constants import MAX_GRID_YZ_BLOCKS, VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel, KernelInterface
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["MaskedScatterCall", "MaskedScatterFwdInterface", "MaskedScatterKernel", "mask_runs"]


def mask_runs(shape: tuple, mask_shape: tuple) -> tuple[int, int]:
    """Split the broadcast *shape* at ``lead``: each run of ``run_len`` elements, the trailing
    axes over which *mask* has extent 1, shares one mask value.

    Returns:
        ``(lead, run_len)``: the mask holds one value per position of ``shape[:lead]``.
    """
    aligned = (1,) * (len(shape) - len(mask_shape)) + tuple(mask_shape)
    lead = max((i + 1 for i, extent in enumerate(aligned) if extent != 1), default=0)
    return lead, math.prod(shape[lead:])


@dataclasses.dataclass(frozen=True)
class MaskedScatterCall(CallSpec):
    """The run length and dtype of one call; the run count and source size are dynamic."""

    run_len: int = 0
    dtype: torch.dtype | None = None


class MaskedScatterFwdInterface(KernelInterface):
    """Write source elements into the true positions of a broadcast mask."""

    request = MaskedScatterCall

    @abstractmethod
    def forward(
        self, input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Return ``input`` broadcast with ``mask``, its true positions filled from ``source``.

        Args:
            input: Any shape, in ``call.dtype``, contiguous.
            mask: bool, broadcastable with ``input``, contiguous; the output's trailing runs
                of ``call.run_len`` elements each share one mask value.
            source: Any shape in ``call.dtype``, contiguous, holding at least as many
                elements as the broadcast mask has true positions.

        Returns:
            A new tensor of the broadcast shape in ``call.dtype``; the k-th true position in
            row-major order holds ``source.flatten()[k]``.
        """


@target_cache(maxsize=32)
def _masked_scatter_kernel(run_len, dtype):
    # Bounds are checked only where written here: a mask byte before the block lies below
    # `start`, the block's own runs and columns are tested, and a true run's source row exists
    # by the op's contract. TileLang's own check on the mask chunk, whose bound it cannot
    # derive from `bx`, splits each 16-byte load into 16 guarded byte loads.
    @tilelang.jit(
        out_idx=[4],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True},
    )
    def _func(block_runs, col_tile, vec, threads, unroll, counted):
        runs = T.dynamic("runs", "int64")
        blocks = T.dynamic("blocks", "int64")
        # Rows of run_len source elements; a true run k reads row k.
        src_rows = T.dynamic("src_rows", "int64")
        warps = threads // WARP_LANES
        per = (block_runs + threads - 1) // threads  # runs one thread scans
        tile_vecs = block_runs * (col_tile // vec)
        col_tiles = (run_len + col_tile - 1) // col_tile
        # A run of more column tiles than the grid's y axis holds takes several passes.
        grid_cols = min(col_tiles, MAX_GRID_YZ_BLOCKS)

        @T.prim_func
        def masked_scatter(
            Input: T.Tensor((runs, run_len), dtype),
            Mask: T.Tensor((runs,), "uint8"),
            Counts: T.Tensor((blocks,), "int32"),
            Source: T.Tensor((src_rows, run_len), dtype),
            Out: T.Tensor((runs, run_len), dtype),
        ):
            with T.Kernel(T.ceildiv(runs, block_runs), grid_cols, threads=threads) as (bx, by):
                tx = T.get_thread_binding()
                lane = tx % WARP_LANES
                warp = tx // WARP_LANES
                warp_sum = T.alloc_shared((warps,), "int64")
                rank_s = T.alloc_shared((block_runs,), "int32")
                mask_s = T.alloc_shared((block_runs,), "uint8")
                start = T.cast(bx, "int64") * block_runs

                # The true runs before this block, from the block counts or the mask itself. A
                # mask byte is true when it is nonzero, as torch reads a bool.
                before = T.alloc_var("int64")
                before = 0
                if counted:
                    for p in T.serial(T.ceildiv(T.cast(bx, "int64"), threads)):
                        b = p * threads + tx
                        if b < bx:
                            before += T.cast(Counts[b], "int64")
                else:
                    chunk = T.alloc_local((VECTOR_ACCESS_BYTES,), "uint8")
                    for p in T.serial(T.ceildiv(start, threads * VECTOR_ACCESS_BYTES)):
                        base = (p * threads + tx) * VECTOR_ACCESS_BYTES
                        if base + VECTOR_ACCESS_BYTES <= start:
                            for i in T.vectorized(VECTOR_ACCESS_BYTES):
                                chunk[i] = Mask[base + i]
                            for i in T.unroll(VECTOR_ACCESS_BYTES):
                                before += T.cast(chunk[i] != 0, "int64")
                        else:
                            for i in T.unroll(VECTOR_ACCESS_BYTES):
                                if base + i < start:
                                    before += T.cast(Mask[base + i] != 0, "int64")
                before = T.warp_reduce_sum(before)
                if lane == 0:
                    warp_sum[warp] = before
                T.sync_threads()
                offset = T.alloc_var("int64")
                offset = 0
                for w in T.unroll(warps):
                    offset += warp_sum[w]
                T.sync_threads()

                # An exclusive scan of the block's own runs; thread tx scans `per` of them.
                own = T.alloc_var("int32")
                own = 0
                for i in T.unroll(per):
                    j = tx * per + i
                    if j < block_runs and start + j < runs:
                        mask_s[j] = Mask[start + j]
                        own += T.cast(Mask[start + j] != 0, "int32")
                incl = T.alloc_var("int32")
                incl = own
                for k in T.unroll(WARP_SHUFFLE_STAGES):
                    up = T.shfl_up(incl, 1 << k)
                    if lane >= (1 << k):
                        incl += up
                if lane == WARP_LANES - 1:
                    warp_sum[warp] = incl
                T.sync_threads()
                local = T.alloc_var("int32")
                local = incl - own
                for w in T.unroll(warps):
                    if w < warp:
                        local += T.cast(warp_sum[w], "int32")
                for i in T.unroll(per):
                    j = tx * per + i
                    if j < block_runs and start + j < runs:
                        rank_s[j] = local
                        local += T.cast(mask_s[j] != 0, "int32")
                T.sync_threads()

                # Copy: a thread loads `unroll` vectors into registers, then stores them.
                buf = T.alloc_local((unroll, vec), dtype)
                for q in T.serial(T.ceildiv(col_tiles, grid_cols)):
                    tile = T.cast(q * grid_cols + by, "int64")
                    for it in T.serial(T.ceildiv(tile_vecs, threads * unroll)):
                        for u in T.unroll(unroll):
                            v = (it * unroll + u) * threads + tx
                            j = v // (col_tile // vec)
                            col = tile * col_tile + (v % (col_tile // vec)) * vec
                            if v < tile_vecs and start + j < runs and col < run_len:
                                if mask_s[j] != 0:
                                    for i in T.vectorized(vec):
                                        buf[u, i] = Source[offset + rank_s[j], col + i]
                                else:
                                    for i in T.vectorized(vec):
                                        buf[u, i] = Input[start + j, col + i]
                        for u in T.unroll(unroll):
                            v = (it * unroll + u) * threads + tx
                            j = v // (col_tile // vec)
                            col = tile * col_tile + (v % (col_tile // vec)) * vec
                            if v < tile_vecs and start + j < runs and col < run_len:
                                for i in T.vectorized(vec):
                                    Out[start + j, col + i] = buf[u, i]

        return masked_scatter

    return _func


@target_cache(maxsize=8)
def _block_count_kernel(block_runs, threads):
    @tilelang.jit
    def _func():
        runs = T.dynamic("runs", "int64")
        blocks = T.dynamic("blocks", "int64")
        per = (block_runs + threads - 1) // threads

        @T.prim_func
        def block_count(Mask: T.Tensor((runs,), "uint8"), Counts: T.Tensor((blocks,), "int32")):
            with T.Kernel(blocks, threads=threads) as bx:
                tx = T.get_thread_binding()
                total = T.alloc_shared((1,), "int32")
                own = T.alloc_var("int32")
                own = 0
                start = T.cast(bx, "int64") * block_runs
                for i in T.unroll(per):
                    j = tx * per + i
                    if j < block_runs and start + j < runs:
                        own += T.cast(Mask[start + j] != 0, "int32")
                own = T.warp_reduce_sum(own)
                if tx == 0:
                    total[0] = 0
                T.sync_threads()
                if tx % WARP_LANES == 0:
                    T.atomic_add(total[0], own)
                T.sync_threads()
                if tx == 0:
                    Counts[bx] = total[0]

        return block_count

    return _func()


class MaskedScatterKernel(Kernel, MaskedScatterFwdInterface):
    """A block of runs a CTA: it counts the true runs before it, scans its own, then copies.

    Each output vector is read once, from source where the mask is true and from input
    where it is false.
    """

    @classmethod
    def entry_for(cls, call: MaskedScatterCall) -> Entry:
        index = call.device_index
        return (call.run_len, call.dtype, index), lambda: cls(
            call.run_len, call.dtype, device_index=index
        )

    def __init__(self, run_len, dtype, config=None, device_index=None):
        super().__init__(device_index=device_index)
        self.run_len, self.dtype = run_len, dtype
        self.kernel = _masked_scatter_kernel(run_len, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self):
        # 256 threads divide every architecture's per-SM thread limit, so whole CTAs fill an
        # SM; 8 vectors of 16 bytes a thread keep 32 KB of loads in flight a CTA before its
        # stores.
        threads, unroll = 256, 8
        block_bytes = threads * unroll * VECTOR_ACCESS_BYTES
        vec = math.gcd(self.run_len, VECTOR_ACCESS_BYTES // self.dtype.itemsize)
        # Whole runs while they fit a block, as many as a power of two allows, and no more
        # than a thread scans in one round of `unroll`; a wider run is split into column
        # tiles of one block.
        fit = min(block_bytes // (self.run_len * self.dtype.itemsize), threads * unroll)
        block_runs = 1 << max(0, fit.bit_length() - 1)
        block_elems = block_bytes // self.dtype.itemsize
        col_tile = self.run_len if block_runs > 1 else min(self.run_len, block_elems)
        return {
            "block_runs": block_runs,
            "col_tile": col_tile,
            "vec": vec,
            "threads": threads,
            "unroll": unroll,
        }

    def forward(self, input, mask, source):
        shape = torch.broadcast_shapes(input.shape, mask.shape)
        lead, run_len = mask_runs(shape, mask.shape)
        runs = math.prod(shape[:lead])
        flags = mask.reshape((1,) * (len(shape) - mask.dim()) + tuple(mask.shape))
        flags = flags[(slice(None),) * lead + (0,) * (len(shape) - lead)]
        flags = flags.expand(shape[:lead]).reshape(runs).contiguous().view(torch.uint8)
        rows = input.expand(shape).reshape(runs, run_len).contiguous()
        src = source.reshape(-1)
        src = src[: src.numel() // run_len * run_len].view(-1, run_len)
        # Each block reads the mask before it, about runs**2 / (2 * block_runs) bytes in all,
        # against 2 * runs * run_len * itemsize bytes the copy moves. Past four times a
        # block's bytes in runs, the mask reads outweigh the copy, so a count pass first sums
        # each block and the scatter reads those sums instead.
        cfg = self.config
        counted = runs > 4 * cfg["block_runs"] * run_len * input.element_size()
        count, scatter = self.program(self._compile, counted)
        blocks = -(-runs // cfg["block_runs"]) if counted else 1
        counts = torch.empty(blocks, dtype=torch.int32, device=flags.device)
        if counted:
            count(flags, counts)
        return scatter(rows, flags, counts, src).view(shape)

    def _compile(self, cfg: dict, counted: bool):
        """The scatter program, and with *counted* the block count it reads its offset from."""
        count = (
            CompiledProgram(_block_count_kernel(cfg["block_runs"], cfg["threads"]))
            if counted
            else None
        )
        return count, CompiledProgram(self.kernel(**cfg, counted=counted))
