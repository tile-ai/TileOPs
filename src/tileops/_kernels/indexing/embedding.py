"""Row gather of an embedding table: ``out[i, :] = weight[input[i], :]``."""

import dataclasses
import math
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface
from tileops._utils import WARP_LANES

__all__ = ["EmbeddingCall", "EmbeddingFwdInterface", "EmbeddingKernel"]


@dataclasses.dataclass(frozen=True)
class EmbeddingCall(CallSpec):
    """The row width and dtypes of one lookup; the index count and table rows are dynamic."""

    dim: int = 0
    dtype: torch.dtype | None = None
    index_dtype: torch.dtype | None = None
    renorm: bool = False


class EmbeddingFwdInterface(KernelInterface):
    """Gather the table rows the indices name."""

    request = EmbeddingCall

    @abstractmethod
    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Copy row ``input[...]`` of *weight* for every index.

        Args:
            input: Indices in ``[0, V)``, any shape, in ``call.index_dtype``, contiguous.
            weight: The table ``[V, call.dim]`` in ``call.dtype``, contiguous.

        Returns:
            A new tensor ``[*input.shape, call.dim]`` in ``call.dtype`` on ``call.device``.
        """


@target_cache(maxsize=32)
def _embedding_kernel(dim, dtype, index_dtype):
    # A thread reads back only the shared memory it landed itself, so no barrier orders it.
    @tilelang.jit(
        out_idx=[2],
        compile_flags=csrc_include("gather_copy.h"),
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True},
    )
    def _func(block_rows, block_vectors, vec, threads):
        total = T.dynamic("total", "int64")
        # Row offsets reach past 2**31 elements in a large table.
        rows = T.dynamic("rows", "int64")
        col_tiles = -(-(dim // vec) // block_vectors)
        tiles = T.ceildiv(total, block_rows) * col_tiles
        # One linear tile index over x, then y: x holds 2**20 tiles and y (at most 65535)
        # the rest, so no row count or row width meets a CUDA grid limit.
        grid_x = T.min(tiles, 1 << 20)
        slots = block_rows * block_vectors
        passes = T.ceildiv(slots, threads)
        width = vec * DataType(dtype).bits // 8
        # Table rows land evict-last in L2 and the output streams past it: a table that fits
        # in L2 is read from DRAM once however many lookups name its rows.
        land = f"tileops::cp_async_evict_last<{width}>"
        store = f"tileops::store_streaming<{width}>"

        @T.macro
        def move(fn, dst, dst_row, dst_col, src, src_row, src_col):
            # cp.async and the streaming store move 4, 8 or 16 bytes; a 2-byte vector is
            # copied element by element.
            if width >= 4:
                T.call_extern(  # lint: extern-target allow: fn is land or store, bound above
                    "handle", fn, T.access_ptr(dst[dst_row, dst_col], "w", vec),
                    T.access_ptr(src[src_row, src_col], "r", vec),
                )  # fmt: skip
            else:
                for i in T.vectorized(vec):
                    dst[dst_row, dst_col + i] = src[src_row, src_col + i]

        @T.prim_func
        def embedding(
            Index: T.Tensor((total,), index_dtype),
            Weight: T.Tensor((rows, dim), dtype),
            Out: T.Tensor((total, dim), dtype),
        ):
            with T.Kernel(grid_x, T.ceildiv(tiles, grid_x), threads=threads) as (bx, by):
                tx = T.get_thread_binding()
                stage = T.alloc_shared((passes * threads, vec), dtype)
                tile = T.cast(by, "int64") * grid_x + bx
                first_row = tile // col_tiles * block_rows
                first_col = tile % col_tiles * block_vectors
                # Each vector's output row and column, -1 for a slot past the tile, and its
                # table row; the in-tile split is 32-bit, which the 64-bit tile arithmetic
                # would widen. The table row stays in the id dtype until the address is
                # formed: a widening beside its load makes the thread wait for that id
                # before it issues the next.
                row_of = T.alloc_local((passes,), "int64")
                col_of = T.alloc_local((passes,), "int64")
                src_of = T.alloc_local((passes,), index_dtype)
                # Every id load of the thread is in flight before its first copy: a copy's
                # memory clobber keeps a later load from moving above it.
                for p in T.unroll(passes):
                    s = p * threads + tx
                    in_row = T.alloc_var("int32")
                    in_row = s // block_vectors
                    in_col = T.alloc_var("int32")
                    in_col = s % block_vectors
                    row_of[p] = first_row + in_row
                    col_of[p] = (first_col + in_col) * vec
                    if tile >= tiles or s >= slots or row_of[p] >= total or col_of[p] >= dim:
                        row_of[p] = -1
                    if row_of[p] >= 0:
                        src_of[p] = Index[row_of[p]]
                # Every copy of the thread is in flight before its one wait.
                for p in T.unroll(passes):
                    if row_of[p] >= 0:
                        move(land, stage, p * threads + tx, 0, Weight, src_of[p], col_of[p])
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                for p in T.unroll(passes):
                    if row_of[p] >= 0:
                        move(store, Out, row_of[p], col_of[p], stage, p * threads + tx, 0)

        return embedding

    return _func


class EmbeddingKernel(Kernel, EmbeddingFwdInterface):
    """Each thread copies its own vectors of a tile of rows through shared memory."""

    @classmethod
    def refusal(cls, call: EmbeddingCall) -> str | None:
        if call.renorm:
            return "does not renormalize rows by max_norm"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: EmbeddingCall) -> Entry:
        index = call.device.index
        return (call.dim, call.dtype, call.index_dtype, index), lambda: cls(
            call.dim, call.dtype, call.index_dtype, device_index=index
        )

    def __init__(self, dim, dtype, index_dtype, device_index=None):
        super().__init__(device_index=device_index)
        self.dim, self.dtype = dim, dtype
        self.kernel = _embedding_kernel(dim, self.dtype_str, self.dtype_to_str(index_dtype))
        self.init_config()

    @property
    def default_config(self):
        # A thread copies vectors of up to 16 bytes, as wide as the row width allows.
        vec = math.gcd(self.dim, VECTOR_ACCESS_BYTES // self.dtype.itemsize)
        # 256 threads divide every architecture's per-SM thread limit, so whole CTAs fill an SM.
        max_threads = 256
        # A thread holds at most four vectors, so its per-vector row, column and id arrays stay
        # in registers; a wider row takes column tiles.
        max_passes = 4
        block_vectors = min(self.dim // vec, max_passes * max_threads)
        # Narrow rows: as many whole rows, a power of two, as leave a thread at most one vector.
        # Such a tile is bound by the id-then-row latency, which more threads hide.
        block_rows = 1
        while 2 * block_rows * block_vectors <= max_threads:
            block_rows *= 2
        # Threads divide the tile's vectors when a count of at least a warp does within the
        # pass bound, so no thread idles in the last pass; otherwise whole warps cover the tile
        # and the last pass is guarded.
        slots = block_rows * block_vectors
        divisors = [
            t
            for t in range(WARP_LANES, max_threads + 1)
            if slots % t == 0 and slots <= max_passes * t
        ]
        warps = -(-slots // (max_passes * WARP_LANES))
        threads = max(divisors) if divisors else min(WARP_LANES * warps, max_threads)
        # Every thread of a one-row tile waits on the tile's one id before its row loads.
        # spread_threads, the fewest whole warps that divide the row, put up to max_passes
        # loads a thread behind that id. Past one wave of CTAs an SM holds more of these
        # smaller CTAs; within one wave the extra loads lengthen the call's single round
        # trip, so ``forward`` launches spread_threads only past one wave.
        whole_warps = [t for t in divisors if t % WARP_LANES == 0]
        spread_threads = threads
        if block_rows == 1 and whole_warps:
            spread_threads = min(whole_warps)
        return {
            "block_rows": block_rows,
            "block_vectors": block_vectors,
            "vec": vec,
            "threads": threads,
            "spread_threads": spread_threads,
        }

    def forward(self, input, weight):
        ids = input.view(-1)
        spread = ids.numel() > self._spread_above(self.config)
        program = self.program(self._compile, spread)
        return program(ids, weight).view(*input.shape, self.dim)

    def _spread_above(self, cfg: dict) -> float:
        """The id count above which ``forward`` launches the ``spread_threads`` program,
        infinity where *cfg* does not spread."""
        if cfg["spread_threads"] == cfg["threads"]:
            return math.inf
        # A config that spreads has one-row tiles of the whole row, one CTA an id, so a call
        # runs in waves above the CTAs of the ``threads`` program the device holds at once,
        # which the SM's thread limit bounds; an SM holds a CTA in whole warps.
        target = self.build_target
        warps = -(-cfg["threads"] // WARP_LANES)
        return target.sm_count * (target.threads_per_sm // WARP_LANES // warps)

    def _compile(self, cfg: dict, spread: bool):
        """The program ``forward`` launches, compiled from *cfg*, with ``spread_threads`` for
        ``threads`` when *spread* holds."""
        threads = cfg["spread_threads"] if spread else cfg["threads"]
        return self.kernel(cfg["block_rows"], cfg["block_vectors"], cfg["vec"], threads)
