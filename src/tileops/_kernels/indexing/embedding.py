"""Row gather of an embedding table: ``out[i, :] = weight[input[i], :]``."""

import dataclasses
import math
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

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
    def _func(block_rows, block_vectors, vec, threads, blocks_per_sm):
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
        width = vec * T.DataType(dtype).bits // 8
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
                if blocks_per_sm:
                    T.annotate_min_blocks_per_sm(blocks_per_sm)
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

    # 256 threads divide every architecture's per-SM thread limit, so whole CTAs fill an SM.
    _MAX_THREADS = 256
    # A thread holds at most four vectors, so its per-vector row, column and id arrays stay in
    # registers; a wider row takes column tiles.
    _MAX_PASSES = 4

    @classmethod
    def refusal(cls, call: EmbeddingCall) -> str | None:
        if call.renorm:
            return "does not renormalize rows by max_norm"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: EmbeddingCall) -> Entry:
        index = call.device.index
        return (call.dim, call.dtype, call.index_dtype, index), lambda: cls(
            call.dim,
            call.dtype,
            call.index_dtype,
            sm_count=call.sm_count,
            threads_per_sm=call.threads_per_sm,
            device_index=index,
        )

    def __init__(self, dim, dtype, index_dtype, *, sm_count, threads_per_sm, device_index=None):
        super().__init__(device_index=device_index)
        self.dim, self.dtype = dim, dtype
        self._sm_count, self._threads_per_sm = sm_count, threads_per_sm
        self._resident_threads = sm_count * threads_per_sm
        self._wave_tiles: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        self.kernel = _embedding_kernel(dim, self.dtype_str, self.dtype_to_str(index_dtype))
        self.init_config()

    @property
    def default_config(self):
        # A thread copies vectors of up to 16 bytes, as wide as the row width allows.
        vec = math.gcd(self.dim, VECTOR_ACCESS_BYTES // self.dtype.itemsize)
        return {
            "vec": vec,
            "block_vectors": min(self.dim // vec, self._MAX_PASSES * self._MAX_THREADS),
        }

    def vectors_per_thread(self, ids: int) -> int:
        """The vectors a thread takes in a call of *ids* indices."""
        # A call of more vectors than the device holds resident threads runs in waves, each
        # waiting one id-then-row round trip. When up to four vectors a thread fit the call in
        # one wave, a thread takes that many; otherwise the call runs in waves either way, and
        # like a smaller call it gives each thread one vector.
        # The tile rounds its rows to a power of two and its threads to a divisor, so a count
        # is taken only where its tile's CTAs fit the resident ones, and only where it gives
        # the tile more rows than one vector a thread does: a row too wide to share a tile
        # already holds its vectors a thread, and keeps the one-vector program.
        passes = max(1, -(-ids * (self.dim // self.config["vec"]) // self._resident_threads))
        if passes == 1:
            return 1
        single_rows = self._wave_tile(1)[0]
        for count in range(passes, self._MAX_PASSES + 1):
            block_rows, col_tiles, resident = self._wave_tile(count)
            if block_rows > single_rows and -(-ids // block_rows) * col_tiles <= resident:
                return count
        return 1

    def _wave_tile(self, passes: int) -> tuple[int, int, int]:
        """The rows and column tiles of the tile at *passes* vectors a thread, and the CTAs of
        it the device holds at once; computed once per config."""
        key = (self.config["vec"], self.config["block_vectors"], passes)
        found = self._wave_tiles.get(key)
        if found is None:
            tile = self.tile(self.config, passes)
            col_tiles = -(-(self.dim // tile["vec"]) // tile["block_vectors"])
            resident = self._sm_count * (self._threads_per_sm // tile["threads"])
            found = self._wave_tiles[key] = (tile["block_rows"], col_tiles, resident)
        return found

    def tile(self, cfg: dict, passes: int) -> dict:
        """The launch tile of *cfg* at *passes* vectors a thread: rows, vectors, width, threads,
        and the CTAs an SM must hold at once."""
        block_vectors = cfg["block_vectors"]
        # As many whole rows, a power of two, as that many vectors a thread hold; a row of more
        # vectors than half of them takes a tile of its own.
        block_rows = 1
        while 2 * block_rows * block_vectors <= passes * self._MAX_THREADS:
            block_rows *= 2
        # Threads divide the tile's vectors when a count of at least a warp does within the
        # pass bound, so no thread idles in the last pass; otherwise whole warps cover the tile
        # and the last pass is guarded.
        slots = block_rows * block_vectors
        divisors = [
            t
            for t in range(WARP_LANES, self._MAX_THREADS + 1)
            if slots % t == 0 and slots <= self._MAX_PASSES * t
        ]
        warps = -(-slots // (self._MAX_PASSES * WARP_LANES))
        threads = max(divisors) if divisors else min(WARP_LANES * warps, self._MAX_THREADS)
        # A count above one is taken where the tile's CTAs fit the device's resident threads,
        # so the program is held to the registers that let that many CTAs reside on an SM.
        return {
            "block_rows": block_rows,
            "block_vectors": block_vectors,
            "vec": cfg["vec"],
            "threads": threads,
            "blocks_per_sm": self._threads_per_sm // threads if passes > 1 else 0,
        }

    def forward(self, input, weight):
        ids = input.view(-1)
        program = self.program(self._compile, self.vectors_per_thread(ids.numel()))
        return program(ids, weight).view(*input.shape, self.dim)

    def _compile(self, cfg: dict, passes: int):
        """The program ``forward`` launches for *passes* vectors a thread."""
        return self.kernel(**self.tile(cfg, passes))
