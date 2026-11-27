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
from tileops._kernels.constants import SM_RESIDENT_BLOCKS, VECTOR_ACCESS_BYTES
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
    def _func(block_rows, block_vectors, vec, threads, min_blocks):
        total = T.dynamic("total", "int64")
        # Row offsets reach past 2**31 elements in a large table.
        rows = T.dynamic("rows", "int64")
        col_tiles = -(-(dim // vec) // block_vectors)
        tiles = T.ceildiv(total, block_rows) * col_tiles
        # One linear tile index over x, then y: x holds 2**20 tiles and y (at most 65535)
        # the rest, so no row count or row width meets a CUDA grid limit.
        x_tiles = 1 << 20
        grid_x = T.min(tiles, x_tiles)
        slots = block_rows * block_vectors
        passes = T.ceildiv(slots, threads)
        width = vec * DataType(dtype).bits // 8
        # A thread lands its vectors in its registers and stores them from there, with no
        # shared-memory round trip between a vector's load and its store. A program that asks
        # for its CTA limit spends the register budget of that limit on the index state of its
        # passes (``default_config``'s ``wave_threads``), so it lands them in shared memory.
        staged = min_blocks > 0
        # Table rows land evict-last in L2 and the output streams past it: a table that fits
        # in L2 is read from DRAM once however many lookups name its rows.
        land = "cp_async_evict_last" if staged else "load_vector_evict_last"
        land = f"tileops::{land}<{width}>"
        store = f"tileops::store_streaming<{width}>"
        # The registers of one vector: 4-byte lanes, or the element of a 2-byte vector.
        lanes = {4: "uint32", 8: "uint32x2", 16: "uint32x4"}.get(width, dtype)

        def slot(p, tx):
            # Where a thread holds its p-th vector: its row of the stage, or its p-th register.
            return p * threads + tx if staged else p

        def pointer(buf, row, col, access):
            if buf.scope() == "local":
                return T.address_of(buf[row, col])
            return T.access_ptr(buf[row, col], access, vec)

        @T.macro
        def move(fn, dst, dst_row, dst_col, src, src_row, src_col):
            # cp.async, the vector load and the streaming store move 4, 8 or 16 bytes; a 2-byte
            # vector is copied element by element.
            if width >= 4:
                T.call_extern(  # lint: extern-target allow: fn is land or store, bound above
                    "handle", fn, pointer(dst, dst_row, dst_col, "w"),
                    pointer(src, src_row, src_col, "r"),
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
                if min_blocks:
                    T.annotate_min_blocks_per_sm(min_blocks)
                if staged:
                    held = T.alloc_shared((passes * threads, vec), dtype)
                else:
                    held = T.alloc_local((passes, 1), lanes)
                # by is 0 unless grid_x is x_tiles, so this is by * grid_x + bx without the
                # arithmetic on the id count that grid_x takes.
                tile = T.cast(by, "int64") * x_tiles + bx
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
                    # A tile past the last starts at a row past the last, so the row bound
                    # also covers it.
                    if s >= slots or row_of[p] >= total or col_of[p] >= dim:
                        row_of[p] = -1
                    if row_of[p] >= 0:
                        src_of[p] = Index[row_of[p]]
                # Every load of the thread is in flight before its first store.
                for p in T.unroll(passes):
                    if row_of[p] >= 0:
                        move(land, held, slot(p, tx), 0, Weight, src_of[p], col_of[p])
                if staged:
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                for p in T.unroll(passes):
                    if row_of[p] >= 0:
                        move(store, Out, row_of[p], col_of[p], held, slot(p, tx), 0)

        return embedding

    return _func


class EmbeddingKernel(Kernel, EmbeddingFwdInterface):
    """Each thread copies its own vectors of a tile of rows.

    A program that asks for its CTA limit lands them in shared memory; every other program
    lands them in registers.
    """

    @classmethod
    def refusal(cls, call: EmbeddingCall) -> str | None:
        if call.renorm:
            return "does not renormalize rows by max_norm"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: EmbeddingCall) -> Entry:
        index = call.device_index
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
        # wave_threads: the CTA size at which an SM reaches its CTA limit, so a call's CTAs
        # fit in fewer waves. The register budget at that limit bounds the passes; six
        # registers a pass (output row, column, table row) is a conservative estimate.
        target = self.build_target
        wave_threads = threads
        if block_rows == 1 and target.arch in SM_RESIDENT_BLOCKS:
            resident_threads = target.threads_per_sm // SM_RESIDENT_BLOCKS[target.arch]
            register_unit = 8
            budget = target.regs_per_sm // target.threads_per_sm // register_unit * register_unit
            pinned_passes = budget // 6
            if (
                resident_threads < spread_threads
                and slots % resident_threads == 0
                and slots <= pinned_passes * resident_threads
            ):
                wave_threads = resident_threads
        return {
            "block_rows": block_rows,
            "block_vectors": block_vectors,
            "vec": vec,
            "threads": threads,
            "spread_threads": spread_threads,
            "wave_threads": wave_threads,
        }

    def forward(self, input, weight):
        ids = input.view(-1)
        cfg, tiles = self.config, ids.numel()
        # wave_threads wins when the call fits in one wave of them and the spread program's
        # waves outnumber their pass ratio. Taking a CTA's life as proportional to its passes
        # is conservative: a thread waits once for all its copies.
        slots = cfg["block_rows"] * cfg["block_vectors"]
        spread, wave = cfg["spread_threads"], cfg["wave_threads"]
        threads, min_blocks = spread, 0
        if tiles <= self._resident(cfg, cfg["threads"]):
            threads = cfg["threads"]
        elif tiles <= (held := self._resident(cfg, wave, staged=True)):
            spread_passes, wave_passes = -(-slots // spread), -(-slots // wave)
            if tiles * spread_passes > self._resident(cfg, spread) * wave_passes:
                threads, min_blocks = wave, held // self.build_target.sm_count
        program = self.program(self._compile, threads, min_blocks)
        return program(ids, weight).view(*input.shape, self.dim)

    def _resident(self, cfg: dict, threads: int, staged: bool = False) -> int:
        """CTAs of *cfg*'s tile on *threads* threads the device holds at once.

        A *staged* program's CTAs each also hold their vectors in shared memory.
        """
        target = self.build_target
        warps = -(-threads // WARP_LANES)
        per_sm = target.threads_per_sm // WARP_LANES // warps
        if target.arch in SM_RESIDENT_BLOCKS:
            slots = cfg["block_rows"] * cfg["block_vectors"]
            stage = 0
            if staged:
                stage = -(-slots // threads) * threads * cfg["vec"] * self.dtype.itemsize
            # Each resident CTA also reserves the gap to the per-block opt-in limit.
            reserved = target.smem_per_sm - target.smem_budget
            per_sm = min(
                per_sm,
                SM_RESIDENT_BLOCKS[target.arch],
                target.smem_per_sm // (stage + reserved),
            )
        return target.sm_count * per_sm

    def _compile(self, cfg: dict, threads: int, min_blocks: int):
        """*cfg*'s program on *threads* threads, asking for *min_blocks* CTAs an SM if nonzero."""
        return self.kernel(cfg["block_rows"], cfg["block_vectors"], cfg["vec"], threads, min_blocks)
