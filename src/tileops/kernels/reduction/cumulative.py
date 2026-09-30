"""Cumulative scan kernels (cumsum, cumprod) using TileLang."""

import functools
import itertools
import math
from dataclasses import dataclass
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import (
    SHARED_BANK_SPAN_BYTES,
    STATIC_SHARED_BYTES,
    VECTOR_ACCESS_BYTES,
)
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    align_up,
    restore_same_shape,
    rows_for_axes,
    torch_dtype_nbytes,
)
from tileops.kernels.reduction.call_spec import (
    CumprodFwdInterface,
    CumsumFwdInterface,
    CumulativeCall,
)
from tileops.utils import WARP_LANES

__all__ = ["CumsumParallelScanKernel", "CumulativeKernel", "CumulativeRowScanKernel"]


@dataclass(frozen=True)
class CumulativeScanPolicy:
    """Shape and shared-memory heuristics for cumulative scan kernels."""

    # Multiple of DEFAULT_ALIGNMENT for T.copy shared memory alignment.
    default_block_n: int = 128

    # Breaks shared-memory bank conflicts for fp16/bf16 row staging.
    smem_pad: int = 8

    # Elements per thread the split aims for.
    row_scan_chunk: int = 64
    # Block width past which the split lengthens the chunk instead of adding warps.
    row_scan_wide_threads: int = 256
    # Longest chunk a thread takes. It lives in fp32 registers, so 256 would spill.
    row_scan_max_chunk: int = 128
    # Pads row_scan_pad chooses between, in vector accesses. A whole vector access is
    # what keeps a chunk 16-byte aligned, which the shared access needs to stay 128-bit.
    row_scan_pad_vectors: tuple = (1, 2)
    row_scan_min_threads: int = 64
    row_scan_max_threads: int = 1024


_SCAN_POLICY = CumulativeScanPolicy()


def row_scan_pad(chunk: int, elem_bytes: int) -> int:
    """Return the padding, in elements, each staged chunk gets.

    Neighbouring lanes read one chunk each, so they sit ``(chunk + pad) * elem_bytes``
    apart, and the conflict ways grow with what that stride shares with
    ``SHARED_BANK_SPAN_BYTES``. This returns the candidate pad that shares least, which
    depends on *chunk*: a fixed pad leaves some chunks striding the whole span.
    """
    candidates = [k * VECTOR_ACCESS_BYTES // elem_bytes for k in _SCAN_POLICY.row_scan_pad_vectors]
    return min(
        candidates,
        key=lambda pad: (math.gcd((chunk + pad) * elem_bytes, SHARED_BANK_SPAN_BYTES), pad),
    )


def row_scan_chunk_ok(chunk: int, elem_bytes: int, threads: int) -> bool:
    """Whether a block of *threads* may give each thread a chunk of *chunk* elements.

    Up to ``row_scan_chunk`` always. A block already ``row_scan_wide_threads`` wide may
    go as far as ``row_scan_max_chunk``, but only with a chunk of whole vector accesses,
    which is what keeps the chunk 16-byte aligned.
    """
    if chunk <= _SCAN_POLICY.row_scan_chunk:
        return True
    return (
        threads >= _SCAN_POLICY.row_scan_wide_threads
        and chunk <= _SCAN_POLICY.row_scan_max_chunk
        and (chunk * elem_bytes) % VECTOR_ACCESS_BYTES == 0
    )


def row_scan_threads(N_padded: int, elem_bytes: int) -> int:
    """Threads the whole-row kernel gives a row of *N_padded*.

    The narrowest block that divides the row and whose chunk :func:`row_scan_chunk_ok`
    accepts, or the widest divisor tried when no block qualifies -- which
    :func:`row_scan_fits` then declines.
    """
    threads = _SCAN_POLICY.row_scan_min_threads
    widest = threads
    while threads <= _SCAN_POLICY.row_scan_max_threads:
        if N_padded % threads == 0:
            widest = threads
            if row_scan_chunk_ok(N_padded // threads, elem_bytes, threads):
                return threads
        threads *= 2
    return widest


@functools.lru_cache(maxsize=32)
def _row_scan_kernel(M: int, N: int, op_kind: str, dtype: str, threads: int):
    """Build a one-block-per-row inclusive prefix scan.

    Three phases, with no serial dependency between blocks:

    1. The row is staged in shared memory at its own dtype, one padded row of the tile
       per thread chunk, and each thread scans its chunk into registers in fp32.
    2. The chunk totals are scanned with shuffles inside each warp, and one barrier
       carries each warp's total to the warps above it.
    3. Each thread adds the prefix ahead of its chunk and writes the chunk back to the
       staging tile, which returns it to global memory.

    Both transfers between global and the tile run over the row in column-major thread
    order, so consecutive threads read consecutive columns and the access coalesces; the
    tile's padding is what keeps a thread's own chunk off one bank. Staging at the input
    dtype halves the shared traffic against fp32, and the write-back reuses the tile
    because a scalar global store per element does not coalesce.

    Args:
        M: Rows to scan.
        N: Length of the scanned axis; must equal the padded width the caller
            allocated for, and be divisible by *threads*.
        op_kind: One of "sum", "prod".
        dtype: TileLang dtype string.
        threads: Threads per row, from :func:`row_scan_threads`.
    """
    chunk_len = N // threads
    pad = row_scan_pad(chunk_len, torch_dtype_nbytes(dtype))
    # Shuffle steps to scan one warp's lanes, and the warps a block holds.
    n_steps = WARP_LANES.bit_length() - 1
    n_warps = max(threads // WARP_LANES, 1)
    identity = 0.0 if op_kind == "sum" else 1.0
    combine = (lambda a, b: a + b) if op_kind == "sum" else (lambda a, b: a * b)

    @tilelang.jit(out_idx=[1])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                staged = T.alloc_shared((threads, chunk_len + pad), dtype)
                totals = T.alloc_shared((n_warps,), "float32")
                chunk = T.alloc_local((chunk_len,), "float32")
                running = T.alloc_local((1,), "float32")
                ahead = T.alloc_local((1,), "float32")
                offset = T.alloc_local((1,), "float32")
                before = T.alloc_local((1,), "float32")

                for i, j in T.Parallel(threads, chunk_len):
                    staged[i, j] = x[row, i * chunk_len + j]
                T.sync_threads()

                running[0] = T.cast(identity, "float32")
                for j in T.serial(chunk_len):
                    running[0] = combine(running[0], T.cast(staged[tx, j], "float32"))
                    chunk[j] = running[0]

                for step in T.serial(n_steps):
                    stride = T.shift_left(T.int32(1), step)
                    ahead[0] = T.shfl_up(running[0], stride)
                    running[0] = T.if_then_else(
                        tx % WARP_LANES >= stride, combine(running[0], ahead[0]), running[0]
                    )
                if tx % WARP_LANES == WARP_LANES - 1:
                    totals[tx // WARP_LANES] = running[0]
                T.sync_threads()

                # One more shuffle turns the inclusive scan into the exclusive one a
                # chunk needs. Subtracting the thread's own total would do it for sum
                # and not for prod; a shuffle holds for any combine.
                before[0] = T.shfl_up(running[0], 1)
                before[0] = T.if_then_else(
                    tx % WARP_LANES == 0, T.cast(identity, "float32"), before[0]
                )
                ahead[0] = T.cast(identity, "float32")
                for w in T.serial(n_warps):
                    ahead[0] = T.if_then_else(
                        w < tx // WARP_LANES, combine(ahead[0], totals[w]), ahead[0]
                    )
                offset[0] = combine(before[0], ahead[0])
                for j in T.serial(chunk_len):
                    staged[tx, j] = T.cast(combine(chunk[j], offset[0]), dtype)
                T.sync_threads()
                for i, j in T.Parallel(threads, chunk_len):
                    y[row, i * chunk_len + j] = staged[i, j]

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _cumulative_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Build a TileLang inclusive prefix scan kernel.

    Accepts an ``(M, N)`` input tensor.  When ``N`` is not a multiple of
    ``block_n`` (which must divide ``N_padded``), the last tile uses
    element-wise ``T.if_then_else`` loads that substitute the identity
    element (0 for sum, 1 for prod) for out-of-bounds columns.  Preceding
    tiles use the fast vectorized ``T.copy`` path since their columns are
    fully in-bounds.

    Uses a tiled approach: the N dimension is divided into tiles of
    ``block_n`` elements. Each tile is loaded via shared memory, scanned
    sequentially with the running accumulator, and written back.

    Args:
        M: Number of rows (product of all leading dimensions).
        N: Hidden dimension (last dim, unpadded).
        op_kind: One of "sum", "prod".
        dtype: TileLang dtype string (e.g. "float16", "bfloat16", "float32").

    Returns:
        A TileLang JIT-compiled kernel factory accepting (block_m, block_n, threads).
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _identity = 0.0 if op_kind == "sum" else 1.0

    if op_kind == "sum":

        @tilelang.jit(out_idx=[1])
        def _func(block_m, block_n, threads):
            n_tiles = N_padded // block_n
            # The last tile may have out-of-bounds columns when N is not
            # a multiple of block_n.
            _needs_mask = (n_tiles * block_n) > N

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_in = T.alloc_shared((block_m, block_n + _SCAN_POLICY.smem_pad), dtype)
                    shared_out = T.alloc_shared((block_m, block_n + _SCAN_POLICY.smem_pad), dtype)
                    tile_f32 = T.alloc_fragment((block_m, block_n), "float32")
                    out_f32 = T.alloc_fragment((block_m, block_n), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")

                    for i in T.Parallel(block_m):
                        acc[i] = T.float32(0)

                    for tile_idx in T.Serial(n_tiles):
                        if _needs_mask:
                            with T.If((tile_idx + 1) * block_n <= N):
                                with T.Then():
                                    for i, j in T.Parallel(block_m, block_n):
                                        with T.If(pid_m * block_m + i < M):  # noqa: SIM117
                                            with T.Then():
                                                shared_in[i, j] = x[
                                                    pid_m * block_m + i, tile_idx * block_n + j
                                                ]
                                    for i, j in T.Parallel(block_m, block_n):
                                        tile_f32[i, j] = T.cast(shared_in[i, j], "float32")
                                with T.Else():
                                    for i, j in T.Parallel(block_m, block_n):
                                        tile_f32[i, j] = T.if_then_else(
                                            T.And(
                                                pid_m * block_m + i < M,
                                                tile_idx * block_n + j < N,
                                            ),
                                            T.cast(
                                                x[pid_m * block_m + i, tile_idx * block_n + j],
                                                "float32",
                                            ),
                                            T.cast(_identity, "float32"),
                                        )
                        else:
                            for i, j in T.Parallel(block_m, block_n):
                                with T.If(pid_m * block_m + i < M):  # noqa: SIM117
                                    with T.Then():
                                        shared_in[i, j] = x[
                                            pid_m * block_m + i, tile_idx * block_n + j
                                        ]
                            for i, j in T.Parallel(block_m, block_n):
                                tile_f32[i, j] = T.cast(shared_in[i, j], "float32")

                        for i in T.Parallel(block_m):
                            for j in T.Serial(block_n):
                                acc[i] = acc[i] + tile_f32[i, j]
                                out_f32[i, j] = acc[i]

                        # Cast back to shared_out (padded) then write to global with M bounds check
                        for i, j in T.Parallel(block_m, block_n):
                            shared_out[i, j] = T.cast(out_f32[i, j], dtype)
                        for i, j in T.Parallel(block_m, block_n):
                            # TileLang requires T.If/T.Then as nested context managers
                            with T.If(pid_m * block_m + i < M):  # noqa: SIM117
                                with T.Then():
                                    y[pid_m * block_m + i, tile_idx * block_n + j] = shared_out[
                                        i, j
                                    ]

            return main

    else:  # prod

        @tilelang.jit(out_idx=[1])
        def _func(block_m, block_n, threads):
            n_tiles = N_padded // block_n
            _needs_mask = (n_tiles * block_n) > N

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_in = T.alloc_shared((block_m, block_n + _SCAN_POLICY.smem_pad), dtype)
                    shared_out = T.alloc_shared((block_m, block_n + _SCAN_POLICY.smem_pad), dtype)
                    tile_f32 = T.alloc_fragment((block_m, block_n), "float32")
                    out_f32 = T.alloc_fragment((block_m, block_n), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")

                    for i in T.Parallel(block_m):
                        acc[i] = T.float32(1)

                    # Process N dimension in tiles
                    for tile_idx in T.Serial(n_tiles):
                        if _needs_mask:
                            with T.If((tile_idx + 1) * block_n <= N):
                                with T.Then():
                                    for i, j in T.Parallel(block_m, block_n):
                                        with T.If(pid_m * block_m + i < M):  # noqa: SIM117
                                            with T.Then():
                                                shared_in[i, j] = x[
                                                    pid_m * block_m + i, tile_idx * block_n + j
                                                ]
                                    for i, j in T.Parallel(block_m, block_n):
                                        tile_f32[i, j] = T.cast(shared_in[i, j], "float32")
                                with T.Else():
                                    for i, j in T.Parallel(block_m, block_n):
                                        tile_f32[i, j] = T.if_then_else(
                                            T.And(
                                                pid_m * block_m + i < M,
                                                tile_idx * block_n + j < N,
                                            ),
                                            T.cast(
                                                x[pid_m * block_m + i, tile_idx * block_n + j],
                                                "float32",
                                            ),
                                            T.cast(_identity, "float32"),
                                        )
                        else:
                            for i, j in T.Parallel(block_m, block_n):
                                with T.If(pid_m * block_m + i < M):  # noqa: SIM117
                                    with T.Then():
                                        shared_in[i, j] = x[
                                            pid_m * block_m + i, tile_idx * block_n + j
                                        ]
                            for i, j in T.Parallel(block_m, block_n):
                                tile_f32[i, j] = T.cast(shared_in[i, j], "float32")

                        for i in T.Parallel(block_m):
                            for j in T.Serial(block_n):
                                acc[i] = acc[i] * tile_f32[i, j]
                                out_f32[i, j] = acc[i]

                        # Cast back to shared_out (padded) then write to global with M bounds check
                        for i, j in T.Parallel(block_m, block_n):
                            shared_out[i, j] = T.cast(out_f32[i, j], dtype)
                        for i, j in T.Parallel(block_m, block_n):
                            # TileLang requires T.If/T.Then as nested context managers
                            with T.If(pid_m * block_m + i < M):  # noqa: SIM117
                                with T.Then():
                                    y[pid_m * block_m + i, tile_idx * block_n + j] = shared_out[
                                        i, j
                                    ]

            return main

    return _func


class _CumulativeKernelBase(Kernel):
    """Inclusive prefix scan (cumsum / cumprod) of ``call.axis``: what the three scans share.

    ``forward`` takes the tensor the op declares and scans ``call.axis`` of it; moving that
    axis to the end, flattening to rows and putting the result back are this kernel's
    business, so both sides of the op/backend boundary speak the declared shape.
    Boundary handling for a row that is not a multiple of the 256-element copy alignment
    is a masked load filling the identity element (0 for sum, 1 for prod).

    Args:
        call: The call this implementation serves.
        config: Optional kernel configuration dict.
    """

    @staticmethod
    def row_scan_fits(N_padded: int, elem_bytes: int, smem_budget: int) -> bool:
        """Whether a row of *N_padded* can be scanned by one thread block."""
        threads = row_scan_threads(N_padded, elem_bytes)
        if N_padded % threads:
            return False
        chunk = N_padded // threads
        staged = threads * (chunk + row_scan_pad(chunk, elem_bytes)) * elem_bytes
        return staged <= smem_budget and row_scan_chunk_ok(chunk, elem_bytes, threads)

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def stages_whole_row(cls, call: CumulativeCall) -> bool:
        """Whether one thread block stages the whole row, aligned, in shared memory."""
        n_padded = align_up(call.n, DEFAULT_ALIGNMENT)
        return n_padded == call.n and cls.row_scan_fits(
            n_padded, torch_dtype_nbytes(call.dtype), call.smem_budget
        )

    def __init__(self, call: CumulativeCall, config: Optional[dict] = None):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        if call.op_kind not in ("sum", "prod"):
            raise ValueError(
                f"Unsupported op_kind '{call.op_kind}'. Expected one of 'sum', 'prod'."
            )
        self.M = call.m
        self.N = call.n
        self.op_kind = call.op_kind
        self.dtype = call.dtype
        self.scan_axis = call.axis
        self.N_padded = align_up(self.N, DEFAULT_ALIGNMENT)
        self._elem_bytes = torch_dtype_nbytes(self.dtype)
        self.kernel = self._program()
        self.init_config(config)

    def _program(self) -> object:
        """The jit factory this implementation launches, or None for several."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan *scan_axis* of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device.

        Returns:
            A tensor shaped like *x*.

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        axes = (self.scan_axis,)
        y = self._scan_rows(rows_for_axes(x, axes))
        # The prim_func may write an alignment-padded row; the surplus columns are trimmed.
        y = y[:, : self.N] if y.shape[1] > self.N else y
        return restore_same_shape(y, in_shape, axes)

    def _scan_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Scan the trailing axis of an ``(M, N)`` buffer."""
        raise NotImplementedError


class CumulativeRowScanKernel(_CumulativeKernelBase, CumsumFwdInterface, CumprodFwdInterface):
    """One block scans one whole row: chunk scans, then a warp-shuffle scan of the chunk
    totals carried across warps. Serves the aligned rows :meth:`stages_whole_row`; the
    program takes no tunable parameter."""

    @classmethod
    def applies(cls, call: CumulativeCall) -> bool:
        return cls.stages_whole_row(call)

    def _program(self) -> object:
        # The chunk length is a compile-time bound, so the thread count is baked in.
        self._threads = row_scan_threads(self.N_padded, self._elem_bytes)
        return _row_scan_kernel(self.M, self.N, self.op_kind, self.dtype_str, self._threads)

    @property
    def default_config(self) -> dict:
        return {"threads": self._threads}

    def _scan_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel()(x)


class CumsumParallelScanKernel(_CumulativeKernelBase, CumsumFwdInterface):
    """Cumsum of a few long rows in three launches: every tile scans at once, then the tile
    totals scan, then each tile adds its carry. Serves ``m < 128`` rows longer than 8192
    that one block cannot stage; the launch shape is fixed."""

    @classmethod
    def applies(cls, call: CumulativeCall) -> bool:
        return call.m < 128 and call.n > 8192 and not cls.stages_whole_row(call)

    def _program(self) -> None:
        return None

    @property
    def default_config(self) -> dict:
        block_n = 256 if self.N > 16384 else 128
        smem_per_row = (block_n + _SCAN_POLICY.smem_pad) * 4  # fp32 intermediate
        max_block_m = STATIC_SHARED_BYTES // smem_per_row
        block_m = max(1, min(16, self.M, max_block_m))
        return {"block_m": block_m, "block_n": block_n, "threads": 256}

    def _scan_rows(self, x: torch.Tensor) -> torch.Tensor:
        block_m, block_n = self.config["block_m"], self.config["block_n"]
        threads = self.config["threads"]
        n_tiles = align_up(self.N, DEFAULT_ALIGNMENT) // block_n
        local = _parallel_scan_local_kernel(self.M, self.N, "sum", self.dtype_str)
        y_local, tile_sums = local(block_m, block_n, threads)(x)
        carries = _parallel_scan_carry_kernel(self.M, n_tiles)(threads)(tile_sums)
        propagate = _parallel_scan_propagate_kernel(self.M, self.N, self.dtype_str)
        return propagate(block_m, block_n, threads)(y_local, carries)


class CumulativeKernel(_CumulativeKernelBase, CumsumFwdInterface, CumprodFwdInterface):
    """The general scan: ``block_m`` rows a block, each walked serially over ``block_n``
    tiles staged through shared memory with the running accumulator carried across."""

    general = True

    def _program(self) -> object:
        return _cumulative_kernel(self.M, self.N, self.op_kind, self.dtype_str)

    @property
    def default_config(self) -> dict:
        block_n = _SCAN_POLICY.default_block_n
        elem_size = torch_dtype_nbytes(self.dtype)
        smem_per_row = 2 * (block_n + _SCAN_POLICY.smem_pad) * elem_size
        max_block_m = STATIC_SHARED_BYTES // smem_per_row

        if self.M < 128:
            block_m = max(1, min(self.M, min(2, max_block_m)))
        else:
            block_m = 1
            for bm in [1, 2, 4, 8, 16]:
                if bm <= max_block_m:
                    block_m = bm

        return {"block_m": block_m, "block_n": block_n, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        elem_size = torch_dtype_nbytes(self.dtype)
        configs = []
        for block_n in [128, 256]:
            # block_n must evenly divide N_padded
            if self.N_padded % block_n != 0:
                continue
            # Account for padding in shared memory budget calculation
            smem_per_row = 2 * (block_n + _SCAN_POLICY.smem_pad) * elem_size
            max_block_m = STATIC_SHARED_BYTES // smem_per_row
            block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
            threads_list = [128, 256]
            for bm, t in itertools.product(block_ms, threads_list):
                configs.append({"block_m": bm, "block_n": block_n, "threads": t})
        return configs

    def _scan_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(self.config["block_m"], self.config["block_n"], self.config["threads"])(
            x
        )


# ---------------------------------------------------------------------------
# Parallel scan kernels for small-M, large-N workloads
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _parallel_scan_local_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Pass 1: scan each tile independently, emitting its total to ``tile_sums``.

    Grid is (ceildiv(M, block_m), n_tiles), so all tiles scan concurrently.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _identity = 0.0 if op_kind == "sum" else 1.0

    @tilelang.jit(out_idx=[1, 2])
    def _func(block_m, block_n, threads):
        n_tiles = N_padded // block_n
        _needs_mask = (n_tiles * block_n) > N

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y_local: T.Tensor[(M, N_padded), "float32"],  # noqa: F821
            tile_sums: T.Tensor[(M, n_tiles), "float32"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, block_m), n_tiles, threads=threads) as (pid_m, tile_idx):
                tile_shared = T.alloc_shared((block_m, block_n + _SCAN_POLICY.smem_pad), "float32")
                tile_frag = T.alloc_fragment((block_m, block_n), "float32")

                for i, j in T.Parallel(block_m, block_n):
                    row = pid_m * block_m + i
                    col = tile_idx * block_n + j
                    in_bounds = T.And(row < M, col < N) if _needs_mask else row < M
                    tile_shared[i, j] = T.if_then_else(
                        in_bounds,
                        T.cast(x[row, col], "float32"),
                        T.cast(_identity, "float32"),
                    )
                T.sync_threads()

                for i, j in T.Parallel(block_m, block_n):
                    tile_frag[i, j] = tile_shared[i, j]
                T.sync_threads()

                T.cumsum(tile_frag, dim=1)
                T.sync_threads()

                # Back through shared memory: the tile_sums writer below reads
                # column block_n - 1, which lives in another thread's registers.
                for i, j in T.Parallel(block_m, block_n):
                    tile_shared[i, j] = tile_frag[i, j]
                T.sync_threads()

                for i, j in T.Parallel(block_m, block_n):
                    row = pid_m * block_m + i
                    col = tile_idx * block_n + j
                    with T.If(row < M), T.Then():
                        y_local[row, col] = tile_shared[i, j]

                for i in T.Parallel(block_m):
                    row = pid_m * block_m + i
                    with T.If(row < M), T.Then():
                        tile_sums[row, tile_idx] = tile_shared[i, block_n - 1]

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _parallel_scan_carry_kernel(M: int, n_tiles: int):
    """Pass 2: exclusive scan of the per-tile totals.

    ``out[i, 0] = 0``, ``out[i, j] = sum(in[i, :j])``.  Scanned serially since
    ``n_tiles`` is small (128-256); one thread owns one row, so the writes
    cannot race.
    """

    @tilelang.jit(out_idx=[1])
    def _func(threads):
        @T.prim_func
        def main(
            tile_sums: T.Tensor[(M, n_tiles), "float32"],  # noqa: F821
            tile_carries: T.Tensor[(M, n_tiles), "float32"],  # noqa: F821
        ):
            with T.Kernel(T.ceildiv(M, threads), threads=threads) as pid:
                tx = T.get_thread_binding()
                row = pid * threads + tx

                with T.If(row < M), T.Then():
                    tile_carries[row, 0] = T.float32(0.0)
                    running_sum = T.alloc_var("float32", init=0.0)
                    for j in T.Serial(n_tiles - 1):
                        running_sum = running_sum + tile_sums[row, j]
                        tile_carries[row, j + 1] = running_sum

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _parallel_scan_propagate_kernel(M: int, N: int, dtype: str):
    """Pass 3: add each tile's carry and cast to the output dtype."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)

    @tilelang.jit(out_idx=[2])
    def _func(block_m, block_n, threads):
        n_tiles = N_padded // block_n

        @T.prim_func
        def main(
            y_local: T.Tensor[(M, N_padded), "float32"],  # noqa: F821
            tile_carries: T.Tensor[(M, n_tiles), "float32"],  # noqa: F821
            y_final: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), n_tiles, threads=threads) as (pid_m, tile_idx):
                for i, j in T.Parallel(block_m, block_n):
                    row = pid_m * block_m + i
                    col = tile_idx * block_n + j
                    with T.If(row < M), T.Then():
                        y_final[row, col] = T.cast(
                            y_local[row, col] + tile_carries[row, tile_idx], dtype
                        )

        return main

    return _func
