"""Cumulative scan kernels (cumsum, cumprod) using TileLang."""

import functools
import itertools
import math
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.constants import (
    SHARED_BANK_SPAN_BYTES,
    STATIC_SHARED_BYTES,
    VECTOR_ACCESS_BYTES,
)
from tileops._kernels.kernel_base import CompiledProgram, Kernel
from tileops._kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    restore_same_shape,
    rows_for_axes,
    torch_dtype_nbytes,
)
from tileops._kernels.reduction.call_spec import (
    CumprodFwdInterface,
    CumsumFwdInterface,
    CumulativeCall,
)
from tileops._kernels.tiling import align_up
from tileops._utils import WARP_LANES

__all__ = ["CumsumParallelScanKernel", "CumulativeKernel", "CumulativeRowScanKernel"]


def _row_scan_pad(chunk: int, elem_bytes: int) -> int:
    """Return the padding, in elements, each staged chunk gets.

    Neighbouring lanes read one chunk each, so they sit ``(chunk + pad) * elem_bytes``
    apart, and the conflict ways grow with what that stride shares with
    ``SHARED_BANK_SPAN_BYTES``. This returns the candidate pad that shares least, which
    depends on *chunk*: a fixed pad leaves some chunks striding the whole span.
    """
    # One or two vector accesses: a whole access keeps a chunk 16-byte aligned, which the
    # shared access needs to stay 128-bit.
    candidates = [k * VECTOR_ACCESS_BYTES // elem_bytes for k in (1, 2)]
    return min(
        candidates,
        key=lambda pad: (math.gcd((chunk + pad) * elem_bytes, SHARED_BANK_SPAN_BYTES), pad),
    )


def _row_scan_chunk_ok(chunk: int, elem_bytes: int, out_bytes: int, threads: int) -> bool:
    """Whether a block of *threads* may give each thread a chunk of *chunk* elements.

    Up to 64 elements and under 256 bytes of output (*out_bytes* each) always. A block
    already 256 threads wide lengthens the chunk rather than adding warps, as far as 128
    elements, the longest fp32 chunk the registers hold without spilling; only with a
    chunk of whole vector accesses of the narrower dtype (*elem_bytes*), which is what
    keeps the chunk 16-byte aligned.
    """
    if chunk <= 64 and chunk * out_bytes < 256:
        return True
    return threads >= 256 and chunk <= 128 and (chunk * elem_bytes) % VECTOR_ACCESS_BYTES == 0


def _row_scan_threads(N_padded: int, elem_bytes: int, out_bytes: int) -> int:
    """Threads the whole-row kernel gives a row of *N_padded*.

    The narrowest block that divides the row and whose chunk :func:`_row_scan_chunk_ok`
    accepts, or the widest divisor tried when no block qualifies -- which
    :func:`row_scan_fits` then declines.
    """
    threads = 64
    widest = threads
    while threads <= 1024:
        if N_padded % threads == 0:
            widest = threads
            if _row_scan_chunk_ok(N_padded // threads, elem_bytes, out_bytes, threads):
                return threads
        threads *= 2
    return widest


def _chunk_vec(chunk_len: int, dtype: str) -> int:
    """The widest access, up to 16 bytes, that divides a chunk of *dtype* elements.

    The staging pad is whole vectors, so every chunk then starts on one.
    """
    vec = VECTOR_ACCESS_BYTES // torch_dtype_nbytes(dtype)
    while chunk_len % vec:
        vec //= 2
    return vec


@functools.lru_cache(maxsize=32)
def _row_scan_kernel(
    M: int,
    N: int,
    op_kind: str,
    dtype: str,
    out_dtype: str,
    threads: int,
    chunk_len: Optional[int] = None,
):
    """Build an inclusive prefix scan of each row, one block a segment of it.

    A segment is ``threads * chunk_len`` columns; without *chunk_len* it is the whole row.
    Three phases, with no serial dependency between blocks:

    1. The segment is staged in shared memory at its own dtype, one padded row of the tile
       per thread chunk, and each thread scans its chunk into registers in fp32.
    2. The chunk totals are scanned with shuffles inside each warp, and one barrier
       carries each warp's total to the warps above it.
    3. Each thread adds the prefix ahead of its chunk and writes the chunk back to a
       staging tile at the output dtype, which returns it to global memory.

    A row of several segments is a cumsum: the block also takes the segment totals of
    :func:`_segment_totals_kernel` and adds those of the segments before its own to every
    chunk. Columns past *N* of the last segment read as zero and are not written. The
    segments of a row lie on the grid's first dimension, as in that kernel.

    Both transfers between global and the tile run over the segment in column-major thread
    order, so consecutive threads read consecutive columns and the access coalesces; the
    tile's padding is what keeps a thread's own chunk off one bank. Staging at the input
    dtype halves the shared traffic against fp32, and the write-back goes through a tile
    because a scalar global store per element does not coalesce; it reuses the input's
    tile when the two dtypes agree.

    Args:
        M: Rows to scan.
        N: Length of the scanned axis. Without *chunk_len* it must equal the padded width
            the caller allocated for, and be divisible by *threads*.
        op_kind: One of "sum", "prod"; "sum" when the row has several segments.
        dtype: TileLang dtype string of the input.
        out_dtype: TileLang dtype string of the output.
        threads: Threads a block: :func:`_row_scan_threads` for a whole row.
        chunk_len: Elements a thread scans, for a row cut into segments.
    """
    if chunk_len is None:
        chunk_len = N // threads
    seg_len = threads * chunk_len
    n_seg = -(-N // seg_len)
    ragged = n_seg * seg_len > N
    assert n_seg == 1 or op_kind == "sum", "only a cumsum carries segment totals"
    # The totals ahead are summed over a fragment, and a fragment's layout gives every
    # thread the same number of its elements.
    n_ahead = -(-n_seg // threads) * threads
    pad = _row_scan_pad(chunk_len, torch_dtype_nbytes(dtype))
    out_pad = _row_scan_pad(chunk_len, torch_dtype_nbytes(out_dtype))
    vec = _chunk_vec(chunk_len, dtype)
    out_vec = _chunk_vec(chunk_len, out_dtype)
    # Shuffle steps to scan one warp's lanes, and the warps a block holds.
    n_steps = WARP_LANES.bit_length() - 1
    n_warps = max(threads // WARP_LANES, 1)
    identity = 0.0 if op_kind == "sum" else 1.0
    combine = (lambda a, b: a + b) if op_kind == "sum" else (lambda a, b: a * b)

    @tilelang.jit(out_idx=[1] if n_seg == 1 else [2])
    def _func():
        @T.macro
        def scan_segment(x, seg_totals, ahead_vals, carry, y, row, seg):
            tx = T.get_thread_binding()
            staged = T.alloc_shared((threads, chunk_len + pad), dtype)
            leaving = (
                staged
                if out_dtype == dtype
                else T.alloc_shared((threads, chunk_len + out_pad), out_dtype)
            )
            totals = T.alloc_shared((n_warps,), "float32")
            chunk = T.alloc_local((chunk_len,), "float32")
            piece = T.alloc_local((vec,), dtype)
            piece_out = T.alloc_local((out_vec,), out_dtype)
            running = T.alloc_local((1,), "float32")
            ahead = T.alloc_local((1,), "float32")
            offset = T.alloc_local((1,), "float32")
            before = T.alloc_local((1,), "float32")
            base = seg * seg_len

            for i, j in T.Parallel(threads, chunk_len):
                if ragged:
                    staged[i, j] = T.if_then_else(
                        base + i * chunk_len + j < N,
                        x[row, base + i * chunk_len + j],
                        T.cast(identity, dtype),
                    )
                else:
                    staged[i, j] = x[row, base + i * chunk_len + j]
            T.sync_threads()

            # A thread reads and writes its own chunk a vector at a time, up to 16
            # bytes an access instead of one element.
            running[0] = T.cast(identity, "float32")
            for v in T.unroll(chunk_len // vec):
                for i in T.vectorized(vec):
                    piece[i] = staged[tx, v * vec + i]
                for i in T.unroll(vec):
                    running[0] = combine(running[0], T.cast(piece[i], "float32"))
                    chunk[v * vec + i] = running[0]

            for step in T.serial(n_steps):
                stride = T.shift_left(T.int32(1), step)
                ahead[0] = T.shfl_up(running[0], stride)
                running[0] = T.if_then_else(
                    tx % WARP_LANES >= stride, combine(running[0], ahead[0]), running[0]
                )
            if tx % WARP_LANES == WARP_LANES - 1:
                totals[tx // WARP_LANES] = running[0]
            if n_seg > 1:
                for k in T.Parallel(n_ahead):
                    ahead_vals[k] = T.if_then_else(
                        k < seg, seg_totals[row, T.min(k, n_seg - 1)], 0.0
                    )
                T.reduce_sum(ahead_vals, carry, dim=0)
            T.sync_threads()

            # One more shuffle turns the inclusive scan into the exclusive one a
            # chunk needs. Subtracting the thread's own total would do it for sum
            # and not for prod; a shuffle holds for any combine.
            before[0] = T.shfl_up(running[0], 1)
            before[0] = T.if_then_else(tx % WARP_LANES == 0, T.cast(identity, "float32"), before[0])
            if n_seg > 1:
                ahead[0] = carry[0]
            else:
                ahead[0] = T.cast(identity, "float32")
            for w in T.serial(n_warps):
                ahead[0] = T.if_then_else(
                    w < tx // WARP_LANES, combine(ahead[0], totals[w]), ahead[0]
                )
            offset[0] = combine(before[0], ahead[0])
            for v in T.unroll(chunk_len // out_vec):
                for i in T.unroll(out_vec):
                    piece_out[i] = T.cast(combine(chunk[v * out_vec + i], offset[0]), out_dtype)
                for i in T.vectorized(out_vec):
                    leaving[tx, v * out_vec + i] = piece_out[i]
            T.sync_threads()
            for i, j in T.Parallel(threads, chunk_len):
                if ragged:
                    if base + i * chunk_len + j < N:
                        y[row, base + i * chunk_len + j] = leaving[i, j]
                else:
                    y[row, base + i * chunk_len + j] = leaving[i, j]

        if n_seg == 1:

            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                y: T.Tensor[(M, N), out_dtype],
            ):
                with T.Kernel(M, threads=threads) as row:
                    scan_segment(x, None, None, None, y, row, 0)

            return main

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            seg_totals: T.Tensor[(M, n_seg), "float32"],  # noqa: F821
            y: T.Tensor[(M, N), out_dtype],
        ):
            with T.Kernel(n_seg, M, threads=threads) as (seg, row):
                ahead_vals = T.alloc_fragment((n_ahead,), "float32")
                carry = T.alloc_fragment((1,), "float32")
                scan_segment(x, seg_totals, ahead_vals, carry, y, row, seg)

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _segment_totals_kernel(M: int, N: int, dtype: str, threads: int, chunk_len: int):
    """Sum each segment of ``threads * chunk_len`` columns of every row into ``totals``.

    One block a segment; columns past *N* read as zero. The segments of a row lie on the
    grid's first dimension, which takes far more blocks than the 65535 of the second.
    """
    seg_len = threads * chunk_len
    n_seg = -(-N // seg_len)
    ragged = n_seg * seg_len > N

    @tilelang.jit(out_idx=[1])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            totals: T.Tensor[(M, n_seg), "float32"],  # noqa: F821
        ):
            with T.Kernel(n_seg, M, threads=threads) as (seg, row):
                vals = T.alloc_fragment((seg_len,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(seg_len):
                    col = seg * seg_len + j
                    if ragged:
                        vals[j] = T.if_then_else(col < N, T.cast(x[row, col], "float32"), 0.0)
                    else:
                        vals[j] = T.cast(x[row, col], "float32")
                T.reduce_sum(vals, total, dim=0)
                for i in T.Parallel(1):
                    totals[row, seg] = total[i]

        return main

    return _func


@functools.lru_cache(maxsize=32)
def _cumulative_kernel(M: int, N: int, op_kind: str, dtype: str, out_dtype: str, smem_pad: int):
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
        dtype: TileLang dtype string of the input (e.g. "float16", "bfloat16", "float32").
        out_dtype: TileLang dtype string of the output.

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
                y: T.Tensor[(M, N_padded), out_dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_in = T.alloc_shared((block_m, block_n + smem_pad), dtype)
                    shared_out = T.alloc_shared((block_m, block_n + smem_pad), out_dtype)
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
                            shared_out[i, j] = T.cast(out_f32[i, j], out_dtype)
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
                y: T.Tensor[(M, N_padded), out_dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_in = T.alloc_shared((block_m, block_n + smem_pad), dtype)
                    shared_out = T.alloc_shared((block_m, block_n + smem_pad), out_dtype)
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
                            shared_out[i, j] = T.cast(out_f32[i, j], out_dtype)
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
    def row_scan_fits(
        N_padded: int, dtype: torch.dtype, out_dtype: torch.dtype, smem_budget: int
    ) -> bool:
        """Whether a row of *N_padded* can be scanned by one thread block."""
        out_bytes = torch_dtype_nbytes(out_dtype)
        narrow = min(torch_dtype_nbytes(dtype), out_bytes)
        threads = _row_scan_threads(N_padded, narrow, out_bytes)
        if N_padded % threads:
            return False
        chunk = N_padded // threads
        # The input's staging tile and the output's never live at once, so they share
        # one allocation.
        staged = max(
            threads * (chunk + _row_scan_pad(chunk, torch_dtype_nbytes(t))) * torch_dtype_nbytes(t)
            for t in (dtype, out_dtype)
        )
        return staged <= smem_budget and _row_scan_chunk_ok(chunk, narrow, out_bytes, threads)

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def stages_whole_row(cls, call: CumulativeCall) -> bool:
        """Whether one thread block stages the whole row, aligned, in shared memory."""
        n_padded = align_up(call.n, DEFAULT_ALIGNMENT)
        return n_padded == call.n and cls.row_scan_fits(
            n_padded, call.dtype, call.out_dtype, call.smem_budget
        )

    @classmethod
    def segments_row(cls, call: CumulativeCall) -> bool:
        """Whether a cumsum row goes to the segmented scan though one block could stage it.

        It does when the row moves at least 64 KiB in the wider of its two dtypes and there
        are no more rows than 4 KiB parts of one row: one block a row then leaves most SMs
        idle while each walks a long chunk, and spreading the row over segments pays for
        the segmented scan's second launch.
        """
        row_bytes = call.n * max(torch_dtype_nbytes(call.dtype), torch_dtype_nbytes(call.out_dtype))
        return call.op_kind == "sum" and row_bytes >= 64 * 1024 and call.m * 4096 <= row_bytes

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
        self.out_dtype_str = self.dtype_to_str(call.out_dtype)
        self.scan_axis = call.axis
        self.N_padded = align_up(self.N, DEFAULT_ALIGNMENT)
        self._elem_bytes = torch_dtype_nbytes(self.dtype)
        self._out_elem_bytes = torch_dtype_nbytes(call.out_dtype)
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
        self._require_devices(x=x)
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
    def refusal(cls, call: CumulativeCall) -> "str | None":
        if not cls.stages_whole_row(call):
            return f"one block does not stage a row of {call.n}"
        if cls.segments_row(call):
            return "the segmented scan serves this few long rows"
        return super().refusal(call)

    def _program(self) -> object:
        # The chunk length is a compile-time bound, so the thread count is baked in.
        self._threads = _row_scan_threads(
            self.N_padded, min(self._elem_bytes, self._out_elem_bytes), self._out_elem_bytes
        )
        return _row_scan_kernel(
            self.M, self.N, self.op_kind, self.dtype_str, self.out_dtype_str, self._threads
        )

    @property
    def default_config(self) -> dict:
        return {"threads": self._threads}

    def _scan_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.program(self._compile)(x)

    def _compile(self, cfg: dict):
        """The program."""
        return self.kernel()


class CumsumParallelScanKernel(_CumulativeKernelBase, CumsumFwdInterface):
    """Cumsum of a few long rows cut into segments, in two launches: every segment's total,
    then every segment's scan seeded with the totals ahead of it in its row. Serves
    ``m < 128`` rows longer than 8192 that one block cannot stage, and the few long rows
    :meth:`segments_row` takes from the whole-row scan.

    A block is 128 threads; the chunk a thread scans is the longest of 16, 8 and 4 elements
    whose segments still give every SM three blocks, and 4 when none does: a long segment
    amortizes the block's fixed steps, a short one spreads few rows over the SMs.
    """

    def __init__(self, call: CumulativeCall, config: Optional[dict] = None):
        self._sm_count = call.sm_count
        super().__init__(call, config)

    @classmethod
    def refusal(cls, call: CumulativeCall) -> "str | None":
        staged = cls.stages_whole_row(call) and not cls.segments_row(call)
        if call.m >= 128 or call.n <= 8192 or staged:
            return "serves fewer than 128 rows longer than 8192 the whole-row scan does not take"
        return super().refusal(call)

    def _program(self) -> None:
        return None

    @property
    def default_config(self) -> dict:
        # Small blocks keep more of them resident, which spreads a few rows over the SMs.
        threads = 128
        for chunk in (16, 8, 4):
            if self.M * -(-self.N // (threads * chunk)) >= 3 * self._sm_count:
                break
        return {"threads": threads, "chunk": chunk}

    def _scan_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.program(self._launcher)(x)

    def _launcher(self, cfg: dict) -> Callable:
        """The two launches *cfg* sizes: the segment totals, then the seeded segment scans."""
        threads, chunk = cfg["threads"], cfg["chunk"]
        totals = _segment_totals_kernel(self.M, self.N, self.dtype_str, threads, chunk)
        totals = CompiledProgram(totals())
        scan = _row_scan_kernel(
            self.M, self.N, "sum", self.dtype_str, self.out_dtype_str, threads, chunk
        )
        scan = CompiledProgram(scan())

        def launch(x: torch.Tensor) -> torch.Tensor:
            return scan(x, totals(x))

        return launch


class CumulativeKernel(_CumulativeKernelBase, CumsumFwdInterface, CumprodFwdInterface):
    """The general scan: ``block_m`` rows a block, each walked serially over ``block_n``
    tiles staged through shared memory with the running accumulator carried across."""

    general = True
    # Columns padding each row of a shared tile against the bank conflicts of 16-bit staging;
    # the tiles and the budgets that size them must agree on it.
    _SMEM_PAD = 8

    def _program(self) -> object:
        return _cumulative_kernel(
            self.M, self.N, self.op_kind, self.dtype_str, self.out_dtype_str, self._SMEM_PAD
        )

    @property
    def default_config(self) -> dict:
        # A multiple of DEFAULT_ALIGNMENT, so T.copy keeps the shared tile aligned.
        block_n = 128
        # The input's and the output's shared tile.
        smem_per_row = (block_n + self._SMEM_PAD) * (self._elem_bytes + self._out_elem_bytes)
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
        configs = []
        for block_n in [128, 256]:
            # block_n must evenly divide N_padded
            if self.N_padded % block_n != 0:
                continue
            # Account for padding in shared memory budget calculation
            smem_per_row = (block_n + self._SMEM_PAD) * (self._elem_bytes + self._out_elem_bytes)
            max_block_m = STATIC_SHARED_BYTES // smem_per_row
            block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
            threads_list = [128, 256]
            for bm, t in itertools.product(block_ms, threads_list):
                configs.append({"block_m": bm, "block_n": block_n, "threads": t})
        return configs

    def _scan_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.program(self._compile)(x)

    def _compile(self, cfg: dict):
        """The program *cfg* tiles."""
        return self.kernel(cfg["block_m"], cfg["block_n"], cfg["threads"])
