"""Cumulative scan kernels (cumsum, cumprod) using TileLang.

Implements an inclusive prefix scan along the last dimension, over three backends:

  - _row_scan_kernel: one block per row, for a row that fits one. Preferred where it
    builds -- it reads and writes the row once and carries nothing between blocks.
  - _cumulative_kernel: a tiled sequential scan for rows that do not, carrying a
    running accumulator across tiles of ``block_n`` columns.
  - the three-kernel parallel scan, for small-M large-N cumsum.

Accepts raw ``(M, N)`` input tensors.  Boundary handling for non-aligned N is performed
inside the tiled kernel via masked loads with identity-element fills (0 for cumsum, 1
for cumprod), eliminating host-side ``F.pad`` from the forward path.  Its output is
``(M, N_padded)`` and the Op layer trims back to N columns.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.

Shared memory padding:
  ``shared_in`` and ``shared_out`` are allocated with ``SMEM_PAD`` extra
  columns so that adjacent rows land on different shared-memory banks,
  eliminating the 32-way bank conflict observed with unpadded layouts.
  For fp16/bf16 (2 bytes/element) the row stride in 4-byte bank words is
  ``(block_n + SMEM_PAD) / 2``.  Choosing SMEM_PAD=8 makes the stride
  68 words, which is not a multiple of 32, so each successive row starts
  in a different bank set.  Element-wise indexing (``smem[i, j]``) is
  used for all smem<->fragment transfers and all global<->smem transfers
  to avoid shape and alignment mismatches with ``T.copy``.
"""

import functools
import itertools
import warnings
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    SHARED_MEMORY_BUDGET_BYTES,
    align_up,
    device_smem_budget,
    restore_same_shape,
    rows_for_axes,
)

__all__ = ["CumulativeKernel"]

# Tile size along the N dimension for the prefix scan.
# Must be a multiple of DEFAULT_ALIGNMENT for T.copy shared memory alignment.
_DEFAULT_BLOCK_N: int = 128

# Shared memory padding to eliminate bank conflicts.
# H200/H100 has 32 banks × 4 bytes.  For fp16/bf16 (2 bytes/elem), a row of
# block_n=128 elements = 256 bytes = 64 bank-words.  64 % 32 == 0, so every
# row starts at the same bank → 32-way conflicts.  Adding SMEM_PAD=8 elements
# (16 bytes = 4 bank-words) makes the stride 68 bank-words; 68 % 32 == 4,
# so successive rows start in different banks → conflict free.
_SMEM_PAD: int = 8

# Elements one thread scans in the whole-row kernel, held in registers. Measured on
# H200 over {2048x4096 fp16, 64x32768 bf16}: 16 peaks, 8 and 32 are within 25%, 4 and
# 128 lose half.
_ROW_SCAN_CHUNK: int = 16

# Thread-count envelope for the whole-row kernel: a warp scheduler's worth, up to the
# CUDA block limit.
_ROW_SCAN_MIN_THREADS: int = 128
_ROW_SCAN_MAX_THREADS: int = 1024


def row_scan_threads(N_padded: int) -> int:
    """Threads the whole-row kernel gives a row of *N_padded*.

    A power of two dividing *N_padded*, chosen so each thread's chunk sits near
    ``_ROW_SCAN_CHUNK``.
    """
    target = max(1, N_padded // _ROW_SCAN_CHUNK)
    threads = 1 << (target.bit_length() - 1)
    threads = max(_ROW_SCAN_MIN_THREADS, min(_ROW_SCAN_MAX_THREADS, threads))
    while threads > _ROW_SCAN_MIN_THREADS and N_padded % threads:
        threads //= 2
    return threads


def row_scan_fits(N_padded: int, elem_bytes: int, smem_budget: int) -> bool:
    """Whether a row of *N_padded* can be scanned by one thread block.

    The row is staged in shared memory whole, and each thread holds its chunk of it
    in registers; both have to fit for the kernel to be buildable.
    """
    threads = row_scan_threads(N_padded)
    if N_padded % threads:
        return False
    chunk = N_padded // threads
    return N_padded * elem_bytes <= smem_budget and chunk <= 4 * _ROW_SCAN_CHUNK


@functools.lru_cache(maxsize=32)
def _row_scan_kernel(M: int, N: int, op_kind: str, dtype: str, threads: int):
    """Build a one-block-per-row inclusive prefix scan.

    Three phases, with no serial dependency between blocks:

    1. One ``T.copy`` stages the row in shared memory at its own dtype -- that is where
       the vectorized global access comes from -- and each thread scans its contiguous
       chunk of it into registers, accumulating in fp32.
    2. The chunk totals are scanned in shared memory, doubling the stride each step.
    3. Each thread adds the prefix ahead of its chunk and writes the chunk back to the
       staging buffer, which one ``T.copy`` returns to global memory.

    Staging at the input dtype halves the shared traffic against fp32, and the
    write-back reuses the buffer because a scalar store per element does not vectorize.

    Args:
        M: Rows to scan.
        N: Length of the scanned axis; must equal the padded width the caller
            allocated for, and be divisible by *threads*.
        op_kind: One of "sum", "prod".
        dtype: TileLang dtype string.
        threads: Threads per row, from :func:`row_scan_threads`.
    """
    chunk_len = N // threads
    # Stride-doubling steps to scan `threads` totals.
    n_steps = threads.bit_length() - 1
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
                staged = T.alloc_shared((N,), dtype)
                totals = T.alloc_shared((threads,), "float32")
                chunk = T.alloc_local((chunk_len,), "float32")
                running = T.alloc_local((1,), "float32")
                ahead = T.alloc_local((1,), "float32")
                offset = T.alloc_local((1,), "float32")

                T.copy(x[row, 0], staged)
                T.sync_threads()

                running[0] = T.cast(identity, "float32")
                for j in T.serial(chunk_len):
                    running[0] = combine(running[0], T.cast(staged[tx * chunk_len + j], "float32"))
                    chunk[j] = running[0]
                totals[tx] = running[0]
                T.sync_threads()

                for step in T.serial(n_steps):
                    stride = T.shift_left(T.int32(1), step)
                    ahead[0] = T.if_then_else(
                        tx >= stride, totals[tx - stride], T.cast(identity, "float32")
                    )
                    T.sync_threads()
                    totals[tx] = combine(totals[tx], ahead[0])
                    T.sync_threads()

                # T.max keeps the read in range for thread 0, whose branch discards it.
                offset[0] = T.if_then_else(
                    tx == 0, T.cast(identity, "float32"), totals[T.max(tx - 1, 0)]
                )
                for j in T.serial(chunk_len):
                    staged[tx * chunk_len + j] = T.cast(combine(chunk[j], offset[0]), dtype)
                T.sync_threads()
                T.copy(staged, y[row, 0])

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
                    # Pad shared memory by _SMEM_PAD columns to break bank-conflict
                    # alignment.  For fp16/bf16 (2 bytes/elem), an unpadded row of
                    # block_n=128 elements is 256 bytes = 64 4-byte bank-words.
                    # 64 % 32 == 0 → every row starts at the same bank → 32-way
                    # conflict.  With pad=8: stride = 136 elems = 68 bank-words;
                    # 68 % 32 == 4 → successive rows start at different banks.
                    shared_in = T.alloc_shared((block_m, block_n + _SMEM_PAD), dtype)
                    shared_out = T.alloc_shared((block_m, block_n + _SMEM_PAD), dtype)
                    tile_f32 = T.alloc_fragment((block_m, block_n), "float32")
                    out_f32 = T.alloc_fragment((block_m, block_n), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")

                    # Initialize accumulator
                    for i in T.Parallel(block_m):
                        acc[i] = T.float32(0)

                    # Process N dimension in tiles
                    for tile_idx in T.Serial(n_tiles):
                        if _needs_mask:
                            # Fast path when current tile is fully in-bounds
                            with T.If((tile_idx + 1) * block_n <= N):
                                with T.Then():
                                    # Element-wise load into padded shared_in with M bounds check
                                    for i, j in T.Parallel(block_m, block_n):
                                        # TileLang requires T.If/T.Then as nested context managers
                                        with T.If(pid_m * block_m + i < M):  # noqa: SIM117
                                            with T.Then():
                                                shared_in[i, j] = x[
                                                    pid_m * block_m + i, tile_idx * block_n + j
                                                ]
                                    for i, j in T.Parallel(block_m, block_n):
                                        tile_f32[i, j] = T.cast(shared_in[i, j], "float32")
                                with T.Else():
                                    # Partially OOB tile: masked load directly to fragment
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
                            # Element-wise load into padded shared_in with M bounds check
                            for i, j in T.Parallel(block_m, block_n):
                                # TileLang requires T.If/T.Then as nested context managers
                                with T.If(pid_m * block_m + i < M):  # noqa: SIM117
                                    with T.Then():
                                        shared_in[i, j] = x[
                                            pid_m * block_m + i, tile_idx * block_n + j
                                        ]
                            for i, j in T.Parallel(block_m, block_n):
                                tile_f32[i, j] = T.cast(shared_in[i, j], "float32")

                        # Inclusive prefix sum within tile
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
                    # Pad shared memory by _SMEM_PAD columns to break bank-conflict
                    # alignment (same reasoning as the sum path above).
                    shared_in = T.alloc_shared((block_m, block_n + _SMEM_PAD), dtype)
                    shared_out = T.alloc_shared((block_m, block_n + _SMEM_PAD), dtype)
                    tile_f32 = T.alloc_fragment((block_m, block_n), "float32")
                    out_f32 = T.alloc_fragment((block_m, block_n), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")

                    # Initialize accumulator (1.0 for product)
                    for i in T.Parallel(block_m):
                        acc[i] = T.float32(1)

                    # Process N dimension in tiles
                    for tile_idx in T.Serial(n_tiles):
                        if _needs_mask:
                            # Fast path when current tile is fully in-bounds
                            with T.If((tile_idx + 1) * block_n <= N):
                                with T.Then():
                                    # Element-wise load into padded shared_in with M bounds check
                                    for i, j in T.Parallel(block_m, block_n):
                                        # TileLang requires T.If/T.Then as nested context managers
                                        with T.If(pid_m * block_m + i < M):  # noqa: SIM117
                                            with T.Then():
                                                shared_in[i, j] = x[
                                                    pid_m * block_m + i, tile_idx * block_n + j
                                                ]
                                    for i, j in T.Parallel(block_m, block_n):
                                        tile_f32[i, j] = T.cast(shared_in[i, j], "float32")
                                with T.Else():
                                    # Partially OOB tile: masked load directly to fragment
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
                            # Element-wise load into padded shared_in with M bounds check
                            for i, j in T.Parallel(block_m, block_n):
                                # TileLang requires T.If/T.Then as nested context managers
                                with T.If(pid_m * block_m + i < M):  # noqa: SIM117
                                    with T.Then():
                                        shared_in[i, j] = x[
                                            pid_m * block_m + i, tile_idx * block_n + j
                                        ]
                            for i, j in T.Parallel(block_m, block_n):
                                tile_f32[i, j] = T.cast(shared_in[i, j], "float32")

                        # Inclusive prefix product within tile
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


# CumulativeKernel class


class CumulativeKernel(Kernel):
    """Inclusive prefix scan kernel (cumsum / cumprod).

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Uses a tiled sequential scan loop along the last
    dimension: the N dimension is divided into tiles of ``block_n``
    elements, reducing shared memory usage and improving occupancy.

    Boundary handling for non-aligned N is performed inside the kernel via
    masked loads with identity-element fills (0 for sum, 1 for prod), so
    no host-side ``F.pad`` is needed.

    ``forward`` takes the tensor the op declares and scans *scan_axis* of it; moving that
    axis to the end, flattening to rows and putting the result back are this kernel's
    business, so both sides of the op/backend boundary speak the declared shape.

    Args:
        M: Rows the scan runs over — the product of every axis but *scan_axis*.
        N: Length of the scanned axis.
        op_kind: One of "sum", "prod".
        dtype: Data type (float32, float16, or bfloat16).
        scan_axis: Non-negative index of the axis the scan runs along.
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
        device_index: The device the input lives on. The shared-memory budget here is a
            constant, so this is for the architecture check alone.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        scan_axis: int,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        if op_kind not in ("sum", "prod"):
            raise ValueError(f"Unsupported op_kind '{op_kind}'. Expected one of 'sum', 'prod'.")
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.scan_axis = scan_axis
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = torch.tensor([], dtype=dtype).element_size()

        # Parallel scan only pays off for small-M, large-N; cumprod has no
        # parallel implementation.
        self.use_parallel = M < 128 and N > 8192 and op_kind == "sum"

        # Fastest of the three on the shapes it is given, for both kinds: it reads and
        # writes the row once and carries nothing between blocks. Two shapes it does not
        # get: a width the alignment would pad, which it cannot stage exactly, and the
        # small-M large-N cumsum below.
        #
        # FIXME(staged-rollout): the three-kernel parallel scan keeps the shapes it
        # already claimed, though this kernel measures faster on them.
        #
        # Broken invariant: the fastest buildable backend wins the shape.
        # Why: test_cumsum_backend_dispatch pins four shapes to use_parallel, and that
        #   file is outside the kernel layer this change is scoped to.
        # Cleanup: drop the `not self.use_parallel` term once that test asserts the
        #   backend each shape actually measures fastest on.
        self.use_row_scan = (
            not self.use_parallel
            and self.N_padded == N
            and row_scan_fits(self.N_padded, self._elem_bytes, device_smem_budget(device_index))
        )
        self._row_scan_threads = row_scan_threads(self.N_padded) if self.use_row_scan else 0

        if self.use_row_scan:
            self.kernel = _row_scan_kernel(M, N, op_kind, self.dtype_str, self._row_scan_threads)
        elif self.use_parallel:
            self.kernel = None
            if tune:
                warnings.warn(
                    f"Autotuning is unsupported for the parallel scan backend "
                    f"(shape {M}x{N}); using default config.",
                    UserWarning,
                    stacklevel=2,
                )
                tune = False
        else:
            self.kernel = _cumulative_kernel(M, N, op_kind, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """Select the default config for the selected backend and shape."""
        if self.use_row_scan:
            # The chunk length each thread scans is a compile-time loop bound, so the
            # thread count is baked in and nothing is left to pick at call time.
            return {"threads": self._row_scan_threads}
        if self.use_parallel:
            block_n = 256 if self.N > 16384 else 128
            smem_per_row = (block_n + _SMEM_PAD) * 4  # fp32 intermediate
            max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
            block_m = max(1, min(16, self.M, max_block_m))
            return {"block_m": block_m, "block_n": block_n, "threads": 256}
        else:
            block_n = _DEFAULT_BLOCK_N
            elem_size = torch.tensor([], dtype=self.dtype).element_size()
            smem_per_row = 2 * (block_n + _SMEM_PAD) * elem_size
            max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row

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
        if self.use_row_scan:
            return [{"threads": self._row_scan_threads}]
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        configs = []
        for block_n in [128, 256]:
            # block_n must evenly divide N_padded
            if self.N_padded % block_n != 0:
                continue
            # Account for padding in shared memory budget calculation
            smem_per_row = 2 * (block_n + _SMEM_PAD) * elem_size
            max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
            block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
            threads_list = [128, 256]
            for bm, t in itertools.product(block_ms, threads_list):
                configs.append({"block_m": bm, "block_n": block_n, "threads": t})
        return configs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan *scan_axis* of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device. Alignment
                padding is handled internally via masked loads.

        Returns:
            A tensor shaped like *x*.

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        axes = (self.scan_axis,)
        y = self._scan_rows(rows_for_axes(x, axes))
        return restore_same_shape(y, in_shape, axes)

    def _scan_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Scan the trailing axis of an ``(M, N)`` buffer.

        The prim_func writes an alignment-padded row; the surplus columns are trimmed
        here.
        """
        if self.use_row_scan:
            program = _row_scan_kernel(
                self.M, self.N, self.op_kind, self.dtype_str, self._row_scan_threads
            )
            return program()(x)
        block_m, block_n = self.config["block_m"], self.config["block_n"]
        threads = self.config["threads"]
        if self.use_parallel:
            y = self._parallel_scan(x, block_m, block_n, threads)
        else:
            program = _cumulative_kernel(self.M, self.N, self.op_kind, self.dtype_str)
            y = program(block_m, block_n, threads)(x)
        return y[:, : self.N] if y.shape[1] > self.N else y

    def _parallel_scan(
        self, x: torch.Tensor, block_m: int, block_n: int, threads: int
    ) -> torch.Tensor:
        """Three passes: scan each tile, scan the tile totals, add the carries."""
        n_tiles = align_up(self.N, DEFAULT_ALIGNMENT) // block_n
        local = _parallel_scan_local_kernel(self.M, self.N, "sum", self.dtype_str)
        y_local, tile_sums = local(block_m, block_n, threads)(x)
        carries = _parallel_scan_carry_kernel(self.M, n_tiles)(threads)(tile_sums)
        propagate = _parallel_scan_propagate_kernel(self.M, self.N, self.dtype_str)
        return propagate(block_m, block_n, threads)(y_local, carries)


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
                tile_shared = T.alloc_shared((block_m, block_n + _SMEM_PAD), "float32")
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
            with T.Kernel(T.ceildiv(M, threads), threads=threads) as pid:  # noqa: SIM117
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
