"""Cumulative scan kernels (cumsum, cumprod) using TileLang.

Implements an inclusive prefix scan along the last dimension.  Each row
is processed independently (``T.Parallel`` over rows) with a tiled
sequential inner loop that processes the N dimension in chunks of
``block_n`` elements.  Within each tile, a ``T.Serial`` loop maintains
a per-row running accumulator.  The tiled approach reduces shared memory
usage (``block_m * block_n`` instead of ``block_m * N_padded``) and
improves memory access patterns for better GPU utilization.

Accepts raw ``(M, N)`` input tensors.  Boundary handling for non-aligned
N is performed inside the kernel via masked loads with identity-element
fills (0 for cumsum, 1 for cumprod), eliminating host-side ``F.pad``
from the forward path.  Output is ``(M, N_padded)`` and the Op layer
trims back to N columns.

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
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    SHARED_MEMORY_BUDGET_BYTES,
    align_up,
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
                                                shared_in[i, j] = x[pid_m * block_m + i, tile_idx * block_n + j]
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
                                        shared_in[i, j] = x[pid_m * block_m + i, tile_idx * block_n + j]
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
                                    y[pid_m * block_m + i, tile_idx * block_n + j] = shared_out[i, j]

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
                                                shared_in[i, j] = x[pid_m * block_m + i, tile_idx * block_n + j]
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
                                        shared_in[i, j] = x[pid_m * block_m + i, tile_idx * block_n + j]
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
                                    y[pid_m * block_m + i, tile_idx * block_n + j] = shared_out[i, j]

            return main

    return _func


# ---------------------------------------------------------------------------
# custom_op wrappers for torch.compile compatibility
# ---------------------------------------------------------------------------


@torch.library.custom_op("top::cumulative_fwd", mutates_args=())
def _cumulative_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    block_n: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _cumulative_kernel(M, N, op_kind, dtype_str)(block_m, block_n, threads)(x)


@_cumulative_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, block_n, threads, x):
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    return torch.empty((M, N_padded), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# CumulativeKernel class
# ---------------------------------------------------------------------------


class CumulativeKernel(Kernel):
    """Inclusive prefix scan kernel (cumsum / cumprod).

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Uses a tiled sequential scan loop along the last
    dimension: the N dimension is divided into tiles of ``block_n``
    elements, reducing shared memory usage and improving occupancy.

    Boundary handling for non-aligned N is performed inside the kernel via
    masked loads with identity-element fills (0 for sum, 1 for prod), so
    no host-side ``F.pad`` is needed.  The ``forward()`` method accepts
    raw ``(M, N)`` tensors and returns ``(M, N_padded)`` output.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "sum", "prod".
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        if op_kind not in ("sum", "prod"):
            raise ValueError(f"Unsupported op_kind '{op_kind}'. Expected one of 'sum', 'prod'.")
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)

        # Adaptive dispatch: use parallel scan for small-M, large-N
        # Only for cumsum (cumprod not yet supported for parallel)
        self.use_parallel = (M < 128 and N > 8192 and op_kind == "sum")

        if self.use_parallel:
            # Three-pass parallel scan
            self.kernel_local = _parallel_scan_local_kernel(M, N, op_kind, self.dtype_str)
            self.kernel_propagate = _parallel_scan_propagate_kernel(M, N, self.dtype_str)
            self.kernel = None  # Not used for parallel
            tune = False  # Disable autotuning for parallel path (uses optimized defaults)
        else:
            # Sequential scan (original implementation)
            self.kernel = _cumulative_kernel(M, N, op_kind, self.dtype_str)
            self.kernel_local = None
            self.kernel_propagate = None

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """Select default config based on shape and backend.

        For parallel scan: use larger block_n for better parallelism.
        For sequential scan: use adaptive block_m based on M.
        """
        if self.use_parallel:
            # Parallel scan config: larger tiles for better parallelism
            block_n = 256 if self.N > 16384 else 128
            block_m = 16
            elem_size = torch.tensor([], dtype=self.dtype).element_size()
            smem_per_row = (block_n + _SMEM_PAD) * 4  # float32 intermediate
            max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
            block_m = min(block_m, max_block_m)
            return {"block_m": block_m, "block_n": block_n, "threads": 256}
        else:
            # Sequential scan config (original logic)
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
        """Run the cumulative scan kernel (sequential or parallel).

        Args:
            x: Input tensor of shape (M, N).  Alignment padding is
                handled internally via masked loads.

        Returns:
            Output tensor of shape (M, N_padded).
        """
        if self.use_parallel:
            # Three-pass parallel scan
            block_m = self.config["block_m"]
            block_n = self.config["block_n"]
            threads = self.config["threads"]
            n_tiles = self.N_padded // block_n

            # Pass 1: Local scan within each tile (JIT kernel auto-allocates outputs)
            local_fn = self.kernel_local(block_m, block_n, threads)
            y_local, tile_sums = local_fn(x)

            # Pass 2: Exclusive scan of tile sums (PyTorch)
            tile_carries = torch.cumsum(tile_sums, dim=1, dtype=torch.float32)
            tile_carries_exclusive = torch.zeros_like(tile_carries)
            if n_tiles > 1:
                tile_carries_exclusive[:, 1:] = tile_carries[:, :-1]

            # Pass 3: Propagate carries (JIT kernel auto-allocates output)
            propagate_fn = self.kernel_propagate(block_m, block_n, threads)
            y_final = propagate_fn(y_local, tile_carries_exclusive)

            return y_final
        else:
            # Sequential scan (original implementation)
            return _cumulative_fwd_wrapped(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_str,
                self.config["block_m"],
                self.config["block_n"],
                self.config["threads"],
                x,
            )


# ---------------------------------------------------------------------------
# Parallel scan kernels for small-M, large-N workloads
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _parallel_scan_local_kernel(M: int, N: int, op_kind: str, dtype: str):
    """Pass 1: Local scan within each tile (parallel across all tiles).

    Grid: (M/block_m, N_padded/block_n) for massive parallelism.
    Each thread block independently scans one tile.
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

                # Load tile
                if _needs_mask:
                    with T.If((tile_idx + 1) * block_n <= N):
                        with T.Then():
                            for i, j in T.Parallel(block_m, block_n):
                                row = pid_m * block_m + i
                                col = tile_idx * block_n + j
                                tile_shared[i, j] = T.if_then_else(
                                    row < M,
                                    T.cast(x[row, col], "float32"),
                                    T.cast(_identity, "float32"),
                                )
                        with T.Else():
                            for i, j in T.Parallel(block_m, block_n):
                                row = pid_m * block_m + i
                                col = tile_idx * block_n + j
                                tile_shared[i, j] = T.if_then_else(
                                    T.And(row < M, col < N),
                                    T.cast(x[row, col], "float32"),
                                    T.cast(_identity, "float32"),
                                )
                else:
                    for i, j in T.Parallel(block_m, block_n):
                        row = pid_m * block_m + i
                        col = tile_idx * block_n + j
                        tile_shared[i, j] = T.if_then_else(
                            row < M,
                            T.cast(x[row, col], "float32"),
                            T.cast(_identity, "float32"),
                        )

                T.sync_threads()

                # Copy to fragment and scan
                for i, j in T.Parallel(block_m, block_n):
                    tile_frag[i, j] = tile_shared[i, j]
                T.sync_threads()

                T.cumsum(tile_frag, dim=1)
                T.sync_threads()

                # Copy back
                for i, j in T.Parallel(block_m, block_n):
                    tile_shared[i, j] = tile_frag[i, j]
                T.sync_threads()

                # Write results
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
def _parallel_scan_propagate_kernel(M: int, N: int, dtype: str):
    """Pass 3: Add carry values and cast to output dtype."""
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
                tile_shared = T.alloc_shared((block_m, block_n + _SMEM_PAD), "float32")

                # Load y_local
                for i, j in T.Parallel(block_m, block_n):
                    row = pid_m * block_m + i
                    col = tile_idx * block_n + j
                    with T.If(row < M), T.Then():
                        tile_shared[i, j] = y_local[row, col]

                T.sync_threads()

                # Add carry and write
                for i, j in T.Parallel(block_m, block_n):
                    row = pid_m * block_m + i
                    col = tile_idx * block_n + j
                    with T.If(row < M), T.Then():
                        carry = tile_carries[row, tile_idx]
                        result_f32 = tile_shared[i, j] + carry
                        y_final[row, col] = T.cast(result_f32, dtype)

        return main

    return _func
