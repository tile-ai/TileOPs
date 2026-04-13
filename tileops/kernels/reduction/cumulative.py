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
                    shared_in = T.alloc_shared((block_m, block_n), dtype)
                    shared_out = T.alloc_shared((block_m, block_n), dtype)
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
                                    T.copy(x[pid_m * block_m, tile_idx * block_n], shared_in)
                                    for i, j in T.Parallel(block_m, block_n):
                                        tile_f32[i, j] = T.cast(shared_in[i, j], "float32")
                                with T.Else():
                                    # Partially OOB tile: masked load
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
                            T.copy(x[pid_m * block_m, tile_idx * block_n], shared_in)
                            for i, j in T.Parallel(block_m, block_n):
                                tile_f32[i, j] = T.cast(shared_in[i, j], "float32")

                        # Inclusive prefix sum within tile
                        for i in T.Parallel(block_m):
                            for j in T.Serial(block_n):
                                acc[i] = acc[i] + tile_f32[i, j]
                                out_f32[i, j] = acc[i]

                        # Cast back and write tile via shared memory
                        for i, j in T.Parallel(block_m, block_n):
                            shared_out[i, j] = T.cast(out_f32[i, j], dtype)
                        T.copy(shared_out, y[pid_m * block_m, tile_idx * block_n])

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
                    shared_in = T.alloc_shared((block_m, block_n), dtype)
                    shared_out = T.alloc_shared((block_m, block_n), dtype)
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
                                    T.copy(x[pid_m * block_m, tile_idx * block_n], shared_in)
                                    for i, j in T.Parallel(block_m, block_n):
                                        tile_f32[i, j] = T.cast(shared_in[i, j], "float32")
                                with T.Else():
                                    # Partially OOB tile: masked load
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
                            T.copy(x[pid_m * block_m, tile_idx * block_n], shared_in)
                            for i, j in T.Parallel(block_m, block_n):
                                tile_f32[i, j] = T.cast(shared_in[i, j], "float32")

                        # Inclusive prefix product within tile
                        for i in T.Parallel(block_m):
                            for j in T.Serial(block_n):
                                acc[i] = acc[i] * tile_f32[i, j]
                                out_f32[i, j] = acc[i]

                        # Cast back and write tile via shared memory
                        for i, j in T.Parallel(block_m, block_n):
                            shared_out[i, j] = T.cast(out_f32[i, j], dtype)
                        T.copy(shared_out, y[pid_m * block_m, tile_idx * block_n])

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
        self.kernel = _cumulative_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """Select default config based on shared memory budget.

        Uses tiled scan with block_n tiles along N.  Shared memory usage
        is ``block_m * block_n * element_size`` (two buffers: in + out).
        """
        block_n = _DEFAULT_BLOCK_N
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        # Two shared buffers: shared_in + shared_out, each block_m * block_n
        smem_per_row = 2 * block_n * elem_size
        max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
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
            smem_per_row = 2 * block_n * elem_size
            max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row
            block_ms = [bm for bm in [1, 2, 4, 8, 16] if bm <= max_block_m]
            threads_list = [128, 256]
            for bm, t in itertools.product(block_ms, threads_list):
                configs.append({"block_m": bm, "block_n": block_n, "threads": t})
        return configs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the cumulative scan kernel.

        Args:
            x: Input tensor of shape (M, N).  Alignment padding is
                handled internally via masked loads.

        Returns:
            Output tensor of shape (M, N_padded).
        """
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
