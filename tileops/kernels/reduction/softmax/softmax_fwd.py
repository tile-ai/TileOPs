"""Softmax / log-softmax forward kernel using TileLang.

Implements a 2-pass online softmax algorithm for two operations:
  - softmax:     y[i,j] = exp(x[i,j] - max_i) / sum_i(exp(x[i,j] - max_i))
  - log_softmax: y[i,j] = x[i,j] - max_i - log(sum_i(exp(x[i,j] - max_i)))

Supports arbitrarily large N dimensions by tiling over N when the full
N_padded does not fit in shared memory.  Uses the online softmax recurrence
(track running max and rescaled running sum) across N-tiles.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions. Padded columns are filled with -infinity so they
contribute 0 to exp-sums and never win the max reduction.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    SHARED_MEMORY_BUDGET_BYTES,
    align_up,
    compute_tile_n,
)

__all__ = ["SoftmaxKernel"]


# ---------------------------------------------------------------------------
# Single-tile kernel (N fits in shared memory) -- original fast path
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _softmax_kernel_single(M: int, N: int, op_kind: str, dtype: str):
    """Build a single-tile softmax/log_softmax kernel (N fits in smem)."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)

    if op_kind == "softmax":

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    row_max = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")

                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    T.fill(row_max, -T.infinity("float32"))
                    T.reduce_max(x_f32, row_max, dim=1, clear=False)

                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.exp(x_f32[i, j] - row_max[i])
                    T.reduce_sum(x_f32, row_sum, dim=1)

                    out_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    for i, j in T.Parallel(block_m, N_padded):
                        out_f32[i, j] = x_f32[i, j] / row_sum[i]

                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = out_f32[i, j]
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

            return main

    else:  # log_softmax

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N_padded), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    row_max = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")

                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    T.fill(row_max, -T.infinity("float32"))
                    T.reduce_max(x_f32, row_max, dim=1, clear=False)

                    for i, j in T.Parallel(block_m, N_padded):
                        x_f32[i, j] = T.exp(x_f32[i, j] - row_max[i])
                    T.reduce_sum(x_f32, row_sum, dim=1)

                    out_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    for i, j in T.Parallel(block_m, N_padded):
                        out_f32[i, j] = T.log(x_f32[i, j] / row_sum[i])

                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = out_f32[i, j]
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

            return main

    return _func


# ---------------------------------------------------------------------------
# Multi-tile kernel (N tiled over shared memory)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _softmax_kernel_tiled(M: int, N: int, op_kind: str, dtype: str, tile_n: int):
    """Build a multi-tile softmax/log_softmax kernel.

    Uses online softmax recurrence across N-tiles:
      Pass 1 (all tiles): compute running max and rescaled running sum.
      Pass 2 (all tiles): normalize using global max and sum.

    The input/output tensor column count is total_cols = num_tiles * tile_n,
    which may be larger than N_padded. Extra columns must be -inf padded.

    NOTE: Pass 2 uses a dedicated shared memory buffer AND dedicated register
    fragments. TileLang's allocator may alias both shared buffers and register
    fragments across T.Serial loop boundaries, corrupting pass-1 accumulators
    (row_max, row_sum) if the same names are reused.  The dual-buffer shared
    memory cost is accounted for by passing ``num_buffers=2`` to
    ``compute_tile_n``.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    total_cols = num_tiles * tile_n

    if op_kind == "softmax":

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, total_cols), dtype],
                y: T.Tensor[(M, total_cols), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    # --- Pass 1 fragments ---
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    tile_local = T.alloc_fragment((block_m, tile_n), dtype)
                    tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")

                    row_max = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")
                    prev_max = T.alloc_fragment((block_m,), "float32")
                    tile_max = T.alloc_fragment((block_m,), "float32")
                    tile_sum = T.alloc_fragment((block_m,), "float32")

                    T.fill(row_max, -T.infinity("float32"))
                    T.fill(row_sum, 0.0)

                    # Pass 1: compute global max and sum using online recurrence
                    for t in T.Serial(num_tiles):
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        T.copy(shared_buf, tile_local)

                        for i, j in T.Parallel(block_m, tile_n):
                            tile_f32[i, j] = T.cast(tile_local[i, j], "float32")

                        T.fill(tile_max, -T.infinity("float32"))
                        T.reduce_max(tile_f32, tile_max, dim=1, clear=False)

                        for i in T.Parallel(block_m):
                            prev_max[i] = row_max[i]
                            row_max[i] = T.max(row_max[i], tile_max[i])

                        for i, j in T.Parallel(block_m, tile_n):
                            tile_f32[i, j] = T.exp(tile_f32[i, j] - row_max[i])
                        T.reduce_sum(tile_f32, tile_sum, dim=1)

                        for i in T.Parallel(block_m):
                            row_sum[i] = (
                                row_sum[i] * T.exp(prev_max[i] - row_max[i]) + tile_sum[i]
                            )

                    # --- Pass 2: dedicated shared + register fragments ---
                    # TileLang's allocator aliases both shared buffers and
                    # register fragments across T.Serial loop boundaries.
                    # Using separate allocations for pass 2 prevents
                    # corruption of pass-1 accumulators (row_max, row_sum).
                    # compute_tile_n accounts for 2x shared memory via
                    # num_buffers=2.
                    p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                    p2_local = T.alloc_fragment((block_m, tile_n), dtype)
                    p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                    p2_out = T.alloc_fragment((block_m, tile_n), "float32")

                    # Pass 2: normalize
                    for t in T.Serial(num_tiles):
                        T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                        T.copy(p2_shared, p2_local)

                        for i, j in T.Parallel(block_m, tile_n):
                            p2_f32[i, j] = T.cast(p2_local[i, j], "float32")

                        for i, j in T.Parallel(block_m, tile_n):
                            p2_out[i, j] = T.exp(p2_f32[i, j] - row_max[i]) / row_sum[i]

                        for i, j in T.Parallel(block_m, tile_n):
                            p2_local[i, j] = p2_out[i, j]
                        T.copy(p2_local, p2_shared)
                        T.copy(p2_shared, y[pid_m * block_m, t * tile_n])

            return main

    else:  # log_softmax

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, total_cols), dtype],
                y: T.Tensor[(M, total_cols), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    # --- Pass 1 fragments ---
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    tile_local = T.alloc_fragment((block_m, tile_n), dtype)
                    tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")

                    row_max = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")
                    prev_max = T.alloc_fragment((block_m,), "float32")
                    tile_max = T.alloc_fragment((block_m,), "float32")
                    tile_sum = T.alloc_fragment((block_m,), "float32")

                    T.fill(row_max, -T.infinity("float32"))
                    T.fill(row_sum, 0.0)

                    # Pass 1: compute global max and sum
                    for t in T.Serial(num_tiles):
                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                        T.copy(shared_buf, tile_local)

                        for i, j in T.Parallel(block_m, tile_n):
                            tile_f32[i, j] = T.cast(tile_local[i, j], "float32")

                        T.fill(tile_max, -T.infinity("float32"))
                        T.reduce_max(tile_f32, tile_max, dim=1, clear=False)

                        for i in T.Parallel(block_m):
                            prev_max[i] = row_max[i]
                            row_max[i] = T.max(row_max[i], tile_max[i])

                        for i, j in T.Parallel(block_m, tile_n):
                            tile_f32[i, j] = T.exp(tile_f32[i, j] - row_max[i])
                        T.reduce_sum(tile_f32, tile_sum, dim=1)

                        for i in T.Parallel(block_m):
                            row_sum[i] = (
                                row_sum[i] * T.exp(prev_max[i] - row_max[i]) + tile_sum[i]
                            )

                    # Precompute log(sum) to avoid recomputing per-element
                    log_sum = T.alloc_fragment((block_m,), "float32")
                    for i in T.Parallel(block_m):
                        log_sum[i] = T.log(row_sum[i])

                    # --- Pass 2: dedicated shared + register fragments ---
                    # (Same aliasing workaround as softmax — see note above.)
                    p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                    p2_local = T.alloc_fragment((block_m, tile_n), dtype)
                    p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                    p2_out = T.alloc_fragment((block_m, tile_n), "float32")

                    # Pass 2: log-normalize
                    for t in T.Serial(num_tiles):
                        T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                        T.copy(p2_shared, p2_local)

                        for i, j in T.Parallel(block_m, tile_n):
                            p2_f32[i, j] = T.cast(p2_local[i, j], "float32")

                        for i, j in T.Parallel(block_m, tile_n):
                            p2_out[i, j] = p2_f32[i, j] - row_max[i] - log_sum[i]

                        for i, j in T.Parallel(block_m, tile_n):
                            p2_local[i, j] = p2_out[i, j]
                        T.copy(p2_local, p2_shared)
                        T.copy(p2_shared, y[pid_m * block_m, t * tile_n])

            return main

    return _func


# ---------------------------------------------------------------------------
# Dispatch: choose single-tile or multi-tile kernel based on tile_n
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _softmax_kernel(M: int, N: int, op_kind: str, dtype: str, tile_n: int = 0):
    """Build the appropriate softmax kernel.

    If tile_n == 0, the full N fits in shared memory and the single-tile
    kernel is used. Otherwise, the multi-tile kernel is used.
    """
    if tile_n == 0:
        return _softmax_kernel_single(M, N, op_kind, dtype)
    return _softmax_kernel_tiled(M, N, op_kind, dtype, tile_n)


# ---------------------------------------------------------------------------
# custom_op wrappers for torch.compile compatibility
# ---------------------------------------------------------------------------


def _compute_padded_cols(N: int, tile_n: int) -> int:
    """Compute the total column count (may exceed N_padded for tiled path)."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    if tile_n == 0:
        return N_padded
    num_tiles = (N_padded + tile_n - 1) // tile_n
    return num_tiles * tile_n


@torch.library.custom_op("top::softmax_fwd", mutates_args=())
def _softmax_fwd_wrapped(
    M: int,
    N: int,
    op_kind: str,
    dtype_str: str,
    block_m: int,
    threads: int,
    tile_n: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _softmax_kernel(M, N, op_kind, dtype_str, tile_n)(block_m, threads)(x)


@_softmax_fwd_wrapped.register_fake
def _(M, N, op_kind, dtype_str, block_m, threads, tile_n, x):
    total_cols = _compute_padded_cols(N, tile_n)
    return torch.empty((M, total_cols), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------


def _elem_bytes(dtype: torch.dtype) -> int:
    """Return bytes per element for the given dtype."""
    return torch.tensor([], dtype=dtype).element_size()


class SoftmaxKernel(Kernel):
    """Softmax / log-softmax forward kernel.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Implements a 2-pass online softmax algorithm.

    For large N that does not fit in shared memory, tiles over N using
    the online softmax recurrence (running max + rescaled sum).

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: One of "softmax", "log_softmax".
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
        if op_kind not in ("softmax", "log_softmax"):
            raise ValueError(
                f"Unsupported op_kind '{op_kind}'. Expected one of 'softmax', 'log_softmax'."
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = _elem_bytes(dtype)

        # Build self.kernel BEFORE init_config: when tune=True, init_config
        # delegates to autotune() which requires self.kernel to exist.
        #
        # tile_n is baked into the kernel at build time, so we pre-compute
        # it from the heuristic block_m in default_config. The matching
        # `autotune_configs` property below filters out any block_m whose
        # tile_n differs from this pre-built one, so the autotuner stays
        # within a single tiling regime by design.
        #
        # Cross-tile_n autotune exploration is tracked as a known follow-up:
        # see issue #821. The single-regime restriction is inherited from
        # #801, not introduced by this fix.
        self._tile_n = self.default_config["tile_n"]
        self.kernel = _softmax_kernel(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self._tile_n,
        )

        self.init_config(config, tune)

        # If a user-provided config picked a block_m mapping to a different
        # tile_n, rebuild the kernel for the new tile_n. _softmax_kernel is
        # lru_cache'd, so this is cheap on the common path.
        new_tile_n = self._tile_n_for_block_m(self.config["block_m"])
        if new_tile_n != self._tile_n:
            self._tile_n = new_tile_n
            self.kernel = _softmax_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_str,
                self._tile_n,
            )
        self.config["tile_n"] = self._tile_n

    # Tiled softmax/log_softmax allocates 2 shared buffers (one per pass)
    # due to TileLang allocator aliasing — see _softmax_kernel_tiled docstring.
    _NUM_SHARED_BUFFERS = 2

    def _tile_n_for_block_m(self, block_m: int) -> int:
        """Return tile_n for a given block_m (0 means no tiling needed)."""
        # First check if N fits in a single buffer (no tiling → single-tile path,
        # which only allocates 1 shared buffer).
        single = compute_tile_n(block_m, self._elem_bytes, self.N_padded)
        if single == self.N_padded:
            return 0
        # Tiled path: budget must accommodate num_buffers shared allocations.
        return compute_tile_n(
            block_m, self._elem_bytes, self.N_padded,
            num_buffers=self._NUM_SHARED_BUFFERS,
        )

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget."""
        smem_per_row = self.N_padded * self._elem_bytes
        max_block_m = SHARED_MEMORY_BUDGET_BYTES // smem_per_row if smem_per_row > 0 else 16
        if max_block_m == 0:
            block_m = 1
        else:
            block_m = 1
            for bm in [1, 2, 4, 8, 16]:
                if bm <= max_block_m:
                    block_m = bm
        tile_n = self._tile_n_for_block_m(block_m)
        return {"block_m": block_m, "threads": 256, "tile_n": tile_n}

    @property
    def autotune_configs(self) -> list[dict]:
        """Generate autotune configs (block_m, threads only).

        tile_n is baked into the kernel at build time, so we only expose
        block_m and threads to the autotuner. All configs must be compatible
        with the kernel's tile_n.
        """
        # Use the same tile_n that the kernel was built with
        tile_n = getattr(self, "_tile_n", self.default_config["tile_n"])
        smem_per_row = self.N_padded * self._elem_bytes
        max_block_m_no_tile = SHARED_MEMORY_BUDGET_BYTES // smem_per_row if smem_per_row > 0 else 16
        threads_list = [128, 256]

        configs = []
        for bm in [1, 2, 4, 8, 16]:
            try:
                compute_tile_n(bm, self._elem_bytes, self.N_padded)
            except ValueError:
                continue
            bm_tile_n = self._tile_n_for_block_m(bm)
            # Only include configs compatible with the built kernel's tile_n
            if bm_tile_n != tile_n:
                continue
            if tile_n == 0 and bm > max_block_m_no_tile:
                continue
            for t in threads_list:
                configs.append({"block_m": bm, "threads": t})

        if not configs:
            configs = [{"block_m": 1, "threads": 256}]

        return configs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the softmax/log_softmax kernel."""
        tile_n = self._tile_n

        # For the tiled path, the input may need extra padding beyond N_padded
        # to make columns a multiple of tile_n.
        total_cols = _compute_padded_cols(self.N, tile_n)
        if x.shape[1] < total_cols:
            import torch.nn.functional as _F

            x = _F.pad(x, (0, total_cols - x.shape[1]), value=float("-inf"))

        y = _softmax_fwd_wrapped(
            self.M,
            self.N,
            self.op_kind,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            tile_n,
            x,
        )

        # Trim back to N_padded (the op layer will trim to N)
        if y.shape[1] > self.N_padded:
            y = y[:, : self.N_padded]

        return y
