"""Softmax / log-softmax forward kernel using TileLang.

Implements a 2-pass online softmax algorithm for two operations:
  - softmax:     y[i,j] = exp(x[i,j] - max_i) / sum_i(exp(x[i,j] - max_i))
  - log_softmax: y[i,j] = x[i,j] - max_i - log(sum_i(exp(x[i,j] - max_i)))

Supports arbitrarily large N dimensions by tiling over N when the full
N_padded does not fit in shared memory.  Uses the online softmax recurrence
(track running max and rescaled running sum) across N-tiles.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions.  Boundary handling for non-aligned N is performed
inside the kernel via masked loads and -inf fills, eliminating host-side
``F.pad`` from the forward path.  In the multi-tile path, only the last
tile uses element-wise masked loads; all preceding tiles use the fast
vectorized T.copy path since their columns are fully in-bounds.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    MAX_SINGLE_TILE_COLS,
    align_up,
    compute_tile_n,
    device_smem_budget,
)

__all__ = ["SoftmaxKernel"]


# ---------------------------------------------------------------------------
# Single-tile kernel (N fits in shared memory) -- original fast path
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _softmax_kernel_single(M: int, N: int, op_kind: str, dtype: str):
    """Build a single-tile softmax/log_softmax kernel (N fits in smem).

    Accepts an ``(M, N)`` input tensor.  When ``N`` is not a multiple of
    ``DEFAULT_ALIGNMENT``, the kernel fills shared memory with ``-inf``
    and loads only the valid ``N`` columns (kernel-side boundary handling).
    When ``N`` is already aligned, the fast ``T.copy`` path is used.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    # Reinterpret -inf as the target dtype at Python level (compile-time
    # constant) to avoid runtime reinterpret inside the kernel.
    _neg_inf = float("-inf")

    if op_kind == "softmax":

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    row_max = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")

                    if _needs_pad:
                        # Kernel-side boundary handling: element-wise load
                        # with T.if_then_else masking for padding columns.
                        for i, j in T.Parallel(block_m, N_padded):
                            x_f32[i, j] = T.if_then_else(
                                j < N,
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_neg_inf, "float32"),
                            )
                    else:
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
                x: T.Tensor[(M, N), dtype],
                y: T.Tensor[(M, N_padded), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_local = T.alloc_fragment((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    row_max = T.alloc_fragment((block_m,), "float32")
                    row_sum = T.alloc_fragment((block_m,), "float32")

                    if _needs_pad:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_f32[i, j] = T.if_then_else(
                                j < N,
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_neg_inf, "float32"),
                            )
                    else:
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


@functools.lru_cache(maxsize=64)
def _softmax_kernel_tiled(M: int, N: int, op_kind: str, dtype: str, tile_n: int):
    """Build a multi-tile softmax/log_softmax kernel.

    Uses online softmax recurrence across N-tiles:
      Pass 1 (all tiles): compute running max and rescaled running sum.
      Pass 2 (all tiles): normalize using global max and sum.

    The input tensor has the raw shape ``(M, N)`` (no host-side padding).
    Boundary handling for the last tile (where ``t * tile_n + j`` may
    exceed ``N``) is performed inside the kernel via ``T.if_then_else``
    masked loads.  Output columns are ``total_cols = num_tiles * tile_n``.

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
    # The last tile may extend beyond N; boundary masking is needed when
    # total_cols > N (which is always true when N is not aligned, and also
    # when tile_n does not evenly divide N_padded).
    _needs_mask = total_cols > N
    _neg_inf = float("-inf")

    if op_kind == "softmax":

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
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
                        if _needs_mask:
                            # Only the last tile may have out-of-bounds columns.
                            # Use fast vectorized T.copy for all earlier tiles,
                            # and element-wise T.if_then_else only for the last.
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                    T.copy(shared_buf, tile_local)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        tile_f32[i, j] = T.cast(tile_local[i, j], "float32")
                                with T.Else():
                                    for i, j in T.Parallel(block_m, tile_n):
                                        tile_f32[i, j] = T.if_then_else(
                                            t * tile_n + j < N,
                                            T.cast(
                                                x[pid_m * block_m + i, t * tile_n + j], "float32"
                                            ),
                                            T.cast(_neg_inf, "float32"),
                                        )
                        else:
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

                    # Precompute reciprocal to replace division with
                    # multiplication in the per-element normalisation.
                    inv_sum = T.alloc_fragment((block_m,), "float32")
                    for i in T.Parallel(block_m):
                        inv_sum[i] = 1.0 / row_sum[i]

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

                    # Pass 2: normalize (cast + compute fused, write-back
                    # reuses p2_f32 -> p2_local to avoid an extra fragment)
                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                    T.copy(p2_shared, p2_local)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        p2_f32[i, j] = T.exp(
                                            T.cast(p2_local[i, j], "float32") - row_max[i]
                                        ) * inv_sum[i]
                                with T.Else():
                                    for i, j in T.Parallel(block_m, tile_n):
                                        p2_f32[i, j] = T.if_then_else(
                                            t * tile_n + j < N,
                                            T.exp(
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                )
                                                - row_max[i]
                                            )
                                            * inv_sum[i],
                                            0.0,
                                        )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                            T.copy(p2_shared, p2_local)
                            for i, j in T.Parallel(block_m, tile_n):
                                p2_f32[i, j] = T.exp(
                                    T.cast(p2_local[i, j], "float32") - row_max[i]
                                ) * inv_sum[i]

                        for i, j in T.Parallel(block_m, tile_n):
                            p2_local[i, j] = p2_f32[i, j]
                        T.copy(p2_local, p2_shared)
                        T.copy(p2_shared, y[pid_m * block_m, t * tile_n])

            return main

    else:  # log_softmax

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
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
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                    T.copy(shared_buf, tile_local)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        tile_f32[i, j] = T.cast(tile_local[i, j], "float32")
                                with T.Else():
                                    for i, j in T.Parallel(block_m, tile_n):
                                        tile_f32[i, j] = T.if_then_else(
                                            t * tile_n + j < N,
                                            T.cast(
                                                x[pid_m * block_m + i, t * tile_n + j], "float32"
                                            ),
                                            T.cast(_neg_inf, "float32"),
                                        )
                        else:
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
                    # (Same aliasing workaround as softmax -- see note above.)
                    p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                    p2_local = T.alloc_fragment((block_m, tile_n), dtype)
                    p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")

                    # Pass 2: log-normalize (cast + compute fused)
                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                    T.copy(p2_shared, p2_local)
                                    for i, j in T.Parallel(block_m, tile_n):
                                        p2_f32[i, j] = (
                                            T.cast(p2_local[i, j], "float32")
                                            - row_max[i]
                                            - log_sum[i]
                                        )
                                with T.Else():
                                    for i, j in T.Parallel(block_m, tile_n):
                                        p2_f32[i, j] = T.if_then_else(
                                            t * tile_n + j < N,
                                            T.cast(
                                                x[pid_m * block_m + i, t * tile_n + j], "float32"
                                            )
                                            - row_max[i]
                                            - log_sum[i],
                                            T.cast(_neg_inf, "float32"),
                                        )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                            T.copy(p2_shared, p2_local)
                            for i, j in T.Parallel(block_m, tile_n):
                                p2_f32[i, j] = (
                                    T.cast(p2_local[i, j], "float32") - row_max[i] - log_sum[i]
                                )

                        for i, j in T.Parallel(block_m, tile_n):
                            p2_local[i, j] = p2_f32[i, j]
                        T.copy(p2_local, p2_shared)
                        T.copy(p2_shared, y[pid_m * block_m, t * tile_n])

            return main

    return _func


# ---------------------------------------------------------------------------
# Dispatch: choose single-tile or multi-tile kernel based on tile_n
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
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

    Boundary handling for non-aligned N is performed inside the kernel
    via masked loads and ``-inf`` fills, so no host-side ``F.pad`` is
    needed.

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
        self._smem_budget = device_smem_budget()

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
    # due to TileLang allocator aliasing -- see _softmax_kernel_tiled docstring.
    _NUM_SHARED_BUFFERS = 2

    def _tile_n_for_block_m(self, block_m: int) -> int:
        """Return tile_n for a given block_m (0 means no tiling needed).

        Uses the device's actual shared memory budget (not the
        conservative 48 KiB default) so that large-N workloads can
        use fewer, larger tiles or even the single-tile fast path.

        Both paths are subject to the MAX_SINGLE_TILE_COLS column
        cap (TileLang's vectorizer fails at the 32768 column boundary).
        """
        budget = self._smem_budget
        # Single-tile path requires the full row in register fragments.
        # Cap by column count (vectorizer limit) and smem budget.
        if self.N_padded <= MAX_SINGLE_TILE_COLS:
            single = compute_tile_n(
                block_m, self._elem_bytes, self.N_padded, budget=budget,
            )
            if single == self.N_padded:
                return 0
        # Tiled path: budget must accommodate num_buffers shared allocations.
        # Cap the smem budget so tile_n stays within the column limit.
        col_budget = MAX_SINGLE_TILE_COLS * self._NUM_SHARED_BUFFERS * block_m * self._elem_bytes
        effective_budget = min(budget, col_budget)
        return compute_tile_n(
            block_m, self._elem_bytes, self.N_padded,
            num_buffers=self._NUM_SHARED_BUFFERS,
            budget=effective_budget,
        )

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget.

        For the single-tile path (tile_n == 0), prefer the largest
        block_m that fits in shared memory -- row-level parallelism is
        free when the full row is in registers.

        For the tiled path, prefer the block_m that **minimises the
        number of N-tiles** (i.e. maximises tile_n).  Fewer tiles means
        fewer global memory passes in the 2-pass algorithm, which
        dominates latency on bandwidth-bound workloads.  Among configs
        with equal tile count, prefer *smaller* block_m: the tiled
        kernel is bandwidth-bound, and smaller shared-memory footprint
        per block improves occupancy.
        """
        best_bm = 1
        best_tile_n = self._tile_n_for_block_m(1)

        for bm in [2, 4, 8, 16]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn == 0 and best_tile_n == 0:
                # Both single-tile: prefer larger block_m (row reuse is free)
                best_bm = bm
                best_tile_n = tn
            elif tn == 0 and best_tile_n != 0:
                # Switching from tiled to single-tile is always better
                best_bm = bm
                best_tile_n = tn
            elif tn != 0 and best_tile_n == 0:
                # Don't give up single-tile for tiled
                pass
            else:
                # Both tiled: prefer strictly fewer tiles only.
                best_num = (self.N_padded + best_tile_n - 1) // best_tile_n
                curr_num = (self.N_padded + tn - 1) // tn
                if curr_num < best_num:
                    best_bm = bm
                    best_tile_n = tn

        return {"block_m": best_bm, "threads": 256, "tile_n": best_tile_n}

    @property
    def autotune_configs(self) -> list[dict]:
        """Generate autotune configs (block_m, threads only).

        tile_n is baked into the kernel at build time, so we only expose
        block_m and threads to the autotuner. All configs must be compatible
        with the kernel's tile_n.
        """
        # Use the same tile_n that the kernel was built with
        tile_n = getattr(self, "_tile_n", self.default_config["tile_n"])
        budget = self._smem_budget
        smem_per_row = self.N_padded * self._elem_bytes
        max_block_m_no_tile = budget // smem_per_row if smem_per_row > 0 else 16
        threads_list = [128, 256]

        configs = []
        for bm in [1, 2, 4, 8, 16]:
            try:
                compute_tile_n(bm, self._elem_bytes, self.N_padded, budget=budget)
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
        """Run the softmax/log_softmax kernel.

        Accepts an ``(M, N)`` tensor.  Boundary handling for non-aligned
        ``N`` is performed inside the GPU kernel (masked loads + ``-inf``
        fill), so no host-side ``F.pad`` is needed.
        """
        tile_n = self._tile_n

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

        # Trim back to N_padded if needed (only when tile_n does not
        # divide N_padded, which the divisor-aware compute_tile_n avoids
        # for all practical cases).
        if y.shape[1] > self.N_padded:
            y = y[:, : self.N_padded]

        return y
