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
    ``DEFAULT_ALIGNMENT``, the kernel uses element-wise ``T.if_then_else``
    loads that substitute ``-inf`` for out-of-bounds columns (kernel-side
    boundary handling).  When ``N`` is already aligned, the fast ``T.copy``
    path is used.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    # Compile-time Python constant used for padding; it is still cast to
    # the kernel dtype where needed inside the generated kernel.
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
                        # with T.if_then_else masking for padding columns
                        # and row-tail safety (M % block_m != 0).
                        for i, j in T.Parallel(block_m, N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
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
                                T.And(pid_m * block_m + i < M, j < N),
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

                    # Compute exp(x - max) into a separate fragment so x_f32
                    # retains the original values for the stable log_softmax
                    # formula: (x - max) - log(sum(exp(x - max))).
                    exp_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    for i, j in T.Parallel(block_m, N_padded):
                        exp_f32[i, j] = T.exp(x_f32[i, j] - row_max[i])
                    T.reduce_sum(exp_f32, row_sum, dim=1)

                    log_sum = T.alloc_fragment((block_m,), "float32")
                    for i in T.Parallel(block_m):
                        log_sum[i] = T.log(row_sum[i])

                    # log_softmax = (x - max) - log(sum), avoids log(0) on
                    # padding columns (they stay at -inf via subtraction).
                    out_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    for i, j in T.Parallel(block_m, N_padded):
                        out_f32[i, j] = x_f32[i, j] - row_max[i] - log_sum[i]

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
                                            T.And(pid_m * block_m + i < M, t * tile_n + j < N),
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
                                            T.And(pid_m * block_m + i < M, t * tile_n + j < N),
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
                                            T.And(pid_m * block_m + i < M, t * tile_n + j < N),
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
                                            T.And(pid_m * block_m + i < M, t * tile_n + j < N),
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
        device_index: CUDA device index for shared memory budget query.
            When ``None``, ``torch.cuda.current_device()`` is used.
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
        device_index: int | None = None,
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
        self._smem_budget = device_smem_budget(device_index)

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

        # When tune=True, autotune() already set self._tile_n and
        # self.config["tile_n"], and rebuilt the kernel.  Only apply
        # the post-init tile_n fixup for user-provided configs.
        if not tune:
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
    _MAX_TILE_N_CANDIDATES = 3

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

    def _tile_n_candidates(self) -> list[int]:
        """Return candidate tile_n values for autotune exploration.

        Includes the heuristic tile_n plus divisors of N_padded that fit
        within the shared memory budget (accounting for num_buffers).
        tile_n=0 means single-tile (no tiling).  All candidates are
        de-duplicated and sorted descending for deterministic ordering.

        Each distinct tile_n value requires a full kernel recompilation,
        which is expensive for large-N workloads (compilations can take
        minutes each).  To keep autotuner wall time practical we cap
        the total number of tile_n candidates at ``_MAX_TILE_N_CANDIDATES``
        (currently 3).

        - When the heuristic default tile_n is 0 (single-tile / small N),
          return ``[0]`` -- the autotuner varies only block_m and threads.
        - Otherwise collect distinct tile_n values from block_m=1..4 and
          return up to ``_MAX_TILE_N_CANDIDATES`` candidates (always
          including the heuristic default).
        """
        default_tn = self._tile_n_for_block_m(1)
        if default_tn == 0:
            return [0]

        candidates: set[int] = {default_tn}
        # Explore tile_n values implied by small block_m values.
        # Higher block_m → smaller tile_n (more N-tiles but better row reuse).
        for bm in (2, 4):
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn > 0 and tn != default_tn:
                candidates.add(tn)

        # Also try half of the default tile_n (rounded to alignment) as a
        # search point when block_m exploration didn't yield alternatives.
        if len(candidates) < 2:
            from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT
            half_tn = (default_tn // 2 // DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT
            if half_tn > 0 and half_tn != default_tn:
                candidates.add(half_tn)

        # Cap to avoid excessive compilation time.
        sorted_candidates = sorted(candidates, reverse=True)
        return sorted_candidates[:self._MAX_TILE_N_CANDIDATES]

    @property
    def autotune_configs(self) -> list[dict]:
        """Generate autotune configs including tile_n candidates.

        tile_n is baked into the kernel at build time, so the autotuner
        rebuilds the kernel for each tile_n value.  Configs include
        ``tile_n`` alongside ``block_m`` and ``threads``.
        """
        budget = self._smem_budget
        smem_per_row = self.N_padded * self._elem_bytes
        max_block_m_no_tile = budget // smem_per_row if smem_per_row > 0 else 16
        threads_list = [128, 256]

        configs = []
        for tile_n in self._tile_n_candidates():
            if tile_n == 0:
                # Single-tile regime: explore multiple block_m values.
                for bm in [1, 2, 4, 8, 16]:
                    try:
                        compute_tile_n(bm, self._elem_bytes, self.N_padded, budget=budget)
                    except ValueError:
                        continue
                    bm_tile_n = self._tile_n_for_block_m(bm)
                    if bm_tile_n != 0:
                        continue
                    if bm > max_block_m_no_tile:
                        continue
                    for t in threads_list:
                        configs.append({"block_m": bm, "threads": t, "tile_n": 0})
            else:
                # Tiled regime: use block_m=1 with each tile_n candidate.
                # Each distinct tile_n triggers a kernel recompilation, so
                # we only vary threads within each tile_n regime.
                for t in threads_list:
                    configs.append({"block_m": 1, "threads": t, "tile_n": tile_n})

        if not configs:
            configs = [{"block_m": 1, "threads": 256, "tile_n": self._tile_n}]

        return configs

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune across tile_n candidates by rebuilding the kernel per regime.

        Groups configs by tile_n, benchmarks each group with its own kernel,
        and picks the overall best (block_m, threads, tile_n) config.
        """
        from tilelang.autotuner import autotune as tl_autotune

        configs = self.autotune_configs
        if not configs:
            return

        # Group configs by tile_n
        by_tile_n: dict[int, list[dict]] = {}
        for cfg in configs:
            tn = cfg["tile_n"]
            by_tile_n.setdefault(tn, []).append(
                {"block_m": cfg["block_m"], "threads": cfg["threads"]}
            )

        best_time = float("inf")
        best_config = None

        for tile_n, group_cfgs in by_tile_n.items():
            kernel = _softmax_kernel(
                self.M, self.N, self.op_kind, self.dtype_str, tile_n,
            )
            autotune_kwargs: dict = dict(
                configs=group_cfgs, warmup=warmup, rep=rep,
            )
            if self.autotune_supply_prog is not None:
                autotune_kwargs["supply_prog"] = self.autotune_supply_prog
            autotuned = tl_autotune(**autotune_kwargs)(kernel)
            tuned = autotuned()
            latency = tuned.latency
            if latency < best_time:
                best_time = latency
                best_config = {**tuned.config, "tile_n": tile_n}

        if best_config is not None:
            self.config = best_config
            # Rebuild kernel for the winning tile_n
            winning_tile_n = best_config["tile_n"]
            if winning_tile_n != self._tile_n:
                self._tile_n = winning_tile_n
                self.kernel = _softmax_kernel(
                    self.M, self.N, self.op_kind, self.dtype_str, self._tile_n,
                )
        print(f'Best config: {self.config}')

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

        # Trim back to N_padded if needed (when tile_n does not divide
        # N_padded, e.g. compute_tile_n chose tile_n_max over a small
        # divisor for better shared memory utilisation).
        if y.shape[1] > self.N_padded:
            y = y[:, : self.N_padded]

        return y
