"""LogSumExp forward kernel using TileLang.

Implements a 2-pass online algorithm for:
  - logsumexp: y[i] = max_i + log(sum_i(exp(x[i,j] - max_i)))

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

__all__ = ["LogSumExpKernel"]


# ---------------------------------------------------------------------------
# Single-tile kernel (N fits in shared memory) -- original fast path
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _logsumexp_kernel_single(M: int, N: int, dtype: str):
    """Build a single-tile logsumexp kernel (N fits in smem).

    Accepts an ``(M, N)`` input tensor.  When ``N`` is not a multiple of
    ``DEFAULT_ALIGNMENT``, the kernel uses element-wise ``T.if_then_else``
    loads that substitute ``-inf`` for out-of-bounds columns (kernel-side
    boundary handling).  When ``N`` is already aligned, the fast ``T.copy``
    path is used.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    _neg_inf = float("-inf")

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
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

                out_local = T.alloc_fragment((block_m,), dtype)
                for i in T.Parallel(block_m):
                    out_local[i] = row_max[i] + T.log(row_sum[i])

                T.copy(out_local, y[pid_m * block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Multi-tile kernel (N tiled over shared memory)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int):
    """Build a multi-tile logsumexp kernel.

    Uses online softmax recurrence across N-tiles:
      Single pass: compute running max and rescaled running sum.
      Then: logsumexp = max + log(sum).

    The input tensor has the raw shape ``(M, N)`` (no host-side padding).
    Boundary handling for the last tile is performed via masked loads.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    num_tiles = (N_padded + tile_n - 1) // tile_n
    total_cols = num_tiles * tile_n
    _needs_mask = total_cols > N
    _neg_inf = float("-inf")

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
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

                # logsumexp = max + log(sum)
                out_local = T.alloc_fragment((block_m,), dtype)
                for i in T.Parallel(block_m):
                    out_local[i] = row_max[i] + T.log(row_sum[i])

                T.copy(out_local, y[pid_m * block_m])

        return main

    return _func


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _logsumexp_kernel(M: int, N: int, dtype: str, tile_n: int = 0):
    """Build the appropriate logsumexp kernel."""
    if tile_n == 0:
        return _logsumexp_kernel_single(M, N, dtype)
    return _logsumexp_kernel_tiled(M, N, dtype, tile_n)


def _compute_padded_cols(N: int, tile_n: int) -> int:
    """Compute the total column count (may exceed N_padded for tiled path)."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    if tile_n == 0:
        return N_padded
    num_tiles = (N_padded + tile_n - 1) // tile_n
    return num_tiles * tile_n


# ---------------------------------------------------------------------------
# custom_op wrappers for torch.compile compatibility
# ---------------------------------------------------------------------------


@torch.library.custom_op("top::logsumexp_fwd", mutates_args=())
def _logsumexp_fwd_wrapped(
    M: int,
    N: int,
    dtype_str: str,
    block_m: int,
    threads: int,
    tile_n: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _logsumexp_kernel(M, N, dtype_str, tile_n)(block_m, threads)(x)


@_logsumexp_fwd_wrapped.register_fake
def _(M, N, dtype_str, block_m, threads, tile_n, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------


def _elem_bytes(dtype: torch.dtype) -> int:
    """Return bytes per element for the given dtype."""
    return torch.tensor([], dtype=dtype).element_size()


class LogSumExpKernel(Kernel):
    """LogSumExp forward kernel.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Implements a 2-pass online algorithm.

    For large N that does not fit in shared memory, tiles over N using
    the online softmax recurrence (running max + rescaled sum).

    Boundary handling for non-aligned N is performed inside the kernel
    via masked loads and ``-inf`` fills, so no host-side ``F.pad`` is
    needed.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: Must be "logsumexp" (kept for API consistency with SoftmaxKernel).
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
        device_index: CUDA device index for shared memory budget query.
            When ``None``, ``torch.cuda.current_device()`` is used.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    _MAX_TILE_N_CANDIDATES = 3

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
        if op_kind != "logsumexp":
            raise ValueError(f"Unsupported op_kind '{op_kind}'. Expected 'logsumexp'.")
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
        self.kernel = _logsumexp_kernel(
            self.M,
            self.N,
            self.dtype_str,
            self._tile_n,
        )

        self.init_config(config, tune)

        # When tune=True, autotune() already set self._tile_n and
        # self.config["tile_n"], and rebuilt the kernel.  Only apply
        # the post-init tile_n fixup for user-provided configs.
        if not tune:
            # If the caller supplied an explicit tile_n (e.g. from a
            # previous autotuner result), honour it.  Only fall back to
            # the heuristic when tile_n was not provided.
            caller_tile_n = config.get("tile_n") if config is not None else None
            if caller_tile_n is not None:
                target_tile_n = caller_tile_n
            else:
                target_tile_n = self._tile_n_for_block_m(self.config["block_m"])
            if target_tile_n != self._tile_n:
                self._tile_n = target_tile_n
                self.kernel = _logsumexp_kernel(
                    self.M,
                    self.N,
                    self.dtype_str,
                    self._tile_n,
                )
            self.config["tile_n"] = self._tile_n

    def _tile_n_for_block_m(self, block_m: int) -> int:
        """Return tile_n for a given block_m (0 means no tiling needed).

        Uses the device's actual shared memory budget (not the
        conservative 48 KiB default) so that large-N workloads can
        use fewer, larger tiles or even the single-tile fast path.

        Both paths are subject to the MAX_SINGLE_TILE_COLS column
        cap (TileLang's vectorizer fails at the 32768 column boundary).
        """
        budget = self._smem_budget
        # Single-tile path: cap by column count and smem budget.
        if self.N_padded <= MAX_SINGLE_TILE_COLS:
            tile_n = compute_tile_n(block_m, self._elem_bytes, self.N_padded, budget=budget)
            if tile_n == self.N_padded:
                return 0
        # Tiled path (logsumexp uses 1 shared buffer).
        # Cap the smem budget so tile_n stays within the column limit.
        col_budget = MAX_SINGLE_TILE_COLS * block_m * self._elem_bytes
        effective_budget = min(budget, col_budget)
        return compute_tile_n(
            block_m, self._elem_bytes, self.N_padded, budget=effective_budget,
        )

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget.

        For the single-tile path (tile_n == 0), prefer the largest
        block_m that fits in shared memory.

        For the tiled path, prefer the block_m that minimises the
        number of N-tiles (maximises tile_n) to reduce global memory
        passes.  Among configs with equal tile count, prefer smaller
        block_m for better occupancy.
        """
        best_bm = 1
        best_tile_n = self._tile_n_for_block_m(1)

        for bm in [2, 4, 8, 16]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn == 0:
                # Single-tile is always better: prefer larger block_m
                best_bm = bm
                best_tile_n = tn
            elif best_tile_n == 0:
                pass
            else:
                best_num = (self.N_padded + best_tile_n - 1) // best_tile_n
                curr_num = (self.N_padded + tn - 1) // tn
                if curr_num < best_num:
                    best_bm = bm
                    best_tile_n = tn

        return {"block_m": best_bm, "threads": 256, "tile_n": best_tile_n}

    def _tile_n_candidates(self) -> list[int]:
        """Return candidate tile_n values for autotune exploration.

        Includes the heuristic tile_n (from block_m=1) plus alternative
        tile_n values derived from ``_tile_n_for_block_m(2)`` and
        ``_tile_n_for_block_m(4)``, with a half-step fallback aligned to
        ``DEFAULT_ALIGNMENT`` when block_m exploration yields no
        alternatives.  tile_n=0 means single-tile (no tiling).  All
        candidates are de-duplicated and sorted descending for
        deterministic ordering.

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
            kernel = _logsumexp_kernel(
                self.M, self.N, self.dtype_str, tile_n,
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
                self.kernel = _logsumexp_kernel(
                    self.M, self.N, self.dtype_str, self._tile_n,
                )
        print(f'Best config: {self.config}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the logsumexp kernel.

        Accepts an ``(M, N)`` tensor.  Boundary handling for non-aligned
        ``N`` is performed inside the GPU kernel (masked loads + ``-inf``
        fill), so no host-side ``F.pad`` is needed.
        """
        tile_n = self._tile_n

        return _logsumexp_fwd_wrapped(
            self.M,
            self.N,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            tile_n,
            x,
        )
