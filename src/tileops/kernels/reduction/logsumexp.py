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

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    BlockConfigPlanner,
    RowTiledAutotuneMixin,
    align_up,
    ceildiv_int,
    device_smem_budget,
    restore_reduced,
    rows_for_axes,
)
from tileops.kernels.reduction.softmax import _softmax_split_partials_kernel, split_seg_n

# These two kernels bake tile_n in at build time and default to the wider
# thread block; AUTOTUNE_THREADS still bounds what the sweep explores.
_DEFAULT_TUNE_THREADS = 256

__all__ = ["LogSumExpKernel"]


@functools.lru_cache(maxsize=64)
def _logsumexp_split_fold_kernel(M: int, N: int, dtype: str, seg_n: int):
    """Fold per-segment ``(max, sum)`` into one logsumexp per row.

    The fold is over a few hundred fp32 pairs, so one warp per row is enough;
    unlike softmax there is no second pass over the input.
    """
    num_segs = ceildiv_int(N, seg_n)

    @tilelang.jit(out_idx=[2])
    def _func():
        @T.prim_func
        def main(
            seg_max: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            seg_sum: T.Tensor[(M * num_segs,), "float32"],  # noqa: F821
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(M, threads=32) as pid_m:
                tx = T.get_thread_binding()
                row_max = T.alloc_local((1,), "float32")
                row_sum = T.alloc_local((1,), "float32")

                if tx == 0:
                    row_max[0] = -T.infinity("float32")
                    for s in T.serial(num_segs):
                        row_max[0] = T.max(row_max[0], seg_max[pid_m * num_segs + s])
                    row_sum[0] = 0.0
                    # An all--inf segment contributes nothing; folding it
                    # through exp(-inf - row_max) would turn NaN when row_max
                    # is -inf too. An all--inf row then reads -inf + log(0),
                    # which is torch's -inf.
                    for s in T.serial(num_segs):
                        row_sum[0] = row_sum[0] + T.if_then_else(
                            seg_max[pid_m * num_segs + s] == -T.infinity("float32"),
                            T.cast(0.0, "float32"),
                            seg_sum[pid_m * num_segs + s]
                            * T.exp(seg_max[pid_m * num_segs + s] - row_max[0]),
                        )
                    y[pid_m] = T.cast(row_max[0] + T.log(row_sum[0]), dtype)

        return main

    return _func


# Single-tile kernel (N fits in shared memory) -- original fast path


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
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_neg_inf, "float32"),
                            )
                else:
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(x_local[i, j], "float32")

                T.fill(row_max, -T.infinity("float32"))
                T.reduce_max(x_f32, row_max, dim=1, clear=False)

                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        x_f32[i, j] = T.exp(x_f32[i, j] - row_max[i])
                T.reduce_sum(x_f32, row_sum, dim=1)

                out_local = T.alloc_fragment((block_m,), dtype)
                for i in T.Parallel(block_m):
                    out_local[i] = row_max[i] + T.log(row_sum[i])

                T.copy(out_local, y[pid_m * block_m])

        return main

    return _func


# Multi-tile kernel (N tiled over shared memory)


@functools.lru_cache(maxsize=64)
def _logsumexp_kernel_tiled(M: int, N: int, dtype: str, tile_n: int):
    """Build a multi-tile logsumexp kernel.

    Uses online softmax recurrence across N-tiles:
      Single pass: compute running max and rescaled running sum.
      Then: logsumexp = max + log(sum).

    The input tensor has the raw shape $[M \\times N]$ (no host-side padding).
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
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        tile_f32[i, j] = T.cast(tile_local[i, j], "float32")
                            with T.Else():
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
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
                        for i in T.serial(block_m):
                            for j in T.Parallel(tile_n):
                                tile_f32[i, j] = T.cast(tile_local[i, j], "float32")

                    T.fill(tile_max, -T.infinity("float32"))
                    T.reduce_max(tile_f32, tile_max, dim=1, clear=False)

                    for i in T.Parallel(block_m):
                        prev_max[i] = row_max[i]
                        row_max[i] = T.max(row_max[i], tile_max[i])

                    for i in T.serial(block_m):
                        for j in T.Parallel(tile_n):
                            tile_f32[i, j] = T.exp(tile_f32[i, j] - row_max[i])
                    T.reduce_sum(tile_f32, tile_sum, dim=1)

                    for i in T.Parallel(block_m):
                        row_sum[i] = row_sum[i] * T.exp(prev_max[i] - row_max[i]) + tile_sum[i]

                # logsumexp = max + log(sum)
                out_local = T.alloc_fragment((block_m,), dtype)
                for i in T.Parallel(block_m):
                    out_local[i] = row_max[i] + T.log(row_sum[i])

                T.copy(out_local, y[pid_m * block_m])

        return main

    return _func


# Dispatch


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


def _elem_bytes(dtype: torch.dtype) -> int:
    """Return bytes per element for the given dtype."""
    return torch.tensor([], dtype=dtype).element_size()


class LogSumExpKernel(RowTiledAutotuneMixin, Kernel):
    """LogSumExp forward kernel.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Implements a 2-pass online algorithm.

    For large N that does not fit in shared memory, tiles over N using
    the online softmax recurrence (running max + rescaled sum).

    Boundary handling for non-aligned N is performed inside the kernel
    via masked loads and ``-inf`` fills, so no host-side ``F.pad`` is
    needed.

    ``forward`` takes the tensor the op declares and reduces *reduce_axes* of it; moving
    those axes to the end, flattening to rows and shaping the result back are this
    kernel's business, so both sides of the op/backend boundary speak the declared shape.

    Args:
        M: Rows the reduction leaves.
        N: Elements each row reduces.
        op_kind: Must be "logsumexp" (kept for API consistency with SoftmaxKernel).
        dtype: Data type (float32, float16, or bfloat16).
        reduce_axes: Non-negative axis indices, ascending, that the reduction runs over.
        keepdim: Whether a reduced axis stays as a length-1 axis.
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
        reduce_axes: "tuple[int, ...]",
        keepdim: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: int | None = None,
    ):
        super().__init__(device_index=device_index)
        if op_kind != "logsumexp":
            raise ValueError(f"Unsupported op_kind '{op_kind}'. Expected 'logsumexp'.")
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.reduce_axes = tuple(reduce_axes)
        self.keepdim = keepdim
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = _elem_bytes(dtype)
        self._smem_budget = device_smem_budget(device_index)
        self._planner = BlockConfigPlanner(
            self.N_padded,
            self._elem_bytes,
            self._smem_budget,
        )

        # Build self.kernel BEFORE init_config: when tune=True, init_config
        # delegates to autotune() which requires self.kernel to exist.
        #
        # tile_n is baked into the kernel at build time, so pre-compute it from
        # default_config; autotune() rebuilds once per candidate width.
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
            if caller_tile_n == 0:
                caller_tile_n = None
            if caller_tile_n is not None:
                reason = self._planner.reject_tile_n(
                    self.config["block_m"],
                    caller_tile_n,
                    self.config.get("threads", _DEFAULT_TUNE_THREADS),
                )
                if reason:
                    raise ValueError(reason)
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
            if not self._planner.layout_ok(bm, self.N_padded, _DEFAULT_TUNE_THREADS):
                continue
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

        return {"block_m": best_bm, "threads": _DEFAULT_TUNE_THREADS, "tile_n": best_tile_n}

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune across tile_n candidates by rebuilding the kernel per regime.

        Groups configs by tile_n, benchmarks each group with its own kernel,
        and picks the overall best (block_m, threads, tile_n) config.
        """
        from tilelang.autotuner import autotune as tl_autotune

        # The split pair bypasses the tuned kernel; measuring candidates for
        # a path forward will not take is waste, so the default config
        # (whose threads the split kernels share) stands.
        default = self.default_config
        if split_seg_n(self.M, self.N, self.N_padded, ceildiv_int(self.M, default["block_m"])):
            self.config = default
            return

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
                self.M,
                self.N,
                self.dtype_str,
                tile_n,
            )
            autotune_kwargs: dict = dict(
                configs=group_cfgs,
                warmup=warmup,
                rep=rep,
            )
            tunable_params = list(self._autotune_initial_kwargs(kernel, group_cfgs[0]).keys())
            if tunable_params:
                autotune_kwargs["do_not_specialize"] = tunable_params
            if self.autotune_supply_prog is not None:
                autotune_kwargs["supply_prog"] = self.autotune_supply_prog
            autotuned = tl_autotune(**autotune_kwargs)(kernel)
            tuned = self._call_autotuned_kernel(autotuned, kernel, group_cfgs[0])
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
                    self.M,
                    self.N,
                    self.dtype_str,
                    self._tile_n,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce *reduce_axes* of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device. Boundary
                handling for non-aligned ``N`` is performed inside the GPU kernel
                (masked loads + ``-inf`` fill), so no host-side ``F.pad`` is needed.

        Returns:
            The reduced tensor.

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        y = self._reduce_rows(rows_for_axes(x, self.reduce_axes))
        return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)

    def _reduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the trailing axis of an ``(M, N)`` buffer.

        A handful of long rows goes to the split pair: softmax's per-segment
        statistics, then a per-row fold.
        """
        seg_n = split_seg_n(
            self.M,
            self.N,
            self.N_padded,
            (self.M + self.config["block_m"] - 1) // self.config["block_m"],
        )
        if seg_n:
            threads = self.config.get("threads", _DEFAULT_TUNE_THREADS)
            seg_max, seg_sum = _softmax_split_partials_kernel(
                self.M, self.N, seg_n, self.dtype_str, threads
            )()(x)
            return _logsumexp_split_fold_kernel(self.M, self.N, self.dtype_str, seg_n)()(
                seg_max, seg_sum
            )
        program = _logsumexp_kernel(self.M, self.N, self.dtype_str, self._tile_n)
        return program(self.config["block_m"], self.config["threads"])(x)
