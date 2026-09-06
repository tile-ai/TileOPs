"""Reduce kernels (sum, mean, amin, amax, prod, std, var, var_mean)."""

import functools
from dataclasses import dataclass
from math import prod
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    FP32_EXACT_INT_LIMIT,
    VECTOR_ACCESS_BYTES,
    BlockConfigPlanner,
    align_up,
    ceildiv_int,
    device_smem_budget,
    edge_axis_plan,
    edge_axis_split,
    identity_for,
    reduce_down_rows,
    restore_reduced,
    rows_for_axes,
    torch_dtype_nbytes,
    tune_by_forward,
)
from tileops.utils import WARP_LANES

__all__ = ["ReduceKernel"]

_WELFORD_KINDS = {"std", "var", "var_mean"}


# Stride-halving shuffle steps that reduce one warp.
_WARP_STAGES = WARP_LANES.bit_length() - 1

_FRAG_SLOTS = {
    "std": 2,
    "var": 2,
    "var_mean": 2,
}


@dataclass(frozen=True)
class ProductReducePolicy:
    """Launch heuristics for product reductions."""

    cols_per_thread: int = 8

    # Rows below this leave the grid too small to hide the vectorized path's
    # per-thread latency; the shared staging keeps more loads in flight there.
    min_rows_for_vectorized: int = 256


_PROD_POLICY = ProductReducePolicy()


# Simple reduce kernel


_LEADING_AXIS_KINDS = frozenset({"sum", "mean", "amax", "amin"})


class ReduceKernel(Kernel):
    """Unified reduce kernel supporting sum/mean/amin/amax/prod/std/var/var_mean.

    Supports SM80+ architectures. Uses 256-element alignment for shared memory
    copies. Dispatches to simple or Welford kernel based on op_kind.

    A row that exceeds any of the capacities ``BlockConfigPlanner`` tracks goes to a
    tiled variant, which iterates over N in chunks of ``tile_n`` columns.

    Boundary handling for non-aligned N is performed inside the kernel via
    masked loads with identity-element fills, so no host-side ``F.pad`` is
    needed.

    ``forward`` takes the tensor the op declares and reduces *reduce_axes* of it: moving
    those axes to the end, flattening to ``(M, N)`` and shaping the result back are this
    kernel's business, so both sides of the op/backend boundary speak the declared shape.

    Args:
        M: Rows the reduction leaves — the product of the axes it keeps.
        N: Elements each row reduces — the product of *reduce_axes*.
        op_kind: One of sum, mean, amin, amax, prod, std, var, var_mean.
        dtype: Element type of the input.
        reduce_axes: Non-negative axis indices, ascending, that the reduction runs over.
        keepdim: Whether a reduced axis stays as a length-1 axis.
        correction: Bessel's correction, for the Welford kinds.
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).
        device_index: CUDA device the input lives on, for the shared-memory budget.
            ``None`` reads the current device.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        reduce_axes: "tuple[int, ...]",
        keepdim: bool = False,
        correction: int = 1,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.reduce_axes = tuple(reduce_axes)
        self.keepdim = keepdim
        self.correction = correction
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._is_welford = op_kind in _WELFORD_KINDS
        self._is_prod = op_kind == "prod"
        # Whether the axes are a proper prefix needs the rank, so forward settles it.
        self._leading_axis_kind = op_kind in _LEADING_AXIS_KINDS and self.reduce_axes == tuple(
            range(len(self.reduce_axes))
        )
        # Axes hugging both edges also run in the tensor's own layout; the rank
        # check again waits for forward. Welford kinds merge fp32 (mean, M2)
        # partials instead of fp32 scalars.
        self._edge_axis_kind = op_kind in _LEADING_AXIS_KINDS or self._is_welford
        self._elem_bytes = torch_dtype_nbytes(dtype)
        self._smem_budget = device_smem_budget(device_index)
        self._planner = BlockConfigPlanner(
            self.N_padded,
            self._elem_bytes,
            self._smem_budget,
            num_buffers=2 if self._is_welford else 1,
            frag_slots=_FRAG_SLOTS.get(op_kind, 1),
        )
        self._needs_tiling = self._planner.needs_tiling
        self.strategy = self._select_strategy()

        if self.strategy == "prod":
            self.kernel = self._prod_reduce_kernel(self.M, self.N, self.dtype_str, DEFAULT_THREADS)
        elif self.strategy == "welford":
            self.kernel = self._welford_reduce_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.correction,
                self.dtype_str,
            )
        elif self.strategy == "simple":
            self.kernel = self._simple_reduce_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_str,
            )
        # For tiled path, kernel is built lazily using tile_n from config.
        # Tiled kernels use wrapped dispatch functions (not a single self.kernel),
        # so standard autotune via self.kernel is not applicable -- see autotune().
        self.init_config(config, tune)

        # After init_config, ensure tile_n is consistent with the chosen block_m.
        # A caller-provided config may have block_m without tile_n.
        if self._needs_tiling and not tune:
            bm = self.config.get("block_m", 1)
            threads = self.config.get("threads", DEFAULT_THREADS)
            if "tile_n" not in self.config or self.config["tile_n"] == 0:
                self.config["tile_n"] = self._planner.tile_n_for(bm, threads)
            reason = self._planner.reject_tile_n(bm, self.config["tile_n"], threads)
            if reason:
                raise ValueError(reason)

    def _select_strategy(self) -> str:
        """Return the row-wise reduction backend selected by op kind and shape."""
        if self._is_prod:
            return "prod"
        if self._is_welford:
            return "welford_tiled" if self._needs_tiling else "welford"
        return "simple_tiled" if self._needs_tiling else "simple"

    @property
    def default_config(self) -> dict:
        return self._planner.default_config()

    @property
    def autotune_configs(self) -> list[dict]:
        return self._planner.autotune_configs()

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Autotune the reduce kernel by benchmarking candidate configs."""
        if not self._needs_tiling:
            return super().autotune(warmup=warmup, rep=rep)
        x = torch.randn(self.M, self.N, dtype=self.dtype, device=torch.cuda.current_device())
        tune_by_forward(self, x, warmup=warmup, rep=rep, forward=self._reduce_rows)

    def forward(self, x: torch.Tensor) -> object:
        """Reduce *reduce_axes* of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device.

        Returns:
            The reduced tensor, or ``(var, mean)`` for ``op_kind="var_mean"``.

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        in_shape = tuple(x.shape)
        if self._leading_axis_kind and self.reduces_leading_axes(x.ndim, self.reduce_axes):
            columns = self._reduce_leading_axes(x)
            return restore_reduced(columns, in_shape, self.reduce_axes, self.keepdim)
        if self._edge_axis_kind:
            k, j = edge_axis_split(x.ndim, self.reduce_axes)
            # The Welford merge folds counts in fp32; past fp32's integer
            # range the weights drift, so those sizes keep the permute path.
            if k and self._is_welford:
                reduced_count = x.numel() // prod(x.shape[k : x.ndim - j])
                if reduced_count > FP32_EXACT_INT_LIMIT:
                    k = 0
            if k:
                if self._is_welford:
                    y = self._welford_edge_axes(x, k, j)
                    if self.op_kind == "var_mean":
                        var, mean = y
                        return (
                            restore_reduced(var, in_shape, self.reduce_axes, self.keepdim),
                            restore_reduced(mean, in_shape, self.reduce_axes, self.keepdim),
                        )
                else:
                    y = self._reduce_edge_axes(x, k, j)
                return restore_reduced(y, in_shape, self.reduce_axes, self.keepdim)
        rows = rows_for_axes(x, self.reduce_axes)
        result = self._reduce_rows(rows)
        if self.op_kind == "var_mean":
            var, mean = result
            return (
                restore_reduced(var, in_shape, self.reduce_axes, self.keepdim),
                restore_reduced(mean, in_shape, self.reduce_axes, self.keepdim),
            )
        return restore_reduced(result, in_shape, self.reduce_axes, self.keepdim)

    def _reduce_edge_axes(self, x: torch.Tensor, k: int, j: int) -> torch.Tensor:
        """Reduce a prefix and a suffix of the axes without permuting the tensor.

        Two passes in the tensor's own layout: the trailing axes reduce as
        contiguous rows into fp32 partials, then the leading axes reduce down
        the columns of those partials. The permute a single pass needs would
        read and write the whole tensor before reducing anything; the partials
        here are ``lead * kept`` values against the ``lead * kept * trail`` the
        first pass reads.

        Everything between the loads and the one cast at the end stays fp32,
        like the single-pass kernels. ``mean`` runs the first pass as ``sum``
        and divides in the second by the full reduced count.
        """
        lead, kept, trail, planner, cfg = edge_axis_plan(
            tuple(x.shape), k, j, self._elem_bytes, self._smem_budget
        )
        inner_kind = "sum" if self.op_kind == "mean" else self.op_kind
        if planner.needs_tiling:
            rows_stage = self._simple_reduce_kernel_tiled(
                lead * kept,
                trail,
                inner_kind,
                self.dtype_str,
                cfg["tile_n"],
                "float32",
            )
        else:
            rows_stage = self._simple_reduce_kernel(
                lead * kept,
                trail,
                inner_kind,
                self.dtype_str,
                "float32",
            )
        partials = rows_stage(cfg["block_m"], cfg["threads"])(x.reshape(lead * kept, trail))
        divisor = float(lead * trail) if self.op_kind == "mean" else 0.0
        # The columns pass writes the storage dtype itself; leaving it in fp32 and
        # casting after costs a third kernel launch.
        return reduce_down_rows(
            partials.reshape(lead, kept),
            self.op_kind,
            "float32",
            self.dtype_str,
            divisor,
        )

    def _welford_edge_axes(self, x: torch.Tensor, k: int, j: int) -> object:
        """Variance over edge axes without permuting the tensor.

        The rows pass leaves fp32 ``(mean, M2)`` per ``(lead, kept)`` slice;
        Chan's merge folds the slices, so the result is the single-pass
        Welford statistic, not a naive sum of squares.
        """
        lead, kept, trail, planner, cfg = edge_axis_plan(
            tuple(x.shape),
            k,
            j,
            self._elem_bytes,
            self._smem_budget,
            num_buffers=2,
            frag_slots=_FRAG_SLOTS.get(self.op_kind, 1),
        )
        if planner.needs_tiling:
            stage = self._welford_reduce_kernel_tiled(
                lead * kept, trail, "partials", 0, self.dtype_str, cfg["tile_n"]
            )
        else:
            stage = self._welford_reduce_kernel(lead * kept, trail, "partials", 0, self.dtype_str)
        m2_p, mean_p = stage(cfg["block_m"], cfg["threads"])(x.reshape(lead * kept, trail))
        merge = self._welford_merge_kernel(
            lead,
            kept,
            self.op_kind,
            self.correction,
            trail,
            self.dtype_str,
            DEFAULT_THREADS,
        )
        stat, mean = merge()(mean_p.reshape(lead, kept), m2_p.reshape(lead, kept))
        return (stat, mean) if self.op_kind == "var_mean" else stat

    def _reduce_leading_axes(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the leading axes of *x* down to one value per kept column."""
        reduced, kept = self.leading_axis_split(tuple(x.shape), self.reduce_axes)
        divisor = float(reduced) if self.op_kind == "mean" else 0.0
        return reduce_down_rows(
            x.reshape(reduced, kept),
            self.op_kind,
            self.dtype_str,
            self.dtype_str,
            divisor,
        )

    def _stream_vec(self) -> int:
        """Elements a thread folds at once: one vector access of them."""
        return VECTOR_ACCESS_BYTES // self._elem_bytes

    def _stream_fold_applies(self, x: torch.Tensor) -> bool:
        """Whether the rows can be folded straight into registers.

        The tiled path copies the row into shared memory and widens it to fp32
        across a fragment as wide as the tile before reducing that. A fold needs
        neither, and needs only a contiguous row of whole vectors owned by one
        block.
        """
        return (
            self.strategy == "simple_tiled"
            and self.op_kind in ("sum", "mean")
            and self.config["block_m"] == 1
            and x.is_contiguous()
            and self.N % self._stream_vec() == 0
        )

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _stream_kernel(m: int, n: int, op_kind: str, dtype: str, out_dtype: str, vec: int):
        """Build a sum/mean kernel that folds the row into registers as it reads.

        The row is summed in a different order from the tiled path: ``threads *
        vec`` partial sums, each over a strided subsequence, then a tree over
        those. A different order, not a stricter one.
        """

        @tilelang.jit(out_idx=[1])
        def _func(threads: int):
            step = threads * vec
            steps = n // step
            exact = steps * step == n

            @T.prim_func
            def main(x: T.Tensor((m, n), dtype), out: T.Tensor((m,), out_dtype)):
                with T.Kernel(m, threads=threads) as bm:
                    acc = T.alloc_fragment((threads, vec), "float32")
                    lane = T.alloc_fragment((threads,), "float32")
                    red = T.alloc_fragment((1,), "float32")
                    T.clear(acc)
                    for k in T.serial(steps):
                        for i, j in T.Parallel(threads, vec):
                            acc[i, j] += T.Cast("float32", x[bm, (k * threads + i) * vec + j])
                    if not exact:
                        for i, j in T.Parallel(threads, vec):
                            col = steps * step + i * vec + j
                            with T.If(col < n):  # noqa: SIM117
                                with T.Then():
                                    acc[i, j] += T.Cast("float32", x[bm, col])
                    T.reduce_sum(acc, lane, dim=1)
                    T.reduce_sum(lane, red, dim=0)
                    total = red[0] / float(n) if op_kind == "mean" else red[0]
                    out[bm] = T.Cast(out_dtype, total)

            return main

        return _func

    def _reduce_rows(self, x: torch.Tensor) -> object:
        """Reduce the trailing axis of an ``(M, N)`` buffer."""
        if self.strategy == "prod":
            return self.kernel()(x)
        block_m, threads = self.config["block_m"], self.config["threads"]
        if self.strategy in {"welford", "welford_tiled"}:
            if self.strategy == "welford_tiled":
                program = self._welford_reduce_kernel_tiled(
                    self.M,
                    self.N,
                    self.op_kind,
                    self.correction,
                    self.dtype_str,
                    self.config["tile_n"],
                )
            else:
                program = self.kernel
            results = program(block_m, threads)(x)
            if self.op_kind == "var_mean":
                return results[0], results[1]
            return results
        if self._stream_fold_applies(x):
            program = self._stream_kernel(
                self.M,
                self.N,
                self.op_kind,
                self.dtype_str,
                self.dtype_str,
                self._stream_vec(),
            )
            return program(threads)(x)
        if self.strategy == "simple_tiled":
            program = self._simple_reduce_kernel_tiled(
                self.M, self.N, self.op_kind, self.dtype_str, self.config["tile_n"]
            )
        else:
            program = self.kernel
        return program(block_m, threads)(x)

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _simple_reduce_kernel(M, N, op_kind, dtype, out_dtype=None):
        """Build a simple reduce kernel for sum/mean/amax/amin.

        Accepts an ``(M, N)`` input tensor.  When ``N`` is not a multiple of
        ``DEFAULT_ALIGNMENT``, the kernel uses element-wise ``T.if_then_else``
        loads that substitute the identity element for out-of-bounds columns
        (kernel-side boundary handling).  When ``N`` is already aligned, the
        fast ``T.copy`` path is used.

        ``out_dtype`` overrides the output element type, for a caller that keeps
        fp32 partials across a second pass.
        """
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        _needs_pad = N_padded != N
        _pad_val = identity_for(op_kind)
        out_dtype = out_dtype or dtype

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out: T.Tensor[(M,), out_dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")
                    out_local = T.alloc_fragment((block_m,), out_dtype)

                    if _needs_pad:
                        # Kernel-side boundary handling: element-wise load
                        # with T.if_then_else masking for padding columns
                        # and row-tail safety (M % block_m != 0).
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = T.if_then_else(
                                    T.And(pid_m * block_m + i < M, j < N),
                                    T.cast(x[pid_m * block_m + i, j], "float32"),
                                    T.cast(_pad_val, "float32"),
                                )
                    else:
                        # Load via shared memory (fast vectorized path)
                        T.copy(x[pid_m * block_m, 0], shared_buf)

                        # Cast to fp32
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                    if op_kind == "sum":
                        T.reduce_sum(x_f32, acc, dim=1)
                    elif op_kind == "mean":
                        T.reduce_sum(x_f32, acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = acc[i] / float(N)
                    elif op_kind == "amax":
                        T.reduce_max(x_f32, acc, dim=1)
                    elif op_kind == "amin":
                        # Negate, reduce_max, negate back
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                x_f32[i, j] = -x_f32[i, j]
                        T.reduce_max(x_f32, acc, dim=1)
                        for i in T.Parallel(block_m):
                            acc[i] = -acc[i]

                    # Cast back to output dtype
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i], out_dtype)

                    T.copy(out_local, out[pid_m * block_m])

            return main

        return _func

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n, out_dtype=None):
        """Tiled simple reduce for N_padded > MAX_SINGLE_TILE_COLS.

        Iterates over N in chunks of ``tile_n`` columns, accumulating
        partial results.  The last tile uses masked loads when
        ``num_tiles * tile_n > N``.
        """
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        num_tiles = (N_padded + tile_n - 1) // tile_n
        total_cols = num_tiles * tile_n
        _needs_mask = total_cols > N
        _pad_val = identity_for(op_kind)
        out_dtype = out_dtype or dtype

        @tilelang.jit(out_idx=[1])
        def _func(block_m, threads):
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out: T.Tensor[(M,), out_dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                    tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")
                    tile_acc = T.alloc_fragment((block_m,), "float32")
                    out_local = T.alloc_fragment((block_m,), out_dtype)

                    # Initialize accumulator
                    if op_kind in ("sum", "mean"):
                        T.fill(acc, 0.0)
                    elif op_kind == "amax":
                        T.fill(acc, -T.infinity("float32"))
                    elif op_kind == "amin":
                        T.fill(acc, T.infinity("float32"))

                    for t in T.Serial(num_tiles):
                        if _needs_mask:
                            with T.If(t < num_tiles - 1):
                                with T.Then():
                                    T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                with T.Else():
                                    for i in T.serial(block_m):
                                        for j in T.Parallel(tile_n):
                                            tile_f32[i, j] = T.if_then_else(
                                                T.And(
                                                    pid_m * block_m + i < M,
                                                    t * tile_n + j < N,
                                                ),
                                                T.cast(
                                                    x[pid_m * block_m + i, t * tile_n + j],
                                                    "float32",
                                                ),
                                                T.cast(_pad_val, "float32"),
                                            )
                        else:
                            T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        # Tile-local reduce
                        if op_kind in ("sum", "mean"):
                            T.reduce_sum(tile_f32, tile_acc, dim=1)
                            for i in T.Parallel(block_m):
                                acc[i] = acc[i] + tile_acc[i]
                        elif op_kind == "amax":
                            T.reduce_max(tile_f32, tile_acc, dim=1)
                            for i in T.Parallel(block_m):
                                acc[i] = T.max(acc[i], tile_acc[i])
                        elif op_kind == "amin":
                            # Negate, reduce_max, negate back
                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    tile_f32[i, j] = -tile_f32[i, j]
                            T.reduce_max(tile_f32, tile_acc, dim=1)
                            for i in T.Parallel(block_m):
                                acc[i] = T.min(acc[i], -tile_acc[i])

                    # Finalize
                    if op_kind == "mean":
                        for i in T.Parallel(block_m):
                            out_local[i] = T.cast(acc[i] / float(N), out_dtype)
                    else:
                        for i in T.Parallel(block_m):
                            out_local[i] = T.cast(acc[i], out_dtype)

                    T.copy(out_local, out[pid_m * block_m])

            return main

        return _func

    @staticmethod
    def leading_axis_split(shape: "tuple[int, ...]", axes: "tuple[int, ...]") -> "tuple[int, int]":
        """Split *shape* into ``(reduced, kept)`` element counts for a leading-axis reduce.

        Only valid when *axes* is ``(0, 1, ... k-1)``: the reduced axes are then the
        outermost ones, so a contiguous tensor reshapes to ``(reduced, kept)`` for free and
        the reduction runs down the rows of that view.
        """
        k = len(axes)
        reduced = 1
        for d in shape[:k]:
            reduced *= d
        kept = 1
        for d in shape[k:]:
            kept *= d
        return reduced, kept

    @staticmethod
    def reduces_leading_axes(ndim: int, axes: "tuple[int, ...]") -> bool:
        """Whether *axes* is a non-empty proper prefix of the axes of an *ndim* tensor."""
        return 0 < len(axes) < ndim and tuple(axes) == tuple(range(len(axes)))

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _prod_reduce_kernel(M: int, N: int, dtype: str, threads: int):
        """Build a product reduce: one block per row, multiplying in fp32.

        With enough rows to fill the device, each thread reads its
        ``cols_per_thread`` consecutive elements per tile with vectorized accesses
        straight into registers. Below ``min_rows_for_vectorized`` rows, or when
        the tile does not divide the row, the row is staged through shared memory
        instead. The two bodies cannot be merged: TileLang rejects a
        ``T.vectorized`` loop and a guarded tail load of the same buffer in one
        kernel.
        """
        chunk = threads * _PROD_POLICY.cols_per_thread
        vectorized = N % chunk == 0 and _PROD_POLICY.min_rows_for_vectorized <= M
        full_tiles = N // chunk if vectorized else 0
        tiles = ceildiv_int(N, chunk)
        exact = tiles * chunk == N
        num_warps = threads // WARP_LANES
        # One vector access is at most 16 bytes; wider per-thread spans split into
        # several back-to-back vector loads.
        vec_elems = min(
            _PROD_POLICY.cols_per_thread, VECTOR_ACCESS_BYTES // torch_dtype_nbytes(dtype)
        )
        vec_groups = _PROD_POLICY.cols_per_thread // vec_elems

        @tilelang.jit(out_idx=[1])
        def _func():
            @T.prim_func
            def main(
                x: T.Tensor[(M, N), dtype],
                out: T.Tensor[(M,), dtype],
            ):
                with T.Kernel(M, threads=threads) as row:
                    tx = T.get_thread_binding()
                    running = T.alloc_local((1,), "float32")
                    warp_prod = T.alloc_shared((num_warps,), "float32")
                    held = T.alloc_local((_PROD_POLICY.cols_per_thread,), dtype)
                    staged = T.alloc_shared((1 if vectorized else chunk,), dtype)
                    # One independent fp32 chain per slot; a single running product
                    # would serialize every multiply behind the previous one.
                    slots = T.alloc_local((_PROD_POLICY.cols_per_thread,), "float32")

                    for c in T.serial(_PROD_POLICY.cols_per_thread):
                        slots[c] = T.cast(1.0, "float32")
                    if vectorized:
                        for t in T.serial(full_tiles):
                            for g in T.serial(vec_groups):
                                for c in T.vectorized(vec_elems):
                                    held[g * vec_elems + c] = x[
                                        row,
                                        t * chunk
                                        + tx * _PROD_POLICY.cols_per_thread
                                        + g * vec_elems
                                        + c,
                                    ]
                            for c in T.serial(_PROD_POLICY.cols_per_thread):
                                slots[c] = slots[c] * T.cast(held[c], "float32")
                    else:
                        for t in T.serial(tiles):
                            for i in T.Parallel(chunk):
                                staged[i] = x[row, t * chunk + i]
                            T.sync_threads()
                            for c in T.serial(_PROD_POLICY.cols_per_thread):
                                kept = staged[tx * _PROD_POLICY.cols_per_thread + c]
                                if exact:
                                    slots[c] = slots[c] * T.cast(kept, "float32")
                                else:
                                    col = (t * threads + tx) * _PROD_POLICY.cols_per_thread + c
                                    slots[c] = slots[c] * T.if_then_else(
                                        col < N, T.cast(kept, "float32"), T.cast(1.0, "float32")
                                    )
                            T.sync_threads()

                    running[0] = slots[0]
                    for c in T.serial(1, _PROD_POLICY.cols_per_thread):
                        running[0] = running[0] * slots[c]

                    for stage in T.serial(_WARP_STAGES):
                        running[0] = running[0] * T.shfl_xor(
                            running[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES
                        )
                    if tx % WARP_LANES == 0:
                        warp_prod[tx // WARP_LANES] = running[0]
                    T.sync_threads()
                    if tx == 0:
                        for w in T.serial(1, num_warps):
                            warp_prod[0] = warp_prod[0] * warp_prod[w]
                        out[row] = T.cast(warp_prod[0], dtype)

            return main

        return _func

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _welford_reduce_kernel(M, N, op_kind, correction, dtype):
        """Build a Welford-based reduce kernel for std/var/var_mean.

        Accepts an ``(M, N)`` input tensor.  Padding columns are filled with
        ``0.0`` via masked loads when ``N`` is not aligned.  The padding
        correction (subtracting ``pad_count * mean^2`` from the variance sum)
        is applied analytically, so the result is exact regardless of padding.

        ``op_kind="partials"`` writes each row's fp32 ``(M2, mean)`` undivided,
        for a cross-row merge; *correction* is unused there.
        """
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        _needs_pad = N_padded != N
        _pair = op_kind in ("var_mean", "partials")
        _out_dtype = "float32" if op_kind == "partials" else dtype

        out_idx = [1, 2] if _pair else [1]

        @tilelang.jit(out_idx=out_idx)
        def _func(block_m, threads):
            if _pair:

                @T.prim_func
                def main(
                    x: T.Tensor[(M, N), dtype],
                    out_var: T.Tensor[(M,), _out_dtype],
                    out_mean: T.Tensor[(M,), _out_dtype],
                ):
                    with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                        shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                        x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                        row_sum = T.alloc_fragment((block_m,), "float32")
                        mean_val = T.alloc_fragment((block_m,), "float32")
                        sq_diff = T.alloc_fragment((block_m, N_padded), "float32")
                        var_sum = T.alloc_fragment((block_m,), "float32")
                        out_v = T.alloc_fragment((block_m,), _out_dtype)
                        out_m = T.alloc_fragment((block_m,), _out_dtype)

                        if _needs_pad:
                            for i in T.serial(block_m):
                                for j in T.Parallel(N_padded):
                                    x_f32[i, j] = T.if_then_else(
                                        T.And(pid_m * block_m + i < M, j < N),
                                        T.cast(x[pid_m * block_m + i, j], "float32"),
                                        T.cast(0.0, "float32"),
                                    )
                        else:
                            T.copy(x[pid_m * block_m, 0], shared_buf)

                            for i in T.serial(block_m):
                                for j in T.Parallel(N_padded):
                                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.reduce_sum(x_f32, row_sum, dim=1)
                        for i in T.Parallel(block_m):
                            mean_val[i] = row_sum[i] / float(N)

                        # Variance: sum((x - mean)^2) / (N - correction)
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                dev = x_f32[i, j] - mean_val[i]
                                sq_diff[i, j] = dev * dev

                        T.reduce_sum(sq_diff, var_sum, dim=1)

                        # Correct for padding: padded elements contribute mean^2 each
                        pad_count = N_padded - N
                        for i in T.Parallel(block_m):
                            corrected_sum = (
                                var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            )
                            if op_kind == "partials":
                                out_v[i] = corrected_sum
                                out_m[i] = mean_val[i]
                            else:
                                out_v[i] = T.cast(corrected_sum / float(N - correction), _out_dtype)
                                out_m[i] = T.cast(mean_val[i], _out_dtype)

                        T.copy(out_v, out_var[pid_m * block_m])
                        T.copy(out_m, out_mean[pid_m * block_m])

            else:
                # std or var (single output)
                @T.prim_func
                def main(
                    x: T.Tensor[(M, N), dtype],
                    out: T.Tensor[(M,), dtype],
                ):
                    with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                        shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                        x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                        row_sum = T.alloc_fragment((block_m,), "float32")
                        mean_val = T.alloc_fragment((block_m,), "float32")
                        sq_diff = T.alloc_fragment((block_m, N_padded), "float32")
                        var_sum = T.alloc_fragment((block_m,), "float32")
                        out_local = T.alloc_fragment((block_m,), dtype)

                        if _needs_pad:
                            for i in T.serial(block_m):
                                for j in T.Parallel(N_padded):
                                    x_f32[i, j] = T.if_then_else(
                                        T.And(pid_m * block_m + i < M, j < N),
                                        T.cast(x[pid_m * block_m + i, j], "float32"),
                                        T.cast(0.0, "float32"),
                                    )
                        else:
                            T.copy(x[pid_m * block_m, 0], shared_buf)

                            for i in T.serial(block_m):
                                for j in T.Parallel(N_padded):
                                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                        T.reduce_sum(x_f32, row_sum, dim=1)
                        for i in T.Parallel(block_m):
                            mean_val[i] = row_sum[i] / float(N)

                        # Variance
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                dev = x_f32[i, j] - mean_val[i]
                                sq_diff[i, j] = dev * dev

                        T.reduce_sum(sq_diff, var_sum, dim=1)

                        pad_count = N_padded - N
                        if op_kind == "var":
                            for i in T.Parallel(block_m):
                                corrected_sum = (
                                    var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                                )
                                out_local[i] = T.cast(corrected_sum / float(N - correction), dtype)
                        else:  # std
                            for i in T.Parallel(block_m):
                                corrected_sum = (
                                    var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                                )
                                out_local[i] = T.cast(
                                    T.sqrt(corrected_sum / float(N - correction)), dtype
                                )

                        T.copy(out_local, out[pid_m * block_m])

            return main

        return _func

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _welford_reduce_kernel_tiled(M, N, op_kind, correction, dtype, tile_n):
        """Tiled Welford reduce for N_padded > MAX_SINGLE_TILE_COLS.

        Two-pass approach over N tiles:
          Pass 1: accumulate row sum for mean computation.
          Pass 2: accumulate sum of squared deviations from the mean.

        ``op_kind="partials"`` as in ``_welford_reduce_kernel``.
        """
        N_padded = align_up(N, DEFAULT_ALIGNMENT)
        num_tiles = (N_padded + tile_n - 1) // tile_n
        total_cols = num_tiles * tile_n
        _needs_mask = total_cols > N
        _pair = op_kind in ("var_mean", "partials")
        _out_dtype = "float32" if op_kind == "partials" else dtype

        out_idx = [1, 2] if _pair else [1]

        @tilelang.jit(out_idx=out_idx)
        def _func(block_m, threads):
            if _pair:

                @T.prim_func
                def main(
                    x: T.Tensor[(M, N), dtype],
                    out_var: T.Tensor[(M,), _out_dtype],
                    out_mean: T.Tensor[(M,), _out_dtype],
                ):
                    with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                        shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                        tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                        tile_sum = T.alloc_fragment((block_m,), "float32")
                        row_sum = T.alloc_fragment((block_m,), "float32")
                        mean_val = T.alloc_fragment((block_m,), "float32")
                        sq_diff = T.alloc_fragment((block_m, tile_n), "float32")
                        tile_sq = T.alloc_fragment((block_m,), "float32")
                        var_sum = T.alloc_fragment((block_m,), "float32")
                        out_v = T.alloc_fragment((block_m,), _out_dtype)
                        out_m = T.alloc_fragment((block_m,), _out_dtype)

                        T.fill(row_sum, 0.0)

                        # Pass 1: compute row sums for mean
                        for t in T.Serial(num_tiles):
                            if _needs_mask:
                                with T.If(t < num_tiles - 1):
                                    with T.Then():
                                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                    with T.Else():
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                tile_f32[i, j] = T.if_then_else(
                                                    T.And(
                                                        pid_m * block_m + i < M,
                                                        t * tile_n + j < N,
                                                    ),
                                                    T.cast(
                                                        x[pid_m * block_m + i, t * tile_n + j],
                                                        "float32",
                                                    ),
                                                    0.0,
                                                )
                            else:
                                T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                            T.reduce_sum(tile_f32, tile_sum, dim=1)
                            for i in T.Parallel(block_m):
                                row_sum[i] = row_sum[i] + tile_sum[i]

                        for i in T.Parallel(block_m):
                            mean_val[i] = row_sum[i] / float(N)

                        # Pass 2: dedicated buffers to avoid TileLang aliasing
                        p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                        p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                        T.fill(var_sum, 0.0)

                        for t in T.Serial(num_tiles):
                            if _needs_mask:
                                with T.If(t < num_tiles - 1):
                                    with T.Then():
                                        T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")
                                    with T.Else():
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                p2_f32[i, j] = T.if_then_else(
                                                    T.And(
                                                        pid_m * block_m + i < M,
                                                        t * tile_n + j < N,
                                                    ),
                                                    T.cast(
                                                        x[pid_m * block_m + i, t * tile_n + j],
                                                        "float32",
                                                    ),
                                                    0.0,
                                                )
                            else:
                                T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")

                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    sq_diff[i, j] = (p2_f32[i, j] - mean_val[i]) * (
                                        p2_f32[i, j] - mean_val[i]
                                    )
                            T.reduce_sum(sq_diff, tile_sq, dim=1)
                            for i in T.Parallel(block_m):
                                var_sum[i] = var_sum[i] + tile_sq[i]

                        # Correct for padding: out-of-bound elements were filled
                        # with 0.0, so each contributes mean^2 to the sq_diff sum.
                        pad_count = total_cols - N
                        for i in T.Parallel(block_m):
                            corrected = var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            if op_kind == "partials":
                                out_v[i] = corrected
                                out_m[i] = mean_val[i]
                            else:
                                out_v[i] = T.cast(corrected / float(N - correction), _out_dtype)
                                out_m[i] = T.cast(mean_val[i], _out_dtype)

                        T.copy(out_v, out_var[pid_m * block_m])
                        T.copy(out_m, out_mean[pid_m * block_m])

            else:
                # std or var (single output)
                @T.prim_func
                def main(
                    x: T.Tensor[(M, N), dtype],
                    out: T.Tensor[(M,), dtype],
                ):
                    with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                        shared_buf = T.alloc_shared((block_m, tile_n), dtype)
                        tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                        tile_sum = T.alloc_fragment((block_m,), "float32")
                        row_sum = T.alloc_fragment((block_m,), "float32")
                        mean_val = T.alloc_fragment((block_m,), "float32")
                        sq_diff = T.alloc_fragment((block_m, tile_n), "float32")
                        tile_sq = T.alloc_fragment((block_m,), "float32")
                        var_sum = T.alloc_fragment((block_m,), "float32")
                        out_local = T.alloc_fragment((block_m,), dtype)

                        T.fill(row_sum, 0.0)

                        # Pass 1: compute row sums for mean
                        for t in T.Serial(num_tiles):
                            if _needs_mask:
                                with T.If(t < num_tiles - 1):
                                    with T.Then():
                                        T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                                    with T.Else():
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                tile_f32[i, j] = T.if_then_else(
                                                    T.And(
                                                        pid_m * block_m + i < M,
                                                        t * tile_n + j < N,
                                                    ),
                                                    T.cast(
                                                        x[pid_m * block_m + i, t * tile_n + j],
                                                        "float32",
                                                    ),
                                                    0.0,
                                                )
                            else:
                                T.copy(x[pid_m * block_m, t * tile_n], shared_buf)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        tile_f32[i, j] = T.cast(shared_buf[i, j], "float32")

                            T.reduce_sum(tile_f32, tile_sum, dim=1)
                            for i in T.Parallel(block_m):
                                row_sum[i] = row_sum[i] + tile_sum[i]

                        for i in T.Parallel(block_m):
                            mean_val[i] = row_sum[i] / float(N)

                        # Pass 2: dedicated buffers
                        p2_shared = T.alloc_shared((block_m, tile_n), dtype)
                        p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")
                        T.fill(var_sum, 0.0)

                        for t in T.Serial(num_tiles):
                            if _needs_mask:
                                with T.If(t < num_tiles - 1):
                                    with T.Then():
                                        T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")
                                    with T.Else():
                                        for i in T.serial(block_m):
                                            for j in T.Parallel(tile_n):
                                                p2_f32[i, j] = T.if_then_else(
                                                    T.And(
                                                        pid_m * block_m + i < M,
                                                        t * tile_n + j < N,
                                                    ),
                                                    T.cast(
                                                        x[pid_m * block_m + i, t * tile_n + j],
                                                        "float32",
                                                    ),
                                                    0.0,
                                                )
                            else:
                                T.copy(x[pid_m * block_m, t * tile_n], p2_shared)
                                for i in T.serial(block_m):
                                    for j in T.Parallel(tile_n):
                                        p2_f32[i, j] = T.cast(p2_shared[i, j], "float32")

                            for i in T.serial(block_m):
                                for j in T.Parallel(tile_n):
                                    sq_diff[i, j] = (p2_f32[i, j] - mean_val[i]) * (
                                        p2_f32[i, j] - mean_val[i]
                                    )
                            T.reduce_sum(sq_diff, tile_sq, dim=1)
                            for i in T.Parallel(block_m):
                                var_sum[i] = var_sum[i] + tile_sq[i]

                        pad_count = total_cols - N
                        if op_kind == "var":
                            for i in T.Parallel(block_m):
                                corrected = (
                                    var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                                )
                                out_local[i] = T.cast(corrected / float(N - correction), dtype)
                        else:  # std
                            for i in T.Parallel(block_m):
                                corrected = (
                                    var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                                )
                                out_local[i] = T.cast(
                                    T.sqrt(corrected / float(N - correction)),
                                    dtype,
                                )

                        T.copy(out_local, out[pid_m * block_m])

            return main

        return _func

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def _welford_merge_kernel(A, B, op_kind, correction, count_per, out_dtype, threads):
        """Merge per-slice Welford partials down the lead axis with Chan's method.

        ``mean_in`` / ``m2_in`` hold ``A`` slices of ``B`` columns, each statistic
        over ``count_per`` elements. One thread owns a kept column and folds the
        slices serially; adjacent threads read adjacent columns, so every step of
        the walk is one coalesced pass over a buffer of ``A * B`` values.

        Every kind writes the ``(statistic, mean)`` pair -- ``B`` extra values --
        so one prim_func serves the three; the caller keeps what its op returns.
        """
        denom = float(A * count_per - correction)
        nb = float(count_per)

        @tilelang.jit(out_idx=[2, 3])
        def _func():
            @T.prim_func
            def main(
                mean_in: T.Tensor[(A, B), "float32"],  # noqa: F821
                m2_in: T.Tensor[(A, B), "float32"],  # noqa: F821
                out_stat: T.Tensor[(B,), out_dtype],
                out_mean: T.Tensor[(B,), out_dtype],
            ):
                with T.Kernel(T.ceildiv(B, threads), threads=threads) as pid_b:
                    tx = T.get_thread_binding()
                    n_acc = T.alloc_local((1,), "float32")
                    mean_acc = T.alloc_local((1,), "float32")
                    m2_acc = T.alloc_local((1,), "float32")
                    delta = T.alloc_local((1,), "float32")

                    n_acc[0] = 0.0
                    mean_acc[0] = 0.0
                    m2_acc[0] = 0.0
                    with T.If(pid_b * threads + tx < B):  # noqa: SIM117
                        with T.Then():
                            for a in T.serial(A):
                                delta[0] = mean_in[a, pid_b * threads + tx] - mean_acc[0]
                                m2_acc[0] = (
                                    m2_acc[0]
                                    + m2_in[a, pid_b * threads + tx]
                                    + delta[0] * delta[0] * (n_acc[0] * nb / (n_acc[0] + nb))
                                )
                                mean_acc[0] = mean_acc[0] + delta[0] * (nb / (n_acc[0] + nb))
                                n_acc[0] = n_acc[0] + nb
                            if op_kind == "std":
                                out_stat[pid_b * threads + tx] = T.cast(
                                    T.sqrt(m2_acc[0] / denom), out_dtype
                                )
                            else:
                                out_stat[pid_b * threads + tx] = T.cast(
                                    m2_acc[0] / denom, out_dtype
                                )
                            out_mean[pid_b * threads + tx] = T.cast(mean_acc[0], out_dtype)

            return main

        return _func
