"""Reduce kernels (sum, mean, amin, amax, prod, std, var, var_mean).

Each class is one launch sequence and states the calls it serves over a `ReduceCall`.
``forward`` takes the tensor the op declares and reduces ``call.axes`` of it; the layout
each class runs in and the shape of its result are its own business.
"""

import functools
import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    FP32_EXACT_INT_LIMIT,
    BlockConfigPlanner,
    align_up,
    ceildiv_int,
    down_rows_once,
    down_rows_split,
    down_rows_splits,
    edge_axis_split,
    fold_rows_kernel,
    identity_for,
    restore_reduced,
    rows_for_axes,
    torch_dtype_nbytes,
    tune_by_forward,
)
from tileops.kernels.reduction.call_spec import FOLD_KINDS, SIMPLE_KINDS, WELFORD_KINDS, ReduceCall
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = [
    "ReduceEdgeKernel",
    "ReduceFoldKernel",
    "ReduceKernel",
    "ReduceKernelBase",
    "ReduceLeadingKernel",
    "ReduceProdKernel",
    "RowReduceKernelBase",
    "WelfordEdgeKernel",
    "WelfordReduceKernel",
]


class ReduceKernelBase(Kernel):
    """Built from a `ReduceCall`; shapes an ``[M]`` result the way the op declares it.

    Args:
        call: The call this implementation serves.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(self, call: ReduceCall):
        self.call = call
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.M = call.m
        self.N = call.n
        self.op_kind = call.op_kind
        self.dtype = call.dtype
        self.out_dtype_str = self.dtype_to_str(call.out_dtype or call.dtype)
        self.reduce_axes = call.axes
        self.keepdim = call.keepdim
        self.correction = call.correction

    @classmethod
    def row_planner(cls, call: ReduceCall) -> BlockConfigPlanner:
        """The block planner for the ``(m, n)`` rows the reduced axes flatten to."""
        slots = 2 if call.op_kind in WELFORD_KINDS else 1
        return BlockConfigPlanner(
            align_up(call.n, DEFAULT_ALIGNMENT),
            torch_dtype_nbytes(call.dtype),
            call.smem_budget,
            num_buffers=slots,
            frag_slots=slots,
        )

    @classmethod
    def edge_plan(cls, call: ReduceCall) -> tuple:
        """An edge-axis call's ``(lead, kept, trail)`` extents, rows-pass planner and its config."""
        k, j = edge_axis_split(len(call.shape), call.axes)
        end = len(call.shape) - j
        lead, kept, trail = (math.prod(call.shape[a:b]) for a, b in ((0, k), (k, end), (end, None)))
        slots = 2 if call.op_kind in WELFORD_KINDS else 1
        planner = BlockConfigPlanner(
            align_up(trail, DEFAULT_ALIGNMENT),
            torch_dtype_nbytes(call.dtype),
            call.smem_budget,
            num_buffers=slots,
            frag_slots=slots,
        )
        return lead, kept, trail, planner, planner.default_config()

    @classmethod
    def whole_vector_rows(cls, call: ReduceCall) -> bool:
        """Whether each reduced row is a whole number of 16-byte vectors: the fold's side."""
        return call.n * torch_dtype_nbytes(call.dtype) % VECTOR_ACCESS_BYTES == 0

    @classmethod
    def reduces_edge_axes(cls, call: ReduceCall) -> bool:
        """Whether the axes are a prefix plus a suffix around kept axes, reduced in place.

        The Welford merge folds counts in fp32, whose weights drift past its integer range.
        """
        if edge_axis_split(len(call.shape), call.axes) == (0, 0):
            return False
        return call.op_kind not in WELFORD_KINDS or call.n <= FP32_EXACT_INT_LIMIT

    def _check_arch(self) -> None:
        """Reject construction for an architecture the call's device does not report."""
        if self.call.arch not in self.supported_archs:
            raise ValueError(
                f"{type(self).__name__} is built for architectures "
                f"{sorted(self.supported_archs)}, but the call's device reports {self.call.arch}"
            )

    def forward(self, x: torch.Tensor) -> object:
        """Reduce ``call.axes`` of *x*.

        Args:
            x: The tensor the op declares, contiguous, on a CUDA device.

        Returns:
            The reduced tensor, or ``(var, mean)`` for ``op_kind="var_mean"``.

        Raises:
            ValueError: *x* is not on a CUDA device.
        """
        self._require_cuda(x=x)
        # The 16-byte loads need a storage start on a vector boundary.
        x = x.clone() if x.data_ptr() % VECTOR_ACCESS_BYTES else x
        in_shape = tuple(x.shape)
        result = self._reduce(x)
        if self.op_kind == "var_mean":
            return tuple(
                restore_reduced(y, in_shape, self.reduce_axes, self.keepdim) for y in result
            )
        return restore_reduced(result, in_shape, self.reduce_axes, self.keepdim)

    def _reduce(self, x: torch.Tensor) -> object:
        raise NotImplementedError


class RowReduceKernelBase(ReduceKernelBase):
    """A shared-memory program over the ``(M, N)`` rows the reduced axes flatten to.

    A row one block pass holds is staged whole, ``block_m`` rows a block, and tuned by
    TileLang's autotuner; a longer row is reduced ``tile_n`` columns a step, and each tile
    width is timed through a call.

    Args:
        call: The call this implementation serves.
        config: Optional kernel configuration dict.
    """

    def __init__(self, call: ReduceCall, config: Optional[dict] = None):
        super().__init__(call)
        self._planner = self.row_planner(call)
        self._needs_tiling = self._planner.needs_tiling
        self.kernel = None if self._needs_tiling else self._untiled()
        self.init_config(config, call.tune)
        # A caller-provided config may have block_m without tile_n.
        if self._needs_tiling and not call.tune:
            bm = self.config.get("block_m", 1)
            threads = self.config.get("threads", DEFAULT_THREADS)
            if "tile_n" not in self.config or self.config["tile_n"] == 0:
                self.config["tile_n"] = self._planner.tiled_tile_n(bm, threads)
            reason = self._planner.reject_tile_n(bm, self.config["tile_n"], threads)
            if reason:
                raise ValueError(reason)

    def _untiled(self) -> object:
        """The program for rows one block pass holds."""
        raise NotImplementedError

    def _tiled(self, tile_n: int) -> object:
        """The program for longer rows, at *tile_n* columns a step."""
        raise NotImplementedError

    @property
    def default_config(self) -> dict:
        return self._planner.default_config()

    @property
    def autotune_configs(self) -> list[dict]:
        return self._planner.autotune_configs()

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        if not self._needs_tiling:
            return super().autotune(warmup=warmup, rep=rep)
        probe = torch.randn(self.M, self.N, dtype=self.dtype, device=torch.cuda.current_device())
        tune_by_forward(self, probe, warmup=warmup, rep=rep, forward=self._reduce_rows)

    def _reduce(self, x: torch.Tensor) -> object:
        return self._reduce_rows(rows_for_axes(x, self.reduce_axes))

    def _reduce_rows(self, x: torch.Tensor) -> object:
        program = self._tiled(self.config["tile_n"]) if self._needs_tiling else self.kernel
        results = program(self.config["block_m"], self.config["threads"])(x)
        return tuple(results) if self.op_kind == "var_mean" else results


class ReduceFoldKernel(ReduceKernelBase):
    """Each row of whole 16-byte vectors folded into registers as it is read, one block a row.

    Serves sum, mean, amax, amin, prod and the l1, l2 and inf norms. Tunes the block's
    thread count.
    """

    general = True
    # Vector loads each thread keeps in flight; a grid of few rows is bound by it.
    _UNROLL = 16

    @classmethod
    def applies(cls, call: ReduceCall) -> bool:
        return call.op_kind in FOLD_KINDS and cls.whole_vector_rows(call)

    def __init__(self, call: ReduceCall, config: Optional[dict] = None):
        super().__init__(call)
        self._planner = self.row_planner(call)
        self.kernel = fold_rows_kernel(
            self.M, self.N, self.op_kind, self.dtype_str, self.out_dtype_str, self._UNROLL
        )
        self.init_config(config, call.tune)

    @property
    def default_config(self) -> dict:
        return {"threads": DEFAULT_THREADS}

    @property
    def autotune_configs(self) -> list[dict]:
        # The thread counts the planner admits for a one-row block (any block for prod).
        threads = dict.fromkeys(
            c["threads"]
            for c in self._planner.autotune_configs()
            if self.op_kind == "prod" or c["block_m"] == 1
        )
        return [{"threads": t} for t in threads]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        probe = torch.randn(self.M, self.N, dtype=self.dtype, device=torch.cuda.current_device())
        tune_by_forward(self, probe, warmup=warmup, rep=rep, forward=self._reduce_rows)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self._reduce_rows(rows_for_axes(x, self.reduce_axes))

    def _reduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(self.config["threads"])(x)


class ReduceKernel(RowReduceKernelBase):
    """sum / mean / amax / amin over rows that are not whole vectors, through shared memory.

    Boundary handling for a row that is not a multiple of the 256-element copy alignment is
    a masked load filling the identity element.
    """

    general = True

    @classmethod
    def applies(cls, call: ReduceCall) -> bool:
        return call.op_kind in SIMPLE_KINDS and not cls.whole_vector_rows(call)

    def _untiled(self) -> object:
        return _simple_reduce_kernel(
            self.M, self.N, self.op_kind, self.dtype_str, self.out_dtype_str
        )

    def _tiled(self, tile_n: int) -> object:
        return _simple_reduce_kernel_tiled(
            self.M, self.N, self.op_kind, self.dtype_str, tile_n, self.out_dtype_str
        )


class ReduceProdKernel(ReduceKernelBase):
    """Product of each row that is not whole vectors, staged through shared memory, one block a row."""

    general = True
    # Columns each thread multiplies per staged tile, each into its own fp32 chain.
    _COLS_PER_THREAD = 8

    @classmethod
    def applies(cls, call: ReduceCall) -> bool:
        return call.op_kind == "prod" and not cls.whole_vector_rows(call)

    def __init__(self, call: ReduceCall):
        super().__init__(call)
        self.kernel = _prod_reduce_kernel(
            self.M,
            self.N,
            self.dtype_str,
            self.out_dtype_str,
            DEFAULT_THREADS,
            self._COLS_PER_THREAD,
        )

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel()(rows_for_axes(x, self.reduce_axes))


class WelfordReduceKernel(RowReduceKernelBase):
    """std / var / var_mean of rows, through shared memory; long rows take two tiled passes."""

    general = True

    @classmethod
    def applies(cls, call: ReduceCall) -> bool:
        return call.op_kind in WELFORD_KINDS

    def _untiled(self) -> object:
        return _welford_reduce_kernel(self.M, self.N, self.op_kind, self.correction, self.dtype_str)

    def _tiled(self, tile_n: int) -> object:
        return _welford_reduce_kernel_tiled(
            self.M, self.N, self.op_kind, self.correction, self.dtype_str, tile_n
        )


class ReduceLeadingKernel(ReduceKernelBase):
    """A reduction down the leading axes in the tensor's own ``(N, M)`` layout.

    The down-rows pass splits the reduced axis into row slices where one launch would
    leave the grid short, and finishes the fp32 partials in a second launch.
    """

    @classmethod
    def applies(cls, call: ReduceCall) -> bool:
        return (
            call.op_kind in SIMPLE_KINDS | {"prod"}
            and 0 < len(call.axes) < len(call.shape)
            and call.axes == tuple(range(len(call.axes)))
        )

    def __init__(self, call: ReduceCall):
        super().__init__(call)
        self._splits = down_rows_splits(self.N, self.M)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(self.N, self.M)
        divisor = float(self.N) if self.op_kind == "mean" else 0.0
        if self._splits == 1:
            return down_rows_once(rows, self.op_kind, self.dtype_str, self.out_dtype_str, divisor)
        return down_rows_split(
            rows, self.op_kind, self.dtype_str, self.out_dtype_str, divisor, self._splits
        )


class ReduceEdgeKernel(ReduceKernelBase):
    """A prefix and a suffix of the axes reduced without permuting the tensor.

    The trailing axes reduce as contiguous rows into fp32 partials, tiled where a row
    exceeds one block pass; the leading axes then reduce down the columns of those
    partials, split as `ReduceLeadingKernel` splits. ``mean`` runs the rows pass as
    ``sum`` and divides in the second by the full reduced count.
    """

    @classmethod
    def applies(cls, call: ReduceCall) -> bool:
        return call.op_kind in SIMPLE_KINDS and cls.reduces_edge_axes(call)

    def __init__(self, call: ReduceCall):
        super().__init__(call)
        self._lead, self._kept, self._trail, planner, cfg = self.edge_plan(call)
        inner_kind = "sum" if self.op_kind == "mean" else self.op_kind
        rows = self._lead * self._kept
        if planner.needs_tiling:
            builder = _simple_reduce_kernel_tiled(
                rows, self._trail, inner_kind, self.dtype_str, cfg["tile_n"], "float32"
            )
        else:
            builder = _simple_reduce_kernel(
                rows, self._trail, inner_kind, self.dtype_str, "float32"
            )
        self._rows_pass = builder(cfg["block_m"], cfg["threads"])
        self._splits = down_rows_splits(self._lead, self._kept)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        lead, kept = self._lead, self._kept
        partials = self._rows_pass(x.reshape(lead * kept, self._trail)).reshape(lead, kept)
        divisor = float(lead * self._trail) if self.op_kind == "mean" else 0.0
        # The columns pass writes the storage dtype itself; leaving it in fp32 and
        # casting after costs a third kernel launch.
        if self._splits == 1:
            return down_rows_once(partials, self.op_kind, "float32", self.out_dtype_str, divisor)
        return down_rows_split(
            partials, self.op_kind, "float32", self.out_dtype_str, divisor, self._splits
        )


class WelfordEdgeKernel(ReduceKernelBase):
    """Variance over a prefix and a suffix of the axes without permuting the tensor.

    The rows pass, tiled where a row exceeds one block pass, leaves fp32 ``(mean, M2)``
    per ``(lead, kept)`` slice; Chan's merge folds the slices, so the result is the
    single-pass Welford statistic.
    """

    @classmethod
    def applies(cls, call: ReduceCall) -> bool:
        return call.op_kind in WELFORD_KINDS and cls.reduces_edge_axes(call)

    def __init__(self, call: ReduceCall):
        super().__init__(call)
        lead, kept, self._trail, planner, cfg = self.edge_plan(call)
        self._lead, self._kept = lead, kept
        if planner.needs_tiling:
            builder = _welford_reduce_kernel_tiled(
                lead * kept, self._trail, "partials", 0, self.dtype_str, cfg["tile_n"]
            )
        else:
            builder = _welford_reduce_kernel(
                lead * kept, self._trail, "partials", 0, self.dtype_str
            )
        self._rows_pass = builder(cfg["block_m"], cfg["threads"])
        self._merge = _welford_merge_kernel(
            lead, kept, self.op_kind, self.correction, self._trail, self.dtype_str, DEFAULT_THREADS
        )()

    def _reduce(self, x: torch.Tensor) -> object:
        lead, kept = self._lead, self._kept
        m2_p, mean_p = self._rows_pass(x.reshape(lead * kept, self._trail))
        stat, mean = self._merge(mean_p.reshape(lead, kept), m2_p.reshape(lead, kept))
        return (stat, mean) if self.op_kind == "var_mean" else stat


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


@functools.lru_cache(maxsize=32)
def _prod_reduce_kernel(
    M: int, N: int, dtype: str, out_dtype: str, threads: int, cols_per_thread: int
):
    """Build a product reduce: one block per row, multiplying in fp32.

    Serves rows that are not whole vectors, staging each tile through shared
    memory and filling the columns past the row end with 1.
    """
    chunk = threads * cols_per_thread
    tiles = ceildiv_int(N, chunk)
    num_warps = threads // WARP_LANES

    @tilelang.jit(out_idx=[1])
    def _func():
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                running = T.alloc_local((1,), "float32")
                warp_prod = T.alloc_shared((num_warps,), "float32")
                staged = T.alloc_shared((chunk,), dtype)
                # One independent fp32 chain per slot; a single running product
                # would serialize every multiply behind the previous one.
                slots = T.alloc_local((cols_per_thread,), "float32")

                for c in T.serial(cols_per_thread):
                    slots[c] = T.cast(1.0, "float32")
                for t in T.serial(tiles):
                    for i in T.Parallel(chunk):
                        staged[i] = x[row, t * chunk + i]
                    T.sync_threads()
                    for c in T.serial(cols_per_thread):
                        kept = staged[tx * cols_per_thread + c]
                        col = (t * threads + tx) * cols_per_thread + c
                        slots[c] = slots[c] * T.if_then_else(
                            col < N, T.cast(kept, "float32"), T.cast(1.0, "float32")
                        )
                    T.sync_threads()

                running[0] = slots[0]
                for c in T.serial(1, cols_per_thread):
                    running[0] = running[0] * slots[c]

                for stage in T.serial(WARP_SHUFFLE_STAGES):
                    running[0] = running[0] * T.shfl_xor(
                        running[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES
                    )
                if tx % WARP_LANES == 0:
                    warp_prod[tx // WARP_LANES] = running[0]
                T.sync_threads()
                if tx == 0:
                    for w in T.serial(1, num_warps):
                        warp_prod[0] = warp_prod[0] * warp_prod[w]
                    out[row] = T.cast(warp_prod[0], out_dtype)

        return main

    return _func


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
                        corrected_sum = var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
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
                            corrected = var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            out_local[i] = T.cast(corrected / float(N - correction), dtype)
                    else:  # std
                        for i in T.Parallel(block_m):
                            corrected = var_sum[i] - float(pad_count) * mean_val[i] * mean_val[i]
                            out_local[i] = T.cast(
                                T.sqrt(corrected / float(N - correction)),
                                dtype,
                            )

                    T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


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
                            out_stat[pid_b * threads + tx] = T.cast(m2_acc[0] / denom, out_dtype)
                        out_mean[pid_b * threads + tx] = T.cast(mean_acc[0], out_dtype)

        return main

    return _func
