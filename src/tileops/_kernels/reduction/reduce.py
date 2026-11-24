"""Reduce kernels (sum, mean, amin, amax, prod, std, var, var_mean).

Each class is one launch sequence and states the calls it serves over a `ReduceCall`.
``forward`` takes the tensor the op declares and reduces ``call.axes`` of it; the layout
each class runs in and the shape of its result are its own business.
"""

import math
import operator
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import CompiledProgram, Kernel
from tileops._kernels.reduction._primitives import (
    AUTOTUNE_THREADS,
    DEFAULT_THREADS,
    FP32_EXACT_INT_LIMIT,
    BlockConfigPlanner,
    block_fold,
    down_rows_launcher,
    down_rows_splits,
    edge_axis_split,
    extrema_combine,
    fold_rows_kernel,
    identity_for,
    reduced_shape,
    row_layout,
    tune_by_forward,
)
from tileops._kernels.reduction.call_spec import (
    ProdFwdInterface,
    ReduceCall,
    ReduceFwdInterface,
    VarianceFwdInterface,
    VarMeanFwdInterface,
    VectorNormFwdInterface,
)
from tileops._kernels.tiling import ALIGNMENT, align_up
from tileops._kernels.warp_reduce import warp_reduce
from tileops._utils import WARP_LANES
from tileops.manifest.primitives import ceil_div

__all__ = [
    "EdgeReduceKernelBase",
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
        super().__init__(device_index=call.device_index)
        self.M = call.m
        self.N = call.n
        self.op_kind = call.op_kind
        self.dtype = call.dtype
        self.out_dtype_str = self.dtype_to_str(call.out_dtype or call.dtype)
        self.correction = call.correction
        self._permutation, _, _ = row_layout(call.shape, call.axes)
        self._resultshape = reduced_shape(call.shape, call.axes, call.keepdim)

    @classmethod
    def row_planner(cls, call: ReduceCall) -> BlockConfigPlanner:
        """The block planner for the ``(m, n)`` rows the reduced axes flatten to."""
        return cls._planner_for(call, call.n)

    @staticmethod
    def _planner_for(call: ReduceCall, n: int) -> BlockConfigPlanner:
        """The block planner for rows of *n* elements of *call*'s dtype and kind."""
        slots = 2 if call.op_kind in ("std", "var", "var_mean") else 1
        # The tiled simple reduce's reduction across threads takes one fp32 per thread of
        # shared memory besides its tile, at the widest thread count the tuner offers.
        # Welford's plan already counts a second tile, held only in its second pass and
        # wider than this.
        tiled_workspace_bytes = max(AUTOTUNE_THREADS) * 4
        return BlockConfigPlanner(
            align_up(n, ALIGNMENT),
            call.dtype.itemsize,
            call.smem_budget,
            num_buffers=slots,
            frag_slots=slots,
            workspace_bytes=0 if slots == 2 else tiled_workspace_bytes,
            split_workspace=True,
        )

    @classmethod
    def edge_plan(cls, call: ReduceCall) -> tuple:
        """An edge-axis call's ``(lead, kept, trail)`` extents, rows-pass planner and its config."""
        k, j = edge_axis_split(len(call.shape), call.axes)
        end = len(call.shape) - j
        lead, kept, trail = (math.prod(call.shape[a:b]) for a, b in ((0, k), (k, end), (end, None)))
        planner = cls._planner_for(call, trail)
        return lead, kept, trail, planner, planner.default_config()

    @classmethod
    def reduces_edge_axes(cls, call: ReduceCall) -> bool:
        """Whether the axes are a prefix plus a suffix around kept axes, reduced in place."""
        return edge_axis_split(len(call.shape), call.axes) != (0, 0)

    def _check_arch(self) -> None:
        """Reject construction for an architecture the call's device does not report."""
        if self.call.arch not in self.supported_archs:
            from tileops.backend import OpNotAvailableError

            raise OpNotAvailableError(
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
        self._require_devices(x=x)
        result = self._reduce(x)
        if self.op_kind == "var_mean":
            return tuple(y.reshape(self._resultshape) for y in result)
        return result.reshape(self._resultshape)

    def _rows(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permutation).contiguous().reshape(self.M, self.N)

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

    def __init__(self, call: ReduceCall, config: dict | None = None):
        super().__init__(call)
        self._planner = self.row_planner(call)
        self._needs_tiling = self._planner.needs_tiling
        self.kernel = None if self._needs_tiling else self._untiled()
        self.init_config(config)
        # A caller-provided config may have block_m without tile_n.
        if self._needs_tiling:
            bm = self.config.get("block_m", 1)
            threads = self.config.get("threads", DEFAULT_THREADS)
            if "tile_n" not in self.config or self.config["tile_n"] == 0:
                tile_n = self._planner.tiled_tile_n(bm, threads)
                self.config = {**self.config, "tile_n": tile_n}

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
        return self._reduce_rows(self._rows(x))

    def _reduce_rows(self, x: torch.Tensor) -> object:
        results = self.program(self._compile)(x)
        return tuple(results) if self.op_kind == "var_mean" else results

    def _compile(self, cfg: dict):
        """The program *cfg* tiles."""
        builder = self._tiled(cfg["tile_n"]) if self._needs_tiling else self.kernel
        return builder(cfg["block_m"], cfg["threads"])


class ReduceFoldKernel(
    ReduceKernelBase, ReduceFwdInterface, ProdFwdInterface, VectorNormFwdInterface
):
    """Each row of whole 16-byte vectors folded into registers as it is read, one block a row.

    Serves sum, mean, amax, amin, prod and the l1, l2 and inf norms. A reduced prefix and
    suffix around kept axes fold in place: each kept index reads its ``lead`` contiguous
    suffix segments, with no permute and no second launch. Tunes the block's thread count.
    """

    # Vector loads each thread keeps in flight; a grid of few rows is bound by it.
    _UNROLL = 16

    @classmethod
    def refusal(cls, call: ReduceCall) -> "str | None":
        segment = cls.edge_plan(call)[2] if cls.reduces_edge_axes(call) else call.n
        if segment * call.dtype.itemsize % VECTOR_ACCESS_BYTES:
            return f"requires a row of whole {VECTOR_ACCESS_BYTES}-byte vectors"
        return super().refusal(call)

    def __init__(self, call: ReduceCall, config: dict | None = None):
        super().__init__(call)
        self._planner = self.row_planner(call)
        self._lead = self.edge_plan(call)[0] if self.reduces_edge_axes(call) else 1
        self.kernel = fold_rows_kernel(
            self.M,
            self.N // self._lead,
            self.op_kind,
            self.dtype_str,
            self.out_dtype_str,
            self._UNROLL,
            self._lead,
        )
        self.init_config(config)

    @staticmethod
    def _default_threads(call: ReduceCall) -> int:
        """512 threads where each still folds at least four vectors of the row, else the default.

        A wider block puts more of a long row in flight; on a short row it leaves each
        thread a vector or two, and the block's final reduction dominates. A row that long
        takes 1024 threads on a grid of at least 1024 rows, which runs several waves; on
        fewer rows the wider block leaves more of the grid's SMs idle.
        """
        wide_threads, min_vectors_per_thread, many_rows = 512, 4, 1024
        vectors = call.n * call.dtype.itemsize // VECTOR_ACCESS_BYTES
        if vectors < wide_threads * min_vectors_per_thread:
            return DEFAULT_THREADS
        return 2 * wide_threads if call.m >= many_rows else wide_threads

    @classmethod
    def holds_edge_row_in_flight(cls, call: ReduceCall) -> bool:
        """Whether a 512-thread block holds an edge-axis call's row in flight at once.

        That is at most ``_UNROLL`` vector loads a thread, a row of at most 128 KB, which one
        launch reads in about one memory latency. A longer row is bound by one block's
        stream rate, so it stays on the edge kernels, whose rows pass spreads it over
        ``lead * kept`` blocks.
        """
        if not cls.reduces_edge_axes(call) or cls.refusal(call) is not None:
            return False
        lead, _, trail, _, _ = cls.edge_plan(call)
        vectors = trail * call.dtype.itemsize // VECTOR_ACCESS_BYTES
        block_threads = 512
        return lead * ceil_div(vectors, block_threads) <= cls._UNROLL

    @property
    def default_config(self) -> dict:
        return {"threads": self._default_threads(self.call)}

    @property
    def autotune_configs(self) -> list[dict]:
        # The thread counts the planner admits for a one-row block (any block for prod),
        # and the default, which may be wider than any of them.
        threads = dict.fromkeys(
            [
                *(
                    c["threads"]
                    for c in self._planner.autotune_configs()
                    if self.op_kind == "prod" or c["block_m"] == 1
                ),
                self._default_threads(self.call),
            ]
        )
        return [{"threads": t} for t in threads]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        shape = (self.M, self.N) if self._lead == 1 else (self._lead, self.M, self.N // self._lead)
        probe = torch.randn(shape, dtype=self.dtype, device=torch.cuda.current_device())
        tune_by_forward(self, probe, warmup=warmup, rep=rep, forward=self._reduce_rows)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self._lead > 1:
            return self._reduce_rows(x.reshape(self._lead, self.M, self.N // self._lead))
        return self._reduce_rows(self._rows(x))

    def _reduce_rows(self, x: torch.Tensor) -> torch.Tensor:
        return self.program(self._compile)(x)

    def _compile(self, cfg: dict):
        """The program at *cfg*'s thread count."""
        return self.kernel(cfg["threads"])


class ReduceKernel(RowReduceKernelBase, ReduceFwdInterface):
    """sum / mean / amax / amin of rows, through shared memory.

    Boundary handling for a row that is not a multiple of the 256-element copy alignment is
    a masked load filling the identity element.
    """

    general = True

    def _untiled(self) -> object:
        return _simple_reduce_kernel(
            self.M, self.N, self.op_kind, self.dtype_str, self.out_dtype_str
        )

    def _tiled(self, tile_n: int) -> object:
        return _simple_reduce_kernel_tiled(
            self.M, self.N, self.op_kind, self.dtype_str, tile_n, self.out_dtype_str
        )


class ReduceProdKernel(ReduceKernelBase, ProdFwdInterface):
    """Product of each row, staged through shared memory, one block a row."""

    general = True
    # Columns each thread multiplies per staged tile, each into its own fp32 chain.
    _COLS_PER_THREAD = 8

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
        return self.program(self._compile)(self._rows(x))

    def _compile(self, cfg: dict):
        """The program."""
        return self.kernel()


class WelfordReduceKernel(RowReduceKernelBase, VarianceFwdInterface, VarMeanFwdInterface):
    """std / var / var_mean of rows, through shared memory; long rows take two tiled passes."""

    general = True

    def _untiled(self) -> object:
        return _welford_reduce_kernel(self.M, self.N, self.op_kind, self.correction, self.dtype_str)

    def _tiled(self, tile_n: int) -> object:
        return _welford_reduce_kernel_tiled(
            self.M, self.N, self.op_kind, self.correction, self.dtype_str, tile_n
        )


class ReduceLeadingKernel(ReduceKernelBase, ReduceFwdInterface, ProdFwdInterface):
    """A reduction down the leading axes in the tensor's own ``(N, M)`` layout.

    A call `down_rows_cluster` admits runs in one launch. Where it returns more than 1, the
    reduced axis splits across the CTAs of a thread-block cluster, which fold their partials
    in shared memory; at 1, each CTA walks every row of its tile. Elsewhere, where one launch
    over the columns would leave the grid short, the down-rows pass splits the reduced axis
    into row slices whose fp32 partials a second launch finishes.
    """

    preferred_over = frozenset({"reduce_fold"})

    @classmethod
    def refusal(cls, call: ReduceCall) -> "str | None":
        if not 0 < len(call.axes) < len(call.shape) or call.axes != tuple(range(len(call.axes))):
            return "serves a reduction over leading axes that keeps at least one"
        return super().refusal(call)

    def __init__(self, call: ReduceCall):
        super().__init__(call)
        self._splits = down_rows_splits(self.N, self.M)

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.program(self._launcher)(x.reshape(self.N, self.M))

    def _launcher(self, cfg: dict) -> Callable:
        """The down-rows reduction of the ``(N, M)`` input."""
        divisor = float(self.N) if self.op_kind == "mean" else 0.0
        return down_rows_launcher(
            self.N,
            self.M,
            self.op_kind,
            self.dtype_str,
            self.out_dtype_str,
            divisor,
            self._splits,
            arch=self.call.arch,
            sm_count=self.call.sm_count,
        )


class EdgeReduceKernelBase(ReduceKernelBase):
    """A reduction of a prefix and a suffix of the axes around kept axes, in place.

    One launch sequence, built once; a subclass states it in ``_launcher`` and the edge
    calls it declines in ``_edge_refusal``.
    """

    @classmethod
    def refusal(cls, call: ReduceCall) -> "str | None":
        if not cls.reduces_edge_axes(call):
            return "serves a reduced prefix and suffix around kept axes"
        return cls._edge_refusal(call) or super().refusal(call)

    @classmethod
    def _edge_refusal(cls, call: ReduceCall) -> "str | None":
        """Why this implementation declines an edge-axis *call*, or None."""
        return None

    def __init__(self, call: ReduceCall):
        super().__init__(call)
        self.program(self._launcher)

    def _reduce(self, x: torch.Tensor) -> object:
        return self.program(self._launcher)(x)

    def _launcher(self, cfg: dict) -> Callable:
        raise NotImplementedError


class ReduceEdgeKernel(EdgeReduceKernelBase, ReduceFwdInterface):
    """A prefix and a suffix of the axes reduced without permuting the tensor.

    The trailing axes reduce as contiguous rows into fp32 partials, tiled where a row
    exceeds one block pass; the leading axes then reduce down the columns of those
    partials, split as `ReduceLeadingKernel` splits. ``mean`` runs the rows pass as
    ``sum`` and divides in the second by the full reduced count.
    """

    preferred_over = frozenset({"reduce_fold"})

    @classmethod
    def _edge_refusal(cls, call: ReduceCall) -> "str | None":
        if ReduceFoldKernel.holds_edge_row_in_flight(call):
            return "leaves a row ReduceFoldKernel holds in flight to its one launch"
        return None

    def _launcher(self, cfg: dict) -> Callable:
        """The rows pass into fp32 partials, then the columns pass down them."""
        lead, kept, trail, planner, rows_cfg = self.edge_plan(self.call)
        inner_kind = "sum" if self.op_kind == "mean" else self.op_kind
        if planner.needs_tiling:
            builder = _simple_reduce_kernel_tiled(
                lead * kept, trail, inner_kind, self.dtype_str, rows_cfg["tile_n"], "float32"
            )
        else:
            builder = _simple_reduce_kernel(
                lead * kept, trail, inner_kind, self.dtype_str, "float32"
            )
        rows_pass = CompiledProgram(builder(rows_cfg["block_m"], rows_cfg["threads"]))
        divisor = float(lead * trail) if self.op_kind == "mean" else 0.0
        # The columns pass writes the storage dtype itself; leaving it in fp32 and
        # casting after costs a third kernel launch.
        splits = down_rows_splits(lead, kept)
        columns_pass = down_rows_launcher(
            lead,
            kept,
            self.op_kind,
            "float32",
            self.out_dtype_str,
            divisor,
            splits,
            arch=self.call.arch,
            sm_count=self.call.sm_count,
        )

        def launch(x: torch.Tensor) -> torch.Tensor:
            partials = rows_pass(x.reshape(lead * kept, trail)).reshape(lead, kept)
            return columns_pass(partials)

        return launch


class WelfordEdgeKernel(EdgeReduceKernelBase, VarianceFwdInterface, VarMeanFwdInterface):
    """Variance over a prefix and a suffix of the axes without permuting the tensor.

    The rows pass, tiled where a row exceeds one block pass, leaves fp32 ``(mean, M2)``
    per ``(lead, kept)`` slice; Chan's merge folds the slices, so the result is the
    single-pass Welford statistic.
    """

    @classmethod
    def _edge_refusal(cls, call: ReduceCall) -> "str | None":
        # The merge folds counts in fp32, whose weights drift past its integer range.
        if call.n > FP32_EXACT_INT_LIMIT:
            return f"folds counts in float32, exact up to {FP32_EXACT_INT_LIMIT}"
        return None

    def _reduce(self, x: torch.Tensor) -> object:
        stat, mean = super()._reduce(x)
        return (stat, mean) if self.op_kind == "var_mean" else stat

    def _launcher(self, cfg: dict) -> Callable:
        """The rows pass into fp32 ``(mean, M2)`` partials, then Chan's merge of them."""
        lead, kept, trail, planner, rows_cfg = self.edge_plan(self.call)
        if planner.needs_tiling:
            builder = _welford_reduce_kernel_tiled(
                lead * kept, trail, "partials", 0, self.dtype_str, rows_cfg["tile_n"]
            )
        else:
            builder = _welford_reduce_kernel(lead * kept, trail, "partials", 0, self.dtype_str)
        rows_pass = CompiledProgram(builder(rows_cfg["block_m"], rows_cfg["threads"]))
        merge = CompiledProgram(
            _welford_merge_kernel(
                lead, kept, self.op_kind, self.correction, trail, self.dtype_str, DEFAULT_THREADS
            )()
        )

        def launch(x: torch.Tensor) -> object:
            m2_p, mean_p = rows_pass(x.reshape(lead * kept, trail))
            return merge(mean_p.reshape(lead, kept), m2_p.reshape(lead, kept))

        return launch


def _tile_loader(M, N, block_m, tile_n, pad):
    """The macro reading tile ``t`` of a block's rows into an fp32 fragment.

    Whole tiles stage through shared memory by ``T.copy``. The tile holding the row end,
    padded to a multiple of ``tile_n``, is read element-wise with *pad* past the row end
    and past row ``M``.
    """
    num_tiles = ceil_div(align_up(N, ALIGNMENT), tile_n)
    masked = num_tiles * tile_n > N

    @T.macro
    def copied(x, staged, values, pid_m, t):
        T.copy(x[pid_m * block_m, t * tile_n], staged)
        for i in T.serial(block_m):
            for j in T.Parallel(tile_n):
                values[i, j] = T.cast(staged[i, j], "float32")

    @T.macro
    def guarded(x, values, pid_m, t):
        for i in T.serial(block_m):
            for j in T.Parallel(tile_n):
                values[i, j] = T.if_then_else(
                    T.And(pid_m * block_m + i < M, t * tile_n + j < N),
                    T.cast(x[pid_m * block_m + i, t * tile_n + j], "float32"),
                    T.cast(pad, "float32"),
                )

    @T.macro
    def load(x, staged, values, pid_m, t):
        if not masked:
            copied(x, staged, values, pid_m, t)
        elif num_tiles == 1:
            guarded(x, values, pid_m, t)
        else:
            with T.If(t < num_tiles - 1):
                with T.Then():
                    copied(x, staged, values, pid_m, t)
                with T.Else():
                    guarded(x, values, pid_m, t)

    return load


@target_cache(maxsize=32)
def _simple_reduce_kernel(M, N, op_kind, dtype, out_dtype=None):
    """Build a reduce of each row for sum/mean/amax/amin, the row held in one tile.

    A row that is not a multiple of ``ALIGNMENT`` is read element-wise, the identity
    standing in for the columns past its end. amax and amin are `_extrema_rows_kernel`'s.

    ``out_dtype`` overrides the output element type, for a caller that keeps
    fp32 partials across a second pass.
    """
    N_padded = align_up(N, ALIGNMENT)
    out_dtype = out_dtype or dtype
    if op_kind in ("amax", "amin"):
        return _extrema_rows_kernel(M, N, op_kind, dtype, out_dtype, N_padded)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        load = _tile_loader(M, N, block_m, N_padded, identity_for(op_kind))

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

                load(x, shared_buf, x_f32, pid_m, 0)
                T.reduce_sum(x_f32, acc, dim=1)
                if op_kind == "mean":
                    for i in T.Parallel(block_m):
                        acc[i] = acc[i] / float(N)
                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(acc[i], out_dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


@target_cache(maxsize=32)
def _simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n, out_dtype=None):
    """Tiled simple reduce for rows one tile does not hold.

    Iterates over N in chunks of ``tile_n`` columns, accumulating partial results.
    amax and amin are `_extrema_rows_kernel`'s.
    """
    out_dtype = out_dtype or dtype
    if op_kind in ("amax", "amin"):
        return _extrema_rows_kernel(M, N, op_kind, dtype, out_dtype, tile_n)
    num_tiles = ceil_div(align_up(N, ALIGNMENT), tile_n)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        load = _tile_loader(M, N, block_m, tile_n, identity_for(op_kind))

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

                T.fill(acc, 0.0)
                for t in T.Serial(num_tiles):
                    load(x, shared_buf, tile_f32, pid_m, t)
                    T.reduce_sum(tile_f32, tile_acc, dim=1)
                    for i in T.Parallel(block_m):
                        acc[i] = acc[i] + tile_acc[i]

                if op_kind == "mean":
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i] / float(N), out_dtype)
                else:
                    for i in T.Parallel(block_m):
                        out_local[i] = T.cast(acc[i], out_dtype)

                T.copy(out_local, out[pid_m * block_m])

        return main

    return _func


@target_cache(maxsize=32)
def _extrema_rows_kernel(M, N, op_kind, dtype, out_dtype, tile_n):
    """Build amax or amin of each row of an ``(M, N)`` buffer, ``tile_n`` columns a step.

    ``T.reduce_max`` combines fp32 with ``fmaxf``, which drops NaN, so threads fold their
    columns with `extrema_combine` and fold each other's partials the same way, once per
    row after the last tile. Whole tiles stage through shared memory by ``T.copy``. The
    block splits into one group of ``lanes`` threads per row, as many groups as it has
    rows while each keeps at least a warp; a group folds one row at a time, each thread
    reading ``width`` contiguous elements a step, a power of two up to a 16-byte vector
    that spreads the tile over the group, into as many independent slots.
    """
    whole = N // tile_n
    last_cols = N - whole * tile_n
    itemsize = getattr(torch, dtype).itemsize
    vec = VECTOR_ACCESS_BYTES // itemsize
    # Elements of the widest load every row start is aligned to: rows of N elements start
    # on multiples of math.gcd(N * itemsize, 16) bytes.
    row_vec = math.gcd(N * itemsize, VECTOR_ACCESS_BYTES) // itemsize

    @tilelang.jit(out_idx=[1], compile_flags=csrc_include("nan_extrema.h"))
    def _func(block_m, threads):
        groups = 1
        while groups * 2 <= min(block_m, threads // WARP_LANES):
            groups *= 2
        lanes = threads // groups
        width = 1
        while width * 2 <= min(vec, tile_n // lanes):
            width *= 2
        passes = ceil_div(block_m, groups)
        step = lanes * width
        identity = T.cast(identity_for(op_kind), "float32")

        def combine(a, b):
            return extrema_combine(a, b, op_kind)

        def finished(total):
            return T.cast(total, out_dtype)

        block_total = block_fold(combine, identity, threads, finished, M, lanes)

        @T.macro
        def fold_row(x, staged, held, slots, parts, pid_m, p, i, lane, direct):
            # A whole tile is read from ``staged``, ``width`` contiguous elements a thread;
            # the tile past the row end, *direct*, from ``x``, masked, in loads of ``chunk``
            # elements, adjacent threads on adjacent loads: its rows start no wider apart
            # than ``row_vec`` elements, so staging it would buy no wider load for its
            # extra barriers. A load lies wholly inside or past the row, as ``chunk``
            # divides N.
            chunk = min(width, row_vec)
            for c in T.serial(width):
                slots[c] = identity
            span = last_cols if direct else tile_n
            for k in T.serial(ceil_div(span, step)):
                if direct:
                    for c in T.serial(width // chunk):
                        col = whole * tile_n + k * step + (c * lanes + lane) * chunk
                        inside = T.And(pid_m * block_m + i < M, col < N)
                        for e in T.vectorized(chunk):
                            held[e] = T.if_then_else(
                                inside, x[pid_m * block_m + i, col + e], T.cast(0, dtype)
                            )
                        for e in T.serial(chunk):
                            slots[c * chunk + e] = combine(
                                slots[c * chunk + e],
                                T.if_then_else(inside, T.cast(held[e], "float32"), identity),
                            )
                else:
                    base = (k * lanes + lane) * width
                    if base < span:
                        for c in T.vectorized(width):
                            held[c] = staged[i, base + c]
                        for c in T.serial(width):
                            slots[c] = combine(slots[c], T.cast(held[c], "float32"))
            for c in T.serial(width):
                parts[p] = combine(parts[p], slots[c])

        @T.macro
        def fold_tile(x, staged, held, slots, parts, pid_m, tx, direct):
            # Folds one tile of each of the block's rows into ``parts``.
            for p in T.serial(passes):
                i = p * groups + tx // lanes
                if passes * groups > block_m:
                    if i < block_m:
                        fold_row(x, staged, held, slots, parts, pid_m, p, i, tx % lanes, direct)
                else:
                    fold_row(x, staged, held, slots, parts, pid_m, p, i, tx % lanes, direct)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            out: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                tx = T.get_thread_binding()
                staged = T.alloc_shared((block_m, tile_n), dtype)
                warp_total = T.alloc_shared((threads // WARP_LANES,), "float32")
                held = T.alloc_local((width,), dtype)
                slots = T.alloc_local((width,), "float32")
                parts = T.alloc_local((passes,), "float32")
                total = T.alloc_local((1,), "float32")

                for p in T.serial(passes):
                    parts[p] = identity
                for t in T.serial(whole):
                    T.copy(x[pid_m * block_m, t * tile_n], staged)
                    T.sync_threads()
                    fold_tile(x, staged, held, slots, parts, pid_m, tx, False)
                    T.sync_threads()
                if last_cols:
                    fold_tile(x, staged, held, slots, parts, pid_m, tx, True)
                for p in T.serial(passes):
                    i = p * groups + tx // lanes
                    total[0] = parts[p]
                    # A group past the block's rows folds along and stores nothing.
                    block_total(
                        total,
                        warp_total,
                        tx,
                        out,
                        T.if_then_else(i < block_m, pid_m * block_m + i, M),
                    )
                    if passes > 1:
                        T.sync_threads()

        return main

    return _func


@target_cache(maxsize=32)
def _prod_reduce_kernel(
    M: int, N: int, dtype: str, out_dtype: str, threads: int, cols_per_thread: int
):
    """Build a product reduce: one block per row, multiplying in fp32.

    Serves rows that are not whole vectors, staging each tile through shared
    memory and filling the columns past the row end with 1.
    """
    chunk = threads * cols_per_thread
    tiles = ceil_div(N, chunk)
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

                warp_reduce(running, operator.mul, high_first=True, serial=True)
                if tx % WARP_LANES == 0:
                    warp_prod[tx // WARP_LANES] = running[0]
                T.sync_threads()
                if tx == 0:
                    for w in T.serial(1, num_warps):
                        warp_prod[0] = warp_prod[0] * warp_prod[w]
                    out[row] = T.cast(warp_prod[0], out_dtype)

        return main

    return _func


def _welford_program(M, N, op_kind, correction, dtype, block_m, threads, moments):
    """The prim_func writing what *op_kind* returns from the row statistics *moments* takes.

    *moments* is a macro ``(x, pid_m, mean_val, var_sum)`` filling a block's row means and
    sums of squared deviations. ``op_kind="partials"`` writes the fp32 ``(M2, mean)`` of
    each row undivided, for a cross-row merge; *correction* is unused there.
    """
    out_dtype = "float32" if op_kind == "partials" else dtype
    divisor = float(N - correction)

    if op_kind in ("var_mean", "partials"):

        @T.prim_func
        def pair(
            x: T.Tensor[(M, N), dtype],
            out_var: T.Tensor[(M,), out_dtype],
            out_mean: T.Tensor[(M,), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                mean_val = T.alloc_fragment((block_m,), "float32")
                var_sum = T.alloc_fragment((block_m,), "float32")
                out_v = T.alloc_fragment((block_m,), out_dtype)
                out_m = T.alloc_fragment((block_m,), out_dtype)

                moments(x, pid_m, mean_val, var_sum)
                for i in T.Parallel(block_m):
                    if op_kind == "partials":
                        out_v[i] = var_sum[i]
                        out_m[i] = mean_val[i]
                    else:
                        out_v[i] = T.cast(var_sum[i] / divisor, out_dtype)
                        out_m[i] = T.cast(mean_val[i], out_dtype)

                T.copy(out_v, out_var[pid_m * block_m])
                T.copy(out_m, out_mean[pid_m * block_m])

        return pair

    @T.prim_func
    def single(
        x: T.Tensor[(M, N), dtype],
        out: T.Tensor[(M,), dtype],
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
            mean_val = T.alloc_fragment((block_m,), "float32")
            var_sum = T.alloc_fragment((block_m,), "float32")
            out_local = T.alloc_fragment((block_m,), dtype)

            moments(x, pid_m, mean_val, var_sum)
            if op_kind == "var":
                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(var_sum[i] / divisor, dtype)
            else:
                for i in T.Parallel(block_m):
                    out_local[i] = T.cast(T.sqrt(var_sum[i] / divisor), dtype)

            T.copy(out_local, out[pid_m * block_m])

    return single


@target_cache(maxsize=32)
def _welford_reduce_kernel(M, N, op_kind, correction, dtype):
    """Build std/var/var_mean of each row, the row held in one tile.

    Padding columns read as ``0.0`` and are masked out of the centered sum of squares.
    ``op_kind="partials"`` is `_welford_program`'s.
    """
    N_padded = align_up(N, ALIGNMENT)

    @tilelang.jit(out_idx=[1, 2] if op_kind in ("var_mean", "partials") else [1])
    def _func(block_m, threads):
        load = _tile_loader(M, N, block_m, N_padded, 0.0)

        @T.macro
        def moments(x, pid_m, mean_val, var_sum):
            shared_buf = T.alloc_shared((block_m, N_padded), dtype)
            x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
            row_sum = T.alloc_fragment((block_m,), "float32")
            sq_diff = T.alloc_fragment((block_m, N_padded), "float32")

            load(x, shared_buf, x_f32, pid_m, 0)
            T.reduce_sum(x_f32, row_sum, dim=1)
            for i in T.Parallel(block_m):
                mean_val[i] = row_sum[i] / float(N)
            for i in T.serial(block_m):
                for j in T.Parallel(N_padded):
                    dev = x_f32[i, j] - mean_val[i]
                    sq_diff[i, j] = T.if_then_else(j < N, dev * dev, 0.0)
            T.reduce_sum(sq_diff, var_sum, dim=1)

        return _welford_program(M, N, op_kind, correction, dtype, block_m, threads, moments)

    return _func


@target_cache(maxsize=32)
def _welford_reduce_kernel_tiled(M, N, op_kind, correction, dtype, tile_n):
    """Tiled std/var/var_mean for rows one tile does not hold, in two passes over the tiles.

    The first pass sums the row for its mean; the second sums the squared deviations from
    it. ``op_kind="partials"`` is `_welford_program`'s.
    """
    num_tiles = ceil_div(align_up(N, ALIGNMENT), tile_n)

    @tilelang.jit(out_idx=[1, 2] if op_kind in ("var_mean", "partials") else [1])
    def _func(block_m, threads):
        load = _tile_loader(M, N, block_m, tile_n, 0.0)

        @T.macro
        def moments(x, pid_m, mean_val, var_sum):
            shared_buf = T.alloc_shared((block_m, tile_n), dtype)
            tile_f32 = T.alloc_fragment((block_m, tile_n), "float32")
            tile_sum = T.alloc_fragment((block_m,), "float32")
            row_sum = T.alloc_fragment((block_m,), "float32")
            sq_diff = T.alloc_fragment((block_m, tile_n), "float32")
            tile_sq = T.alloc_fragment((block_m,), "float32")
            # The second pass reads into buffers of its own: TileLang aliases the first
            # pass's otherwise.
            p2_shared = T.alloc_shared((block_m, tile_n), dtype)
            p2_f32 = T.alloc_fragment((block_m, tile_n), "float32")

            T.fill(row_sum, 0.0)
            for t in T.Serial(num_tiles):
                load(x, shared_buf, tile_f32, pid_m, t)
                T.reduce_sum(tile_f32, tile_sum, dim=1)
                for i in T.Parallel(block_m):
                    row_sum[i] = row_sum[i] + tile_sum[i]
            for i in T.Parallel(block_m):
                mean_val[i] = row_sum[i] / float(N)

            T.fill(var_sum, 0.0)
            for t in T.Serial(num_tiles):
                load(x, p2_shared, p2_f32, pid_m, t)
                for i in T.serial(block_m):
                    for j in T.Parallel(tile_n):
                        dev = p2_f32[i, j] - mean_val[i]
                        sq_diff[i, j] = T.if_then_else(t * tile_n + j < N, dev * dev, 0.0)
                T.reduce_sum(sq_diff, tile_sq, dim=1)
                for i in T.Parallel(block_m):
                    var_sum[i] = var_sum[i] + tile_sq[i]

        return _welford_program(M, N, op_kind, correction, dtype, block_m, threads, moments)

    return _func


@target_cache(maxsize=32)
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

                with T.If(pid_b * threads + tx < B):  # noqa: SIM117
                    with T.Then():
                        # Seed from a real slice: an empty-state delta can overflow
                        # before its zero weight is applied.
                        n_acc[0] = nb
                        mean_acc[0] = mean_in[0, pid_b * threads + tx]
                        m2_acc[0] = m2_in[0, pid_b * threads + tx]
                        for a in T.serial(1, A):
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
