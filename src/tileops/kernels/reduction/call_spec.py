"""Call records for reduction kernels."""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import ClassVar, Mapping

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.reduction._primitives import (
    AUTOTUNE_THREADS,
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    BlockConfigPlanner,
    align_up,
    ceildiv_int,
    device_smem_budget,
    edge_axis_split,
)
from tileops.kernels.reduction._split_softmax import edge_split_view, fused_split_plan, split_seg_n

__all__ = [
    "STREAMING_LOGSUMEXP",
    "LogSumExpCall",
    "LogicalReduceCall",
    "SoftmaxCall",
    "StreamingLogSumExpPolicy",
]


@dataclasses.dataclass(frozen=True)
class LogicalReduceCall(CallSpec):
    """A logical reduction call: the input's shape and dtype, the axes it reduces."""

    # The fused edge pass runs one block per kept column and has no other parallelism:
    # the fewest kept columns that fill the device, per calibrated board.
    _EDGE_FUSED_MIN_KEPT: ClassVar[Mapping[str, int]] = {"h200": 32}

    shape: tuple[int, ...] = ()
    # Non-negative and ascending.
    axes: tuple[int, ...] = ()
    op_kind: str = ""
    dtype: torch.dtype = torch.float16
    keepdim: bool = False

    @property
    def device_index(self) -> "int | None":
        return self.device.index if self.device is not None else None

    @property
    def edge_kept(self) -> int:
        """The kept extent between a reduced prefix and suffix of axes, or 0 for other layouts."""
        k, j = edge_axis_split(len(self.shape), self.axes)
        return math.prod(self.shape[k : len(self.shape) - j]) if k else 0

    @property
    def edge_fused_min_kept(self) -> float:
        """The fewest kept columns at which the fused edge pass fills this call's board.

        Infinite on a board with no calibrated entry.
        """
        return self._EDGE_FUSED_MIN_KEPT.get(self.calibration, math.inf)


@dataclasses.dataclass(frozen=True)
class StreamingLogSumExpPolicy:
    """Launch shape and eligibility gate of the streaming kernel.

    The launch pair is fixed rather than tuned, and ``LogSumExpCall.streams`` keeps
    the kernel on the shapes that pair suits.
    """

    threads: int = 128

    cols_per_thread: int = 8

    # Enough rows to fill the device with one block per row.
    min_rows: int = 256

    # Rows long enough that the tiled kernel's staging measurably loses.
    min_cols: int = 16384

    # Seed of the running max: below every finite fp16/bf16 value, but
    # finite, so an all--inf row keeps a zero sum and folds to
    # max_floor + log(0) = -inf, matching torch.
    max_floor: float = -3.4e38


STREAMING_LOGSUMEXP = StreamingLogSumExpPolicy()


@dataclasses.dataclass(frozen=True)
class _RowPlanCall(CallSpec):
    """A call whose row kernels plan against the device's shared memory.

    ``smem_budget`` is a device fact like ``sm_count``: a record that states none reads
    it when it is built.
    """

    smem_budget: int = 0

    # Blocks per SM a split aims for; under this the grid runs the device empty.
    _SPLIT_BLOCKS_PER_SM: ClassVar[int] = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.smem_budget <= 0:
            index = self.device.index if self.device is not None else None
            object.__setattr__(self, "smem_budget", device_smem_budget(index))

    @property
    def split_target(self) -> int:
        """The block count a split of the rows aims for."""
        return self._SPLIT_BLOCKS_PER_SM * self.sm_count


@dataclasses.dataclass(frozen=True)
class LogSumExpCall(_RowPlanCall):
    """A logsumexp call: the input as the manifest declares it.

    ``axes`` are the reduced axes, ascending and non-negative.
    """

    shape: tuple[int, ...] = ()
    axes: tuple[int, ...] = ()
    keepdim: bool = False
    dtype: torch.dtype = torch.float16

    @property
    def n(self) -> int:
        """Elements each output element reduces."""
        return math.prod(self.shape[a] for a in self.axes)

    @property
    def m(self) -> int:
        """Output elements: the kept extents' product."""
        return math.prod(self.shape) // self.n

    @property
    def edge_view(self) -> "tuple[int, int, int] | None":
        """The ``(outer, kept, inner)`` view an edge-axis reduction reads in place, or None."""
        k, j = edge_axis_split(len(self.shape), self.axes)
        return edge_split_view(self.shape, k, j, DEFAULT_THREADS) if k else None

    @property
    def streams(self) -> bool:
        """The rows are long and many enough for the streaming launch shape."""
        policy = STREAMING_LOGSUMEXP
        return (
            self.dtype in (torch.float16, torch.bfloat16)
            and policy.min_rows <= self.m
            and policy.min_cols <= self.n
            and self.n % (policy.threads * policy.cols_per_thread) == 0
        )

    @property
    def row_plan(self) -> "tuple[int, int]":
        """The row kernels' untuned ``(block_m, tile_n)``; ``tile_n == 0`` is one tile."""
        return self._plan_rows(
            align_up(self.n, DEFAULT_ALIGNMENT), self.dtype.itemsize, self.smem_budget
        )

    @property
    def split_seg_n(self) -> int:
        """The segment width a split of the rows takes, or 0 when the untuned row grid fills."""
        return split_seg_n(self.m, self.n, self.row_plan[0], self.split_target)

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _plan_rows(n_padded: int, elem_bytes: int, smem_budget: int) -> "tuple[int, int]":
        """One tile takes the largest block_m that holds the row; tiled rows take the
        block_m with strictly the fewest tiles, the smallest one on a tie."""
        planner = BlockConfigPlanner(n_padded, elem_bytes, smem_budget)
        threads = max(AUTOTUNE_THREADS)
        best_bm = 1
        best_tile_n = planner.tile_n_for(1, threads)
        for bm in [2, 4, 8, 16]:
            if not planner.layout_ok(bm, n_padded, DEFAULT_THREADS):
                continue
            try:
                tn = planner.tile_n_for(bm, threads)
            except ValueError:
                continue
            if (
                tn == 0
                or best_tile_n != 0
                and ceildiv_int(n_padded, tn) < ceildiv_int(n_padded, best_tile_n)
            ):
                best_bm = bm
                best_tile_n = tn
        return best_bm, best_tile_n


@dataclasses.dataclass(frozen=True)
class SoftmaxCall(_RowPlanCall):
    """A softmax or log_softmax call: the input as the manifest declares it.

    ``axis`` is the normalized axis, non-negative; ``dtype`` is the input's as the kernel
    reads it and ``out_dtype`` the output's.
    """

    shape: tuple[int, ...] = ()
    axis: int = 0
    op_kind: str = "softmax"
    dtype: torch.dtype = torch.float16
    out_dtype: torch.dtype = torch.float16

    @property
    def n(self) -> int:
        """Length of the normalized axis."""
        return self.shape[self.axis]

    @property
    def m(self) -> int:
        """Rows: the product of every other axis."""
        return math.prod(self.shape) // self.n

    @property
    def num_buffers(self) -> int:
        """Row-sized shared buffers the tiled kernel holds.

        Two, one per pass, since TileLang's allocator aliases one buffer across the
        passes; an output of another dtype adds its stage.
        """
        in_bytes = self.dtype.itemsize
        out_stage = 0 if self.out_dtype == self.dtype else -(-self.out_dtype.itemsize // in_bytes)
        return 2 + out_stage

    @property
    def row_plan(self) -> "tuple[int, int]":
        """The row kernels' untuned ``(block_m, tile_n)``; ``tile_n == 0`` is one tile."""
        return self._plan_rows(
            align_up(self.n, DEFAULT_ALIGNMENT),
            self.dtype.itemsize,
            self.smem_budget,
            self.num_buffers,
        )

    @property
    def split_seg_n(self) -> int:
        """The segment width a split of the rows takes, or 0 when the untuned row grid fills."""
        return split_seg_n(self.m, self.n, self.row_plan[0], self.split_target)

    @property
    def fused_split_threads(self) -> "int | None":
        """The width a one-kernel split runs at, or None; see ``fused_split_plan``."""
        return fused_split_plan(self.m, self.n, self.split_seg_n, self.split_target)

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _plan_rows(
        n_padded: int, elem_bytes: int, smem_budget: int, num_buffers: int
    ) -> "tuple[int, int]":
        """One tile keeps the *smallest* block_m: each extra row hands every thread
        another ``N_padded / threads`` registers until the fragment spills. Tiled rows
        take the block_m with strictly the fewest tiles, since fewer tiles means fewer
        global passes, and the smallest one on a tie, for occupancy."""
        planner = BlockConfigPlanner(n_padded, elem_bytes, smem_budget, num_buffers=num_buffers)
        threads = max(AUTOTUNE_THREADS)
        best_bm = 1
        best_tile_n = planner.tile_n_for(1, threads)
        for bm in [2, 4, 8, 16]:
            if not planner.layout_ok(bm, n_padded, DEFAULT_THREADS):
                continue
            try:
                tn = planner.tile_n_for(bm, threads)
            except ValueError:
                continue
            if tn == 0 and not planner.frag_fits(bm, n_padded, DEFAULT_THREADS):
                continue
            if best_tile_n == 0:
                continue
            if tn == 0 or ceildiv_int(n_padded, tn) < ceildiv_int(n_padded, best_tile_n):
                best_bm = bm
                best_tile_n = tn
        return best_bm, best_tile_n
