"""Call records for reduction kernels."""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar, Mapping

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.reduction._primitives import device_smem_budget, edge_axis_split

__all__ = ["LogSumExpCall", "LogicalReduceCall", "SoftmaxCall"]


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
class _SharedMemoryCall(CallSpec):
    """A call whose kernels plan against the device's shared memory.

    ``smem_budget`` is a device fact like ``sm_count``: a record that states none reads
    it when it is built.
    """

    smem_budget: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.smem_budget <= 0:
            index = self.device.index if self.device is not None else None
            object.__setattr__(self, "smem_budget", device_smem_budget(index))


@dataclasses.dataclass(frozen=True)
class LogSumExpCall(_SharedMemoryCall):
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


@dataclasses.dataclass(frozen=True)
class SoftmaxCall(_SharedMemoryCall):
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
