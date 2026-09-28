"""Call records for reduction kernels."""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar, Mapping

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.reduction._primitives import edge_axis_split

__all__ = ["LogicalReduceCall"]


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
