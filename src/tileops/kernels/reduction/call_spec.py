"""Call records and implementation regions for reduction kernels."""

from __future__ import annotations

import dataclasses
import math

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.reduction._primitives import FP32_EXACT_INT_LIMIT, edge_axis_split

__all__ = [
    "LogicalReduceCall",
    "edge_fused_min_kept",
    "logical_edge_fused_region",
    "logical_edge_two_pass_region",
    "logical_reduce_region",
]


# The fused pass runs one block per kept column and has no other parallelism, so
# it takes over only where that alone is enough: the fewest kept columns that fill the
# device, per calibrated board.
_EDGE_FUSED_MIN_KEPT = {"h200": 32}


@dataclasses.dataclass(frozen=True)
class LogicalReduceCall(CallSpec):
    """A logical reduction call: the input's shape and dtype, the axes it reduces."""

    shape: tuple[int, ...] = ()
    # Non-negative and ascending.
    axes: tuple[int, ...] = ()
    op_kind: str = ""
    dtype: torch.dtype = torch.float16
    keepdim: bool = False


def logical_reduce_region(call: LogicalReduceCall) -> bool:
    """The general logical reduction region."""

    return call.op_kind in {"any", "all", "count_nonzero"}


def edge_fused_min_kept(call: LogicalReduceCall) -> float:
    """The fewest kept columns at which the fused edge pass fills the call's board.

    Infinite on a board with no calibrated entry.
    """

    return _EDGE_FUSED_MIN_KEPT.get(call.calibration, math.inf)


def _edge_kept(call: LogicalReduceCall) -> int:
    """The kept extent between a reduced prefix and suffix of axes, or 0 for other layouts."""

    k, j = edge_axis_split(len(call.shape), call.axes)
    return math.prod(call.shape[k : len(call.shape) - j]) if k else 0


def logical_edge_fused_region(call: LogicalReduceCall) -> bool:
    """Edge axes with at least :func:`edge_fused_min_kept` kept columns."""

    kept = _edge_kept(call)
    return logical_reduce_region(call) and kept > 0 and kept >= edge_fused_min_kept(call)


def logical_edge_two_pass_region(call: LogicalReduceCall) -> bool:
    """Edge axes with fewer than :func:`edge_fused_min_kept` kept columns.

    A count crosses between the passes in fp32, so it also needs the elements each output
    reduces to stay within fp32's exact integers.
    """

    kept = _edge_kept(call)
    if not (logical_reduce_region(call) and 0 < kept < edge_fused_min_kept(call)):
        return False
    reduced = math.prod(call.shape) // kept
    return call.op_kind != "count_nonzero" or reduced <= FP32_EXACT_INT_LIMIT
