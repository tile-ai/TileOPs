"""Call records and implementation regions for reduction kernels."""

from __future__ import annotations

import dataclasses

import torch

from tileops.kernels.call_spec import CallSpec

__all__ = [
    "LogicalReduceCall",
    "logical_edge_fused_region",
    "logical_reduce_region",
]


@dataclasses.dataclass(frozen=True)
class LogicalReduceCall(CallSpec):
    """Semantic and shape facts used to select a logical reduction implementation."""

    shape: tuple[int, ...] = ()
    axes: tuple[int, ...] = ()
    op_kind: str = ""
    dtype: torch.dtype = torch.float16
    keepdim: bool = False
    m: int = 0
    edge_axes: bool = False
    kept: int = 0
    trail_needs_tiling: bool = False
    reduced_count: int = 0


def logical_reduce_region(call: LogicalReduceCall) -> bool:
    """The general logical reduction region."""

    return call.op_kind in {"any", "all", "count_nonzero"}


# The fused pass runs one block per kept column and has no other parallelism, so
# it takes over only where that alone is enough: the fewest kept columns that fill the
# device, per calibrated board. A board without an entry uses the general implementation
# until it has a region of its own.
_EDGE_FUSED_MIN_KEPT = {"h200": 32}


def logical_edge_fused_region(call: LogicalReduceCall) -> bool:
    """The edge-axis logical reduction region a calibrated board serves with the fused pass."""

    if not logical_reduce_region(call):
        return False
    min_kept = _EDGE_FUSED_MIN_KEPT.get(call.calibration)
    if min_kept is None:
        return False
    if not call.edge_axes or call.trail_needs_tiling:
        return False
    if call.kept < min_kept:
        return False
    return call.op_kind != "count_nonzero" or call.reduced_count <= 1 << 24
