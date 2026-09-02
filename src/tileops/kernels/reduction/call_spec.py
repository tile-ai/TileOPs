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
    edge_axes: bool = False
    kept: int = 0
    trail_needs_tiling: bool = False
    reduced_count: int = 0
    tune: bool = False


def logical_reduce_region(call: LogicalReduceCall) -> bool:
    """The general logical reduction region."""

    return call.op_kind in {"any", "all", "count_nonzero"}


# The fused pass runs one block per kept column and has no other parallelism, so
# it takes over only where that alone is enough. Other devices use the general
# implementation until they have a region of their own.
_EDGE_FUSED_MIN_KEPT_H200 = 32


def logical_edge_fused_region(call: LogicalReduceCall) -> bool:
    """The H200 edge-axis logical reduction region served by the fused pass."""

    if not logical_reduce_region(call):
        return False
    if not call.h200 or call.tune:
        return False
    if not call.edge_axes or call.trail_needs_tiling:
        return False
    if call.kept < _EDGE_FUSED_MIN_KEPT_H200:
        return False
    return call.op_kind != "count_nonzero" or call.reduced_count <= 1 << 24
