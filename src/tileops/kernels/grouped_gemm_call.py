"""The facts of one grouped-GEMM call, shared by the implementations that serve it."""

import dataclasses
from typing import Optional

import torch

from .call_spec import CallSpec

__all__ = ["GroupedGemmCall"]


@dataclasses.dataclass(frozen=True)
class GroupedGemmCall(CallSpec):
    """One grouped GEMM over a tight, per-group row layout.

    ``numel`` and ``num_experts`` are how the rows are spread, not how many arrive:
    the routing is only known on the device, so a region stated over these is the
    same for every call of one shape.
    """

    numel: int = 0
    num_experts: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None
