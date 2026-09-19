"""The facts of one grouped-GEMM call, shared by the implementations that serve it."""

import dataclasses
from typing import Optional

import torch

from ..call_spec import CallSpec

__all__ = ["GroupedGemmCall"]


@dataclasses.dataclass(frozen=True)
class GroupedGemmCall(CallSpec):
    """One grouped GEMM over a per-group row layout, tight or padded.

    ``numel`` and ``num_experts`` are the declared spread, not the routed one: the
    routing lands on the device, so a region over these holds for every call.
    """

    numel: int = 0
    num_experts: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None
    transpose_a: bool = False
    transpose_b: bool = True
    # The caller stated a's rows are padded so that no tile spans two groups.
    padded: bool = False
