"""The facts of one DeltaNet decode call, gated or not."""

import dataclasses
from typing import Optional

import torch

from .call_spec import CallSpec

__all__ = ["DeltaNetDecodeCall"]


@dataclasses.dataclass(frozen=True)
class DeltaNetDecodeCall(CallSpec):
    """One decode step, as the op knows it after reading its inputs.

    ``tune`` is one of the facts: an implementation with no tunable knobs is not
    the one to run when the caller asked for autotuning.
    """

    batch: int = 0
    heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    dtype: Optional[torch.dtype] = None
    tune: bool = False
