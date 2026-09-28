"""The facts of one INT8 dequantize call."""

import dataclasses
from typing import Literal, Optional

import torch

from ..call_spec import CallSpec

__all__ = ["DequantizeCall"]


@dataclasses.dataclass(frozen=True)
class DequantizeCall(CallSpec):
    """One INT8 dequantize call, as the op knows it after reading ``q``'s shape."""

    m: int = 0
    k: int = 0
    # What one scale covers: the whole tensor, one row, or 128 contiguous elements of a row.
    granularity: Literal["tensor", "channel", "block"] = "tensor"
    out_dtype: Optional[torch.dtype] = None
