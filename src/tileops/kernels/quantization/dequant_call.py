"""The facts of one INT8 dequantize call."""

import dataclasses
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec

__all__ = ["DequantizeCall"]


@dataclasses.dataclass(frozen=True)
class DequantizeCall(CallSpec):
    """One INT8 dequantize call, as the op knows it after reading ``q``'s shape."""

    m: int = 0
    k: int = 0
    out_dtype: Optional[torch.dtype] = None
