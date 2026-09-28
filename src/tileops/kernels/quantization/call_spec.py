"""Call records for quantization kernels."""

from __future__ import annotations

import dataclasses

import torch

from tileops.kernels.call_spec import CallSpec

__all__ = ["QuantizeCall"]


@dataclasses.dataclass(frozen=True)
class QuantizeCall(CallSpec):
    """The facts of one quantize call: the 2-D input's shape and dtype, and its group size.

    The block shapes and output dtypes are fixed by each op's signature, so the op a
    kernel serves states them and the record does not.
    """

    # The input's leading extent, ``M`` or ``N``.
    rows: int = 0
    # The extent the scales group along, ``K``.
    cols: int = 0
    dtype: torch.dtype = torch.float16
    # ``INT4QuantPerGroupFwdOp``'s ``group_size``; ``None`` for an op without one.
    group_size: "int | None" = None
