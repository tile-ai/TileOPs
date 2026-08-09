"""The facts of one GEMM call, and the region the GEMV kernel serves."""

import dataclasses
from typing import Optional

import torch

from .call_spec import CallSpec

__all__ = ["GemmCall", "gemv_region"]


@dataclasses.dataclass(frozen=True)
class GemmCall(CallSpec):
    """One matmul, as the op knows it after inferring ``(m, n, k)``."""

    m: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None
    trans_a: bool = False
    trans_b: bool = False


def gemv_region(call: GemmCall) -> bool:
    """Whether the call is a matrix-vector product the GEMV kernel is written for.

    Either operand may be the vector: ``a`` is a single row with ``b``
    transposed, or ``b`` is a single column with neither transposed. The other
    two layouts have no GEMV form here.
    """
    lhs_row = call.m == 1 and not call.trans_a and call.trans_b
    rhs_col = call.n == 1 and not call.trans_a and not call.trans_b
    return lhs_row or rhs_col
