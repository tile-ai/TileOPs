"""The facts of one GEMM call, and the region the GEMV kernel serves."""

import dataclasses
from typing import Optional

import torch

from .call_spec import CallSpec

__all__ = ["GemmCall", "gemv_region", "small_batch_region"]


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


def small_batch_region(call: GemmCall) -> bool:
    """Whether the small-batch bandwidth kernel serves this call.

    For small ``m`` an NT GEMM is HBM-bound and the warp-specialized kernel
    underfills the GPU: its grid is one M-tile wide, so ~``ceil(n/128)`` CTAs.
    The region is set by that occupancy rather than by ``k``, in two tiers by
    ``m`` (the bandwidth kernel's per-row cost rises with ``m``):

    - ``2 <= m <= 4``: whenever the generic grid underfills a wave.
    - ``5 <= m <= 8``: only when it is severely underfilled (<= ~20% of a
      wave); by ~25% the generic tensor-core kernel is already faster.

    ``m >= 9`` regresses even when severely underfilled, and ``m == 1`` is the
    GEMV region. NT only: the reduction over ``K`` needs ``K`` contiguous.
    """
    if call.trans_a or not call.trans_b:
        return False
    sm_count = (
        torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
        if torch.cuda.is_available() else 132)
    n_ctas = -(-call.n // 128)
    return ((2 <= call.m <= 4 and n_ctas < sm_count)
            or (5 <= call.m <= 8 and 5 * n_ctas <= sm_count))
