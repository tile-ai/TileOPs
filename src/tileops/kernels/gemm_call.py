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

    Its CUDA-core inner loop pays ``m`` FMAs and ``m`` converts per weight
    element, so its lead over the tensor-core kernel shrinks as ``m`` grows.
    Once the analytic small-``m`` configs (split-K / simple, see
    ``gemm_heuristics._tiny_m_config``) lifted the general kernel to cuBLAS
    parity the crossover moved down to ``m == 2`` — measured on H200 (per-rep
    interleaved, full config-grid sweep on the decode shapes): this kernel
    beats the best general config at ``m = 2`` on all three families and loses
    from ``m = 3`` up (gate-up 1.02x vs 1.05x, down 0.81x vs 0.92x).

    The occupancy condition stands: the general grid is one M-tile wide, so
    ~``ceil(n/128)`` CTAs, and with that at a full wave the streaming kernel
    has no idle SMs to reclaim.

    ``m == 1`` is the GEMV region. NT only: the reduction over ``K`` needs
    ``K`` contiguous.
    """
    if call.trans_a or not call.trans_b or call.m != 2:
        return False
    sm_count = (
        torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
        if torch.cuda.is_available() else 132)
    return -(-call.n // 128) < sm_count
