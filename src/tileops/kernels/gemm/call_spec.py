"""The facts of one GEMM call, and the region the GEMV kernel serves."""

import dataclasses
from typing import Optional

import torch


from ..call_spec import CallSpec
from .call_spec import CallSpec
from .heuristics import TINY_M_BLOCK_N
from .heuristics import TINY_M_BLOCK_N, swap_ab_stages
from tileops.utils import get_sm_count

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
    ``heuristics._tiny_m_config``) lifted the general kernel to cuBLAS
    parity the crossover moved down to ``m == 2`` — measured on H200 (per-rep
    interleaved, full config-grid sweep on the decode shapes): this kernel
    beats the best general config at ``m = 2`` on all three families and loses
    from ``m = 3`` up (gate-up 1.02x vs 1.05x, down 0.81x vs 0.92x).

    The occupancy condition stands, priced on the grid this kernel actually
    competes with — the tiny-m generic band's ``block_n`` — because with that
    grid at a full wave the streaming kernel has no idle SMs to reclaim.

    It also steps aside wherever the operand-swapped generic kernel applies:
    that one streams the same weights on a grid twice as wide with no padded
    ``A`` re-read, and measures 1.07-1.08x cuBLAS at ``m = 2`` on the down and
    attention shapes against this kernel's 1.01x.

    ``m == 1`` is the GEMV region. NT only: the reduction over ``K`` needs
    ``K`` contiguous.
    """
    if call.trans_a or not call.trans_b or call.m != 2:
        return False
    sm_count = get_sm_count()
    if swap_ab_stages(call.n, sm_count) is not None:
        return False
    return -(-call.n // TINY_M_BLOCK_N) < sm_count
