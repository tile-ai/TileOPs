"""The facts of one GEMM call, and the regions the bandwidth kernels serve."""

import dataclasses
from typing import Optional

import torch

from tileops.utils import get_sm_count

from .call_spec import CallSpec
from .gemm_heuristics import swap_ab_grid_underfills

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
    """The region the small-batch bandwidth kernel claims: m == 2, NT, narrow n.

    Its CUDA-core inner loop pays ``m`` FMAs and ``m`` converts per weight
    element, so its lead over the tensor-core kernel shrinks as ``m`` grows.
    Once the analytic small-``m`` configs (split-K / simple, see
    ``gemm_heuristics._tiny_m_config``) lifted the general kernel to cuBLAS
    parity the crossover moved down to ``m == 2`` — measured on H200 (per-rep
    interleaved, full config-grid sweep on the decode shapes): this kernel
    beats the best general config at ``m = 2`` on all three families and loses
    from ``m = 3`` up (gate-up 1.02x vs 1.05x, down 0.81x vs 0.92x).

    The n bound is ``swap_ab_grid_underfills``: the claim holds while a
    64-wide n-tiling sits below three-eighths of a wave. From that fill upward
    the lead inverts — a grid that wide streams the same weights with no
    padded ``A`` re-read, and measures 1.07-1.08x cuBLAS at ``m = 2`` on the
    down and attention shapes against this kernel's 1.01x.

    The same bound settles occupancy: a full wave of the tiny-m generic band
    needs ``n >= TINY_M_BLOCK_N * sm_count``, far past it — so nothing claimed
    here is out of idle SMs a generic config could reclaim. Widening the
    claimed band brings that condition back.

    ``m == 1`` is the GEMV region. NT only: the reduction over ``K`` needs
    ``K`` contiguous.
    """
    if call.trans_a or not call.trans_b or call.m != 2:
        return False
    return swap_ab_grid_underfills(call.n, get_sm_count())
