"""Cumulative sum operator (L2 Op layer).

Provides:
  - CumsumFwdOp: y = cumsum(x, dim)

Output has the same shape and dtype as input. Alignment padding is handled
inside the kernel via masked loads.
"""

import torch

from .cumulative_base import CumulativeOp

__all__ = ["CumsumFwdOp"]


class CumsumFwdOp(CumulativeOp):
    """Cumulative sum operator: ``y = cumsum(x, dim)``.

    Output has the same shape and dtype as ``x``. Alignment padding is
    handled inside the kernel via masked loads.

    Args:
        N: Reduction dimension size (statically committed at ctor;
            corresponds to manifest ``static_dims.N = "x.shape[dim]"``).
        dtype: Data type (float32, float16, or bfloat16).
        dim: Reduction axis (default -1). Negative values are normalized
            at forward time.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).

    Example:
        >>> op = CumsumFwdOp(N=4096, dtype=torch.float16)
        >>> x = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
        >>> y = op(x)  # shape: (1024, 4096)
    """

    _op_kind = "sum"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x)
