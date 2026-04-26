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
    """Cumulative sum operator: y = cumsum(x, dim)."""

    _op_kind = "sum"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x)
