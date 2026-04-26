"""Cumulative product operator (L2 Op layer).

Provides:
  - CumprodFwdOp: y = cumprod(x, dim)

Output has the same shape and dtype as input. Alignment padding is handled
inside the kernel via masked loads.
"""

import torch

from .cumulative_base import CumulativeOp

__all__ = ["CumprodFwdOp"]


class CumprodFwdOp(CumulativeOp):
    """Cumulative product operator: y = cumprod(x, dim)."""

    _op_kind = "prod"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x)
