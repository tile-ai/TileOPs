"""Helpers shared by the normalization Op family."""

from typing import Optional, Sequence

import torch

__all__ = ["affine_or_constant"]


def affine_or_constant(
    tensor: Optional[torch.Tensor],
    shape: Sequence[int],
    value: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """*tensor*, or the constant an absent affine tensor stands for: ones for a scale,
    zeros for a shift. Multiplying by one and adding zero leave every value unchanged."""
    if tensor is not None:
        return tensor.contiguous()
    return torch.full(tuple(shape), value, dtype=dtype, device=device)
