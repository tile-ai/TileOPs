"""Helpers shared by the normalization Op family."""

import math
from typing import Sequence

__all__ = ["normalized_shape_to_n"]


def normalized_shape_to_n(normalized_shape: Sequence[int]) -> int:
    """Return the product of ``normalized_shape``, which must be non-empty."""
    shape = tuple(int(d) for d in normalized_shape)
    if len(shape) == 0:
        raise ValueError("normalized_shape must be non-empty")
    return math.prod(shape)
