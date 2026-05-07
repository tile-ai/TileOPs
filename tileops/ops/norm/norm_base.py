"""Helpers shared by the normalization Op family."""

import math
from typing import Sequence

__all__ = ["ALIGNMENT", "normalized_shape_to_n"]

ALIGNMENT = 256


def normalized_shape_to_n(normalized_shape: Sequence[int]) -> int:
    """Return the product of ``normalized_shape``.

    Args:
        normalized_shape: Manifest-style trailing-axis shape; must be
            non-empty.

    Returns:
        Product of every entry in ``normalized_shape``.

    Raises:
        ValueError: If ``normalized_shape`` is empty.
    """
    shape = tuple(int(d) for d in normalized_shape)
    if len(shape) == 0:
        raise ValueError("normalized_shape must be non-empty")
    return math.prod(shape)
