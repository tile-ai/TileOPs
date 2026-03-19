"""Cumulative scan kernel subpackage (cumsum, cumprod)."""

from .fwd import CumulativeKernel

__all__: list[str] = [
    "CumulativeKernel",
]
