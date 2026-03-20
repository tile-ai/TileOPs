"""Reduce kernel subpackage (sum, mean, amin, amax, prod, std, var, var_mean)."""

from .fwd import ReduceKernel

__all__: list[str] = [
    "ReduceKernel",
]
