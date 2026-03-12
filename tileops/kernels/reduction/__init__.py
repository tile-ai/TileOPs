# Copyright (c) Tile-AI. All rights reserved.
"""Reduction kernel primitives and shared utilities.

This package provides the foundational building blocks (macros, constants,
and utility functions) used by all reduction sub-category kernels.
"""

from ._primitives import (
    DEFAULT_ALIGNMENT,
    align_up,
    make_cumulative_scan,
    make_reduce_epilogue,
    make_softmax_epilogue,
    make_welford_update,
)
from .argreduce import ArgreduceKernel
from .reduce import ReduceKernel
from .softmax import LogSumExpKernel, SoftmaxKernel

# Placeholder imports for reduction kernels.
# Each sub-category PR uncomments its own lines.
# from .cumulative import CumulativeKernel
# from .logical_reduce import LogicalReduceKernel
# from .vector_norm import VectorNormKernel

__all__: list[str] = [
    "align_up",
    "DEFAULT_ALIGNMENT",
    "make_reduce_epilogue",
    "make_welford_update",
    "make_softmax_epilogue",
    "make_cumulative_scan",
    "ArgreduceKernel",
    # "CumulativeKernel",
    # "LogicalReduceKernel",
    "ReduceKernel",
    "LogSumExpKernel",
    "SoftmaxKernel",
    # "VectorNormKernel",
]
