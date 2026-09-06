# Copyright (c) Tile-AI. All rights reserved.
"""Reduction kernels, one module per sub-category."""

from ._primitives import (
    DEFAULT_ALIGNMENT,
    SHARED_MEMORY_BUDGET_BYTES,
    align_up,
)
from .argreduce import ArgreduceKernel
from .cumulative import CumulativeKernel
from .logical_reduce import LogicalReduceEdgeFusedKernel, LogicalReduceKernel
from .logsumexp import LogSumExpKernel
from .reduce import ReduceKernel
from .softmax import SoftmaxKernel
from .vector_norm import VectorNormKernel

__all__: list[str] = [
    "DEFAULT_ALIGNMENT",
    "SHARED_MEMORY_BUDGET_BYTES",
    "ArgreduceKernel",
    "CumulativeKernel",
    "LogSumExpKernel",
    "LogicalReduceEdgeFusedKernel",
    "LogicalReduceKernel",
    "ReduceKernel",
    "SoftmaxKernel",
    "VectorNormKernel",
    "align_up",
]
