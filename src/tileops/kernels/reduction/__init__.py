# Copyright (c) Tile-AI. All rights reserved.
"""Reduction kernels, one module per sub-category."""

from ._primitives import (
    DEFAULT_ALIGNMENT,
    SHARED_MEMORY_BUDGET_BYTES,
    align_up,
)
from .argreduce import ArgreduceKernel
from .cumulative import CumulativeKernel
from .logical_reduce import (
    LogicalReduceEdgeFusedKernel,
    LogicalReduceEdgeTwoPassKernel,
    LogicalReduceKernel,
)
from .logsumexp import (
    LogSumExpEdgeSplitKernel,
    LogSumExpKernel,
    LogSumExpSplitKernel,
    LogSumExpStreamingKernel,
)
from .reduce import ReduceKernel
from .softmax import (
    SoftmaxKernel,
    SoftmaxSplitKernel,
)
from .vector_norm import VectorNormKernel

__all__: list[str] = [
    "DEFAULT_ALIGNMENT",
    "SHARED_MEMORY_BUDGET_BYTES",
    "ArgreduceKernel",
    "CumulativeKernel",
    "LogSumExpEdgeSplitKernel",
    "LogSumExpKernel",
    "LogSumExpSplitKernel",
    "LogSumExpStreamingKernel",
    "LogicalReduceEdgeFusedKernel",
    "LogicalReduceEdgeTwoPassKernel",
    "LogicalReduceKernel",
    "ReduceKernel",
    "SoftmaxKernel",
    "SoftmaxSplitKernel",
    "VectorNormKernel",
    "align_up",
]
