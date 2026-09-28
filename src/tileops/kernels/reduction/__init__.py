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
    LogSumExpSingleTileKernel,
    LogSumExpSplitKernel,
    LogSumExpStreamingKernel,
)
from .reduce import ReduceKernel
from .softmax import (
    SoftmaxFusedSplitKernel,
    SoftmaxKernel,
    SoftmaxSingleTileKernel,
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
    "LogSumExpSingleTileKernel",
    "LogSumExpSplitKernel",
    "LogSumExpStreamingKernel",
    "LogicalReduceEdgeFusedKernel",
    "LogicalReduceEdgeTwoPassKernel",
    "LogicalReduceKernel",
    "ReduceKernel",
    "SoftmaxFusedSplitKernel",
    "SoftmaxKernel",
    "SoftmaxSingleTileKernel",
    "SoftmaxSplitKernel",
    "VectorNormKernel",
    "align_up",
]
