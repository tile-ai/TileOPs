# Copyright (c) Tile-AI. All rights reserved.
"""Reduction kernels, one module per sub-category."""

from ._primitives import (
    DEFAULT_ALIGNMENT,
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
from .reduce import (
    ReduceEdgeKernel,
    ReduceFoldKernel,
    ReduceKernel,
    ReduceLeadingKernel,
    ReduceProdKernel,
    WelfordEdgeKernel,
    WelfordReduceKernel,
)
from .softmax import (
    SoftmaxKernel,
    SoftmaxSplitKernel,
)
from .vector_norm import (
    VectorNormEdgeKernel,
    VectorNormKernel,
)

__all__: list[str] = [
    "DEFAULT_ALIGNMENT",
    "ArgreduceKernel",
    "CumulativeKernel",
    "LogSumExpEdgeSplitKernel",
    "LogSumExpKernel",
    "LogSumExpSplitKernel",
    "LogSumExpStreamingKernel",
    "LogicalReduceEdgeFusedKernel",
    "LogicalReduceEdgeTwoPassKernel",
    "LogicalReduceKernel",
    "ReduceEdgeKernel",
    "ReduceFoldKernel",
    "ReduceKernel",
    "ReduceLeadingKernel",
    "ReduceProdKernel",
    "SoftmaxKernel",
    "SoftmaxSplitKernel",
    "VectorNormEdgeKernel",
    "VectorNormKernel",
    "WelfordEdgeKernel",
    "WelfordReduceKernel",
    "align_up",
]
