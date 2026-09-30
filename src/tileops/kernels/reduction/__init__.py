# Copyright (c) Tile-AI. All rights reserved.
"""Reduction kernels, one module per sub-category."""

from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT, align_up
from tileops.kernels.reduction.argreduce import (
    ArgreduceKernel,
    ArgreduceSplitKernel,
    ArgreduceStridedKernel,
)
from tileops.kernels.reduction.cumulative import (
    CumsumParallelScanKernel,
    CumulativeKernel,
    CumulativeRowScanKernel,
)
from tileops.kernels.reduction.logical_reduce import (
    CountNonzeroEdgeTwoPassKernel,
    LogicalReduceEdgeFusedKernel,
    LogicalReduceEdgeTwoPassKernel,
    LogicalReduceKernel,
)
from tileops.kernels.reduction.logsumexp import (
    LogSumExpEdgeSplitKernel,
    LogSumExpKernel,
    LogSumExpSplitKernel,
    LogSumExpStreamingKernel,
)
from tileops.kernels.reduction.reduce import (
    ReduceEdgeKernel,
    ReduceFoldKernel,
    ReduceKernel,
    ReduceLeadingKernel,
    ReduceProdKernel,
    WelfordEdgeKernel,
    WelfordReduceKernel,
)
from tileops.kernels.reduction.softmax import SoftmaxKernel, SoftmaxSplitKernel
from tileops.kernels.reduction.vector_norm import VectorNormEdgeKernel, VectorNormKernel

__all__: list[str] = [
    "DEFAULT_ALIGNMENT",
    "ArgreduceKernel",
    "ArgreduceSplitKernel",
    "ArgreduceStridedKernel",
    "CountNonzeroEdgeTwoPassKernel",
    "CumsumParallelScanKernel",
    "CumulativeKernel",
    "CumulativeRowScanKernel",
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
