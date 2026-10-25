# Copyright (c) Tile-AI. All rights reserved.
"""Reduction kernels, one module per sub-category."""

from tileops._kernels.reduction._primitives import DEFAULT_ALIGNMENT
from tileops._kernels.reduction.argreduce import (
    ArgreduceKernel,
    ArgreduceSplitKernel,
    ArgreduceStridedKernel,
)
from tileops._kernels.reduction.cumulative import (
    CumsumParallelScanKernel,
    CumulativeKernel,
    CumulativeRowScanKernel,
)
from tileops._kernels.reduction.logical_reduce import (
    CountNonzeroEdgeTwoPassKernel,
    LogicalReduceEdgeFusedKernel,
    LogicalReduceEdgeTwoPassKernel,
    LogicalReduceKernel,
)
from tileops._kernels.reduction.logsumexp import (
    LogSumExpEdgeSplitKernel,
    LogSumExpKernel,
    LogSumExpSplitKernel,
    LogSumExpStreamingKernel,
)
from tileops._kernels.reduction.reduce import (
    ReduceEdgeKernel,
    ReduceFoldKernel,
    ReduceKernel,
    ReduceLeadingKernel,
    ReduceProdKernel,
    WelfordEdgeKernel,
    WelfordReduceKernel,
)
from tileops._kernels.reduction.softmax import (
    SoftmaxKernel,
    SoftmaxSplitKernel,
    SoftmaxStreamingKernel,
)
from tileops._kernels.reduction.softmax_on_chip import (
    SoftmaxClusterSplitKernel,
    SoftmaxOnChipKernel,
)
from tileops._kernels.reduction.softmax_warp_rows import SoftmaxWarpRowsKernel
from tileops._kernels.reduction.vector_norm import VectorNormEdgeKernel, VectorNormKernel

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
    "SoftmaxClusterSplitKernel",
    "SoftmaxKernel",
    "SoftmaxOnChipKernel",
    "SoftmaxSplitKernel",
    "SoftmaxStreamingKernel",
    "SoftmaxWarpRowsKernel",
    "VectorNormEdgeKernel",
    "VectorNormKernel",
    "WelfordEdgeKernel",
    "WelfordReduceKernel",
]
