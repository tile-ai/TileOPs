# Copyright (c) Tile-AI. All rights reserved.
"""Reduction op layer (L2) package.

This package will host stateless dispatchers for reduction operators
(sum, max, softmax, variance, prefix-scan, etc.) once their corresponding
kernels are implemented.
"""

# --- LogicalReduceKernel ops ---
# --- ArgreduceKernel ops ---
from tileops.ops.reduction.argreduce import ArgmaxFwdOp, ArgminFwdOp

# --- CumulativeKernel ops ---
from tileops.ops.reduction.cumulative import CumprodFwdOp, CumsumFwdOp
from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp

# --- ReduceKernel ops ---
# --- SoftmaxKernel ops ---
from tileops.ops.reduction.reduce import (
    AmaxFwdOp,
    AminFwdOp,
    MeanFwdOp,
    ProdFwdOp,
    StdFwdOp,
    SumFwdOp,
    VarFwdOp,
    VarMeanFwdOp,
)
from tileops.ops.reduction.softmax import LogSoftmaxFwdOp, LogSumExpFwdOp, SoftmaxFwdOp

# --- VectorNormKernel ops ---
from tileops.ops.reduction.vector_norm import VectorNormFwdOp

__all__: list[str] = [
    # --- LogicalReduceKernel ops ---
    "AllFwdOp",
    "AnyFwdOp",
    "CountNonzeroFwdOp",
    # --- ReduceKernel ops ---
    "AmaxFwdOp",
    "AminFwdOp",
    "MeanFwdOp",
    "ProdFwdOp",
    "StdFwdOp",
    "SumFwdOp",
    "VarMeanFwdOp",
    "VarFwdOp",
    # --- SoftmaxKernel ops ---
    "SoftmaxFwdOp",
    "LogSoftmaxFwdOp",
    "LogSumExpFwdOp",
    # --- ArgreduceKernel ops ---
    "ArgmaxFwdOp",
    "ArgminFwdOp",
    # --- CumulativeKernel ops ---
    "CumsumFwdOp",
    "CumprodFwdOp",
    # --- VectorNormKernel ops ---
    "VectorNormFwdOp",
]
