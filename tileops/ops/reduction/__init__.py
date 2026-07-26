# Copyright (c) Tile-AI. All rights reserved.
"""Reduction op layer (L2) package.

This package will host stateless dispatchers for reduction operators
(sum, max, softmax, variance, prefix-scan, etc.) once their corresponding
kernels are implemented.
"""

# Placeholder imports for reduction ops.
# Each sub-category PR uncomments its own lines.

# --- LogicalReduceKernel ops ---
# --- ArgreduceKernel ops ---
from .argreduce import ArgmaxFwdOp, ArgminFwdOp

# --- CumulativeKernel ops ---
from .cumulative import CumprodFwdOp, CumsumFwdOp
from .logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp

# --- ReduceKernel ops ---
# --- SoftmaxKernel ops ---
from .reduce import (
    AmaxFwdOp,  # ReduceMaxOp
    AminFwdOp,  # ReduceMinOp
    MeanFwdOp,  # ReduceMeanOp
    ProdFwdOp,  # ReduceProdOp
    StdFwdOp,
    SumFwdOp,  # ReduceSumOp
    VarFwdOp,
    VarMeanFwdOp,
)
from .softmax import LogSoftmaxFwdOp, LogSumExpFwdOp, SoftmaxFwdOp

# from .cummax import CummaxOp
# from .cummin import CumminOp
# --- VectorNormKernel ops ---
from .vector_norm import InfNormFwdOp, L1NormFwdOp, L2NormFwdOp

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
    # "ReduceMaxOp",
    # "ReduceMeanOp",
    # "ReduceMinOp",
    # "ReduceProdOp",
    # "ReduceSumOp",
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
    # "CummaxOp",
    # "CumminOp",
    # --- VectorNormKernel ops ---
    "InfNormFwdOp",
    "L1NormFwdOp",
    "L2NormFwdOp",
]
