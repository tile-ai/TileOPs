# Copyright (c) Tile-AI. All rights reserved.
"""Reduction op layer (L2) package.

This package will host stateless dispatchers for reduction operators
(sum, max, softmax, variance, prefix-scan, etc.) once their corresponding
kernels are implemented.
"""

# Placeholder imports for reduction ops.
# Each sub-category PR uncomments its own lines.

# --- LogicalReduceKernel ops ---
from .all_op import AllFwdOp
from .any_op import AnyFwdOp

# --- ArgreduceKernel ops ---
from .argmax import ArgmaxFwdOp
from .argmin import ArgminFwdOp
from .count_nonzero import CountNonzeroFwdOp

# --- CumulativeKernel ops ---
from .cumprod import CumprodFwdOp
from .cumsum import CumsumFwdOp

# from .cummax import CummaxOp
# from .cummin import CumminOp
# --- VectorNormKernel ops ---
from .inf_norm import InfNormFwdOp
from .l1_norm import L1NormFwdOp
from .l2_norm import L2NormFwdOp
from .log_softmax import LogSoftmaxFwdOp
from .logsumexp import LogSumExpFwdOp

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
from .softmax import SoftmaxFwdOp

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
