# Copyright (c) Tile-AI. All rights reserved.
"""Reduction op layer (L2) package.

This package will host stateless dispatchers for reduction operators
(sum, max, softmax, variance, prefix-scan, etc.) once their corresponding
kernels are implemented.
"""

# Placeholder imports for reduction ops.
# Each sub-category PR uncomments its own lines.

# --- LogicalReduceKernel ops ---
from .all_op import AllOp
from .any_op import AnyOp

# --- ArgreduceKernel ops ---
from .argmax import ArgmaxOp
from .argmin import ArgminOp
from .count_nonzero import CountNonzeroOp

# --- CumulativeKernel ops ---
from .cumprod import CumprodOp
from .cumsum import CumsumOp

# from .cummax import CummaxOp
# from .cummin import CumminOp
# --- VectorNormKernel ops ---
from .inf_norm import InfNormOp
from .l1_norm import L1NormOp
from .l2_norm import L2NormOp
from .log_softmax import LogSoftmaxOp
from .logsumexp import LogSumExpOp

# --- ReduceKernel ops ---
# --- SoftmaxKernel ops ---
from .reduce import (
    AmaxOp,  # ReduceMaxOp
    AminOp,  # ReduceMinOp
    MeanOp,  # ReduceMeanOp
    ProdOp,  # ReduceProdOp
    StdOp,
    SumOp,  # ReduceSumOp
    VarMeanOp,
    VarOp,
)
from .softmax import SoftmaxOp

__all__: list[str] = [
    # --- LogicalReduceKernel ops ---
    "AllOp",
    "AnyOp",
    "CountNonzeroOp",
    # --- ReduceKernel ops ---
    "AmaxOp",
    "AminOp",
    "MeanOp",
    "ProdOp",
    "StdOp",
    "SumOp",
    "VarMeanOp",
    "VarOp",
    # "ReduceMaxOp",
    # "ReduceMeanOp",
    # "ReduceMinOp",
    # "ReduceProdOp",
    # "ReduceSumOp",
    # --- SoftmaxKernel ops ---
    "SoftmaxOp",
    "LogSoftmaxOp",
    "LogSumExpOp",
    # --- ArgreduceKernel ops ---
    "ArgmaxOp",
    "ArgminOp",
    # --- CumulativeKernel ops ---
    "CumsumOp",
    "CumprodOp",
    # "CummaxOp",
    # "CumminOp",
    # --- VectorNormKernel ops ---
    "InfNormOp",
    "L1NormOp",
    "L2NormOp",
]
