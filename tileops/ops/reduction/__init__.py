# Copyright (c) Tile-AI. All rights reserved.
"""Reduction op layer (L2) package.

This package will host stateless dispatchers for reduction operators
(sum, max, softmax, variance, prefix-scan, etc.) once their corresponding
kernels are implemented.
"""

# Placeholder imports for reduction ops.
# Each sub-category PR uncomments its own lines.

# --- ReduceKernel ops ---
# --- SoftmaxKernel ops ---
# --- ArgreduceKernel ops ---
from .argmax import ArgmaxOp
from .argmin import ArgminOp
from .log_softmax import LogSoftmaxOp
from .logsumexp import LogSumExpOp
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

# --- CumulativeKernel ops ---
# from .cumsum import CumsumOp
# from .cumprod import CumprodOp
# from .cummax import CummaxOp
# from .cummin import CumminOp

# --- LogicalReduceKernel ops ---
# from .all import AllOp
# from .any import AnyOp
# from .count_nonzero import CountNonzeroOp

# --- VectorNormKernel ops ---
# from .l1_norm import L1NormOp
# from .l2_norm import L2NormOp
# from .inf_norm import InfNormOp

__all__: list[str] = [
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
    # "CumsumOp",
    # "CumprodOp",
    # "CummaxOp",
    # "CumminOp",
    # --- LogicalReduceKernel ops ---
    # "AllOp",
    # "AnyOp",
    # "CountNonzeroOp",
    # --- VectorNormKernel ops ---
    # "L1NormOp",
    # "L2NormOp",
    # "InfNormOp",
]
