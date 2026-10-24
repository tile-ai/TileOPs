# Copyright (c) Tile-AI. All rights reserved.
"""Reduction op layer (L2) package.

This package will host stateless dispatchers for reduction operators
(sum, max, softmax, variance, prefix-scan, etc.) once their corresponding
kernels are implemented.
"""

from tileops.ops.reduction.argreduce import ArgmaxFwdOp, ArgminFwdOp
from tileops.ops.reduction.cumulative import CumprodFwdOp, CumsumFwdOp
from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp
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
from tileops.ops.reduction.vector_norm import VectorNormFwdOp

# Grouped by the kernel that serves each op, which sorting would lose.
__all__: list[str] = [  # noqa: RUF022
    "AllFwdOp",
    "AnyFwdOp",
    "CountNonzeroFwdOp",
    "AmaxFwdOp",
    "AminFwdOp",
    "MeanFwdOp",
    "ProdFwdOp",
    "StdFwdOp",
    "SumFwdOp",
    "VarMeanFwdOp",
    "VarFwdOp",
    "SoftmaxFwdOp",
    "LogSoftmaxFwdOp",
    "LogSumExpFwdOp",
    "ArgmaxFwdOp",
    "ArgminFwdOp",
    "CumsumFwdOp",
    "CumprodFwdOp",
    "VectorNormFwdOp",
]
