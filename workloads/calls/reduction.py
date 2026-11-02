"""Call workloads of the reduction family."""

import functools

from workloads.reduction import CumulativeCall, LogicalCall, ProdCall, ReductionCall

WORKLOADS = {
    **{
        name: ReductionCall
        for name in (
            "ArgmaxFwdOp",
            "ArgminFwdOp",
            "SumFwdOp",
            "MeanFwdOp",
            "AmaxFwdOp",
            "AminFwdOp",
            "StdFwdOp",
            "VarFwdOp",
            "VarMeanFwdOp",
            "SoftmaxFwdOp",
            "LogSoftmaxFwdOp",
            "LogSumExpFwdOp",
            "VectorNormFwdOp",
        )
    },
    "ProdFwdOp": ProdCall,
    **{name: LogicalCall for name in ("AnyFwdOp", "AllFwdOp", "CountNonzeroFwdOp")},
    "CumsumFwdOp": functools.partial(CumulativeCall, op_kind="cumsum"),
    "CumprodFwdOp": functools.partial(CumulativeCall, op_kind="cumprod"),
}
