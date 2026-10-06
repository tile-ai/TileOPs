"""Case factories of the reduction family."""

import functools

from benchmarks._cases import Entry
from workloads.reduction import CumulativeCall, LogicalCall, ProdCall, ReductionCall

ENTRIES = {
    **{
        name: Entry(ReductionCall)
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
    "ProdFwdOp": Entry(ProdCall),
    **{name: Entry(LogicalCall) for name in ("AnyFwdOp", "AllFwdOp", "CountNonzeroFwdOp")},
    "CumsumFwdOp": Entry(functools.partial(CumulativeCall, op_kind="cumsum")),
    "CumprodFwdOp": Entry(functools.partial(CumulativeCall, op_kind="cumprod")),
}
