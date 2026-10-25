"""Case factories of the GEMM family."""

from benchmarks._cases import Entry
from benchmarks.api import Implementation
from workloads.gemm import (
    BMMFP8Workload,
    BMMWorkload,
    GEMMFP8Workload,
    GEMMINT8W8A8Workload,
    GEMMW4A8Workload,
    GEMMW4A16Workload,
    GEMMWorkload,
    GroupedGEMMWorkload,
)


def _prepacked_weight(op, case) -> Implementation:
    """The op called on its own repack of the case's row-major weight, made once, untimed."""
    activation, packed_weight, *scales = case.inputs
    return Implementation(run=op, args=(activation, op.repack(packed_weight), *scales))


ENTRIES = {
    "BMMFwdOp": Entry(BMMWorkload.from_call),
    "BMMFP8FwdOp": Entry(BMMFP8Workload.from_call),
    "GEMMFwdOp": Entry(GEMMWorkload.from_call),
    "GEMMFP8FwdOp": Entry(GEMMFP8Workload.from_call, count_copies=True),
    "GEMMINT8W8A8FwdOp": Entry(GEMMINT8W8A8Workload.from_call),
    "GEMMW4A8FwdOp": Entry(GEMMW4A8Workload.from_call, binder=_prepacked_weight),
    "GEMMW4A16FwdOp": Entry(GEMMW4A16Workload.from_call),
    "GroupedGEMMFwdOp": Entry(GroupedGEMMWorkload.from_call),
}
