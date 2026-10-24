"""Case factories of the GEMM family."""

from benchmarks._cases import Entry
from benchmarks.api import Implementation
from workloads.gemm import (
    BmmFP8Workload,
    BmmWorkload,
    GemmFP8Workload,
    GemmINT8W8A8Workload,
    GemmW4A8Workload,
    GemmW4A16Workload,
    GemmWorkload,
    GroupedGemmWorkload,
)


def _prepacked_weight(op, case) -> Implementation:
    """The op called on its own repack of the case's row-major weight, made once, untimed."""
    activation, packed_weight, *scales = case.inputs
    return Implementation(run=op, args=(activation, op.repack(packed_weight), *scales))


ENTRIES = {
    "BmmFwdOp": Entry(BmmWorkload.from_call),
    "BmmFP8FwdOp": Entry(BmmFP8Workload.from_call),
    "GemmFwdOp": Entry(GemmWorkload.from_call),
    "GemmFP8FwdOp": Entry(GemmFP8Workload.from_call, count_copies=True),
    "GemmINT8W8A8FwdOp": Entry(GemmINT8W8A8Workload.from_call),
    "GemmW4A8FwdOp": Entry(GemmW4A8Workload.from_call, binder=_prepacked_weight),
    "GemmW4A16FwdOp": Entry(GemmW4A16Workload.from_call),
    "GroupedGemmFwdOp": Entry(GroupedGemmWorkload.from_call),
}
