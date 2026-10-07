"""Case factories of the GEMM family."""

from benchmarks._cases import Entry
from workloads.gemm import (
    BmmFP8Workload,
    BmmWorkload,
    GemmFP8Workload,
    GemmW4A16Workload,
    GemmWorkload,
    GroupedGemmWorkload,
)

ENTRIES = {
    "BmmFwdOp": Entry(BmmWorkload.from_call),
    "BmmFP8FwdOp": Entry(BmmFP8Workload.from_call),
    "GemmFwdOp": Entry(GemmWorkload.from_call),
    "GemmFP8FwdOp": Entry(GemmFP8Workload.from_call, count_copies=True),
    "GemmW4A16FwdOp": Entry(GemmW4A16Workload.from_call),
    "GroupedGemmFwdOp": Entry(GroupedGemmWorkload.from_call),
}
