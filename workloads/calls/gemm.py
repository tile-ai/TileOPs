"""Call workloads of the gemm family."""

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

WORKLOADS = {
    "BMMFwdOp": BMMWorkload.from_call,
    "BMMFP8FwdOp": BMMFP8Workload.from_call,
    "GEMMFwdOp": GEMMWorkload.from_call,
    "GEMMFP8FwdOp": GEMMFP8Workload.from_call,
    "GEMMINT8W8A8FwdOp": GEMMINT8W8A8Workload.from_call,
    "GEMMW4A8FwdOp": GEMMW4A8Workload.from_call,
    "GEMMW4A16FwdOp": GEMMW4A16Workload.from_call,
    "GroupedGEMMFwdOp": GroupedGEMMWorkload.from_call,
}
