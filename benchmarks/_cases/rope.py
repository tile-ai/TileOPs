"""Case factories of the rope family."""

from benchmarks._cases import Entry
from workloads.rope import RoPECall

ENTRIES = {
    name: Entry(RoPECall)
    for name in (
        "RoPEFwdOp",
        "RoPELlama31FwdOp",
        "YaRNFwdOp",
        "LongRoPEFwdOp",
        "RoPENeoxPositionIdsFwdOp",
    )
}
