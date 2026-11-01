"""Call workloads of the rope family."""

from workloads.rope import RoPECall

WORKLOADS = {
    name: RoPECall
    for name in (
        "RoPEFwdOp",
        "RoPELlama31FwdOp",
        "YaRNFwdOp",
        "LongRoPEFwdOp",
        "RoPENeoxPositionIdsFwdOp",
    )
}
