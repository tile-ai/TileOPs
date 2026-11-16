"""Call workloads of the indexing family."""

from workloads.indexing import MaskedScatterWorkload

WORKLOADS = {
    "MaskedScatterFwdOp": MaskedScatterWorkload,
}
