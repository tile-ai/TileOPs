"""Call workloads of the indexing family."""

from workloads.indexing import ConcatWorkload, MaskedScatterWorkload

WORKLOADS = {
    "ConcatFwdOp": ConcatWorkload,
    "MaskedScatterFwdOp": MaskedScatterWorkload,
}
