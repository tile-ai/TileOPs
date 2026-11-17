"""Call workloads of the indexing family."""

from workloads.indexing import (
    ConcatWorkload,
    EmbeddingWorkload,
    MaskedScatterWorkload,
)

WORKLOADS = {
    "ConcatFwdOp": ConcatWorkload,
    "EmbeddingFwdOp": EmbeddingWorkload,
    "MaskedScatterFwdOp": MaskedScatterWorkload,
}
