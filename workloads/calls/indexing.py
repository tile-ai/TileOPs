"""Call workloads of the indexing family."""

from workloads.indexing import EmbeddingWorkload

WORKLOADS = {
    "EmbeddingFwdOp": EmbeddingWorkload,
}
