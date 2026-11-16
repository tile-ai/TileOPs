"""Call workloads of the indexing family."""

from workloads.indexing import EmbeddingCall

WORKLOADS = {
    "EmbeddingFwdOp": EmbeddingCall,
}
