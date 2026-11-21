"""Call workloads of the ngram_hash family."""

from workloads.sequence_modeling.ngram_hash import NGramHashWorkloadCall

WORKLOADS = {"NGramHashFwdOp": NGramHashWorkloadCall}
