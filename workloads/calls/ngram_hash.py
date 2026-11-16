"""Manifest call workload for n-gram hashing."""

from workloads.sequence_modeling.ngram_hash import NGramHashWorkloadCall

WORKLOADS = {"NGramHashFwdOp": NGramHashWorkloadCall}
