"""Private context for stateful n-gram benchmark calls."""

from benchmarks._cases import Entry
from benchmarks.baselines import private_inputs
from workloads.sequence_modeling.ngram_hash import ngram_hash_result


def _bind(op, case):
    return private_inputs(lambda *args: ngram_hash_result(op, *args), case.inputs, 1)


ENTRIES = {"NGramHashFwdOp": Entry(binder=_bind)}
