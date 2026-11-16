"""N-gram hashes and context commits against the independent torch baseline."""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import private_inputs
from tileops.sequence_modeling import NGramHashFwdOp


@pytest.mark.parametrize("case", bench.cases(NGramHashFwdOp), ids=lambda case: case.id)
def test_ngram_hash_bench(case):
    op = NGramHashFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            "torch-ref": private_inputs(case.torch, case.inputs, 1),
        }
    )
