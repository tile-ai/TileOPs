"""N-gram hashes and context commits against torch eager and torch.compile."""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference, private_inputs
from tileops.sequence_modeling import NGramHashFwdOp


@pytest.mark.parametrize("case", bench.cases(NGramHashFwdOp))
def test_ngram_hash_bench(case):
    op = NGramHashFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            "torch": private_inputs(case.torch, case.inputs, 1),
            TORCH_COMPILE_TAG: private_inputs(compiled_reference(case.torch), case.inputs, 1),
        }
    )
