"""Benchmark EmbeddingFwdOp against torch.nn.functional.embedding."""

import pytest

from benchmarks import api as bench
from tileops.indexing import EmbeddingFwdOp


@pytest.mark.parametrize("case", bench.cases(EmbeddingFwdOp), ids=lambda case: case.id)
def test_embedding_bench(case) -> None:
    bench.Runner(EmbeddingFwdOp(**case.arguments), case).compare({"torch": case.torch})
