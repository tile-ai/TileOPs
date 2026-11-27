"""Benchmark the TileOPs embedding lookup against ``torch.nn.functional.embedding``."""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from tileops.indexing import EmbeddingFwdOp


@pytest.mark.parametrize("case", bench.cases(EmbeddingFwdOp))
def test_embedding_bench(case) -> None:
    op = EmbeddingFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {"torch": case.torch, TORCH_COMPILE_TAG: compiled_reference(case.torch)}
    )
