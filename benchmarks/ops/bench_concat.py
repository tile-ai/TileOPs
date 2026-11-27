"""Benchmark the TileOPs concatenation against ``torch.cat``, eager and compiled."""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from tileops.indexing import ConcatFwdOp


@pytest.mark.parametrize("case", bench.cases(ConcatFwdOp))
def test_concat_bench(case) -> None:
    op = ConcatFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {"torch": case.torch, TORCH_COMPILE_TAG: compiled_reference(case.torch)}
    )
