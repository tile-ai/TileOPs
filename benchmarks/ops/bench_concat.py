"""Benchmark the TileOPs concatenation against ``torch.cat``, one case per manifest call."""

import pytest

from benchmarks import api as bench
from tileops.indexing import ConcatFwdOp


@pytest.mark.parametrize("case", bench.cases(ConcatFwdOp), ids=lambda case: case.id)
def test_concat_bench(case) -> None:
    op = ConcatFwdOp(**case.arguments)
    bench.Runner(op, case).compare({"torch": case.torch})
