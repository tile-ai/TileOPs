"""Benchmark the masked scatter against torch, eager and compiled, and FlagGems' Triton kernel."""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import FLAGGEMS_TAG, TORCH_COMPILE_TAG, compiled_reference, flaggems_op
from tileops.indexing import MaskedScatterFwdOp


@pytest.mark.parametrize("case", bench.cases(MaskedScatterFwdOp), ids=lambda case: case.id)
def test_masked_scatter_bench(case) -> None:
    op = MaskedScatterFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            "torch": case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
            FLAGGEMS_TAG: flaggems_op("masked_scatter"),
        }
    )
