"""Benchmark for the gated pooling compressor op.

The row is timed against the workload's batched torch program only: no library call returns
this op's outputs and rings, and the program reads the closing tokens' count on the host, so
inductor has no single graph of it.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import private_inputs
from tileops.attention import GatedPoolCompressFwdOp


@pytest.mark.parametrize("case", bench.cases(GatedPoolCompressFwdOp))
def test_gated_pool_compress_bench(case) -> None:
    op = GatedPoolCompressFwdOp(**case.arguments)
    # The torch program writes the rings it is given, as the op does: on private copies.
    bench.Runner(op, case).compare({"torch": private_inputs(case.torch, case.inputs, 3, 4)})
