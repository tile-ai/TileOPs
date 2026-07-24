from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport
from benchmarks.ops.gemm_w4a16_benchmark_utils import (
    FEASIBILITY_CASES,
    StaticGemmW4A16Benchmark,
    W4A16BenchmarkCase,
    dense_a16_memory_bytes,
    w4a16_memory_bytes,
)
from tests.ops.test_gemm import GemmW4A16Test
from tileops.ops import GemmW4A16Op


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.label) for case in FEASIBILITY_CASES],
)
def test_gemm_w4a16_feasibility_bench(case: W4A16BenchmarkCase) -> None:
    """Initial W4A16 feasibility suite.

    These four shapes answer two early questions only:
    1. Can W4A16 run correctly through Tensor Core GEMM?
    2. Does W4 show more value on prefill or decode?

    The result of the experiment was that M=128 prefill was not yet favorable,
    while M=1 decode was worth optimizing further. The later AKO decode suite is
    therefore separate from this feasibility evidence set.
    """
    dtype = torch.float16
    m, n, k = case.m, case.n, case.k
    scenario = case.scenario
    purpose = case.purpose

    test = GemmW4A16Test(m, n, k, dtype)
    inputs = test.gen_inputs()
    op = GemmW4A16Op()

    w4_bm = StaticGemmW4A16Benchmark(test, w4a16_memory_bytes(m, n, k, dtype=dtype))
    result = w4_bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops-w4a16")

    dense_bm = StaticGemmW4A16Benchmark(test, dense_a16_memory_bytes(m, n, k, dtype=dtype))
    result_dense = dense_bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_dense, tag="torch-dequantized-a16")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
