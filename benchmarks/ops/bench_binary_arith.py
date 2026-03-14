"""Benchmarks for binary arithmetic ops (add) across strategies.

Profiles both binary strategies (direct, explicit_parallel) and
compares against PyTorch baseline.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_binary_arith import AddSameShapeTest
from tileops.ops.elementwise import AddOp


class AddBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # Add: 1 flop per element
        return self.test.n_total

    def calculate_memory(self) -> Optional[float]:
        """Read a + read b + write y."""
        t = self.test
        elem_bytes = t.dtype.itemsize
        return 3 * t.n_total * elem_bytes


_ADD_BENCH_PARAMS = [
    pytest.param(4_000_000, torch.float16, id="throughput-fp16"),
    pytest.param(4_000_000, torch.bfloat16, id="throughput-bf16"),
    pytest.param(1_000_000, torch.float32, id="baseline-fp32"),
]


@pytest.mark.parametrize("n_total, dtype", _ADD_BENCH_PARAMS)
def test_add_bench(n_total: int, dtype: torch.dtype) -> None:
    test = AddSameShapeTest(n_total, dtype)
    bm = AddBenchmark(test)
    inputs = test.gen_inputs()

    shape = (n_total,)
    op = AddOp(a_shape=shape, b_shape=shape, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("add", locals(), result, tag="tileops")

    # Baseline: PyTorch add
    def baseline_fn(a, b):
        return a + b

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("add", locals(), result_bl, tag="baseline")


_ADD_STRATEGY_BENCH_PARAMS = [
    pytest.param(4_000_000, torch.float16, "direct", id="direct"),
    pytest.param(4_000_000, torch.float16, "explicit_parallel", id="explicit-parallel"),
]


@pytest.mark.parametrize("n_total, dtype, strategy", _ADD_STRATEGY_BENCH_PARAMS)
def test_add_strategy_bench(n_total: int, dtype: torch.dtype, strategy: str) -> None:
    """Benchmark each binary strategy to determine optimal default."""
    test = AddSameShapeTest(n_total, dtype)
    bm = AddBenchmark(test)
    inputs = test.gen_inputs()

    shape = (n_total,)
    op = AddOp(a_shape=shape, b_shape=shape, dtype=dtype, strategy=strategy)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("add_strategy", locals(), result, tag=f"tileops_{strategy}")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
