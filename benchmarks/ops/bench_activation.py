"""Benchmarks for unary activation ops (relu) across strategies.

Profiles all 3 strategies (direct, explicit_parallel, register_copy) and
compares against PyTorch baseline to determine optimal DEFAULT_STRATEGY.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_activation import ReluTest
from tileops.ops.elementwise import ReluOp


class ReluBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # ReLU: 1 comparison per element
        return self.test.n_total

    def calculate_memory(self) -> Optional[float]:
        """Read x + write y."""
        t = self.test
        elem_bytes = t.dtype.itemsize
        return 2 * t.n_total * elem_bytes


_RELU_BENCH_PARAMS = [
    pytest.param(4_000_000, torch.float16, id="throughput-fp16"),
    pytest.param(4_000_000, torch.bfloat16, id="throughput-bf16"),
    pytest.param(1_000_000, torch.float32, id="baseline-fp32"),
]


@pytest.mark.parametrize("n_total, dtype", _RELU_BENCH_PARAMS)
def test_relu_bench(n_total: int, dtype: torch.dtype) -> None:
    test = ReluTest(n_total, dtype)
    bm = ReluBenchmark(test)
    inputs = test.gen_inputs()

    op = ReluOp(N_total=n_total, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("relu", locals(), result, tag="tileops")

    # Baseline: PyTorch relu
    def baseline_fn(x):
        return torch.relu(x)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record("relu", locals(), result_bl, tag="baseline")


_RELU_STRATEGY_BENCH_PARAMS = [
    pytest.param(4_000_000, torch.float16, "direct", id="direct"),
    pytest.param(4_000_000, torch.float16, "explicit_parallel", id="explicit-parallel"),
    pytest.param(4_000_000, torch.float16, "register_copy", id="register-copy"),
]


@pytest.mark.parametrize("n_total, dtype, strategy", _RELU_STRATEGY_BENCH_PARAMS)
def test_relu_strategy_bench(n_total: int, dtype: torch.dtype, strategy: str) -> None:
    """Benchmark each unary strategy to determine optimal default."""
    test = ReluTest(n_total, dtype)
    bm = ReluBenchmark(test)
    inputs = test.gen_inputs()

    op = ReluOp(N_total=n_total, dtype=dtype, strategy=strategy)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("relu_strategy", locals(), result, tag=f"tileops_{strategy}")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
