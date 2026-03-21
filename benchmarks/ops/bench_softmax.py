"""Benchmarks for softmax-family ops (softmax, log_softmax, logsumexp).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_softmax import (
    LogSoftmaxTest,
    LogSumExpTest,
    SoftmaxTest,
)
from tileops.ops.reduction.log_softmax import LogSoftmaxOp
from tileops.ops.reduction.logsumexp import LogSumExpOp
from tileops.ops.reduction.softmax import SoftmaxOp


class SoftmaxBenchmark(BenchmarkBase):
    """Benchmark for softmax op (4N FLOPs: max, exp, sum, div)."""

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return 4 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Read x (M*N) + write y (M*N)."""
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        return (2 * t.m * t.n) * elem_bytes


class LogSoftmaxBenchmark(BenchmarkBase):
    """Benchmark for log_softmax op (5N FLOPs: max, exp, sum, div, log)."""

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return 5 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Read x (M*N) + write y (M*N)."""
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        return (2 * t.m * t.n) * elem_bytes


class LogSumExpBenchmark(BenchmarkBase):
    """Benchmark for logsumexp op (3N FLOPs: max, exp, sum + 1 log+add)."""

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return 3 * t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        """Read x (M*N) + write y (M)."""
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        return (t.m * t.n + t.m) * elem_bytes


# ===================================================================
# Softmax benchmarks
# ===================================================================


_SOFTMAX_BENCH_PARAMS = [
    pytest.param(1024, 4096, torch.float16, True, id="mainstream-fp16"),
    pytest.param(4096, 4096, torch.bfloat16, True, id="throughput-bf16"),
    pytest.param(1024, 3000, torch.float16, True, id="non-power-of-two"),
    pytest.param(1025, 4096, torch.float16, True, id="tail-m"),
]


@pytest.mark.parametrize("m, n, dtype, tune", _SOFTMAX_BENCH_PARAMS)
def test_softmax_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = SoftmaxTest(m, n, dtype)
    bm = SoftmaxBenchmark(test)
    inputs = test.gen_inputs()

    op = SoftmaxOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return F.softmax(x, dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="baseline")


# ===================================================================
# LogSoftmax benchmarks
# ===================================================================


_LOG_SOFTMAX_BENCH_PARAMS = _SOFTMAX_BENCH_PARAMS


@pytest.mark.parametrize("m, n, dtype, tune", _LOG_SOFTMAX_BENCH_PARAMS)
def test_log_softmax_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LogSoftmaxTest(m, n, dtype)
    bm = LogSoftmaxBenchmark(test)
    inputs = test.gen_inputs()

    op = LogSoftmaxOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return F.log_softmax(x, dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="baseline")


# ===================================================================
# LogSumExp benchmarks
# ===================================================================


_LOGSUMEXP_BENCH_PARAMS = _SOFTMAX_BENCH_PARAMS


@pytest.mark.parametrize("m, n, dtype, tune", _LOGSUMEXP_BENCH_PARAMS)
def test_logsumexp_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LogSumExpTest(m, n, dtype)
    bm = LogSumExpBenchmark(test)
    inputs = test.gen_inputs()

    op = LogSumExpOp(M=m, N=n, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.logsumexp(x, dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
