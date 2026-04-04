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
        dim_normalized = t.dim % len(t.shape)
        n = t.shape[dim_normalized]
        m = 1
        for i, s in enumerate(t.shape):
            if i != dim_normalized:
                m *= s
        return 4 * m * n

    def calculate_memory(self) -> Optional[float]:
        """Read x (M*N) + write y (M*N)."""
        t = self.test
        total_elems = 1
        for s in t.shape:
            total_elems *= s
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        return (2 * total_elems) * elem_bytes


class LogSoftmaxBenchmark(BenchmarkBase):
    """Benchmark for log_softmax op (5N FLOPs: max, exp, sum, div, log)."""

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        dim_normalized = t.dim % len(t.shape)
        n = t.shape[dim_normalized]
        m = 1
        for i, s in enumerate(t.shape):
            if i != dim_normalized:
                m *= s
        return 5 * m * n

    def calculate_memory(self) -> Optional[float]:
        """Read x (M*N) + write y (M*N)."""
        t = self.test
        total_elems = 1
        for s in t.shape:
            total_elems *= s
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        return (2 * total_elems) * elem_bytes


class LogSumExpBenchmark(BenchmarkBase):
    """Benchmark for logsumexp op (3N FLOPs: max, exp, sum + 1 log+add)."""

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        dim_normalized = t.dim % len(t.shape)
        n = t.shape[dim_normalized]
        m = 1
        for i, s in enumerate(t.shape):
            if i != dim_normalized:
                m *= s
        return 3 * m * n

    def calculate_memory(self) -> Optional[float]:
        """Read x (M*N) + write y (M)."""
        t = self.test
        dim_normalized = t.dim % len(t.shape)
        n = t.shape[dim_normalized]
        m = 1
        for i, s in enumerate(t.shape):
            if i != dim_normalized:
                m *= s
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        return (m * n + m) * elem_bytes


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
    test = SoftmaxTest(shape=(m, n), dim=-1, dtype=dtype)
    bm = SoftmaxBenchmark(test)
    inputs = test.gen_inputs()

    op = SoftmaxOp(dim=-1, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return F.softmax(x, dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# LogSoftmax benchmarks
# ===================================================================


_LOG_SOFTMAX_BENCH_PARAMS = _SOFTMAX_BENCH_PARAMS


@pytest.mark.parametrize("m, n, dtype, tune", _LOG_SOFTMAX_BENCH_PARAMS)
def test_log_softmax_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LogSoftmaxTest(shape=(m, n), dim=-1, dtype=dtype)
    bm = LogSoftmaxBenchmark(test)
    inputs = test.gen_inputs()

    op = LogSoftmaxOp(dim=-1, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return F.log_softmax(x, dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# LogSumExp benchmarks
# ===================================================================


_LOGSUMEXP_BENCH_PARAMS = _SOFTMAX_BENCH_PARAMS


@pytest.mark.parametrize("m, n, dtype, tune", _LOGSUMEXP_BENCH_PARAMS)
def test_logsumexp_bench(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LogSumExpTest(shape=(m, n), dim=-1, dtype=dtype)
    bm = LogSumExpBenchmark(test)
    inputs = test.gen_inputs()

    op = LogSumExpOp(dim=-1, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.logsumexp(x, dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
