"""Benchmarks for softmax-family ops (softmax, log_softmax, logsumexp).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from ops_manifest.yaml.
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.manifest import eval_roofline, load_workloads
from tileops.ops.reduction.log_softmax import LogSoftmaxFwdOp
from tileops.ops.reduction.logsumexp import LogSumExpFwdOp
from tileops.ops.reduction.softmax import SoftmaxFwdOp
from workloads.ops.softmax import (
    LogSoftmaxTest,
    LogSumExpTest,
    SoftmaxTest,
)

# ===================================================================
# Benchmark classes — use manifest roofline for FLOP/memory counts
# ===================================================================

_SOFTMAX_OP = "softmax_fwd"
_LOG_SOFTMAX_OP = "log_softmax_fwd"
_LOGSUMEXP_OP = "logsumexp_fwd"


def _roofline_vars(workload) -> dict:
    """Extract roofline variables from a workload (shape + dtype → M, N, elem_bytes)."""
    elem_bytes = torch.tensor([], dtype=workload.dtype).element_size()
    N = workload.shape[-1]
    M = 1
    for s in workload.shape[:-1]:
        M *= s
    return dict(M=M, N=N, elem_bytes=elem_bytes)


class SoftmaxBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _SOFTMAX_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class LogSoftmaxBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _LOG_SOFTMAX_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


class LogSumExpBenchmark(BenchmarkBase):
    _roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = eval_roofline(
                _LOGSUMEXP_OP, **_roofline_vars(self.workload))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


# ===================================================================
# Manifest-driven parametrize helper
# ===================================================================


def _workloads_to_params(workloads):
    """Convert manifest workload dicts to pytest params: (shape, dtype)."""
    params = []
    for w in workloads:
        shape = tuple(w["x_shape"])
        label = w.get("label", "x".join(str(s) for s in shape))
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(
                shape, dtype,
                id=f"{label}-{dtype_str}",
            ))
    return params


# ===================================================================
# Softmax benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_SOFTMAX_OP)))
def test_softmax_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = SoftmaxTest(shape, dtype)
    bm = SoftmaxBenchmark(test)
    inputs = test.gen_inputs()

    op = SoftmaxFwdOp(dtype=dtype, dim=-1, tune=True)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return F.softmax(x, dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# LogSoftmax benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_LOG_SOFTMAX_OP)))
def test_log_softmax_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = LogSoftmaxTest(shape, dtype)
    bm = LogSoftmaxBenchmark(test)
    inputs = test.gen_inputs()

    op = LogSoftmaxFwdOp(dtype=dtype, dim=-1, tune=True)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return F.log_softmax(x, dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# LogSumExp benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", _workloads_to_params(load_workloads(_LOGSUMEXP_OP)))
def test_logsumexp_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = LogSumExpTest(shape, dtype)
    bm = LogSumExpBenchmark(test)
    inputs = test.gen_inputs()

    op = LogSumExpFwdOp(dtype=dtype, dim=-1, tune=True)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.logsumexp(x, dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
