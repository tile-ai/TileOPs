"""Benchmarks for softmax-family ops (softmax, log_softmax, logsumexp).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest (src/tileops/manifest/).
"""

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark, workloads_to_params
from tileops.ops.reduction.softmax import LogSoftmaxFwdOp, LogSumExpFwdOp, SoftmaxFwdOp
from workloads.reduction import (
    LogSoftmaxWorkload,
    LogSumExpWorkload,
    SoftmaxWorkload,
)

# Op name constants

_SOFTMAX_OP = "SoftmaxFwdOp"
_LOG_SOFTMAX_OP = "LogSoftmaxFwdOp"
_LOGSUMEXP_OP = "LogSumExpFwdOp"


# Softmax benchmarks


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_SOFTMAX_OP))
def test_softmax_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = SoftmaxWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = SoftmaxFwdOp(dim=-1, tune=True)
    bm = ManifestBenchmark(_SOFTMAX_OP, op, test)
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


# LogSoftmax benchmarks


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_LOG_SOFTMAX_OP))
def test_log_softmax_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = LogSoftmaxWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = LogSoftmaxFwdOp(dim=-1, tune=True)
    bm = ManifestBenchmark(_LOG_SOFTMAX_OP, op, test)
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


# LogSumExp benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_LOGSUMEXP_OP, include_extra=True),
)
def test_logsumexp_bench(
    shape: tuple, dtype: torch.dtype, op_params: dict
) -> None:
    test = LogSumExpWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = LogSumExpFwdOp(tune=True, **op_params)
    bm = ManifestBenchmark(_LOGSUMEXP_OP, op, test)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
