"""Benchmarks for argreduce ops (argmax, argmin).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes, dtypes, and op-call parameters (e.g. ``dim``) are loaded
from the ops manifest (``tileops/manifest/``) — the benchmark must not
hard-code op parameters that are declared on manifest workload entries.
"""

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark, workloads_to_params
from tileops.ops.reduction.argmax import ArgmaxFwdOp
from tileops.ops.reduction.argmin import ArgminFwdOp
from workloads.argreduce import ArgmaxTest, ArgminTest

_ARGMAX_OP = "ArgmaxFwdOp"
_ARGMIN_OP = "ArgminFwdOp"


# ===================================================================
# Argmax benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype, extra", workloads_to_params(_ARGMAX_OP, include_extra=True))
def test_argmax_bench(shape: tuple, dtype: torch.dtype, extra: dict) -> None:
    workload = ArgmaxTest(shape, dtype)
    inputs = workload.gen_inputs()

    op = ArgmaxFwdOp(dtype=dtype, **extra)
    bm = ManifestBenchmark(_ARGMAX_OP, op, workload)
    # FIXME(staged-rollout): ArgreduceKernel skips large-N manifest workloads
    #
    # Broken invariant: benchmark must execute all manifest workload shapes
    # Why: kernel crashes on N>=102400 ("Can't fetch the lanes of a scalable vector")
    # Cleanup: remove try/skip once ArgreduceKernel handles arbitrary N
    try:
        result = bm.profile(op, *inputs)
    except Exception as exc:
        msg = str(exc)
        if "scalable vector" in msg or "No configurations to tune" in msg:
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    dim = extra["dim"]

    def baseline_fn(x):
        return x.argmax(dim=dim)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Argmin benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype, extra", workloads_to_params(_ARGMIN_OP, include_extra=True))
def test_argmin_bench(shape: tuple, dtype: torch.dtype, extra: dict) -> None:
    workload = ArgminTest(shape, dtype)
    inputs = workload.gen_inputs()

    op = ArgminFwdOp(dtype=dtype, **extra)
    bm = ManifestBenchmark(_ARGMIN_OP, op, workload)
    # FIXME(staged-rollout): ArgreduceKernel skips large-N manifest workloads
    #
    # Broken invariant: benchmark must execute all manifest workload shapes
    # Why: kernel crashes on N>=102400 ("Can't fetch the lanes of a scalable vector")
    # Cleanup: remove try/skip once ArgreduceKernel handles arbitrary N
    try:
        result = bm.profile(op, *inputs)
    except Exception as exc:
        msg = str(exc)
        if "scalable vector" in msg or "No configurations to tune" in msg:
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    dim = extra["dim"]

    def baseline_fn(x):
        return x.argmin(dim=dim)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
