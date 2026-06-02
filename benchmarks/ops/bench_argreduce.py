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


def _is_unsupported_large_argreduce_error(exc: Exception) -> bool:
    """Return True for known staged-rollout large-N argreduce failures."""
    msg = str(exc)
    return (
        "scalable vector" in msg
        or "No configurations to tune" in msg
        or (
            "A single row requires" in msg
            and "shared memory" in msg
            and "exceeds" in msg
        )
    )


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
    # Why: the current single-tile shared-memory kernel cannot fit lm-head
    #      N=102400 rows (204800 bytes per fp16/bf16 row exceeds 49152 bytes).
    # Cleanup: remove try/skip once ArgreduceKernel has a tiled-N path.
    try:
        result = bm.profile(op, *inputs)
    except Exception as exc:
        if _is_unsupported_large_argreduce_error(exc):
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
    # Why: the current single-tile shared-memory kernel cannot fit lm-head
    #      N=102400 rows (204800 bytes per fp16/bf16 row exceeds 49152 bytes).
    # Cleanup: remove try/skip once ArgreduceKernel has a tiled-N path.
    try:
        result = bm.profile(op, *inputs)
    except Exception as exc:
        if _is_unsupported_large_argreduce_error(exc):
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
