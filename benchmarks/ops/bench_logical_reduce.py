"""Benchmarks for logical reduce ops (any, all, count_nonzero).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from ops_manifest.yaml.
"""

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark, workloads_to_params
from tileops.ops.reduction.all_op import AllFwdOp
from tileops.ops.reduction.any_op import AnyFwdOp
from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp
from workloads.logical_reduce import AllTest, AnyTest, CountNonzeroTest

# ===================================================================
# Op name constants
# ===================================================================

_ANY_OP = "AnyFwdOp"
_ALL_OP = "AllFwdOp"
_COUNT_NONZERO_OP = "CountNonzeroFwdOp"


# ===================================================================
# Any benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ANY_OP))
def test_any_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = AnyTest(shape, dtype)
    bm = ManifestBenchmark(_ANY_OP, test)
    inputs = test.gen_inputs()

    op = AnyFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.bool().any(dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# All benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_ALL_OP))
def test_all_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = AllTest(shape, dtype)
    bm = ManifestBenchmark(_ALL_OP, test)
    inputs = test.gen_inputs()

    op = AllFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.bool().all(dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# CountNonzero benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_COUNT_NONZERO_OP))
def test_count_nonzero_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = CountNonzeroTest(shape, dtype)
    bm = ManifestBenchmark(_COUNT_NONZERO_OP, test)
    inputs = test.gen_inputs()

    op = CountNonzeroFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.count_nonzero(x, dim=-1).to(torch.int64)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
