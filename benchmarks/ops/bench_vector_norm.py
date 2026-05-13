"""Benchmarks for vector norm ops (l1_norm, l2_norm, inf_norm).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest (tileops/manifest/).
"""

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark, workloads_to_params
from tileops.ops.reduction.inf_norm import InfNormFwdOp
from tileops.ops.reduction.l1_norm import L1NormFwdOp
from tileops.ops.reduction.l2_norm import L2NormFwdOp
from workloads.vector_norm import InfNormTest, L1NormTest, L2NormTest

# ===================================================================
# Op name constants
# ===================================================================

_L1_NORM_OP = "L1NormFwdOp"
_L2_NORM_OP = "L2NormFwdOp"
_INF_NORM_OP = "InfNormFwdOp"


# ===================================================================
# L1 Norm benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_L1_NORM_OP))
def test_l1_norm_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = L1NormTest(shape, dtype)
    inputs = test.gen_inputs()

    op = L1NormFwdOp(dtype=dtype, dim=-1)
    bm = ManifestBenchmark(_L1_NORM_OP, op, test)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.linalg.vector_norm(x.float(), ord=1, dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# L2 Norm benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_L2_NORM_OP))
def test_l2_norm_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = L2NormTest(shape, dtype)
    inputs = test.gen_inputs()

    op = L2NormFwdOp(dtype=dtype, dim=-1)
    bm = ManifestBenchmark(_L2_NORM_OP, op, test)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Inf Norm benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_INF_NORM_OP))
def test_inf_norm_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = InfNormTest(shape, dtype)
    inputs = test.gen_inputs()

    op = InfNormFwdOp(dtype=dtype, dim=-1)
    bm = ManifestBenchmark(_INF_NORM_OP, op, test)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.linalg.vector_norm(x.float(), ord=float("inf"), dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
