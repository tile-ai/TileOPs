"""Benchmarks for the 8 basic reduce ops.

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from ops_manifest.yaml.
"""

import pytest
import torch

from benchmarks.benchmark import BenchmarkReport, ManifestBenchmark, workloads_to_params
from tileops.ops.reduction.reduce import (
    AmaxFwdOp,
    AminFwdOp,
    MeanFwdOp,
    ProdFwdOp,
    StdFwdOp,
    SumFwdOp,
    VarFwdOp,
    VarMeanFwdOp,
)
from workloads.ops.reduce import (
    AmaxTest,
    AminTest,
    MeanTest,
    ProdTest,
    StdTest,
    SumTest,
    VarMeanTest,
    VarTest,
)

# ===================================================================
# Op name constants
# ===================================================================

_SUM_OP = "SumFwdOp"
_MEAN_OP = "MeanFwdOp"
_AMAX_OP = "AmaxFwdOp"
_AMIN_OP = "AminFwdOp"
_PROD_OP = "ProdFwdOp"
_STD_OP = "StdFwdOp"
_VAR_OP = "VarFwdOp"
_VAR_MEAN_OP = "VarMeanFwdOp"


# ===================================================================
# Sum benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_SUM_OP))
def test_sum_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = SumTest(shape, dtype)
    bm = ManifestBenchmark(_SUM_OP, test)
    inputs = test.gen_inputs()

    op = SumFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.float().sum(dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Mean benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_MEAN_OP))
def test_mean_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = MeanTest(shape, dtype)
    bm = ManifestBenchmark(_MEAN_OP, test)
    inputs = test.gen_inputs()

    op = MeanFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.float().mean(dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Amax benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_AMAX_OP))
def test_amax_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = AmaxTest(shape, dtype)
    bm = ManifestBenchmark(_AMAX_OP, test)
    inputs = test.gen_inputs()

    op = AmaxFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.amax(dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Amin benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_AMIN_OP))
def test_amin_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = AminTest(shape, dtype)
    bm = ManifestBenchmark(_AMIN_OP, test)
    inputs = test.gen_inputs()

    op = AminFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.amin(dim=-1)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Prod benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_PROD_OP))
def test_prod_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = ProdTest(shape, dtype)
    bm = ManifestBenchmark(_PROD_OP, test)
    inputs = test.gen_inputs()

    op = ProdFwdOp(dtype=dtype)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.float().prod(dim=-1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Std benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_STD_OP))
def test_std_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = StdTest(shape, dtype)
    bm = ManifestBenchmark(_STD_OP, test)
    inputs = test.gen_inputs()

    op = StdFwdOp(dtype=dtype, correction=1)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.float().std(dim=-1, correction=1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# Var benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_VAR_OP))
def test_var_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = VarTest(shape, dtype)
    bm = ManifestBenchmark(_VAR_OP, test)
    inputs = test.gen_inputs()

    op = VarFwdOp(dtype=dtype, correction=1)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return x.float().var(dim=-1, correction=1).to(x.dtype)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# VarMean benchmarks
# ===================================================================


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_VAR_MEAN_OP))
def test_var_mean_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = VarMeanTest(shape, dtype)
    bm = ManifestBenchmark(_VAR_MEAN_OP, test)
    inputs = test.gen_inputs()

    op = VarMeanFwdOp(dtype=dtype, correction=1)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        v = x.float().var(dim=-1, correction=1).to(x.dtype)
        m = x.float().mean(dim=-1).to(x.dtype)
        return (v, m)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
