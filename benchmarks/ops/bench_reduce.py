"""Benchmarks for the 8 basic reduce ops.

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest (src/tileops/manifest/).

Every row is timed against torch eager, the same reference through inductor, and
flag_gems' Triton reduction where one exists. amin has none: flag_gems 5.0.2
exposes amax but no amin, and ``min_dim`` also produces indices.

The flag_gems callables cast nothing — like aten, its reductions accumulate in fp32
and return the storage dtype — so each tag is one kernel.
"""

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_dims,
    flaggems_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, workloads_to_params
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
from workloads.reduction import (
    AmaxWorkload,
    AminWorkload,
    MeanWorkload,
    ProdWorkload,
    StdWorkload,
    SumWorkload,
    VarMeanWorkload,
    VarWorkload,
)

# Op name constants

_SUM_OP = "SumFwdOp"
_MEAN_OP = "MeanFwdOp"
_AMAX_OP = "AmaxFwdOp"
_AMIN_OP = "AminFwdOp"
_PROD_OP = "ProdFwdOp"
_STD_OP = "StdFwdOp"
_VAR_OP = "VarFwdOp"
_VAR_MEAN_OP = "VarMeanFwdOp"


def _functors(op, baseline_fn, inputs, dtype: torch.dtype, flaggems_fn=None) -> dict:
    """The op, flag_gems where it has a kernel, and torch eager and compiled."""
    functors = {"tileops": op}
    if flaggems_fn is not None:
        assert_matches_reference(flaggems_fn, baseline_fn, *inputs, **reference_tolerance(dtype))
        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    return functors


# Sum benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_SUM_OP, include_extra=True),
)
def test_sum_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = SumWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)  # baseline below reduces dim=-1
    op = SumFwdOp(**op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params.get("dim", -1)
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.float().sum(dim=dim, keepdim=keepdim).to(x.dtype)

    flaggems_sum = flaggems_op("sum_dim")

    def flaggems_fn(x):
        return flaggems_sum(x, flaggems_dims(dim), keepdim)

    try:
        bm.compare(_functors(op, baseline_fn, inputs, dtype, flaggems_fn), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# Mean benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_MEAN_OP, include_extra=True),
)
def test_mean_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = MeanWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)  # baseline below mirrors the op's dim
    op = MeanFwdOp(**op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.float().mean(dim=dim, keepdim=keepdim).to(x.dtype)

    flaggems_mean = flaggems_op("mean_dim")

    def flaggems_fn(x):
        return flaggems_mean(x, flaggems_dims(dim), keepdim)

    try:
        bm.compare(_functors(op, baseline_fn, inputs, dtype, flaggems_fn), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# Amax benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_AMAX_OP, include_extra=True),
)
def test_amax_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = AmaxWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = AmaxFwdOp(**op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.amax(dim=dim, keepdim=keepdim)

    flaggems_amax = flaggems_op("amax")

    def flaggems_fn(x):
        return flaggems_amax(x, flaggems_dims(dim), keepdim)

    try:
        bm.compare(_functors(op, baseline_fn, inputs, dtype, flaggems_fn), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# Amin benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_AMIN_OP, include_extra=True),
)
def test_amin_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = AminWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = AminFwdOp(**op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.amin(dim=dim, keepdim=keepdim)

    try:
        bm.compare(_functors(op, baseline_fn, inputs, dtype), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# Prod benchmarks


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_PROD_OP))
def test_prod_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = ProdWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = ProdFwdOp()
    bm = ManifestBenchmark(op, test)

    def baseline_fn(x):
        return x.float().prod(dim=-1).to(x.dtype)

    flaggems_prod = flaggems_op("prod_dim")

    def flaggems_fn(x):
        return flaggems_prod(x, -1)

    try:
        bm.compare(_functors(op, baseline_fn, inputs, dtype, flaggems_fn), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# Std benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_STD_OP, include_extra=True),
)
def test_std_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = StdWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = StdFwdOp(correction=1, **op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.float().std(dim=dim, keepdim=keepdim, correction=1).to(x.dtype)

    flaggems_std = flaggems_op("std")

    def flaggems_fn(x):
        return flaggems_std(x, flaggems_dims(dim), correction=1, keepdim=keepdim)

    try:
        bm.compare(_functors(op, baseline_fn, inputs, dtype, flaggems_fn), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# Var benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_VAR_OP, include_extra=True),
)
def test_var_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = VarWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = VarFwdOp(correction=1, **op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.float().var(dim=dim, keepdim=keepdim, correction=1).to(x.dtype)

    flaggems_var = flaggems_op("var_dim")

    def flaggems_fn(x):
        return flaggems_var(x, flaggems_dims(dim), correction=1, keepdim=keepdim)

    try:
        bm.compare(_functors(op, baseline_fn, inputs, dtype, flaggems_fn), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# VarMean benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_VAR_MEAN_OP, include_extra=True),
)
def test_var_mean_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = VarMeanWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = VarMeanFwdOp(correction=1, **op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        v = x.float().var(dim=dim, keepdim=keepdim, correction=1).to(x.dtype)
        m = x.float().mean(dim=dim, keepdim=keepdim).to(x.dtype)
        return (v, m)

    flaggems_var_mean = flaggems_op("var_mean")

    def flaggems_fn(x):
        return flaggems_var_mean(x, flaggems_dims(dim), correction=1, keepdim=keepdim)

    try:
        bm.compare(_functors(op, baseline_fn, inputs, dtype, flaggems_fn), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
