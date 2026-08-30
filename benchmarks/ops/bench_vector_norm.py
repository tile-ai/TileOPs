"""Benchmarks for vector norm ops (l1_norm, l2_norm, inf_norm).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest (src/tileops/manifest/).

Each order is timed against flag_gems' Triton ``vector_norm`` and against torch
eager and inductor.
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
from tileops.ops.reduction.vector_norm import InfNormFwdOp, L1NormFwdOp, L2NormFwdOp
from workloads.reduction import InfNormWorkload, L1NormWorkload, L2NormWorkload

# Op name constants

_L1_NORM_OP = "L1NormFwdOp"
_L2_NORM_OP = "L2NormFwdOp"
_INF_NORM_OP = "InfNormFwdOp"


def _flaggems_vector_norm(ord_value, dim, keepdim: bool):
    """flag_gems' ``vector_norm``, which accumulates in fp32 as the reference does."""
    fn = flaggems_op("vector_norm")
    dims = flaggems_dims(dim)

    def baseline_fn(x):
        return fn(x, ord_value, dims, keepdim)

    return baseline_fn


def _functors(op, baseline_fn, flaggems_fn, inputs, dtype: torch.dtype) -> dict:
    assert_matches_reference(flaggems_fn, baseline_fn, *inputs, **reference_tolerance(dtype))
    return {
        "tileops": op,
        FLAGGEMS_TAG: flaggems_fn,
        "torch": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }


# L1 Norm benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_L1_NORM_OP, include_extra=True),
)
def test_l1_norm_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = L1NormWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = L1NormFwdOp(**op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return torch.linalg.vector_norm(
            x.float(),
            ord=1,
            dim=dim,
            keepdim=keepdim,
        ).to(x.dtype)

    flaggems_fn = _flaggems_vector_norm(1, dim, keepdim)

    try:
        bm.compare(_functors(op, baseline_fn, flaggems_fn, inputs, dtype), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# L2 Norm benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_L2_NORM_OP, include_extra=True),
)
def test_l2_norm_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = L2NormWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = L2NormFwdOp(**op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return torch.linalg.vector_norm(
            x.float(),
            ord=2,
            dim=dim,
            keepdim=keepdim,
        ).to(x.dtype)

    flaggems_fn = _flaggems_vector_norm(2, dim, keepdim)

    try:
        bm.compare(_functors(op, baseline_fn, flaggems_fn, inputs, dtype), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# Inf Norm benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_INF_NORM_OP, include_extra=True),
)
def test_inf_norm_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = InfNormWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = InfNormFwdOp(**op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return torch.linalg.vector_norm(
            x.float(),
            ord=float("inf"),
            dim=dim,
            keepdim=keepdim,
        ).to(x.dtype)

    flaggems_fn = _flaggems_vector_norm(float("inf"), dim, keepdim)

    try:
        bm.compare(_functors(op, baseline_fn, flaggems_fn, inputs, dtype), *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
