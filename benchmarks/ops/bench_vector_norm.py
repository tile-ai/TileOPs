"""Benchmarks for vector norm ops (l1_norm, l2_norm, inf_norm).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest (src/tileops/manifest/).
"""

import pytest
import torch

from benchmarks.benchmark_base import ManifestBenchmark, workloads_to_params
from tileops.ops.reduction.vector_norm import InfNormFwdOp, L1NormFwdOp, L2NormFwdOp
from workloads.reduction import InfNormWorkload, L1NormWorkload, L2NormWorkload

# Op name constants

_L1_NORM_OP = "L1NormFwdOp"
_L2_NORM_OP = "L2NormFwdOp"
_INF_NORM_OP = "InfNormFwdOp"


def _torch_vector_norm(x, *, ord, dim, keepdim):
    return torch.linalg.vector_norm(x, ord=ord, dim=dim, keepdim=keepdim)


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
    bm = ManifestBenchmark(_L1_NORM_OP, op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return _torch_vector_norm(x, ord=1, dim=dim, keepdim=keepdim)

    try:
        bm.compare({"tileops": op, "torch": baseline_fn}, *inputs, record_as=op, params=locals())
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
    bm = ManifestBenchmark(_L2_NORM_OP, op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return _torch_vector_norm(x, ord=2, dim=dim, keepdim=keepdim)

    try:
        bm.compare({"tileops": op, "torch": baseline_fn}, *inputs, record_as=op, params=locals())
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
    bm = ManifestBenchmark(_INF_NORM_OP, op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return _torch_vector_norm(x, ord=float("inf"), dim=dim, keepdim=keepdim)

    try:
        bm.compare({"tileops": op, "torch": baseline_fn}, *inputs, record_as=op, params=locals())
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
