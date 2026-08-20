"""Benchmarks for logical reduce ops (any, all, count_nonzero).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest (src/tileops/manifest/).
"""

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    torch_inductor_baseline,
    workloads_to_params,
)
from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp
from workloads.reduction import AllWorkload, AnyWorkload, CountNonzeroWorkload

# Op name constants

_ANY_OP = "AnyFwdOp"
_ALL_OP = "AllFwdOp"
_COUNT_NONZERO_OP = "CountNonzeroFwdOp"


# Any benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ANY_OP, include_extra=True),
)
def test_any_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = AnyWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = AnyFwdOp(**op_params)
    bm = ManifestBenchmark(_ANY_OP, op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.bool().any(dim=dim, keepdim=keepdim)

    try:
        bm.compare(
            {"tileops": op, "torch-inductor": torch_inductor_baseline(baseline_fn)},
            *inputs,
            record_as=op,
            params=locals(),
        )
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# All benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ALL_OP, include_extra=True),
)
def test_all_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = AllWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = AllFwdOp(**op_params)
    bm = ManifestBenchmark(_ALL_OP, op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.bool().all(dim=dim, keepdim=keepdim)

    try:
        bm.compare(
            {"tileops": op, "torch-inductor": torch_inductor_baseline(baseline_fn)},
            *inputs,
            record_as=op,
            params=locals(),
        )
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# CountNonzero benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_COUNT_NONZERO_OP, include_extra=True),
)
def test_count_nonzero_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = CountNonzeroWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = CountNonzeroFwdOp(**op_params)
    bm = ManifestBenchmark(_COUNT_NONZERO_OP, op, test)
    dim = op_params["dim"]

    def baseline_fn(x):
        return torch.count_nonzero(x, dim=dim).to(torch.int64)

    try:
        bm.compare(
            {"tileops": op, "torch-inductor": torch_inductor_baseline(baseline_fn)},
            *inputs,
            record_as=op,
            params=locals(),
        )
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
