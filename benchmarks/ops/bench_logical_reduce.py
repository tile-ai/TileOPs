"""Benchmarks for logical reduce ops (any, all, count_nonzero).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest (src/tileops/manifest/).

any and all are timed against flag_gems' Triton reductions as well as torch, eager
and compiled. count_nonzero has none: its entry point raises on a list of dims.
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
)
from benchmarks.benchmark_base import ManifestBenchmark, workloads_to_params
from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp
from workloads.reduction import AllWorkload, AnyWorkload, CountNonzeroWorkload

# Op name constants

_ANY_OP = "AnyFwdOp"
_ALL_OP = "AllFwdOp"
_COUNT_NONZERO_OP = "CountNonzeroFwdOp"


def _functors(op, baseline_fn, inputs, flaggems_name=None, dim=None, keepdim=False) -> dict:
    """The op, flag_gems where it has a kernel, and torch eager and compiled.

    A boolean reduction is exact or wrong, so the check takes no tolerance.
    """
    functors = {"tileops": op}
    if flaggems_name is not None:
        fn = flaggems_op(flaggems_name)
        dims = flaggems_dims(dim)

        def flaggems_fn(x):
            return fn(x.bool(), dims, keepdim)

        assert_matches_reference(flaggems_fn, baseline_fn, *inputs)
        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    return functors


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
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.bool().any(dim=dim, keepdim=keepdim)

    try:
        bm.compare(
            _functors(op, baseline_fn, inputs, "any_dims", dim, keepdim),
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
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return x.bool().all(dim=dim, keepdim=keepdim)

    try:
        bm.compare(
            _functors(op, baseline_fn, inputs, "all_dims", dim, keepdim),
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
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]

    def baseline_fn(x):
        return torch.count_nonzero(x, dim=dim).to(torch.int64)

    try:
        bm.compare(
            _functors(op, baseline_fn, inputs),
            *inputs,
            record_as=op,
            params=locals(),
        )
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
