"""Benchmarks for softmax-family ops (softmax, log_softmax, logsumexp).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest (src/tileops/manifest/).

softmax and log_softmax are timed against flag_gems' Triton kernels as well as
torch, eager and compiled. logsumexp has no flag_gems entry point.
"""

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, workloads_to_params
from tileops.ops.reduction.softmax import LogSoftmaxFwdOp, LogSumExpFwdOp, SoftmaxFwdOp
from workloads.reduction import (
    LogSoftmaxWorkload,
    LogSumExpWorkload,
    SoftmaxWorkload,
)

# Op name constants

_SOFTMAX_OP = "SoftmaxFwdOp"
_LOG_SOFTMAX_OP = "LogSoftmaxFwdOp"
_LOGSUMEXP_OP = "LogSumExpFwdOp"


def _flaggems_softmax(name: str, dim: int):
    """Bind a flag_gems softmax entry point to *dim*."""
    fn = flaggems_op(name)

    def baseline_fn(x):
        return fn(x, dim)

    return baseline_fn


# Softmax benchmarks


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_SOFTMAX_OP))
def test_softmax_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = SoftmaxWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = SoftmaxFwdOp(dim=-1, tune=True)
    bm = ManifestBenchmark(op, test)

    def baseline_fn(x):
        return F.softmax(x, dim=-1)

    flaggems_fn = _flaggems_softmax("softmax", -1)
    assert_matches_reference(flaggems_fn, baseline_fn, *inputs, **reference_tolerance(dtype))

    try:
        bm.compare(
            {
                "tileops": op,
                FLAGGEMS_TAG: flaggems_fn,
                "torch": baseline_fn,
                TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
            },
            *inputs,
            record_as=op,
            params=locals(),
        )
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# LogSoftmax benchmarks


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_LOG_SOFTMAX_OP))
def test_log_softmax_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = LogSoftmaxWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = LogSoftmaxFwdOp(dim=-1, tune=True)
    bm = ManifestBenchmark(op, test)

    def baseline_fn(x):
        return F.log_softmax(x, dim=-1)

    flaggems_fn = _flaggems_softmax("log_softmax", -1)
    assert_matches_reference(flaggems_fn, baseline_fn, *inputs, **reference_tolerance(dtype))

    try:
        bm.compare(
            {
                "tileops": op,
                FLAGGEMS_TAG: flaggems_fn,
                "torch": baseline_fn,
                TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
            },
            *inputs,
            record_as=op,
            params=locals(),
        )
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise


# LogSumExp benchmarks


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_LOGSUMEXP_OP, include_extra=True),
)
def test_logsumexp_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = LogSumExpWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op_params.setdefault("dim", -1)
    op = LogSumExpFwdOp(tune=True, **op_params)
    bm = ManifestBenchmark(op, test)
    dim = op_params["dim"]
    keepdim = op_params.get("keepdim", False)

    def baseline_fn(x):
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)

    try:
        bm.compare(
            {
                "tileops": op,
                "torch": baseline_fn,
                TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
            },
            *inputs,
            record_as=op,
            params=locals(),
        )
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
