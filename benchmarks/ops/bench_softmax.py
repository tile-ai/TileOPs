"""Benchmarks for softmax-family ops (softmax, log_softmax, logsumexp).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines. Each case is one
manifest call (src/tileops/manifest/); the roofline comes from ``op.eval_roofline()``.

softmax and log_softmax are timed against flag_gems' Triton kernels as well as
torch, eager and compiled, except on a row passing ``dtype``, which flag_gems' entry points
do not take. logsumexp has no flag_gems entry point.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLAGGEMS_TAG,
    QUACK_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flaggems_op,
    quack_op,
)
from tileops.ops.reduction.softmax import LogSoftmaxFwdOp, LogSumExpFwdOp, SoftmaxFwdOp


def _bench(op_cls: type, case: bench.Case, flaggems_name: "str | None") -> None:
    baseline_fn = case.reference
    inputs = case.inputs
    op = op_cls(**case.arguments)
    op.autotune()
    implementations = {}
    if flaggems_name is not None and (not case.params.get("dtype")):
        fn = flaggems_op(flaggems_name)
        dim = case.params["dim"]

        def flaggems_fn(x):
            return fn(x, dim)

        implementations[FLAGGEMS_TAG] = flaggems_fn
    if op_cls is SoftmaxFwdOp and case.params["dim"] % inputs[0].ndim == inputs[0].ndim - 1:
        softmax = quack_op("softmax")
        dtype = _dtype(case.params)

        def quack_fn(x):
            values = x.to(dtype) if dtype is not None else x
            return softmax(values.reshape(-1, values.shape[-1])).reshape_as(values)

        implementations[QUACK_TAG] = quack_fn
    implementations["torch"] = baseline_fn
    implementations[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    bench.Runner(op, case).compare(implementations)


def _dtype(params: dict) -> "torch.dtype | None":
    return getattr(torch, params["dtype"]) if params.get("dtype") else None


@pytest.mark.parametrize("case", bench.cases(SoftmaxFwdOp), ids=lambda case: case.id)
def test_softmax_bench(case) -> None:
    _bench(SoftmaxFwdOp, case, "softmax")


@pytest.mark.parametrize("case", bench.cases(LogSoftmaxFwdOp), ids=lambda case: case.id)
def test_log_softmax_bench(case) -> None:
    _bench(LogSoftmaxFwdOp, case, "log_softmax")


@pytest.mark.parametrize("case", bench.cases(LogSumExpFwdOp), ids=lambda case: case.id)
def test_logsumexp_bench(case) -> None:
    _bench(LogSumExpFwdOp, case, None)
