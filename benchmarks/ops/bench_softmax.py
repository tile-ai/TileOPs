"""Benchmarks for softmax-family ops (softmax, log_softmax, logsumexp).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines. Each case is one
manifest call (src/tileops/manifest/); the roofline comes from ``op.eval_roofline()``.

softmax and log_softmax are timed against flag_gems' Triton kernels as well as
torch, eager and compiled, except on a row passing ``dtype``, which flag_gems' entry points
do not take. logsumexp has no flag_gems entry point.
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
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.reduction.softmax import LogSoftmaxFwdOp, LogSumExpFwdOp, SoftmaxFwdOp
from workloads.reduction import ReductionCall


def _bench(op_cls: type, call, baseline_fn, flaggems_name: "str | None") -> None:
    workload = ReductionCall(call)
    inputs = workload.gen_inputs()
    op = op_cls(**workload.arguments(), tune=True)
    tolerance = reference_tolerance(inputs[0].dtype)
    functors = {"tileops": op}
    if flaggems_name is not None and not call.params.get("dtype"):
        fn = flaggems_op(flaggems_name)
        dim = call.params["dim"]

        def flaggems_fn(x):
            return fn(x, dim)

        assert_matches_reference(flaggems_fn, baseline_fn, *inputs, **tolerance)
        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    ManifestBenchmark(op, workload).compare(functors, *inputs)


def _dtype(params: dict) -> "torch.dtype | None":
    return getattr(torch, params["dtype"]) if params.get("dtype") else None


@pytest.mark.parametrize("call", manifest_calls(SoftmaxFwdOp))
def test_softmax_bench(call) -> None:
    dim, dtype = call.params["dim"], _dtype(call.params)

    def baseline_fn(x):
        return F.softmax(x, dim=dim, dtype=dtype)

    _bench(SoftmaxFwdOp, call, baseline_fn, "softmax")


@pytest.mark.parametrize("call", manifest_calls(LogSoftmaxFwdOp))
def test_log_softmax_bench(call) -> None:
    dim, dtype = call.params["dim"], _dtype(call.params)

    def baseline_fn(x):
        return F.log_softmax(x, dim=dim, dtype=dtype)

    _bench(LogSoftmaxFwdOp, call, baseline_fn, "log_softmax")


@pytest.mark.parametrize("call", manifest_calls(LogSumExpFwdOp))
def test_logsumexp_bench(call) -> None:
    dim, keepdim = call.params["dim"], call.params["keepdim"]

    def baseline_fn(x):
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)

    _bench(LogSumExpFwdOp, call, baseline_fn, None)
