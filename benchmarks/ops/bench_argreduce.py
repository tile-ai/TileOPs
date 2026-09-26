"""Benchmarks for argreduce ops (argmax, argmin).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines. Each case is one
manifest call (``src/tileops/manifest/``), parameters included.

Each row is timed against flag_gems' Triton argreduce and against torch eager and
inductor.
"""

import pytest

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.reduction.argreduce import ArgmaxFwdOp, ArgminFwdOp
from workloads.reduction import ReductionCall


def _functors(op, baseline_fn, flaggems_name: str, dim: int, keepdim: bool, inputs) -> dict:
    """The op, flag_gems' argreduce, and torch eager and compiled.

    Indices are exact or wrong, so the check takes no tolerance.
    """
    assert_matches_reference(op, baseline_fn, *inputs)
    functors = {"tileops": op}
    # flag_gems' argmin launch fails with an invalid argument on a non-last axis.
    if flaggems_name == "argmax" or dim in (-1, inputs[0].ndim - 1):
        fn = flaggems_op(flaggems_name)

        def flaggems_fn(x):
            return fn(x, dim, keepdim)

        assert_matches_reference(flaggems_fn, baseline_fn, *inputs)
        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    return functors


@pytest.mark.parametrize("call", manifest_calls(ArgmaxFwdOp))
def test_argmax_bench(call) -> None:
    workload = ReductionCall(call)
    inputs = workload.gen_inputs()
    op = ArgmaxFwdOp(**workload.arguments())
    dim, keepdim = call.params["dim"], call.params["keepdim"]

    def baseline_fn(x):
        return x.argmax(dim=dim, keepdim=keepdim)

    ManifestBenchmark(op, workload).compare(
        _functors(op, baseline_fn, "argmax", dim, keepdim, inputs), *inputs
    )


@pytest.mark.parametrize("call", manifest_calls(ArgminFwdOp))
def test_argmin_bench(call) -> None:
    workload = ReductionCall(call)
    inputs = workload.gen_inputs()
    op = ArgminFwdOp(**workload.arguments())
    dim, keepdim = call.params["dim"], call.params["keepdim"]

    def baseline_fn(x):
        return x.argmin(dim=dim, keepdim=keepdim)

    ManifestBenchmark(op, workload).compare(
        _functors(op, baseline_fn, "argmin", dim, keepdim, inputs), *inputs
    )
