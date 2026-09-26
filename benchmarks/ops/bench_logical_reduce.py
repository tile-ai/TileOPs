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
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp
from workloads.reduction import LogicalCall


def _bench(op_cls: type, call, baseline_fn, flaggems_name=None) -> None:
    """Check the op and flag_gems where it has a kernel against torch, then time them.

    A boolean reduction is exact or wrong, so the check takes no tolerance.
    """
    workload = LogicalCall(call)
    inputs = workload.gen_inputs()
    op = op_cls(**workload.arguments())
    assert_matches_reference(op, baseline_fn, *inputs)
    functors = {"tileops": op}
    if flaggems_name is not None:
        fn = flaggems_op(flaggems_name)
        dims = flaggems_dims(call.params["dim"])
        keepdim = call.params["keepdim"]

        def flaggems_fn(x):
            return fn(x.bool(), dims, keepdim)

        assert_matches_reference(flaggems_fn, baseline_fn, *inputs)
        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    ManifestBenchmark(op, workload).compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(AnyFwdOp))
def test_any_bench(call) -> None:
    dim, keepdim = call.params["dim"], call.params["keepdim"]

    def baseline_fn(x):
        return x.bool().any(dim=dim, keepdim=keepdim)

    _bench(AnyFwdOp, call, baseline_fn, "any_dims")


@pytest.mark.parametrize("call", manifest_calls(AllFwdOp))
def test_all_bench(call) -> None:
    dim, keepdim = call.params["dim"], call.params["keepdim"]

    def baseline_fn(x):
        return x.bool().all(dim=dim, keepdim=keepdim)

    _bench(AllFwdOp, call, baseline_fn, "all_dims")


@pytest.mark.parametrize("call", manifest_calls(CountNonzeroFwdOp))
def test_count_nonzero_bench(call) -> None:
    dim = call.params["dim"]

    def baseline_fn(x):
        return torch.count_nonzero(x, dim=dim).to(torch.int64)

    _bench(CountNonzeroFwdOp, call, baseline_fn)
