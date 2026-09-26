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
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.reduction.vector_norm import InfNormFwdOp, L1NormFwdOp, L2NormFwdOp
from workloads.reduction import ReductionCall


def _bench(op_cls: type, call) -> None:
    """Check flag_gems' ``vector_norm`` against torch, then time it and the op.

    flag_gems accumulates in fp32 as the reference does; it takes no output dtype, so a row
    passing ``dtype`` has no flag_gems tag.
    """
    p = call.params
    dtype = getattr(torch, p["dtype"]) if p.get("dtype") else None

    def baseline_fn(x):
        return torch.linalg.vector_norm(
            x.float(), ord=p["ord"], dim=p["dim"], keepdim=p["keepdim"]
        ).to(dtype or x.dtype)

    workload = ReductionCall(call)
    inputs = workload.gen_inputs()
    op = op_cls(**workload.arguments())
    tolerance = reference_tolerance(inputs[0].dtype)
    functors = {"tileops": op}
    if dtype is None:
        fn = flaggems_op("vector_norm")
        dims = flaggems_dims(p["dim"])

        def flaggems_fn(x):
            return fn(x, p["ord"], dims, p["keepdim"])

        assert_matches_reference(flaggems_fn, baseline_fn, *inputs, **tolerance)
        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    ManifestBenchmark(op, workload).compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(L1NormFwdOp))
def test_l1_norm_bench(call) -> None:
    _bench(L1NormFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(L2NormFwdOp))
def test_l2_norm_bench(call) -> None:
    _bench(L2NormFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(InfNormFwdOp))
def test_inf_norm_bench(call) -> None:
    _bench(InfNormFwdOp, call)
