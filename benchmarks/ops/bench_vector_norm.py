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
    compiled_reference,
    flaggems_dims,
    flaggems_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.reduction.vector_norm import VectorNormFwdOp
from workloads.reduction import ReductionCall


def _bench(op_cls: type, call) -> None:
    """Check flag_gems' ``vector_norm`` against torch, then time it and the op.

    flag_gems accumulates in fp32 as the reference does; it takes no output dtype, so a row
    passing ``dtype`` has no flag_gems tag.
    """
    p = call.params
    dtype = getattr(torch, p["dtype"]) if p.get("dtype") else None
    workload = ReductionCall(call)
    baseline_fn = workload.ref_program
    inputs = workload.gen_inputs()
    op = op_cls(**workload.arguments())
    functors = {"tileops": op}
    if dtype is None:
        fn = flaggems_op("vector_norm")
        dims = flaggems_dims(p["dim"])

        def flaggems_fn(x):
            return fn(x, p["ord"], dims, p["keepdim"])

        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    ManifestBenchmark(op, workload).compare(functors, *inputs)


@pytest.mark.parametrize("call", manifest_calls(VectorNormFwdOp))
def test_vector_norm_bench(call) -> None:
    _bench(VectorNormFwdOp, call)
