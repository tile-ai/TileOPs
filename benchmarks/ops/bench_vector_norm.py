"""Benchmarks for vector norm ops (l1_norm, l2_norm, inf_norm).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest (src/tileops/manifest/).

Each order is timed against flag_gems' Triton ``vector_norm`` and against torch
eager and inductor.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flaggems_dims,
    flaggems_op,
)
from tileops.ops.reduction.vector_norm import VectorNormFwdOp


def _bench(op_cls: type, case: bench.Case) -> None:
    """Check flag_gems' ``vector_norm`` against torch, then time it and the op.

    flag_gems accumulates in fp32 as the reference does; it takes no output dtype, so a row
    passing ``dtype`` has no flag_gems tag.
    """
    p = case.params
    dtype = getattr(torch, p["dtype"]) if p.get("dtype") else None
    baseline_fn = case.reference
    op = op_cls(**case.arguments)
    implementations = {}
    if dtype is None:
        fn = flaggems_op("vector_norm")
        dims = flaggems_dims(p["dim"])

        def flaggems_fn(x):
            return fn(x, p["ord"], dims, p["keepdim"])

        implementations[FLAGGEMS_TAG] = flaggems_fn
    implementations["torch"] = baseline_fn
    implementations[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(VectorNormFwdOp), ids=lambda case: case.id)
def test_vector_norm_bench(case) -> None:
    _bench(VectorNormFwdOp, case)
