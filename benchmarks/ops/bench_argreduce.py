"""Benchmarks for argreduce ops (argmax, argmin).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines. Each case is one
manifest call (``src/tileops/manifest/``), parameters included.

Each row is timed against flag_gems' Triton argreduce and against torch eager and
inductor.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flaggems_op,
)
from tileops.ops.reduction.argreduce import ArgmaxFwdOp, ArgminFwdOp


def _functors(baseline_fn, flaggems_name: str, dim: int, keepdim: bool, inputs) -> dict:
    """flag_gems' argreduce, and torch eager and compiled.

    Indices are exact or wrong, so the check takes no tolerance.
    """
    functors = {}
    # flag_gems' argmin launch fails with an invalid argument on a non-last axis.
    if flaggems_name == "argmax" or dim in (-1, inputs[0].ndim - 1):
        fn = flaggems_op(flaggems_name)

        def flaggems_fn(x):
            return fn(x, dim, keepdim)

        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    return functors


@pytest.mark.parametrize("case", bench.cases(ArgmaxFwdOp), ids=lambda case: case.id)
def test_argmax_bench(case) -> None:
    op = ArgmaxFwdOp(**case.arguments)
    dim, keepdim = case.params["dim"], case.params["keepdim"]

    baseline_fn = case.reference

    functors = _functors(baseline_fn, "argmax", dim, keepdim, case.inputs)

    bench.Runner(op, case).compare(functors)


@pytest.mark.parametrize("case", bench.cases(ArgminFwdOp), ids=lambda case: case.id)
def test_argmin_bench(case) -> None:
    op = ArgminFwdOp(**case.arguments)
    dim, keepdim = case.params["dim"], case.params["keepdim"]

    baseline_fn = case.reference

    functors = _functors(baseline_fn, "argmin", dim, keepdim, case.inputs)

    bench.Runner(op, case).compare(functors)
