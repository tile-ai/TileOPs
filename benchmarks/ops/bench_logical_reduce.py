"""Benchmarks for logical reduce ops (any, all, count_nonzero).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest (src/tileops/manifest/).

any and all are timed against flag_gems' Triton reductions as well as torch, eager
and compiled. count_nonzero has none: its entry point raises on a list of dims.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flaggems_dims,
    flaggems_op,
)
from tileops.ops.reduction.logical_reduce import AllFwdOp, AnyFwdOp, CountNonzeroFwdOp


def _bench(op_cls: type, case: bench.Case, flaggems_name=None) -> None:
    """Check the op and flag_gems where it has a kernel against torch, then time them.

    A boolean reduction is exact or wrong, so the check takes no tolerance.
    """
    baseline_fn = case.reference
    op = op_cls(**case.arguments)
    functors = {}
    if flaggems_name is not None:
        fn = flaggems_op(flaggems_name)
        dims = flaggems_dims(case.params["dim"])
        keepdim = case.params["keepdim"]

        def flaggems_fn(x):
            return fn(x.bool(), dims, keepdim)

        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    bench.Runner(op, case).compare(functors)


@pytest.mark.parametrize("case", bench.cases(AnyFwdOp), ids=lambda case: case.id)
def test_any_bench(case) -> None:
    _bench(AnyFwdOp, case, "any_dims")


@pytest.mark.parametrize("case", bench.cases(AllFwdOp), ids=lambda case: case.id)
def test_all_bench(case) -> None:
    _bench(AllFwdOp, case, "all_dims")


@pytest.mark.parametrize("case", bench.cases(CountNonzeroFwdOp), ids=lambda case: case.id)
def test_count_nonzero_bench(case) -> None:
    _bench(CountNonzeroFwdOp, case)
