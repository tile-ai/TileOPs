"""Benchmarks for argreduce ops (argmax, argmin).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes, dtypes, and op-call parameters (e.g. ``dim``) are loaded
from the ops manifest (``src/tileops/manifest/``) — the benchmark must not
hard-code op parameters that are declared on manifest workload entries.

Each row is timed against flag_gems' Triton argreduce and against torch eager and
inductor.
"""

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, workloads_to_params
from tileops.ops.reduction.argreduce import ArgmaxFwdOp, ArgminFwdOp
from workloads.reduction import ArgmaxWorkload, ArgminWorkload

_ARGMAX_OP = "ArgmaxFwdOp"
_ARGMIN_OP = "ArgminFwdOp"


def _functors(op, baseline_fn, flaggems_name: str, dim: int, inputs) -> dict:
    """The op, flag_gems' argreduce, and torch eager and compiled.

    Indices are exact or wrong, so the check takes no tolerance.
    """
    fn = flaggems_op(flaggems_name)

    def flaggems_fn(x):
        return fn(x, dim)

    assert_matches_reference(flaggems_fn, baseline_fn, *inputs)
    return {
        "tileops": op,
        FLAGGEMS_TAG: flaggems_fn,
        "torch": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }


# Argmax benchmarks


@pytest.mark.parametrize("shape, dtype, extra", workloads_to_params(_ARGMAX_OP, include_extra=True))
def test_argmax_bench(shape: tuple, dtype: torch.dtype, extra: dict) -> None:
    workload = ArgmaxWorkload(shape, dtype)
    inputs = workload.gen_inputs()

    op = ArgmaxFwdOp(**extra)
    bm = ManifestBenchmark(op, workload)

    dim = extra["dim"]

    def baseline_fn(x):
        return x.argmax(dim=dim)

    bm.compare(
        _functors(op, baseline_fn, "argmax", dim, inputs),
        *inputs,
        record_as=op,
        params={"shape": shape, "dtype": dtype, "dim": dim},
    )


# Argmin benchmarks


@pytest.mark.parametrize("shape, dtype, extra", workloads_to_params(_ARGMIN_OP, include_extra=True))
def test_argmin_bench(shape: tuple, dtype: torch.dtype, extra: dict) -> None:
    workload = ArgminWorkload(shape, dtype)
    inputs = workload.gen_inputs()

    op = ArgminFwdOp(**extra)
    bm = ManifestBenchmark(op, workload)

    dim = extra["dim"]

    def baseline_fn(x):
        return x.argmin(dim=dim)

    bm.compare(
        _functors(op, baseline_fn, "argmin", dim, inputs),
        *inputs,
        record_as=op,
        params={"shape": shape, "dtype": dtype, "dim": dim},
    )
