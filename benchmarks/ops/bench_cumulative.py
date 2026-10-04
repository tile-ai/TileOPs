"""Benchmarks for cumulative ops (cumsum, cumprod).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest
(src/tileops/manifest/).

cumsum is timed against flag_gems' Triton scan as well as torch, eager and
compiled. cumprod has no flag_gems entry point in 5.0.2.
"""

import pytest

from benchmarks.baselines import FLAGGEMS_TAG, TORCH_COMPILE_TAG, compiled_reference, flaggems_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.reduction.cumulative import CumprodFwdOp, CumsumFwdOp
from workloads.reduction import CumulativeCall


@pytest.mark.parametrize("call", manifest_calls(CumsumFwdOp))
def test_cumsum_bench(call) -> None:
    workload = CumulativeCall(call, "cumsum")
    inputs = workload.gen_inputs()
    op = CumsumFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)
    flaggems_cumsum = flaggems_op("cumsum")

    def flaggems_fn(x):
        return flaggems_cumsum(x, workload.dim)

    bm.compare(
        {
            "tileops": op,
            FLAGGEMS_TAG: flaggems_fn,
            "torch": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(CumprodFwdOp))
def test_cumprod_bench(call) -> None:
    workload = CumulativeCall(call, "cumprod")
    inputs = workload.gen_inputs()

    op = CumprodFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    bm.compare(
        {
            "tileops": op,
            "torch": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )
