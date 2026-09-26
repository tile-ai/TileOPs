"""Benchmarks for cumulative ops (cumsum, cumprod).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest
(src/tileops/manifest/).

cumsum is timed against flag_gems' Triton scan as well as torch, eager and
compiled. cumprod has no flag_gems entry point in 5.0.2.
"""

import math

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.reduction.cumulative import CumprodFwdOp, CumsumFwdOp
from workloads.reduction import CumulativeWorkload


class CumulativeBenchmarkWorkload(CumulativeWorkload):
    def __init__(self, call, op_kind: str):
        shape, dtype = call.tensors["x"]
        super().__init__(shape, getattr(torch, dtype), op_kind)
        self.dim = call.params["dim"]

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        if self.op_kind == "cumsum":
            return x_f32.cumsum(dim=self.dim).to(x.dtype)
        if self.op_kind == "cumprod":
            return x_f32.cumprod(dim=self.dim).to(x.dtype)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


@pytest.mark.parametrize("call", manifest_calls(CumsumFwdOp))
def test_cumsum_bench(call) -> None:
    test = CumulativeBenchmarkWorkload(call, "cumsum")
    inputs = test.gen_inputs()
    dtype = test.dtype

    op = CumsumFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, test)

    flaggems_cumsum = flaggems_op("cumsum")

    def flaggems_fn(x):
        return flaggems_cumsum(x, test.dim)

    # A scan's error grows with the prefix length it sums in another order, so atol scales
    # with the square root of the scanned length.
    tolerance = reference_tolerance(dtype)
    tolerance["atol"] *= math.sqrt(test.shape[test.dim])
    assert_matches_reference(flaggems_fn, test.ref_program, *inputs, **tolerance)

    bm.compare(
        {
            "tileops": op,
            FLAGGEMS_TAG: flaggems_fn,
            "torch": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(CumprodFwdOp))
def test_cumprod_bench(call) -> None:
    test = CumulativeBenchmarkWorkload(call, "cumprod")
    inputs = test.gen_inputs()

    op = CumprodFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, test)

    bm.compare(
        {
            "tileops": op,
            "torch": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
    )
