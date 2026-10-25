"""Benchmarks for cumulative ops (cumsum, cumprod).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest
(src/tileops/manifest/).

cumsum is timed against flag_gems' Triton scan as well as torch, eager and
compiled. cumprod has no flag_gems entry point in 5.0.2.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import FLAGGEMS_TAG, TORCH_COMPILE_TAG, compiled_reference, flaggems_op
from tileops._ops.reduction.cumulative import CumprodFwdOp, CumsumFwdOp


@pytest.mark.parametrize("case", bench.cases(CumsumFwdOp), ids=lambda case: case.id)
def test_cumsum_bench(case) -> None:
    op = CumsumFwdOp(**case.arguments)
    flaggems_cumsum = flaggems_op("cumsum")
    dim = case.workload.dim

    def flaggems_fn(x):
        return flaggems_cumsum(x, dim)

    torch_cumsum = case.torch
    if case.inputs[0].dtype == torch.bfloat16:
        torch_cumsum = bench.Implementation(
            torch_cumsum, noncomparable_reason="torch accumulates a bfloat16 cumsum in bfloat16"
        )
    bench.Runner(op, case).compare(
        {
            FLAGGEMS_TAG: flaggems_fn,
            "torch": torch_cumsum,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
        }
    )


@pytest.mark.parametrize("case", bench.cases(CumprodFwdOp), ids=lambda case: case.id)
def test_cumprod_bench(case) -> None:
    op = CumprodFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            "torch": case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
        }
    )
