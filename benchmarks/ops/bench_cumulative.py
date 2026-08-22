"""Benchmarks for cumulative ops (cumsum, cumprod).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest
(src/tileops/manifest/).

cumsum is timed against flag_gems' Triton scan as well as torch, eager and
compiled. cumprod has no flag_gems entry point in 5.0.2.
"""

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
from benchmarks.benchmark_base import (
    ManifestBenchmark,
    workloads_to_params,
)
from workloads.reduction import CumulativeWorkload

_CUMSUM_OP = "CumsumFwdOp"
_CUMPROD_OP = "CumprodFwdOp"


def _make_op(shape: tuple, dtype: torch.dtype, op_kind: str):
    """Create the appropriate Op for the given op_kind."""
    from tileops.ops.reduction.cumulative import CumprodFwdOp, CumsumFwdOp

    op_map = {
        "cumsum": CumsumFwdOp,
        "cumprod": CumprodFwdOp,
    }
    cls = op_map[op_kind]
    return cls(dim=-1)


class CumulativeBenchmarkWorkload(CumulativeWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        if self.op_kind == "cumsum":
            return x_f32.cumsum(dim=-1).to(x.dtype)
        if self.op_kind == "cumprod":
            return x_f32.cumprod(dim=-1).to(x.dtype)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_CUMSUM_OP))
def test_cumsum_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = CumulativeBenchmarkWorkload(shape, dtype, "cumsum")
    inputs = test.gen_inputs()

    op = _make_op(shape, dtype, "cumsum")
    bm = ManifestBenchmark(_CUMSUM_OP, op, test)

    flaggems_cumsum = flaggems_op("cumsum")

    def flaggems_fn(x):
        return flaggems_cumsum(x, -1)

    assert_matches_reference(flaggems_fn, test.ref_program, *inputs, **reference_tolerance(dtype))

    bm.compare(
        {
            "tileops": op,
            FLAGGEMS_TAG: flaggems_fn,
            "torch": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_CUMPROD_OP))
def test_cumprod_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = CumulativeBenchmarkWorkload(shape, dtype, "cumprod")
    inputs = test.gen_inputs()

    op = _make_op(shape, dtype, "cumprod")
    bm = ManifestBenchmark(_CUMPROD_OP, op, test)

    bm.compare(
        {
            "tileops": op,
            "torch": test.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(test.ref_program),
        },
        *inputs,
        record_as=op,
        params=locals(),
    )
