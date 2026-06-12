"""Benchmarks for cumulative ops (cumsum, cumprod).

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines.
Workload shapes and roofline formulas are loaded from the ops manifest
(tileops/manifest/).
"""

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from workloads.workload_base import WorkloadBase

_CUMSUM_OP = "CumsumFwdOp"
_CUMPROD_OP = "CumprodFwdOp"


class CumulativeBenchTest(WorkloadBase):
    def __init__(self, shape: tuple, dtype: torch.dtype, op_kind: str):
        self.shape = shape
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.op_kind == "cumprod":
            x = torch.rand(*self.shape, dtype=self.dtype, device="cuda") * 0.01 + 0.99
        else:
            x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        if self.op_kind == "cumsum":
            return x_f32.cumsum(dim=-1).to(x.dtype)
        elif self.op_kind == "cumprod":
            return x_f32.cumprod(dim=-1).to(x.dtype)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


def _make_op(shape: tuple, dtype: torch.dtype, op_kind: str):
    """Create the appropriate Op for the given op_kind."""
    from tileops.ops.reduction.cumprod import CumprodFwdOp
    from tileops.ops.reduction.cumsum import CumsumFwdOp

    op_map = {
        "cumsum": CumsumFwdOp,
        "cumprod": CumprodFwdOp,
    }
    cls = op_map[op_kind]
    return cls(N=shape[-1], dtype=dtype, dim=-1)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_CUMSUM_OP))
def test_cumsum_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = CumulativeBenchTest(shape, dtype, "cumsum")
    inputs = test.gen_inputs()

    op = _make_op(shape, dtype, "cumsum")
    bm = ManifestBenchmark(_CUMSUM_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_CUMPROD_OP))
def test_cumprod_bench(shape: tuple, dtype: torch.dtype) -> None:
    test = CumulativeBenchTest(shape, dtype, "cumprod")
    inputs = test.gen_inputs()

    op = _make_op(shape, dtype, "cumprod")
    bm = ManifestBenchmark(_CUMPROD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
