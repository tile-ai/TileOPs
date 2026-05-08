"""Benchmarks for cumulative ops (cumsum, cumprod).

Workload shapes and roofline formulas are loaded from the ops manifest
(``tileops/manifest/scan.yaml``).
"""

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark, workloads_to_params
from tileops.ops.reduction.cumprod import CumprodFwdOp
from tileops.ops.reduction.cumsum import CumsumFwdOp
from workloads.workload_base import WorkloadBase

_CUMSUM_OP = "CumsumFwdOp"
_CUMPROD_OP = "CumprodFwdOp"


class _CumulativeWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        self.m = m
        self.n = n
        self.shape = (m, n)
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.op_kind == "cumprod":
            x = torch.rand(self.m, self.n, dtype=self.dtype, device="cuda") * 0.01 + 0.99
        else:
            x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        if self.op_kind == "cumsum":
            return x_f32.cumsum(dim=-1).to(x.dtype)
        return x_f32.cumprod(dim=-1).to(x.dtype)


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_CUMSUM_OP))
def test_cumsum_bench(shape: tuple, dtype: torch.dtype) -> None:
    m, n = shape
    op_kind = "cumsum"
    test = _CumulativeWorkload(m, n, dtype, op_kind)
    inputs = test.gen_inputs()

    op = CumsumFwdOp(N=n, dtype=dtype)
    bm = ManifestBenchmark(_CUMSUM_OP, op, test)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    # Preserve legacy report column order: m, n, dtype, op_kind.
    report_params = {"m": m, "n": n, "dtype": dtype, "op_kind": op_kind}
    BenchmarkReport.record(op, report_params, result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, report_params, result_bl, tag="torch")


@pytest.mark.parametrize("shape, dtype", workloads_to_params(_CUMPROD_OP))
def test_cumprod_bench(shape: tuple, dtype: torch.dtype) -> None:
    m, n = shape
    op_kind = "cumprod"
    test = _CumulativeWorkload(m, n, dtype, op_kind)
    inputs = test.gen_inputs()

    op = CumprodFwdOp(N=n, dtype=dtype)
    bm = ManifestBenchmark(_CUMPROD_OP, op, test)
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    # Preserve legacy report column order: m, n, dtype, op_kind.
    report_params = {"m": m, "n": n, "dtype": dtype, "op_kind": op_kind}
    BenchmarkReport.record(op, report_params, result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, report_params, result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
