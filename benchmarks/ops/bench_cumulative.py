"""Benchmarks for cumulative ops (cumsum, cumprod)."""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from workloads.base import FixtureBase, WorkloadBase


class CumulativeBenchFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, op_kind",
            [
                pytest.param(1024, 4096, torch.float16, "cumsum"),
                pytest.param(1024, 4096, torch.bfloat16, "cumsum"),
                pytest.param(4096, 4096, torch.float16, "cumsum"),
                pytest.param(1024, 4096, torch.float16, "cumprod"),
                pytest.param(1024, 4096, torch.bfloat16, "cumprod"),
                pytest.param(4096, 4096, torch.float16, "cumprod"),
            ],
        ),
    ]


class CumulativeBenchTest(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        self.m = m
        self.n = n
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
        elif self.op_kind == "cumprod":
            return x_f32.cumprod(dim=-1).to(x.dtype)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class CumulativeBenchmark(BenchmarkBase):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        # Approximate: inclusive scan performs N-1 ops per row, rounded up to M*N
        return t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + write output (M*N)
        return 2 * t.m * t.n * elem_bytes


def _make_op(m: int, n: int, dtype: torch.dtype, op_kind: str):
    """Create the appropriate Op for the given op_kind."""
    from tileops.ops.reduction.cumprod import CumprodFwdOp
    from tileops.ops.reduction.cumsum import CumsumFwdOp

    op_map = {
        "cumsum": CumsumFwdOp,
        "cumprod": CumprodFwdOp,
    }
    cls = op_map[op_kind]
    return cls(M=m, N=n, dtype=dtype)


@CumulativeBenchFixture
def test_cumulative_bench(m: int, n: int, dtype: torch.dtype, op_kind: str) -> None:
    test = CumulativeBenchTest(m, n, dtype, op_kind)
    bm = CumulativeBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_op(m, n, dtype, op_kind)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
