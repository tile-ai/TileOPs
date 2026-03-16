"""Benchmarks for logical reduce ops (any, all, count_nonzero)."""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.test_base import FixtureBase, TestBase


class LogicalReduceBenchFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, op_kind",
            [
                pytest.param(1024, 4096, torch.float16, "any", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, "any", marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, "any", marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, "all", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, "all", marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, "all", marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, "count_nonzero", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, "count_nonzero", marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, "count_nonzero", marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduceBenchTest(TestBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_kind == "any":
            return x.bool().any(dim=-1)
        elif self.op_kind == "all":
            return x.bool().all(dim=-1)
        elif self.op_kind == "count_nonzero":
            return torch.count_nonzero(x, dim=-1).to(torch.int64)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class LogicalReduceBenchmark(BenchmarkBase):
    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Logical reduce: N comparisons per row, M rows
        return t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Output bytes: bool (1 byte) for any/all, int64 (8 bytes) for count_nonzero
        out_elem_bytes = 8 if t.op_kind == "count_nonzero" else 1
        # Read x (M*N) + write output (M * out_elem_bytes)
        return t.m * t.n * elem_bytes + t.m * out_elem_bytes


def _make_op(m: int, n: int, dtype: torch.dtype, op_kind: str):
    """Create the appropriate Op for the given op_kind."""
    from tileops.ops.reduction.all_op import AllOp
    from tileops.ops.reduction.any_op import AnyOp
    from tileops.ops.reduction.count_nonzero import CountNonzeroOp

    op_map = {
        "any": AnyOp,
        "all": AllOp,
        "count_nonzero": CountNonzeroOp,
    }
    cls = op_map[op_kind]
    return cls(M=m, N=n, dtype=dtype)


@LogicalReduceBenchFixture
def test_logical_reduce_bench(m: int, n: int, dtype: torch.dtype, op_kind: str) -> None:
    test = LogicalReduceBenchTest(m, n, dtype, op_kind)
    bm = LogicalReduceBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_op(m, n, dtype, op_kind)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("logical_reduce", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("logical_reduce", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
