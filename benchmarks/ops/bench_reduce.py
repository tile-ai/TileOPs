"""Benchmarks for the 8 basic reduce ops."""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.test_base import FixtureBase, TestBase


class ReduceBenchFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, op_kind",
            [
                pytest.param(1024, 4096, torch.float16, "sum", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, "sum", marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, "sum", marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, "mean", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, "amax", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, "amin", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, "prod", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, "std", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, "var", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, "var_mean", marks=pytest.mark.smoke),
            ],
        ),
    ]


class ReduceBenchTest(TestBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.op_kind == "prod":
            x = torch.rand(self.m, self.n, dtype=self.dtype, device="cuda") * 0.01 + 0.99
        else:
            x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> object:
        x_f32 = x.float()
        if self.op_kind == "sum":
            return x_f32.sum(dim=-1).to(x.dtype)
        elif self.op_kind == "mean":
            return x_f32.mean(dim=-1).to(x.dtype)
        elif self.op_kind == "amax":
            return x_f32.amax(dim=-1).to(x.dtype)
        elif self.op_kind == "amin":
            return x_f32.amin(dim=-1).to(x.dtype)
        elif self.op_kind == "prod":
            return x_f32.prod(dim=-1).to(x.dtype)
        elif self.op_kind == "std":
            return x_f32.std(dim=-1, correction=1).to(x.dtype)
        elif self.op_kind == "var":
            return x_f32.var(dim=-1, correction=1).to(x.dtype)
        elif self.op_kind == "var_mean":
            v = x_f32.var(dim=-1, correction=1).to(x.dtype)
            m = x_f32.mean(dim=-1).to(x.dtype)
            return (v, m)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class ReduceBenchmark(BenchmarkBase):
    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Approximate: N operations per row for reduce, M rows
        if t.op_kind in ("std", "var", "var_mean"):
            return 3 * t.m * t.n  # sum + sq_diff + sum
        return t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + write output (M) -- or (2*M) for var_mean
        out_elems = 2 * t.m if t.op_kind == "var_mean" else t.m
        return (t.m * t.n + out_elems) * elem_bytes


def _make_op(m, n, dtype, op_kind):
    """Create the appropriate Op for the given op_kind."""
    from tileops.ops.reduction.reduce import (
        AmaxOp,
        AminOp,
        MeanOp,
        ProdOp,
        StdOp,
        SumOp,
        VarMeanOp,
        VarOp,
    )

    op_map = {
        "sum": SumOp,
        "mean": MeanOp,
        "amax": AmaxOp,
        "amin": AminOp,
        "prod": ProdOp,
        "std": StdOp,
        "var": VarOp,
        "var_mean": VarMeanOp,
    }
    cls = op_map[op_kind]
    if op_kind in ("std", "var", "var_mean"):
        return cls(M=m, N=n, dtype=dtype, correction=1)
    return cls(M=m, N=n, dtype=dtype)


@ReduceBenchFixture
def test_reduce_bench(m: int, n: int, dtype: torch.dtype, op_kind: str) -> None:
    test = ReduceBenchTest(m, n, dtype, op_kind)
    bm = ReduceBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_op(m, n, dtype, op_kind)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("reduce", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("reduce", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
