"""Benchmarks for argreduce ops (argmax, argmin)."""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.test_base import FixtureBase, TestBase


class ArgreduceBenchFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, op_kind",
            [
                pytest.param(1024, 4096, torch.float16, "argmax", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, "argmax", marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, "argmax", marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float16, "argmin", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, "argmin", marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, "argmin", marks=pytest.mark.full),
            ],
        ),
    ]


class ArgreduceBenchTest(TestBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_kind == "argmax":
            return x.argmax(dim=-1)
        elif self.op_kind == "argmin":
            return x.argmin(dim=-1)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class ArgreduceBenchmark(BenchmarkBase):
    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Argreduce: N comparisons per row, M rows
        return t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + write output indices (M * 8 bytes for int64)
        return t.m * t.n * elem_bytes + t.m * 8


def _make_op(m: int, n: int, dtype: torch.dtype, op_kind: str):
    """Create the appropriate Op for the given op_kind."""
    from tileops.ops.reduction.argmax import ArgmaxOp
    from tileops.ops.reduction.argmin import ArgminOp

    op_map = {
        "argmax": ArgmaxOp,
        "argmin": ArgminOp,
    }
    cls = op_map[op_kind]
    return cls(M=m, N=n, dtype=dtype)


@ArgreduceBenchFixture
def test_argreduce_bench(m: int, n: int, dtype: torch.dtype, op_kind: str) -> None:
    test = ArgreduceBenchTest(m, n, dtype, op_kind)
    bm = ArgreduceBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_op(m, n, dtype, op_kind)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("argreduce", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("argreduce", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
