"""Benchmarks for vector norm ops (l1_norm, l2_norm, inf_norm)."""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from workloads.base import FixtureBase, WorkloadBase


class VectorNormBenchFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, op_kind",
            [
                # --- l1 ---
                pytest.param(1024, 4096, torch.float16, "l1", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, "l1", marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float32, "l1", marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, "l1", marks=pytest.mark.full),
                # --- l2 ---
                pytest.param(1024, 4096, torch.float16, "l2", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, "l2", marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float32, "l2", marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, "l2", marks=pytest.mark.full),
                # --- inf ---
                pytest.param(1024, 4096, torch.float16, "inf", marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, "inf", marks=pytest.mark.full),
                pytest.param(1024, 4096, torch.float32, "inf", marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, "inf", marks=pytest.mark.full),
            ],
        ),
    ]


# Map op_kind to the ord parameter for torch.linalg.vector_norm
_ORD_MAP = {"l1": 1, "l2": 2, "inf": float("inf")}


class VectorNormBenchTest(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        ord_val = _ORD_MAP[self.op_kind]
        return torch.linalg.vector_norm(x, ord=ord_val, dim=-1)


class VectorNormBenchmark(BenchmarkBase):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        # l1: N abs + N-1 adds per row
        # l2: N muls + N-1 adds + 1 sqrt per row
        # inf: N abs + N-1 comparisons per row
        return t.m * t.n

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read x (M*N) + write output (M)
        return t.m * t.n * elem_bytes + t.m * elem_bytes


def _make_op(dtype: torch.dtype, op_kind: str):
    """Create the appropriate Op for the given op_kind."""
    from tileops.ops.reduction.inf_norm import InfNormFwdOp
    from tileops.ops.reduction.l1_norm import L1NormFwdOp
    from tileops.ops.reduction.l2_norm import L2NormFwdOp

    op_map = {
        "l1": L1NormFwdOp,
        "l2": L2NormFwdOp,
        "inf": InfNormFwdOp,
    }
    cls = op_map[op_kind]
    return cls(dtype=dtype)


@VectorNormBenchFixture
def test_vector_norm_bench(m: int, n: int, dtype: torch.dtype, op_kind: str) -> None:
    test = VectorNormBenchTest(m, n, dtype, op_kind)
    bm = VectorNormBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_op(dtype, op_kind)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
