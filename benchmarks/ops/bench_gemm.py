from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, BenchmarkWorkload
from tileops.ops import GemmOp
from workloads.gemm import GemmTest


class _GemmTestBaseline(GemmTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)


class GemmBenchmark(BenchmarkBase[BenchmarkWorkload]):

    def calculate_flops(self) -> Optional[float]:
        return 2.0 * self.workload.m * self.workload.n * self.workload.k

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        return (t.m * t.k + t.k * t.n + t.m * t.n) * t.dtype.itemsize


_GEMM_BENCH_PARAMS = [
    pytest.param(1024, 1024, 1024, torch.float16, False, False, True, id="square-fp16"),
    pytest.param(1, 7168, 16384, torch.float16, False, True, True, id="wide-fp16"),
    pytest.param(1, 18432, 7168, torch.bfloat16, False, True, True, id="wide-alt-bf16"),
    pytest.param(7168, 1, 16384, torch.float16, False, False, True, id="thin-n-fp16"),
    pytest.param(18432, 1, 7168, torch.bfloat16, False, False, True, id="thin-n-alt-bf16"),
]


@pytest.mark.parametrize("m, n, k, dtype, trans_a, trans_b, tune", _GEMM_BENCH_PARAMS)
def test_gemm_bench(m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool,
                    tune: bool) -> None:
    test = _GemmTestBaseline(m, n, k, dtype, trans_a, trans_b)
    bm = GemmBenchmark(test)
    inputs = test.gen_inputs()

    op = GemmOp(m, n, k, trans_a=trans_a, trans_b=trans_b, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-cublas")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
