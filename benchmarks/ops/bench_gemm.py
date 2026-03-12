from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gemm import GemmTest
from tileops.ops import GemmOp


class GemmBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return 2.0 * self.test.m * self.test.n * self.test.k

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return (t.m * t.k + t.k * t.n + t.m * t.n) * t.dtype.itemsize


_GEMM_BENCH_PARAMS = [
    pytest.param(
        1, 18432, 7168, torch.float16, False, True, True,
        marks=pytest.mark.nightly,
        id="bench-nightly-fp16-tuned-wide-alt",
    ),
    pytest.param(
        7168, 1, 16384, torch.float16, False, False, True,
        marks=pytest.mark.nightly,
        id="bench-nightly-fp16-tuned-thin-n",
    ),
    pytest.param(
        18432, 1, 7168, torch.float16, False, False, True,
        marks=pytest.mark.nightly,
        id="bench-nightly-fp16-tuned-thin-n-alt",
    ),
    pytest.param(
        1, 7168, 16384, torch.bfloat16, False, True, True,
        marks=pytest.mark.nightly,
        id="bench-nightly-bf16-tuned-wide",
    ),
    pytest.param(
        1, 18432, 7168, torch.bfloat16, False, True, True,
        marks=pytest.mark.nightly,
        id="bench-nightly-bf16-tuned-wide-alt",
    ),
    pytest.param(
        7168, 1, 16384, torch.bfloat16, False, False, True,
        marks=pytest.mark.nightly,
        id="bench-nightly-bf16-tuned-thin-n",
    ),
    pytest.param(
        18432, 1, 7168, torch.bfloat16, False, False, True,
        marks=pytest.mark.nightly,
        id="bench-nightly-bf16-tuned-thin-n-alt",
    ),
]


@pytest.mark.parametrize(
    "m, n, k, dtype, trans_a, trans_b, tune",
    _GEMM_BENCH_PARAMS,
)
def test_gemm_bench(m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool,
                    tune: bool) -> None:
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    bm = GemmBenchmark(test)
    inputs = test.gen_inputs()

    op = GemmOp(m, n, k, trans_a=trans_a, trans_b=trans_b, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gemm", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gemm", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
