from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from benchmarks.ops.cases.case_fp8_quant import Fp8QuantTest
from tileops.ops import Fp8QuantOp


class Fp8QuantBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return 2 * t.seq_len_kv * t.index_dim + t.seq_len_kv + 4 * t.seq_len_kv * t.index_dim

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return t.seq_len_kv * t.index_dim * t.in_dtype.itemsize


_FP8_QUANT_BENCH_PARAMS = [
    pytest.param(8192, 64, torch.bfloat16, True, marks=pytest.mark.full, id="bench-bf16-base"),
    pytest.param(4096, 128, torch.float32, True, marks=pytest.mark.full, id="bench-fp32-wide"),
    pytest.param(16384, 32, torch.float32, True, marks=pytest.mark.nightly, id="bench-fp32-long"),
]


@pytest.mark.parametrize("seq_len_kv, index_dim, in_dtype, tune", _FP8_QUANT_BENCH_PARAMS)
def test_fp8_quant_bench(seq_len_kv: int, index_dim: int, in_dtype: torch.dtype,
                         tune: bool) -> None:
    test = Fp8QuantTest(seq_len_kv, index_dim, in_dtype)
    bm = Fp8QuantBenchmark(test)
    inputs = test.gen_inputs()

    op = Fp8QuantOp(seq_len_kv=seq_len_kv, index_dim=index_dim, in_dtype=in_dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("fp8_quant", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("fp8_quant", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
