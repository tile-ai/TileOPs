from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_fp8_quant import Fp8QuantTest
from tileops.ops import Fp8QuantOp


class Fp8QuantBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return (2 * t.batch * t.seq_len_kv * t.kv_group * t.index_dim +
                t.batch * t.seq_len_kv * t.kv_group + 4 * t.batch * t.seq_len_kv * t.kv_group * t.index_dim)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return t.batch * t.seq_len_kv * t.kv_group * t.index_dim * t.in_dtype.itemsize


_FP8_QUANT_BENCH_PARAMS = [
    pytest.param(8192, 64, torch.float16, True, id="mainstream-fp16"),
    pytest.param(8192, 64, torch.bfloat16, True, id="mainstream-bf16"),
    pytest.param(4096, 128, torch.float32, True, id="wider-index"),
    pytest.param(16384, 32, torch.float32, True, id="long-sequence"),
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
