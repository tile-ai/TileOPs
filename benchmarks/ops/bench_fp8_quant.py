from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import Fp8QuantOp
from workloads.ops.fp8_quant import Fp8QuantTest


class Fp8QuantBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        return (2 * t.batch * t.seq_len_kv * t.kv_group * t.index_dim +
                t.batch * t.seq_len_kv * t.kv_group + 4 * t.batch * t.seq_len_kv * t.kv_group * t.index_dim)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        return t.batch * t.seq_len_kv * t.kv_group * t.index_dim * t.in_dtype.itemsize


_FP8_QUANT_BENCH_PARAMS = [
    pytest.param(1, 8192, 1, 64, torch.float16, True, id="mainstream-fp16"),
    pytest.param(1, 8192, 1, 64, torch.bfloat16, True, id="mainstream-bf16"),
    pytest.param(1, 4096, 1, 128, torch.float32, True, id="wider-index"),
    pytest.param(1, 16384, 1, 32, torch.float32, True, id="long-sequence"),
]


@pytest.mark.parametrize("batch, seq_len_kv, kv_group, index_dim, in_dtype, tune",
                         _FP8_QUANT_BENCH_PARAMS)
def test_fp8_quant_bench(batch: int, seq_len_kv: int, kv_group: int, index_dim: int,
                         in_dtype: torch.dtype, tune: bool) -> None:
    test = Fp8QuantTest(batch, seq_len_kv, kv_group, index_dim, in_dtype)
    bm = Fp8QuantBenchmark(test)
    inputs = test.gen_inputs()

    op = Fp8QuantOp(batch=batch,
                    seq_len_kv=seq_len_kv,
                    kv_group=kv_group,
                    index_dim=index_dim,
                    in_dtype=in_dtype,
                    tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
