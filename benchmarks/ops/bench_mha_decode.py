from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_mha_decode import MhaDecodeTest
from tileops.ops import MultiHeadAttentionDecodeWithKVCacheOp


class MhaDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len_q * t.seq_len_kv * t.dim
        flops = flops_per_matmul * 2
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Q: batch * seq_len_q * heads * dim
        # K, V: batch * seq_len_kv * heads * dim
        # Output: batch * seq_len_q * heads * dim
        return (t.batch * t.heads * (2 * t.seq_len_q + 2 * t.seq_len_kv) * t.dim *
                t.dtype.itemsize)


_MHA_DECODE_BENCH_PARAMS = [
    pytest.param(1, 32, 128, 8192, 128, torch.float16, True, id="fp16-long-cache"),
    pytest.param(1, 32, 128, 8192, 128, torch.bfloat16, True, id="bf16-long-cache"),
    pytest.param(1, 32, 128, 5, 128, torch.float16, True, id="short-kv-tail"),
]


@pytest.mark.parametrize("b, h, s_q, s_kv, d, dtype, tune", _MHA_DECODE_BENCH_PARAMS)
def test_mha_decode_bench(b: int, h: int, s_q: int, s_kv: int, d: int, dtype: torch.dtype,
                          tune: bool) -> None:
    test = MhaDecodeTest(b, h, s_q, s_kv, d, dtype)
    bm = MhaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionDecodeWithKVCacheOp(b, h, s_q, s_kv, d, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
