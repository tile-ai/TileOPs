from typing import Optional

import torch
import pytest

from tests.ops.test_mha_decode import MhaDecodeFixture, MhaDecodeTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
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


@MhaDecodeFixture
def test_mha_decode_bench(b: int, h: int, s_q: int, s_kv: int, d: int, dtype: torch.dtype,
                          tune: bool) -> None:
    test = MhaDecodeTest(b, h, s_q, s_kv, d, dtype)
    bm = MhaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionDecodeWithKVCacheOp(b, h, s_q, s_kv, d, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("mha_decode", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("mha_decode", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
