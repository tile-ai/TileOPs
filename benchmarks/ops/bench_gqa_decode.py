from typing import Optional

import torch
import pytest

from tests.ops.test_gqa_decode import GqaDecodeFixture, GqaDecodeTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import GroupQueryAttentionDecodeWithKVCacheOp


class GqaDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seq_len_kv * t.dim
        flops = flops_per_matmul * 2
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Q: batch * 1 * heads * dim
        # K, V: batch * seq_len_kv * heads_kv * dim
        # Output: batch * 1 * heads * dim
        return 2 * t.batch * t.dim * t.dtype.itemsize * (
            t.heads + t.heads_kv * t.seq_len_kv)


@GqaDecodeFixture
def test_gqa_decode_bench(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                          dtype: torch.dtype, tune: bool) -> None:
    test = GqaDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dtype)
    bm = GqaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupQueryAttentionDecodeWithKVCacheOp(batch, heads, heads_kv, seq_len_kv, dim, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gqa_decode", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gqa_decode", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
