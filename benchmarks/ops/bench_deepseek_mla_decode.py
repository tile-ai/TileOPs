from typing import Optional

import torch
import pytest

from tests.ops.test_deepseek_mla_decode import MlaDecodeFixture, MlaDecodeTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheOp


class MlaDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # qk_flops = 2 * batch * heads * seq_len_kv * (dim + dim_pe)
        # pv_flops = 2 * batch * heads * seq_len_kv * dim
        qk_flops = 2 * t.batch * t.heads * t.seq_len_kv * (t.dim + t.dim_pe)
        pv_flops = 2 * t.batch * t.heads * t.seq_len_kv * t.dim
        return qk_flops + pv_flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Q: batch * heads * dim
        # Q_pe: batch * heads * dim_pe
        # K: batch * seq_len_kv * heads_kv * dim
        # K_pe: batch * seq_len_kv * heads_kv * dim_pe
        # Output: batch * heads * dim
        return t.batch * (t.heads + t.seq_len_kv * t.heads_kv) * (
            t.dim + t.dim_pe) * t.dtype.itemsize


@MlaDecodeFixture
def test_mla_decode_bench(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                          dim_pe: int, dtype: torch.dtype, tune: bool) -> None:
    test = MlaDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    bm = MlaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = MultiHeadLatentAttentionDecodeWithKVCacheOp(
        batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("mla_decode", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("mla_decode", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
