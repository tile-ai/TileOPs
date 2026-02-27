from typing import Optional

import torch
import pytest

from tests.ops.test_deepseek_dsa_decode import DsaDecodeFixture, DsaDecodeTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import DeepSeekSparseAttentionDecodeWithKVCacheOp


class DsaDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return t.batch * t.seq_len * (2 * t.dim + t.dim_tail) * t.topk * 2 * t.heads

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # q: batch, seq_len, heads, dim + dim_tail
        # kv: batch, seq_len_kv, group_kv, dim + dim_tail
        # indices: batch, seq_len, group_kv, topk
        # Output: batch, seq_len, heads, dim
        q_memory = (
            t.batch * t.seq_len * t.heads * (t.dim + t.dim_tail) * t.dtype.itemsize)
        kv_memory = t.batch * t.seq_len_kv * t.group_kv * (
            t.dim + t.dim_tail) * t.dtype.itemsize
        indices_memory = t.batch * t.seq_len * t.group_kv * t.topk * 4  # int32
        output_memory = t.batch * t.seq_len * t.heads * t.dim * t.dtype.itemsize
        return q_memory + kv_memory + indices_memory + output_memory


@DsaDecodeFixture
def test_dsa_decode_bench(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                          dim_tail: int, topk: int, stride_kv: int, group_kv: int,
                          q_start_index_s: int, sm_scale: float, dtype: torch.dtype,
                          tune: bool) -> None:
    test = DsaDecodeTest(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, group_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    bm = DsaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = DeepSeekSparseAttentionDecodeWithKVCacheOp(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, group_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("dsa_decode", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("dsa_decode", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
