from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_deepseek_dsa_decode import DsaDecodeTest
from tileops.ops import DeepSeekSparseAttentionDecodeWithKVCacheOp


class DsaDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return t.batch * t.seq_len * (2 * t.dim + t.dim_tail) * t.topk * 2 * t.heads

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # q: batch, seq_len, heads, dim + dim_tail
        # kv: batch, seq_len_kv, heads_kv, dim + dim_tail
        # indices: batch, seq_len, heads_kv, topk
        # Output: batch, seq_len, heads, dim
        q_memory = (
            t.batch * t.seq_len * t.heads * (t.dim + t.dim_tail) * t.dtype.itemsize)
        kv_memory = t.batch * t.seq_len_kv * t.heads_kv * (
            t.dim + t.dim_tail) * t.dtype.itemsize
        indices_memory = t.batch * t.seq_len * t.heads_kv * t.topk * 4  # int32
        output_memory = t.batch * t.seq_len * t.heads * t.dim * t.dtype.itemsize
        return q_memory + kv_memory + indices_memory + output_memory


_DSA_DECODE_BENCH_PARAMS = [
    pytest.param(
        1, 128, 1024, 2048, 512, 64, 2048, 1, 1, 1024, None, torch.float16, False,
        id="single-batch-mainstream",
    ),
    pytest.param(
        1, 128, 512, 4096, 512, 64, 1024, 1, 1, 512, None, torch.float16, False,
        id="longer-kv-lower-topk",
    ),
]


@pytest.mark.parametrize(
    "batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, q_start_index_s, sm_scale, dtype, tune",
    _DSA_DECODE_BENCH_PARAMS,
)
def test_dsa_decode_bench(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                          dim_tail: int, topk: int, stride_kv: int, heads_kv: int,
                          q_start_index_s: int, sm_scale: float, dtype: torch.dtype,
                          tune: bool) -> None:
    test = DsaDecodeTest(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    bm = DsaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = DeepSeekSparseAttentionDecodeWithKVCacheOp(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
