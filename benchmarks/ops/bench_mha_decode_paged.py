from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from benchmarks.ops.cases.case_mha_decode_paged import MhaDecodePagedTest
from tileops.ops import MultiHeadAttentionDecodePagedWithKVCacheOp


class MhaDecodePagedBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seqlen_q * t.seqlen_kv * t.dim
        flops = flops_per_matmul * 2
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        num_pages = t.seqlen_kv // t.page_size
        # Q, output: batch * seqlen_q * heads * dim; K,V: seqlen_kv * heads * dim; block_table, real_seqlen_kv: int32
        return (t.batch * t.seqlen_q * t.heads * t.dim * 2 +
                2 * t.seqlen_kv * t.heads * t.dim) * t.dtype.itemsize + \
            t.batch * num_pages * 4 + t.batch * 4


_MHA_DECODE_PAGED_BENCH_PARAMS = [
    pytest.param(
        2, 8, 1, 1024, 64, 256, False, torch.float16, False,
        marks=pytest.mark.full,
        id="bench-fp16-page256",
    ),
    pytest.param(
        2, 16, 1, 4096, 128, 128, False, torch.float16, False,
        marks=pytest.mark.nightly,
        id="bench-fp16-long-page128",
    ),
    pytest.param(
        1, 16, 1, 4096, 128, 256, False, torch.bfloat16, False,
        marks=pytest.mark.nightly,
        id="bench-bf16-long-page256",
    ),
]


@pytest.mark.parametrize(
    "batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune",
    _MHA_DECODE_PAGED_BENCH_PARAMS,
)
def test_mha_decode_paged_bench(batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int,
                                page_size: int, is_causal: bool, dtype: torch.dtype,
                                tune: bool) -> None:
    test = MhaDecodePagedTest(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype)
    bm = MhaDecodePagedBenchmark(test)
    inputs = test.gen_inputs()

    op = MultiHeadAttentionDecodePagedWithKVCacheOp(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("mha_decode_paged", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("mha_decode_paged", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
