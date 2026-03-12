from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gqa_decode_paged import GqaDecodePagedTest
from tileops.ops import GroupQueryAttentionDecodePagedWithKVCacheOp


class GqaDecodePagedBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seqlen_kv * t.dim
        flops = flops_per_matmul * 2
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        num_pages = t.seqlen_kv // t.page_size
        # Q, output: batch * heads * dim; K,V: seqlen_kv * heads_kv * dim; block_table, real_seqlen_kv: int32
        return (t.batch * t.heads * t.dim * 2 +
                2 * t.seqlen_kv * t.heads_kv * t.dim) * t.dtype.itemsize + \
            t.batch * num_pages * 4 + t.batch * 4


_GQA_DECODE_PAGED_BENCH_PARAMS = [
    pytest.param(
        2, 8, 4, 1024, 64, 256, torch.float16, False,
        marks=pytest.mark.full,
        id="bench-fp16-page256",
    ),
    pytest.param(
        2, 16, 8, 4096, 128, 128, torch.float16, False,
        marks=pytest.mark.nightly,
        id="bench-fp16-long-page128",
    ),
    pytest.param(
        1, 32, 8, 4096, 128, 256, torch.bfloat16, False,
        marks=pytest.mark.nightly,
        id="bench-bf16-long-page256",
    ),
]


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune",
    _GQA_DECODE_PAGED_BENCH_PARAMS,
)
def test_gqa_decode_paged_bench(batch: int, heads: int, heads_kv: int, seqlen_kv: int, dim: int,
                                page_size: int, dtype: torch.dtype, tune: bool) -> None:
    test = GqaDecodePagedTest(batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype)
    bm = GqaDecodePagedBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupQueryAttentionDecodePagedWithKVCacheOp(
        batch, heads, heads_kv, seqlen_kv, dim, page_size, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gqa_decode_paged", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("gqa_decode_paged", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
