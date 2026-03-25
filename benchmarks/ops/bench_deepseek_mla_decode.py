from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_deepseek_mla_decode import MlaDecodeTest
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


_MLA_DECODE_BENCH_PARAMS = [
    pytest.param(32, 128, 1, 8192, 512, 64, torch.float16, True, id="mainstream-fp16"),
    pytest.param(16, 128, 1, 4096, 512, 64, torch.float16, True, id="mid-cache-fp16"),
]


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune",
    _MLA_DECODE_BENCH_PARAMS,
)
def test_mla_decode_bench(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                          dim_pe: int, dtype: torch.dtype, tune: bool) -> None:
    test = MlaDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    bm = MlaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = MultiHeadLatentAttentionDecodeWithKVCacheOp(
        batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
