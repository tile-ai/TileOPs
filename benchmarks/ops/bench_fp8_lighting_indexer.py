from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_fp8_lighting_indexer import Fp8LightingIndexerTest
from tileops.ops import Fp8LightingIndexerOp


class Fp8LightingIndexerBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # Flops depend on the actual mask cost which varies per input
        return None

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        dtype = torch.float8_e4m3fn
        accum_dtype = torch.float32
        index_dtype = torch.int32

        index_q_memory = t.batch * t.seq_len * t.heads * t.index_dim * dtype.itemsize
        index_k_memory = t.batch * t.seq_len_kv * t.index_dim * t.kv_group * dtype.itemsize
        index_k_scale_memory = t.batch * t.seq_len_kv * t.kv_group * accum_dtype.itemsize
        logits_memory = t.batch * t.seq_len * t.seq_len_kv * t.kv_group * accum_dtype.itemsize
        weights_memory = t.seq_len * t.heads * accum_dtype.itemsize
        cu_seqlens_ks_memory = t.seq_len * index_dtype.itemsize
        cu_seqlens_ke_memory = t.seq_len * index_dtype.itemsize

        return (index_q_memory + index_k_memory + index_k_scale_memory + logits_memory +
                weights_memory + cu_seqlens_ks_memory + cu_seqlens_ke_memory)


_FP8_LIGHTING_INDEXER_BENCH_PARAMS = [
    pytest.param(1, 4096, 32, 64, 8192, 1, True, None, False, id="default-config"),
    pytest.param(1, 2048, 16, 64, 4096, 1, True, None, False, id="mid-shape"),
]


@pytest.mark.parametrize(
    "batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, config, tune",
    _FP8_LIGHTING_INDEXER_BENCH_PARAMS,
)
def test_fp8_lighting_indexer_bench(batch: int, seq_len: int, heads: int, index_dim: int,
                                    seq_len_kv: int, kv_group: int, clean_logits: bool,
                                    config: Optional[dict], tune: bool) -> None:
    test = Fp8LightingIndexerTest(batch, seq_len, heads, index_dim, seq_len_kv, kv_group,
                                  clean_logits, config)
    bm = Fp8LightingIndexerBenchmark(test)
    inputs = test.gen_inputs()

    op = Fp8LightingIndexerOp(batch=batch,
                              seq_len=seq_len,
                              heads=heads,
                              index_dim=index_dim,
                              seq_len_kv=seq_len_kv,
                              kv_group=kv_group,
                              clean_logits=clean_logits,
                              config=config,
                              tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
