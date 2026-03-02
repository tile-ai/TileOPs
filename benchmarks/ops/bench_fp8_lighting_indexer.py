from typing import Optional

import torch
import pytest

from tests.ops.test_fp8_lighting_indexer import Fp8LightingIndexerFixture, Fp8LightingIndexerTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
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

        index_q_memory = t.seq_len * t.heads * t.index_dim * dtype.itemsize
        index_k_memory = t.seq_len_kv * t.index_dim * dtype.itemsize
        index_k_scale_memory = t.seq_len_kv * accum_dtype.itemsize
        logits_memory = t.seq_len * t.seq_len_kv * accum_dtype.itemsize
        weights_memory = t.seq_len * t.heads * accum_dtype.itemsize
        cu_seqlens_ks_memory = t.seq_len * index_dtype.itemsize
        cu_seqlens_ke_memory = t.seq_len * index_dtype.itemsize

        return (index_q_memory + index_k_memory + index_k_scale_memory + logits_memory +
                weights_memory + cu_seqlens_ks_memory + cu_seqlens_ke_memory)


@Fp8LightingIndexerFixture
def test_fp8_lighting_indexer_bench(seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                                    clean_logits: bool, config: Optional[dict],
                                    tune: bool) -> None:
    test = Fp8LightingIndexerTest(seq_len, heads, index_dim, seq_len_kv, clean_logits, config)
    bm = Fp8LightingIndexerBenchmark(test)
    inputs = test.gen_inputs()

    op = Fp8LightingIndexerOp(seq_len, heads, index_dim, seq_len_kv, clean_logits, config,
                               tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("fp8_lighting_indexer", locals(), result, tag="tileops")

    try:
        result_bl = bm.profile(test.ref_program, *inputs)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        result_bl = None
    if result_bl is not None:
        BenchmarkReport.record("fp8_lighting_indexer", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
