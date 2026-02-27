from typing import Optional

import torch

from tests.ops.test_fp8_lighting_indexer import Fp8LightingIndexerTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


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
