from typing import Optional

from tests.ops.test_deepseek_nsa_fwd import NsaFwdTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


class NsaFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_token = 4 * t.dim * t.selected_blocks * t.block_size
        return flops_per_token * t.c_seq_len * t.heads

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # q, k, v, output, block_indices, block_counts, offsets, token_indices
        # ignore block counts, offsets and token_indices memory
        q_memory = t.heads * t.c_seq_len * t.dim * t.dtype.itemsize
        k_memory = t.head_kv * t.c_seq_len * t.dim * t.dtype.itemsize
        v_memory = t.head_kv * t.c_seq_len * t.dim * t.dtype.itemsize
        output_memory = t.heads * t.c_seq_len * t.dim * t.dtype.itemsize
        block_indices_memory = t.head_kv * t.c_seq_len * t.selected_blocks * 4
        return (q_memory + k_memory + v_memory + output_memory + block_indices_memory)
