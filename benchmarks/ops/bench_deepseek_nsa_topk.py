from typing import Optional

from tests.ops.test_deepseek_nsa_topk import NsaTopkTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


class NsaTopkBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Step 1 (LSE) + Step 2 (Scores)
        # Total: c_seq_len * head_kv * 2 * (2 * group * dim * (c_seq_len / (2 * bs)))
        return (2 * t.heads * t.dim * t.c_seq_len**2) // t.bs

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # q: read once, k_cmp: read twice per preceding block per token, block_indices: write once
        q_read = t.heads * t.c_seq_len * t.dim * t.dtype.itemsize
        k_read = (t.head_kv * t.dim * t.c_seq_len**2 * t.dtype.itemsize) // t.bs
        indices_write = t.c_seq_len * t.head_kv * t.selected_block_num * 4
        return q_read + k_read + indices_write
