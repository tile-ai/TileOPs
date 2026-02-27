from typing import Optional

from tests.ops.test_deepseek_nsa_gqa_window_sliding import GqaWindowSlidingTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


class GqaWindowSlidingBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        total_flops = 2.0 * t.heads * t.uq * t.ukv * t.dim * 2
        if t.is_causal:
            total_flops *= 0.5
        return total_flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        head_kv = t.heads // t.groups
        q_memory = t.uq * t.heads * t.dim * t.dtype.itemsize
        k_memory = t.ukv * head_kv * t.dim * t.dtype.itemsize
        v_memory = t.ukv * head_kv * t.dim * t.dtype.itemsize
        output_memory = t.uq * t.heads * t.dim * t.dtype.itemsize
        return q_memory + k_memory + v_memory + output_memory
