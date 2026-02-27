from typing import Optional

from tests.ops.test_deepseek_nsa_cmp_fwd import NsaCmpFwdTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


class NsaCmpFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return (2 * t.heads * t.dim_k * t.c_seq_len**2) // t.bs

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        q_read = t.heads * t.c_seq_len * t.dim_k * t.dtype.itemsize
        k_read = (t.head_kv * t.dim_k * t.c_seq_len**2 * t.dtype.itemsize) // t.bs
        v_read = (t.head_kv * t.dim_v * t.c_seq_len**2 * t.dtype.itemsize) // t.bs
        return q_read + k_read + v_read
