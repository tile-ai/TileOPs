from typing import Optional

from tests.ops.test_fp8_quant import Fp8QuantTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


class Fp8QuantBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return 2 * t.seq_len_kv * t.index_dim + t.seq_len_kv + 4 * t.seq_len_kv * t.index_dim

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return t.seq_len_kv * t.index_dim * t.in_dtype.itemsize
