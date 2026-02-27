from typing import Optional

from tests.ops.test_gemv import GemvTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


class GemvBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return 2.0 * self.test.n * self.test.k

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return (t.k + t.k * t.n + t.n) * t.dtype.itemsize
