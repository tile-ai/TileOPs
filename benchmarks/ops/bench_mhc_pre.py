from typing import Optional

from tests.ops.test_mhc_pre import MhcPreTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


class MhcPreBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops = 2 * t.batch * (
            (t.n_expand * t.n_expand * t.c_x * t.c_x) *
            (t.n_expand * t.n_expand + 2 * t.n_expand) + t.n_expand * t.c_x)
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return (t.n_expand * 3 + 1) * t.c_x + (t.n_expand * t.c_x) * (
            t.n_expand * t.n_expand + 2 * t.n_expand)
