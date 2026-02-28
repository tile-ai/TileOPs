from typing import Optional

from tests.ops.test_grouped_gemm import GroupedGemmTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


class GroupedGemmBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return 2.0 * t.batch_sum * t.K * t.N

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        if not t.transpose_a:
            # NT/NN: A(batch_sum, K) + B(batch_count, N, K) or (batch_count, K, N) + C(batch_sum, N)
            memory_A = t.batch_sum * t.K * t.dtype.itemsize
            memory_B = t.batch_count * t.N * t.K * t.dtype.itemsize
            memory_C = t.batch_sum * t.N * t.dtype.itemsize
        else:
            # TN/TT: A(batch_sum, N) + C(batch_count, N, K)
            memory_A = t.batch_sum * t.N * t.dtype.itemsize
            memory_C = t.batch_count * t.N * t.K * t.dtype.itemsize
            if t.transpose_b:
                # TT: B(K, batch_sum)
                memory_B = t.K * t.batch_sum * t.dtype.itemsize
            else:
                # TN: B(batch_sum, K)
                memory_B = t.batch_sum * t.K * t.dtype.itemsize
        return memory_A + memory_B + memory_C
