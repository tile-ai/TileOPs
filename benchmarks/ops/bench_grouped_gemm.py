from typing import Optional

from tests.ops.test_grouped_gemm import (
    GroupedGemmNTTest,
    GroupedGemmNNTest,
    GroupedGemmTNTest,
    GroupedGemmTTTest,
    GroupedGemmCompleteTest,
)
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


# ---------------------------------------------------------------------------
# NT benchmark
# ---------------------------------------------------------------------------

class GroupedGemmNTBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return 2.0 * t.batch_sum * t.K * t.N

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        memory_A = t.batch_sum * t.K * t.dtype.itemsize
        memory_B = t.batch_count * t.K * t.N * t.dtype.itemsize
        memory_Y = t.batch_sum * t.N * t.dtype.itemsize
        return memory_A + memory_B + memory_Y


# ---------------------------------------------------------------------------
# NN benchmark
# ---------------------------------------------------------------------------

class GroupedGemmNNBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return 2.0 * t.batch_sum * t.K * t.N

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        memory_A = t.batch_sum * t.N * t.dtype.itemsize
        memory_B = t.batch_count * t.N * t.K * t.dtype.itemsize
        memory_Y = t.batch_sum * t.K * t.dtype.itemsize
        return memory_A + memory_B + memory_Y


# ---------------------------------------------------------------------------
# TN benchmark
# ---------------------------------------------------------------------------

class GroupedGemmTNBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return 2.0 * t.batch_sum * t.K * t.N

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        memory_A = t.K * t.batch_sum * t.dtype.itemsize
        memory_B = t.batch_sum * t.N * t.dtype.itemsize
        memory_Y = t.batch_count * t.K * t.N * t.dtype.itemsize
        return memory_A + memory_B + memory_Y


# ---------------------------------------------------------------------------
# TT benchmark
# ---------------------------------------------------------------------------

class GroupedGemmTTBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        return 2.0 * t.batch_sum * t.N * t.K

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        memory_A = t.batch_sum * t.N * t.dtype.itemsize
        memory_B = t.K * t.batch_sum * t.dtype.itemsize
        memory_Y = t.batch_count * t.N * t.K * t.dtype.itemsize
        return memory_A + memory_B + memory_Y


# ---------------------------------------------------------------------------
# Complete (GroupedGemmFunc) benchmark
# ---------------------------------------------------------------------------

class GroupedGemmCompleteBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Forward (NT) + backward dA (NN) + backward dB (TN)
        return 3 * 2.0 * t.batch_sum * t.K * t.N

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Forward NT memory
        mem_nt = (t.batch_sum * t.K + t.batch_count * t.K * t.N + t.batch_sum * t.N)
        # Backward dA NN memory
        mem_nn = (t.batch_sum * t.N + t.batch_count * t.N * t.K + t.batch_sum * t.K)
        # Backward dB TN memory
        mem_tn = (t.K * t.batch_sum + t.batch_sum * t.N + t.batch_count * t.K * t.N)
        return (mem_nt + mem_nn + mem_tn) * t.dtype.itemsize
