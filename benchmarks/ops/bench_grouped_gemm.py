from typing import Optional

import pytest
import torch

from tests.ops.test_grouped_gemm import (
    GroupedGemmNTFixture,
    GroupedGemmNTTest,
    GroupedGemmNNFixture,
    GroupedGemmNNTest,
    GroupedGemmTNFixture,
    GroupedGemmTNTest,
    GroupedGemmTTFixture,
    GroupedGemmTTTest,
    GroupedGemmCompleteFixture,
    GroupedGemmCompleteTest,
)
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import GroupedGemmNTOp, GroupedGemmNNOp, GroupedGemmTNOp, GroupedGemmTTOp


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


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

@GroupedGemmNTFixture
def test_grouped_gemm_nt_bench(batch_sum: int, batch_count: int, N: int, K: int,
                               dtype: torch.dtype, tune: bool) -> None:
    test = GroupedGemmNTTest(batch_sum, batch_count, N, K, dtype)
    bm = GroupedGemmNTBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedGemmNTOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("grouped_gemm_nt", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("grouped_gemm_nt", locals(), result_bl, tag="baseline")


@GroupedGemmNNFixture
def test_grouped_gemm_nn_bench(batch_sum: int, batch_count: int, N: int, K: int,
                               dtype: torch.dtype, tune: bool) -> None:
    test = GroupedGemmNNTest(batch_sum, batch_count, N, K, dtype)
    bm = GroupedGemmNNBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedGemmNNOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("grouped_gemm_nn", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("grouped_gemm_nn", locals(), result_bl, tag="baseline")


@GroupedGemmTNFixture
def test_grouped_gemm_tn_bench(batch_sum: int, batch_count: int, N: int, K: int,
                               dtype: torch.dtype, tune: bool) -> None:
    test = GroupedGemmTNTest(batch_sum, batch_count, N, K, dtype)
    bm = GroupedGemmTNBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedGemmTNOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("grouped_gemm_tn", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("grouped_gemm_tn", locals(), result_bl, tag="baseline")


@GroupedGemmTTFixture
def test_grouped_gemm_tt_bench(batch_sum: int, batch_count: int, N: int, K: int,
                               dtype: torch.dtype, tune: bool) -> None:
    test = GroupedGemmTTTest(batch_sum, batch_count, N, K, dtype)
    bm = GroupedGemmTTBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedGemmTTOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("grouped_gemm_tt", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("grouped_gemm_tt", locals(), result_bl, tag="baseline")


def _combine_results(bm: GroupedGemmCompleteBenchmark, *results: dict) -> dict:
    """Combine latencies from multiple profiles into a single result."""
    total_latency = sum(r["latency_ms"] for r in results)
    combined = {"latency_ms": total_latency}
    flops = bm.calculate_flops()
    if flops is not None:
        combined["tflops"] = flops / total_latency * 1e-9
    memory = bm.calculate_memory()
    if memory is not None:
        combined["bandwidth_gbs"] = memory / total_latency * 1e-9
    return combined


@GroupedGemmCompleteFixture
def test_grouped_gemm_complete_bench(batch_sum: int, batch_count: int, N: int, K: int,
                                     dtype: torch.dtype, tune: bool) -> None:
    test = GroupedGemmCompleteTest(batch_sum, batch_count, N, K, dtype)
    bm = GroupedGemmCompleteBenchmark(test)

    # Profile NT (forward)
    nt_test = GroupedGemmNTTest(batch_sum, batch_count, N, K, dtype)
    nt_inputs = nt_test.gen_inputs()
    nt_op = GroupedGemmNTOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    result_nt = bm.profile(nt_op, *nt_inputs)

    # Profile NN (backward dA)
    nn_test = GroupedGemmNNTest(batch_sum, batch_count, N, K, dtype)
    nn_inputs = nn_test.gen_inputs()
    nn_op = GroupedGemmNNOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    result_nn = bm.profile(nn_op, *nn_inputs)

    # Profile TN (backward dB)
    tn_test = GroupedGemmTNTest(batch_sum, batch_count, N, K, dtype)
    tn_inputs = tn_test.gen_inputs()
    tn_op = GroupedGemmTNOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    result_tn = bm.profile(tn_op, *tn_inputs)

    result = _combine_results(bm, result_nt, result_nn, result_tn)
    BenchmarkReport.record("grouped_gemm_complete", locals(), result, tag="tileops")

    # Profile baselines
    result_bl = _combine_results(
        bm,
        bm.profile(nt_test.ref_program, *nt_inputs),
        bm.profile(nn_test.ref_program, *nn_inputs),
        bm.profile(tn_test.ref_program, *tn_inputs),
    )
    BenchmarkReport.record("grouped_gemm_complete", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
