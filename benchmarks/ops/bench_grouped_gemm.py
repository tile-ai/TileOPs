from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_grouped_gemm import (
    GroupedGemmCompleteFixture,
    GroupedGemmCompleteTest,
    GroupedGemmFixture,
    GroupedGemmTest,
)
from tileops.ops import GroupedGemmOp


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
# Helper for individual variant benchmarks
# ---------------------------------------------------------------------------

def _run_variant_bench(name: str, batch_sum: int, batch_count: int, N: int, K: int,
                       dtype: torch.dtype, transpose_a: bool, transpose_b: bool,
                       tune: bool) -> None:
    """Run tileops and baseline benchmark for a single grouped GEMM variant."""
    test = GroupedGemmTest(batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b)
    bm = GroupedGemmBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedGemmOp(batch_sum, batch_count, N, K, dtype,
                       transpose_a=transpose_a, transpose_b=transpose_b, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(name, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(name, locals(), result_bl, tag="baseline")


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

@GroupedGemmFixture
def test_grouped_gemm_bench(batch_sum: int, batch_count: int, N: int, K: int,
                            dtype: torch.dtype, transpose_a: bool, transpose_b: bool,
                            tune: bool) -> None:
    layout = ("T" if transpose_a else "N") + ("T" if transpose_b else "N")
    _run_variant_bench(f"grouped_gemm_{layout.lower()}", batch_sum, batch_count, N, K,
                       dtype, transpose_a, transpose_b, tune)


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

    # Profile forward(TT) + forward (NT) + backward dA (NN) + backward dB (TN)
    variants = [
        (True, True),    # TT
        (False, True),   # NT
        (False, False),  # NN
        (True, False),   # TN
    ]

    tileops_results = []
    baseline_results = []
    for transpose_a, transpose_b in variants:
        variant_test = GroupedGemmTest(batch_sum, batch_count, N, K, dtype,
                                       transpose_a, transpose_b)
        inputs = variant_test.gen_inputs()
        op = GroupedGemmOp(batch_sum, batch_count, N, K, dtype,
                           transpose_a=transpose_a, transpose_b=transpose_b, tune=tune)
        tileops_results.append(bm.profile(op, *inputs))
        baseline_results.append(bm.profile(variant_test.ref_program, *inputs))

    result = _combine_results(bm, *tileops_results)
    BenchmarkReport.record("grouped_gemm_complete", locals(), result, tag="tileops")

    result_bl = _combine_results(bm, *baseline_results)
    BenchmarkReport.record("grouped_gemm_complete", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
