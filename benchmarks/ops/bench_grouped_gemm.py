from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import GroupedGemmOp
from workloads.ops.grouped_gemm import (
    GroupedGemmCompleteTest,
    GroupedGemmTest,
)


class _GroupedGemmTestBaseline(GroupedGemmTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor,
                    batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        if not self.transpose_a:
            # NT / NN: output is (batch_sum, N)
            if self.transpose_b:
                # NT: A @ B^T
                assert A.shape[0] == sum(batch_sizes)
                assert B.shape[0] == len(batch_sizes)
                output = torch.empty((sum(batch_sizes), B.shape[1]), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[start:end] = torch.mm(A[start:end], B[i].transpose(0, 1).contiguous())
                    start = end
            else:
                # NN: A @ B
                assert A.shape[0] == sum(batch_sizes)
                assert B.shape[0] == len(batch_sizes)
                output = torch.empty((sum(batch_sizes), B.shape[2]), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[start:end] = torch.mm(A[start:end], B[i])
                    start = end
        else:
            # TN / TT: output is (batch_count, N, K)
            total_batch = int(batch_sizes.sum().item())
            assert A.shape[0] == total_batch
            N = A.shape[1]
            batch_count = len(batch_sizes)

            if self.transpose_b:
                # TT: A^T @ B^T
                K = B.shape[0]
                assert B.shape[1] == total_batch
                output = torch.zeros((batch_count, N, K), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[i] = torch.mm(A[start:end].transpose(0, 1),
                                         B[:, start:end].transpose(0, 1))
                    start = end
            else:
                # TN: A^T @ B
                K = B.shape[1]
                assert B.shape[0] == total_batch
                output = torch.zeros((batch_count, N, K), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[i] = torch.mm(A[start:end].transpose(0, 1), B[start:end])
                    start = end
        return output


class GroupedGemmBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        return 2.0 * t.batch_sum * t.K * t.N

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
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
        t = self.workload
        # Forward (NT) + backward dA (NN) + backward dB (TN)
        return 3 * 2.0 * t.batch_sum * t.K * t.N

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
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
    test = _GroupedGemmTestBaseline(batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b)
    bm = GroupedGemmBenchmark(test)
    inputs = test.gen_inputs()

    op = GroupedGemmOp(batch_sum, batch_count, N, K, dtype,
                       transpose_a=transpose_a, transpose_b=transpose_b, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(name, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(name, locals(), result_bl, tag="torch-ref")


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

_GROUPED_GEMM_BENCH_PARAMS = [
    pytest.param(16384, 4, 4864, 4096, torch.float16, False, True, True, id="nt-fp16"),
    pytest.param(16384, 4, 4864, 4096, torch.float16, False, False, True, id="nn-fp16"),
    pytest.param(16384, 4, 4864, 4096, torch.float16, True, False, True, id="tn-fp16"),
    pytest.param(16384, 4, 4864, 4096, torch.float16, True, True, True, id="tt-fp16"),
]


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b, tune",
    _GROUPED_GEMM_BENCH_PARAMS,
)
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


_GROUPED_GEMM_COMPLETE_BENCH_PARAMS = [
    pytest.param(16384, 4, 4864, 4096, torch.float16, True, id="complete-fp16"),
]


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype, tune",
    _GROUPED_GEMM_COMPLETE_BENCH_PARAMS,
)
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
        variant_test = _GroupedGemmTestBaseline(batch_sum, batch_count, N, K, dtype,
                                       transpose_a, transpose_b)
        inputs = variant_test.gen_inputs()
        op = GroupedGemmOp(batch_sum, batch_count, N, K, dtype,
                           transpose_a=transpose_a, transpose_b=transpose_b, tune=tune)
        tileops_results.append(bm.profile(op, *inputs))
        baseline_results.append(bm.profile(variant_test.ref_program, *inputs))

    result = _combine_results(bm, *tileops_results)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = _combine_results(bm, *baseline_results)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
