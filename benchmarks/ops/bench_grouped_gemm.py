"""Benchmarks for the grouped GEMM op.

Workload shapes, dtypes, and transpose layouts come from the ops manifest;
per-variant roofline FLOP and byte counts come from the op's
``eval_roofline()`` via :class:`ManifestBenchmark`. The composed
forward+backward case keeps a local roofline because it aggregates four
GEMM launches, which no single manifest workload describes.
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops import GroupedGemmOp
from workloads.grouped_gemm import (
    GroupedGemmCompleteTest,
    GroupedGemmTest,
)

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


def _workload_params(workloads: list, keys: tuple) -> list:
    """Turn manifest workload dicts into pytest params.

    First workload is marked ``smoke``, the rest ``full``. Keys ending in
    ``dtype`` are resolved to ``torch.dtype`` values.
    """
    params = []
    for i, w in enumerate(workloads):
        args = [getattr(torch, w[k]) if k.endswith("dtype") else w[k] for k in keys]
        params.append(
            pytest.param(
                *args,
                marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
                id=w["label"],
            )
        )
    return params


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


# Test functions

_GROUPED_GEMM_OP = "GroupedGemmOp"
_GROUPED_GEMM_PARAMS = _workload_params(
    load_workloads(_GROUPED_GEMM_OP),
    ("batch_sum", "batch_count", "n", "k", "dtype", "transpose_a", "transpose_b"),
)


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b",
    _GROUPED_GEMM_PARAMS,
)
def test_grouped_gemm_bench(batch_sum: int, batch_count: int, N: int, K: int,
                            dtype: torch.dtype, transpose_a: bool,
                            transpose_b: bool) -> None:
    layout = ("T" if transpose_a else "N") + ("T" if transpose_b else "N")
    name = f"grouped_gemm_{layout.lower()}"

    test = _GroupedGemmTestBaseline(batch_sum, batch_count, N, K, dtype, transpose_a, transpose_b)
    inputs = test.gen_inputs()

    op = GroupedGemmOp(transpose_a=transpose_a, transpose_b=transpose_b, tune=_TUNE)
    bm = ManifestBenchmark(_GROUPED_GEMM_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(name, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(name, locals(), result_bl, tag="torch-ref")


# Complete (GroupedGemmFunc) benchmark: forward plus both backward GEMMs.

class GroupedGemmCompleteBenchmark(BenchmarkBase[GroupedGemmCompleteTest]):

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


# The composed case reuses the first manifest workload's geometry; the layout
# sweep below is the fwd+bwd chain, not a manifest workload.
_GROUPED_GEMM_COMPLETE_PARAMS = _workload_params(
    load_workloads(_GROUPED_GEMM_OP)[:1],
    ("batch_sum", "batch_count", "n", "k", "dtype"),
)


@pytest.mark.parametrize("batch_sum, batch_count, N, K, dtype", _GROUPED_GEMM_COMPLETE_PARAMS)
def test_grouped_gemm_complete_bench(batch_sum: int, batch_count: int, N: int, K: int,
                                     dtype: torch.dtype) -> None:
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
        op = GroupedGemmOp(transpose_a=transpose_a, transpose_b=transpose_b, tune=_TUNE)
        tileops_results.append(bm.profile(op, *inputs))
        baseline_results.append(bm.profile(variant_test.ref_program, *inputs))

    result = _combine_results(bm, *tileops_results)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = _combine_results(bm, *baseline_results)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
