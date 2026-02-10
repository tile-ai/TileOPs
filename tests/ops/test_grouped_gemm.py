import time

import torch
import pytest

from benchmarks import (
    GroupedGemmBenchmark,
    GroupedGemmNNBenchmark,
    GroupedGemmNTBenchmark,
    GroupedGemmTNBenchmark,
    GroupedGemmTTBenchmark,
)
from top.ops.grouped_gemm import GroupedGemmNNOp, GroupedGemmNTOp, GroupedGemmTNOp, GroupedGemmTTOp


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype, tune",
    [
        (16384, 4, 4864, 4096, torch.float16, False),
    ],
)
def test_grouped_gemm_nt(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                         tune: bool):
    op = GroupedGemmNTOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = GroupedGemmNTBenchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype, tune",
    [
        (16384, 4, 4864, 4096, torch.float16, False),
    ],
)
def test_grouped_gemm_nn(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                         tune: bool):
    op = GroupedGemmNNOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = GroupedGemmNNBenchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype, tune",
    [
        (16384, 4, 4864, 4096, torch.float16, False),
    ],
)
def test_grouped_gemm_tn(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                         tune: bool):
    op = GroupedGemmTNOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = GroupedGemmTNBenchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype, tune",
    [
        (16384, 4, 4864, 4096, torch.float16, False),
    ],
)
def test_grouped_gemm_tt(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                         tune: bool):
    op = GroupedGemmTTOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = GroupedGemmTTBenchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype, tune",
    [
        (16384, 4, 4864, 4096, torch.float16, False),
    ],
)
def test_grouped_gemm_complete(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype,
                               tune: bool):
    from top.functions.grouped_gemm import GroupedGemmFunc

    op = GroupedGemmFunc(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = GroupedGemmBenchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()

    for _ in range(1):
        op(*inputs)
        torch.cuda.synchronize()
    num_iterations = 1
    total_latency = 0.0

    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        op(*inputs)
        torch.cuda.synchronize()
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        total_latency += latency_ms

    avg_latency_ms = total_latency / num_iterations
    total_flops = benchmark.total_flops
    tflops = total_flops / (avg_latency_ms * 1e9)
    print(f"grouped_gemm_fn latency: {avg_latency_ms:.2f} ms")
    print(f"grouped_gemm_fn TFlops: {tflops:.2f} TFlops")
    total_memory = benchmark.total_memory
    bandwidth = total_memory / (avg_latency_ms * 1e6)
    print(f"grouped_gemm_fn Bandwidth: {bandwidth:.2f} GB/s")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
