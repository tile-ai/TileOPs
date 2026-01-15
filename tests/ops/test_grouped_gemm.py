import argparse
import time

import torch

from benchmarks import (
    GroupedGemmBenchmark,
    GroupedGemmNNBenchmark,
    GroupedGemmNTBenchmark,
    GroupedGemmTNBenchmark,
    GroupedGemmTTBenchmark,
)
from top.ops.grouped_gemm import GroupedGemmNNOp, GroupedGemmNTOp, GroupedGemmTNOp, GroupedGemmTTOp
from top.utils import str2dtype


def test_grouped_gemm_nt(batch_sum, batch_count, N, K, dtype, tune=False):
    op = GroupedGemmNTOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = GroupedGemmNTBenchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


def test_grouped_gemm_nn(batch_sum, batch_count, N, K, dtype, tune=False):
    op = GroupedGemmNNOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = GroupedGemmNNBenchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


def test_grouped_gemm_tn(batch_sum, batch_count, N, K, dtype, tune=False):
    op = GroupedGemmTNOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = GroupedGemmTNBenchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


def test_grouped_gemm_tt(batch_sum, batch_count, N, K, dtype, tune=False):
    op = GroupedGemmTTOp(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = GroupedGemmTTBenchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


def test_grouped_gemm_complete(batch_sum, batch_count, N, K, dtype, tune=False):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sum', type=int, default=16384, help='sum of batch_size_list')
    parser.add_argument('--batch_count', type=int, default=4, help='length of batch_size_list')
    parser.add_argument('--N', type=int, default=4864, help='head dim')
    parser.add_argument('--K', type=int, default=4096, help='num heads')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store', default=False, help='enable autotune')
    args = parser.parse_args()

    print("Testing grouped_gemm_nt (forward)...")
    test_grouped_gemm_nt(args.batch_sum, args.batch_count, args.N, args.K, str2dtype[args.dtype],
                         args.tune)
    print("Testing grouped_gemm_nn (backward dA)...")
    test_grouped_gemm_nn(args.batch_sum, args.batch_count, args.N, args.K, str2dtype[args.dtype],
                         args.tune)
    print("Testing grouped_gemm_tn (backward dB)...")
    test_grouped_gemm_tn(args.batch_sum, args.batch_count, args.N, args.K, str2dtype[args.dtype],
                         args.tune)
    print("Testing grouped_gemm_tt (backward dB)...")
    test_grouped_gemm_tt(args.batch_sum, args.batch_count, args.N, args.K, str2dtype[args.dtype],
                         args.tune)
    print("Testing complete grouped_gemm function...")
    test_grouped_gemm_complete(args.batch_sum, args.batch_count, args.N, args.K,
                               str2dtype[args.dtype], args.tune)
