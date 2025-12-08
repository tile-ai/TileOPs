import argparse
from top.ops.grouped_gemm import grouped_gemm_nt, grouped_gemm_nn, grouped_gemm_tn, grouped_gemm_tt
from top.utils import str2dtype
from benchmarks import grouped_gemm_nt_benchmark, grouped_gemm_nn_benchmark, grouped_gemm_tn_benchmark, grouped_gemm_tt_benchmark, grouped_gemm_benchmark
import time
import torch


def test_grouped_gemm_nt(batch_sum, batch_count, N, K, dtype, tune=False):
    op = grouped_gemm_nt(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = grouped_gemm_nt_benchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)

def test_grouped_gemm_nn(batch_sum, batch_count, N, K, dtype, tune=False):
    op = grouped_gemm_nn(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = grouped_gemm_nn_benchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)

def test_grouped_gemm_tn(batch_sum, batch_count, N, K, dtype, tune=False):
    op = grouped_gemm_tn(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = grouped_gemm_tn_benchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)

def test_grouped_gemm_tt(batch_sum, batch_count, N, K, dtype, tune=False):
    op = grouped_gemm_tt(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = grouped_gemm_tt_benchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)

def test_grouped_gemm_complete(batch_sum, batch_count, N, K, dtype, tune=False):
    from top.functions.grouped_gemm import grouped_gemm_fn
    
    op = grouped_gemm_fn(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = grouped_gemm_benchmark(batch_sum, batch_count, N, K, dtype)

    inputs = benchmark.gen_inputs()

    for _ in range(1):
        results = op(*inputs)
        torch.cuda.synchronize()
    num_iterations = 1
    total_latency = 0.0
    
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        results = op(*inputs)
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
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store', default=False, help='enable autotune')
    # parser.add_argument('--test', type=str, default='all', choices=['all', 'nt', 'nn', 'tn', 'complete'], help='which test to run')
    args = parser.parse_args()

    print("Testing grouped_gemm_nt (forward)...")
    test_grouped_gemm_nt(args.batch_sum, args.batch_count, args.N, args.K, str2dtype[args.dtype], args.tune)
    print("Testing grouped_gemm_nn (backward dA)...")
    test_grouped_gemm_nn(args.batch_sum, args.batch_count, args.N, args.K, str2dtype[args.dtype], args.tune)
    print("Testing grouped_gemm_tn (backward dB)...")
    test_grouped_gemm_tn(args.batch_sum, args.batch_count, args.N, args.K, str2dtype[args.dtype], args.tune)
    print("Testing grouped_gemm_tt (backward dB)...")
    test_grouped_gemm_tt(args.batch_sum, args.batch_count, args.N, args.K, str2dtype[args.dtype], args.tune)
    print("Testing complete grouped_gemm function...")
    test_grouped_gemm_complete(args.batch_sum, args.batch_count, args.N, args.K, str2dtype[args.dtype], args.tune)
