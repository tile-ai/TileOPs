import argparse
from top import grouped_gemm_fn
from top.utils import str2dtype
import torch
import math
from benchmarks import grouped_gemm_benchmark


def test_grouped_gemm_fn(batch_sizes_list, N, K, padding_M, dtype, tune=False):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count - 1):
        batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                         math.ceil((batch_sizes_list[i] + 1) / padding_M) *
                                         padding_M)

    fn = grouped_gemm_fn(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = grouped_gemm_benchmark(batch_sum, batch_count, N, K, dtype)
    inputs = benchmark.gen_inputs()
    # print("Testing forward propagation...")
    output = fn(*inputs)
    # print("Testing backward propagation...")
    A, B, batch_sizes, batch_offsets, batch_padded_offsets = inputs
    A.requires_grad = True
    B.requires_grad = True
    output = fn(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    assert A.grad is not None, "Gradient for A should not be None"
    assert B.grad is not None, "Gradient for B should not be None"
    print(f"A.grad shape: {A.grad.shape}")
    print(f"B.grad shape: {B.grad.shape}")
    print(f"A.grad range: [{A.grad.min():.6f}, {A.grad.max():.6f}]")
    print(f"B.grad range: [{B.grad.min():.6f}, {B.grad.max():.6f}]")
    print("Function test passed!")
    print("Profiling...")
    benchmark.profile(fn, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_sizes_list', type=str, default="4096,4096,4096,4096", help='batch size list')
    parser.add_argument('--N', type=int, default=4864, help='N')
    parser.add_argument('--K', type=int, default=8192, help='K')
    parser.add_argument('--padding_M', type=int, default=128, help='padding M')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    batch_sizes_list = [int(x) for x in args.batch_sizes_list.split(',')]

    test_grouped_gemm_fn(
        batch_sizes_list=batch_sizes_list,
        N=args.N,
        K=args.K,
        padding_M=args.padding_M,
        dtype=str2dtype[args.dtype],
        tune=args.tune)
