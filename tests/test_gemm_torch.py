import argparse
import time

import pytest
import torch
import torch.nn as nn


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


def calculate_gemm_flops(M, N, K):
    return 2.0 * M * N * K


@pytest.mark.parametrize(
    "M, N, K, dtype, num_iter",
    [
        (16384, 8192, 13824, torch.float16, 100),
    ],
)
def test_pytorch_gemm(M: int, N: int, K: int, dtype, num_iter: int):
    device = 'cuda'
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    for _ in range(5):
        with torch.no_grad():
            torch.matmul(A, B)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_iter):
        with torch.no_grad():
            torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iter
    flops = calculate_gemm_flops(M, N, K)
    tflops = (flops / 1e12) / elapsed_time
    return elapsed_time, tflops, flops


@pytest.mark.parametrize(
    "M, N, K, dtype, num_iter",
    [
        (16384, 8192, 13824, torch.float16, 100),
    ],
)
def test_cublas_gemm(M: int, N: int, K: int, dtype, num_iter: int):
    device = 'cuda'
    linear = nn.Linear(K, N, bias=False).to(device).to(dtype)
    input_tensor = torch.randn(M, K, device=device, dtype=dtype)
    for _ in range(5):
        with torch.no_grad():
            linear(input_tensor)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iter):
        with torch.no_grad():
            linear(input_tensor)
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / num_iter
    flops = calculate_gemm_flops(M, N, K)
    tflops = (flops / 1e12) / elapsed_time

    return elapsed_time, tflops, flops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GEMM Performance Benchmark')
    parser.add_argument('--M', type=int, default=16384, help='Matrix A rows')
    parser.add_argument('--N', type=int, default=8192, help='Matrix B columns')
    parser.add_argument('--K', type=int, default=13824, help='Matrix A columns / Matrix B rows')
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help='Data type')
    args = parser.parse_args()
    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}
    M = args.M
    N = args.N
    K = args.K
    dtype = dtype_map[args.dtype]
    print("=" * 60)
    print("GEMM Performance Benchmark")
    print("=" * 60)
    print("Configuration:")
    print(f"  M: {M}, N: {N}, K: {K}")
    print(f"  Data type: {dtype}")
    base_time, base_tflops, flops = test_pytorch_gemm(M, N, K, dtype)
    print("\nPyTorch torch.matmul:")
    print(f"  Time: {base_time * 1000:.4f} ms")
    print(f"  Performance: {base_tflops:.2f} TFLOPS")
    print(f"  Total FLOPs: {flops / 1e12:.2f} TFLOPs")
