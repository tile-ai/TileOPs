import time

import pytest
import torch
import torch.nn as nn


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
    pytest.main([__file__, "-vvs"])
