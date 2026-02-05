import argparse
import time

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel_fp16(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    a_mask = (rm[:, None] < M) & (rk[None, :] < K)
    b_mask = (rk[:, None] < K) & (rn[None, :] < N)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(
            b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
        rk += BLOCK_SIZE_K
    acc = acc.to(tl.float16)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=c_mask)


def triton_gemm_fp16(A, B, block_m=64, block_n=64, block_k=32):
    assert A.shape[1] == B.shape[0], "Matrix dimensions do not match"
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    gemm_kernel_fp16[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
    )
    return C


def calculate_gemm_flops(M, N, K):
    return 2.0 * M * N * K


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "M, N, K, dtype, num_iter",
    [
        (4096, 4864, 8192, torch.float16, 100),
    ],
)
def test_benchmark_triton_gemm_fp16(M: int, N: int, K: int, dtype, num_iter: int):
    device = 'cuda'
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    block_configs = [
        (64, 64, 32),
        (64, 128, 32),
        (64, 256, 32),
        (128, 64, 32),
        (128, 128, 32),
        (128, 256, 32),
        (128, 128, 64),
        (128, 128, 128),
        (256, 64, 32),
        (256, 128, 32),
    ]
    print("Triton GEMM performance test (fp16 computation and accumulation)")
    print("=" * 50)
    results = []
    for config_idx, (block_m, block_n, block_k) in enumerate(block_configs, 1):
        try:
            for _ in range(5):
                triton_gemm_fp16(A, B, block_m, block_n, block_k)
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(num_iter):
                triton_gemm_fp16(A, B, block_m, block_n, block_k)
            torch.cuda.synchronize()
            elapsed_time = (time.time() - start_time) / num_iter
            flops = calculate_gemm_flops(M, N, K)
            tflops = (flops / 1e12) / elapsed_time
            gflops = (flops / 1e9) / elapsed_time
            results.append({
                'config': (block_m, block_n, block_k),
                'time_ms': elapsed_time * 1000,
                'tflops': tflops,
                'gflops': gflops
            })
            print(f"Config {config_idx}: Block({block_m:3d},{block_n:3d},{block_k:3d})")
            print(f"  Time: {elapsed_time * 1000:8.4f} ms")
            print(f"  Performance: {tflops:8.2f} TFLOPS")
            print("-" * 40)

        except Exception as e:
            print(
                f"Config {config_idx}: Block({block_m:3d},{block_n:3d},{block_k:3d}) - Failed: {e}")
            print("-" * 40)
    if results:
        best_result = max(results, key=lambda x: x['tflops'])
        print("\nBest config summary (fp16 accumulation):")
        print("=" * 50)
        print(f"Block size: {best_result['config']}")
        print(f"Execution time: {best_result['time_ms']:.4f} ms")
        print(f"Compute performance: {best_result['tflops']:.2f} TFLOPS")
        print(f"Total FLOPs: {calculate_gemm_flops(M, N, K) / 1e12:.2f} TFLOPs")

    return results


@pytest.mark.parametrize(
    "M, N, K, dtype",
    [
        (512, 512, 512, torch.float16),
    ],
)
def test_verify_triton_gemm_fp16(M: int, N: int, K: int, dtype):
    print("Verifying Triton GEMM correctness (fp16 accumulation)...")
    device = 'cuda'
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    with torch.no_grad():
        C_torch = torch.matmul(A, B)
    C_triton = triton_gemm_fp16(A, B)
    error = torch.max(torch.abs(C_triton - C_torch)).item()
    relative_error = torch.mean(torch.abs(C_triton - C_torch) / torch.abs(C_torch)).item()
    print(f"Max absolute error: {error:.6f}")
    print(f"Average relative error: {relative_error:.6f}")
    if error < 1e-2:
        print("✓ Triton GEMM (fp16 accumulation) results are correct")
    else:
        print("✗ Triton GEMM (fp16 accumulation) results are incorrect")
    return error < 1e-2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Triton GEMM performance test - fp16 accumulation')
    parser.add_argument('--M', type=int, default=4096, help='Number of rows in matrix A')
    parser.add_argument('--N', type=int, default=4864, help='Number of columns in matrix B')
    parser.add_argument(
        '--K', type=int, default=8192, help='Number of columns in matrix A / rows in matrix B')
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16'],
        help='Data type (only float16 supported)')
    parser.add_argument('--verify', action='store_true', help='Verify correctness')
    args = parser.parse_args()
    dtype = torch.float16
    M = args.M
    N = args.N
    K = args.K
    print("Triton GEMM standalone performance test (fp16 computation and accumulation)")
    print("=" * 60)
    print(f"Matrix dimensions: A[{M}, {K}] × B[{K}, {N}] = C[{M}, {N}]")
    print(f"Data type: {dtype} (fp16 computation and accumulation)")
    print(f"Total computation: {calculate_gemm_flops(M, N, K) / 1e12:.2f} TFLOPs")
    print()
    if args.verify:
        test_verify_triton_gemm_fp16(M, N, K, dtype)
        print()
    test_benchmark_triton_gemm_fp16(M, N, K, dtype, num_iter=100)
