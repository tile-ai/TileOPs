import time

import pytest
import torch


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


class PyTorchGroupedGEMM:

    def __init__(self):
        pass

    def grouped_gemm_nt(self, a, b, batch_sizes):
        outputs = []
        start = 0
        for i, size in enumerate(batch_sizes):
            end = start + size
            part_A = a[start:end]
            part_B = b[i]
            output = torch.mm(part_A, part_B.transpose(0, 1))
            outputs.append(output)
            start = end
        return torch.cat(outputs, dim=0)

    def grouped_gemm_nn(self, a, b, batch_sizes):
        outputs = []
        start = 0
        for i, size in enumerate(batch_sizes):
            end = start + size
            part_A = a[start:end]
            part_B = b[i].transpose(0, 1)
            output = torch.mm(part_A, part_B)
            outputs.append(output)
            start = end
        return torch.cat(outputs, dim=0)

    def grouped_gemm_tn(self, a, b, batch_sizes):
        batch_count = len(batch_sizes)
        N, K = a.shape[1], b.shape[1]
        outputs = torch.zeros(batch_count, N, K, device=a.device, dtype=a.dtype)

        start = 0
        for i, size in enumerate(batch_sizes):
            end = start + size
            part_A = a[start:end]
            part_B = b[start:end]
            outputs[i] = torch.mm(part_A.transpose(0, 1), part_B)
            start = end
        return outputs

    def grouped_gemm_tt(self, a, b, batch_sizes):
        batch_count = len(batch_sizes)
        N, K = a.shape[1], b.shape[0]
        outputs = torch.zeros(batch_count, N, K, device=a.device, dtype=a.dtype)

        start = 0
        for i, size in enumerate(batch_sizes):
            end = start + size
            part_A = a[start:end]
            part_B = b[:, start:end]
            outputs[i] = torch.mm(part_A.transpose(0, 1), part_B.transpose(0, 1))
            start = end
        return outputs


def calculate_flops(batch_sizes, k, n):
    total_flops = 0
    for batch_size in batch_sizes:
        total_flops += 2 * batch_size * k * n
    return total_flops


def benchmark_single(gemm, a, b, batch_sizes, num_iter=100):
    for _ in range(10):
        with torch.no_grad():
            _ = gemm(a, b, batch_sizes)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iter):
        with torch.no_grad():
            _ = gemm(a, b, batch_sizes)
    torch.cuda.synchronize()

    return (time.time() - start_time) / num_iter


@pytest.mark.parametrize(
    "batch_sum, batch_count, k, n, dtype",
    [
        (4096, 4, 8192, 4864, torch.float16),
    ],
)
def test_all_grouped_gemm(batch_sum, batch_count, k, n, dtype):
    print("=" * 70)
    print("PyTorch Grouped GEMM Performance Test")
    print("=" * 70)
    print(f"Config: batch_sum={batch_sum}, batch_count={batch_count}, K={k}, N={n}, dtype={dtype}")

    base_size = batch_sum // batch_count
    remainder = batch_sum % batch_count
    batch_sizes = [base_size] * batch_count
    for i in range(remainder):
        batch_sizes[i] += 1
    print(f"Batch sizes: {batch_sizes}")
    print("-" * 70)

    gemm = PyTorchGroupedGEMM()
    device = 'cuda'

    print("\n1. Testing NT layout (forward): A[batch_sum, K] @ B[batch_count, N, K]^T")
    print("-" * 50)
    A_nt = torch.randn(batch_sum, k, device=device, dtype=dtype)
    B_nt = torch.randn(batch_count, n, k, device=device, dtype=dtype)

    nt_time = benchmark_single(gemm.grouped_gemm_nt, A_nt, B_nt, batch_sizes)
    nt_flops = calculate_flops(batch_sizes, k, n)
    nt_tflops = (nt_flops / 1e12) / nt_time
    print(f"Time: {nt_time*1000:.2f} ms")
    print(f"Performance: {nt_tflops:.2f} TFLOPS")

    print("\n2. Testing NN layout (backward dA): A[batch_sum, N] @ B[batch_count, K, N]")
    print("-" * 50)
    A_nn = torch.randn(batch_sum, n, device=device, dtype=dtype)
    B_nn = torch.randn(batch_count, k, n, device=device, dtype=dtype)

    nn_time = benchmark_single(gemm.grouped_gemm_nn, A_nn, B_nn, batch_sizes)
    nn_flops = calculate_flops(batch_sizes, k, n)
    nn_tflops = (nn_flops / 1e12) / nn_time
    print(f"Time: {nn_time*1000:.2f} ms")
    print(f"Performance: {nn_tflops:.2f} TFLOPS")

    print("\n3. Testing TN layout (backward dB): A[batch_sum, N]^T @ B[batch_sum, K]")
    print("-" * 50)
    A_tn = torch.randn(batch_sum, n, device=device, dtype=dtype)
    B_tn = torch.randn(batch_sum, k, device=device, dtype=dtype)

    tn_time = benchmark_single(gemm.grouped_gemm_tn, A_tn, B_tn, batch_sizes)
    tn_flops = calculate_flops(batch_sizes, k, n)
    tn_tflops = (tn_flops / 1e12) / tn_time
    print(f"Time: {tn_time*1000:.2f} ms")
    print(f"Performance: {tn_tflops:.2f} TFLOPS")

    print("\n4. Testing TT layout (backward dB): A[batch_sum, N]^T @ (B[K, batch_sum]^T)^T")
    print("-" * 50)
    A_tt = torch.randn(batch_sum, n, device=device, dtype=dtype)
    B_tt = torch.randn(k, batch_sum, device=device, dtype=dtype)

    tt_time = benchmark_single(gemm.grouped_gemm_tt, A_tt, B_tt, batch_sizes)
    tt_flops = calculate_flops(batch_sizes, k, n)
    tt_tflops = (tt_flops / 1e12) / tt_time
    print(f"Time: {tt_time*1000:.2f} ms")
    print(f"Performance: {tt_tflops:.2f} TFLOPS")

    print("\n5. Testing Combined execution with dependencies")
    print("-" * 50)
    print("Flow: NT(forward) -> NN(dA) -> TN(dB) -> TT(dB alternative)")

    A = torch.randn(batch_sum, k, device=device, dtype=dtype)
    B = torch.randn(batch_count, n, k, device=device, dtype=dtype)
    grad_output = torch.ones_like(torch.randn(batch_sum, n, device=device, dtype=dtype))
    B_nn_combined = B.transpose(1, 2).contiguous()
    B_tt_combined = A.transpose(0, 1).contiguous()

    for _ in range(10):
        with torch.no_grad():
            gemm.grouped_gemm_nt(A, B, batch_sizes)
            gemm.grouped_gemm_nn(grad_output, B_nn_combined, batch_sizes)
            gemm.grouped_gemm_tn(grad_output, A, batch_sizes)
            gemm.grouped_gemm_tt(grad_output, B_tt_combined, batch_sizes)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            gemm.grouped_gemm_nt(A, B, batch_sizes)
            gemm.grouped_gemm_nn(grad_output, B_nn_combined, batch_sizes)
            gemm.grouped_gemm_tn(grad_output, A, batch_sizes)
            gemm.grouped_gemm_tt(grad_output, B_tt_combined, batch_sizes)
    torch.cuda.synchronize()

    combined_time = (time.time() - start_time) / 100
    total_flops = nt_flops + nn_flops + tn_flops + tt_flops
    combined_tflops = (total_flops / 1e12) / combined_time

    print(f"Total time: {combined_time*1000:.2f} ms")
    print(f"Total FLOPs: {total_flops/1e12:.2f} TFLOPs")
    print(f"Average performance: {combined_tflops:.2f} TFLOPS")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Layout':<15} {'Time (ms)':<12} {'TFLOPS':<12}")
    print("-" * 70)
    print(f"{'NT (forward)':<15} {nt_time*1000:<12.2f} {nt_tflops:<12.2f}")
    print(f"{'NN (dA)':<15} {nn_time*1000:<12.2f} {nn_tflops:<12.2f}")
    print(f"{'TN (dB)':<15} {tn_time*1000:<12.2f} {tn_tflops:<12.2f}")
    print(f"{'TT (dB)':<15} {tt_time*1000:<12.2f} {tt_tflops:<12.2f}")
    print(f"{'Combined':<15} {combined_time*1000:<12.2f} {combined_tflops:<12.2f}")
    print("=" * 70)


if __name__ == "__main__":
    test_all_grouped_gemm(batch=4096, batch_count=4, k=8192, n=4864, dtype=torch.float16)
