import pytest
import math
import torch

from benchmarks import GroupedGemmBenchmark
from top.functions import GroupedGemmFunc


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "batch_sizes_list, N, K, padding_M, dtype, tune",
    [
        ([4096, 4096, 4096, 4096], 4864, 8192, 128, torch.float16, False),
    ],
)
def test_grouped_gemm_fn(batch_sizes_list: list, N: int, K: int, padding_M: int, dtype: torch.dtype,
                         tune: bool):
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

    fn = GroupedGemmFunc(batch_sum, batch_count, N, K, dtype, tune=tune)
    benchmark = GroupedGemmBenchmark(batch_sum, batch_count, N, K, dtype)
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
    import sys

    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
