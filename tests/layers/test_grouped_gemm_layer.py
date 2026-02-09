import sys
import pytest
import torch

from benchmarks import GroupedGemmBenchmark
from top.layers import GroupedGemmLayer


@pytest.mark.parametrize(
    "batch_sum, batch_count, N, K, dtype",
    [
        (16384, 4, 4864, 8192, torch.float16),
    ],
)
def test_grouped_gemm_layer(batch_sum: int, batch_count: int, N: int, K: int, dtype: torch.dtype):
    grouped_gemm = GroupedGemmLayer(batch_sum, batch_count, N, K, dtype)
    benchmark = GroupedGemmBenchmark(batch_sum, batch_count, N, K, dtype)
    inputs = benchmark.gen_inputs()
    # enable gradients for A and B
    inputs = list(inputs)
    inputs[0] = inputs[0].clone().detach().requires_grad_(True)
    inputs[1] = inputs[1].clone().detach().requires_grad_(True)
    inputs = tuple(inputs)
    benchmark.check_fn(grouped_gemm, *inputs)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
