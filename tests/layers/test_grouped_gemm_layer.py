import argparse
import pytest
import torch

from benchmarks import GroupedGemmBenchmark
from top.layers import GroupedGemmLayer
from top.utils import str2dtype


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sum', type=int, default=16384, help='sum of batch_size_list')
    parser.add_argument('--batch_count', type=int, default=4, help='length of batch_size_list')
    parser.add_argument('--N', type=int, default=4864, help='head dim')
    parser.add_argument('--K', type=int, default=8192, help='num heads')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    args = parser.parse_args()

    test_grouped_gemm_layer(args.batch_sum, args.batch_count, args.N, args.K, str2dtype[args.dtype])
