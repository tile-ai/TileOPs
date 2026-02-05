import argparse
import pytest
import torch

from top.layers import LinearLayer
from top.utils import str2dtype


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "m, n, k, dtype, tune",
    [
        (1024, 1024, 1024, torch.float16, False),
    ],
)
def test_linear(m: int, n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    linear_layer = LinearLayer(m, n, k, dtype=dtype, tune=tune)
    input_tensor = torch.randn(m, k, dtype=dtype, device='cuda', requires_grad=True)

    output = linear_layer(input_tensor)

    loss = output.sum()
    loss.backward()

    print("Output shape:", output.shape)
    print("Gradient shape:", input_tensor.grad.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=1024, help='M')
    parser.add_argument('--N', type=int, default=1024, help='N')
    parser.add_argument('--K', type=int, default=1024, help='K')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_linear(args.M, args.N, args.K, str2dtype[args.dtype], args.tune)
