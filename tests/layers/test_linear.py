import argparse
import torch
from top.layers import LinearLayer
from top.utils import str2dtype


def test_linear(m, n, k, dtype, tune=False):
    linear_layer = LinearLayer(m, n, k, dtype=dtype, tune=tune)
    input = torch.randn(m, k, dtype=dtype, device='cuda', requires_grad=True)

    output = linear_layer(input)

    loss = output.sum()
    loss.backward()

    print("Output shape:", output.shape)
    print("Gradient shape:", input.grad.shape)


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
