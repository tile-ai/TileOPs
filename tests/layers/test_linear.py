import argparse
from top.layers import Linear
from top.utils import str2dtype
from benchmarks import linear_benchmark


def test_linear(M, N, K, dtype, tune=False):
    linear_layer = Linear(M, N, K, dtype=dtype, tune=tune)
    benchmark = linear_benchmark(M, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(linear_layer, *inputs)


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
