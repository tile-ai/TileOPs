import argparse
from top.utils import str2dtype
from top.layers.grouped_gemm import GROUPED_GEMM
from benchmarks import grouped_gemm_benchmark


def test_grouped_gemm_layer(batch_sum, batch_count, N, K, dtype):
    grouped_gemm = GROUPED_GEMM(batch_sum, batch_count, N, K, dtype)
    benchmark = grouped_gemm_benchmark(batch_sum, batch_count, N, K, dtype)
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
