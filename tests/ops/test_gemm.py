import argparse
from top import Gemm
from top.utils import str2dtype
from benchmarks import gemm_benchmark


def test_gemm(M, N, K, dtype):
    op = Gemm(M, N, K, dtype)
    benchmark = gemm_benchmark(M, N, K, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=1024, help='M')
    parser.add_argument('--N', type=int, default=1024, help='N')
    parser.add_argument('--K', type=int, default=1024, help='K')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    args = parser.parse_args()

    test_gemm(args.M, args.N, args.K, str2dtype[args.dtype])