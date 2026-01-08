import argparse

from benchmarks import MatMulBenchmark
from top.functions import MatMulFunc, matmul
from top.utils import str2dtype


def test_matmul(m, n, k, dtype, tune=False):
    benchmark = MatMulBenchmark(m, n, k, dtype)

    inputs = benchmark.gen_inputs()

    print("=========Testing matmul function inference=========")
    benchmark.check_fn(matmul, *inputs)

    print("=========Testing matmul function class=========")
    fn = MatMulFunc(m, n, k, dtype, tune)
    benchmark.check_fn(fn, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=1024, help='M')
    parser.add_argument('--N', type=int, default=1024, help='N')
    parser.add_argument('--K', type=int, default=1024, help='K')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'bfloat16'],
                        help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_matmul(args.M, args.N, args.K, str2dtype[args.dtype], args.tune)
