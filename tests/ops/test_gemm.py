import argparse

from benchmarks import GemmBenchmark
from top.ops import GemmOp
from top.utils import str2dtype


def test_gemm(m, n, k, dtype, trans_a=False, trans_b=False, tune=False):
    op = GemmOp(m, n, k, trans_a=trans_a, trans_b=trans_b, dtype=dtype, tune=tune)
    benchmark = GemmBenchmark(m, n, k, dtype, trans_a=trans_a, trans_b=trans_b)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


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
    parser.add_argument('--trans_A', action='store_true', default=False, help='transpose input A')
    parser.add_argument('--trans_B', action='store_true', default=False, help='transpose input B')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_gemm(args.M, args.N, args.K, str2dtype[args.dtype], args.trans_A, args.trans_B, args.tune)
