import argparse
from top import mla_decode
from top.utils import str2dtype
from benchmarks import mla_decode_benchmark


def test_mla_decode(B, H, kv_head_num, S_kv, D, Pe_D, dtype, tune=False):
    op = mla_decode(B, H, kv_head_num, S_kv, D, Pe_D, dtype, tune=tune)
    benchmark = mla_decode_benchmark(B, H, kv_head_num, S_kv, D, Pe_D, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--kv_head_num', type=int, default=128, help='number of key/value heads')
    parser.add_argument('--seq_len_kv', type=int, default=8192, help='key/value sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--pe_dim', type=int, default=128, help='positional encoding dim')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--tune', action='store_true', default=False, help='enable autotune')
    args = parser.parse_args()

    test_mla_decode(args.batch, args.heads, args.kv_head_num, args.seq_len_kv, args.dim, args.pe_dim, str2dtype[args.dtype], args.tune)
