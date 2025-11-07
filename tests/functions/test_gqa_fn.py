import argparse
from top import gqa_fn
from top.utils import str2dtype
import torch
from benchmarks import gqa_benchmark


def test_gqa_fn(B, S, H, H_kv, D, causal, dtype):
    fn = gqa_fn(B, H, H_kv, S, D, causal, dtype)
    benchmark = gqa_benchmark(B, H, H_kv, S, D, causal, dtype)

    inputs = benchmark.gen_inputs()
    benchmark.check_fn(fn, *inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--heads_kv', type=int, default=32, help='num heads for key/value')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--causal', action='store_true', default=False, help='causal attention')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    args = parser.parse_args()

    test_gqa_fn(args.batch, args.seq_len, args.heads, args.heads_kv, args.dim, args.causal, str2dtype[args.dtype])