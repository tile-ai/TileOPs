import argparse
from top.ops import mha_fwd
from top.utils import str2dtype


def test_mha_kernel_autotune(B, S, H, D, causal, dtype):
    # 1. test autotune at initialization
    op = mha_fwd(B, H, S, D, causal, dtype, tune=True)

    # 2. test op.autotune()
    op.autotune()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--causal', action='store_true', default=False, help='causal attention')
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    args = parser.parse_args()

    test_mha_kernel_autotune(args.batch, args.seq_len, args.heads, args.dim, args.causal,
                             str2dtype[args.dtype])
