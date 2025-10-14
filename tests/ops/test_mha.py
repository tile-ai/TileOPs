import argparse
from top import mha_fwd, mha_fwd_kernel
from top.utils import str2dtype


def test_mha_kernel(B, S, H, D, causal, dtype):
    op = mha_fwd(B, H, S, D, causal, dtype)
    op.check()
    op.profile()


def test_mha_kernel_sm80(B, S, H, D, causal, dtype):
    op = mha_fwd(B, H, S, D, causal, dtype, kernel_map={"mha_fwd_kernel": mha_fwd_kernel})
    op.check()
    op.profile()


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

    test_mha_kernel_sm80(args.batch, args.seq_len, args.heads, args.dim, args.causal,
                         str2dtype[args.dtype])
