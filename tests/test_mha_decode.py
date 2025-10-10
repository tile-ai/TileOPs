import argparse
from top import MHADecodeKernel


def test_mha_decode_kernel(B, S, H, D, S_q, tune):
    kernel = MHADecodeKernel(B, H, S, D, seqlen_q=S_q, tune=tune)
    kernel.check()
    kernel.profile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--seqlen_q', type=int, default=1, help='sequence length')
    parser.add_argument('--seqlen_kv', type=int, default=8192, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--tune', action='store_true', default=True, help='tune the kernel')
    args = parser.parse_args()
    B, S, S_q, H, D, tune = args.batch, args.seqlen_kv, args.seqlen_q, args.heads, args.dim, args.tune

    test_mha_decode_kernel(B, S, H, D, S_q, tune)
