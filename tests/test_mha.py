import argparse
from tla import MHAKernel


def test_mha_kernel(B, S, H, D, causal):
    kernel = MHAKernel(B, H, S, D, causal)
    kernel.check()
    kernel.profile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=64, help='head dim')
    parser.add_argument('--causal', action='store_true', default=False, help='causal attention')
    args = parser.parse_args()
    B, S, H, D, causal = args.batch, args.seq_len, args.heads, args.dim, args.causal

    test_mha_kernel(B, S, H, D, causal)
