import argparse
from top import MHAKernel
from top.utils import str2dtype


def test_mha_kernel(B, S, H, D, dtype, causal, tune):
    kernel = MHAKernel(B, H, S, D, causal, fwd_tune=tune, bwd_tune=tune, dtype=str2dtype[dtype])
    kernel.check()
    kernel.profile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=64, help='head dim')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'], help='data type')
    parser.add_argument('--causal', action='store_true', default=True, help='causal attention')
    parser.add_argument('--tune', action='store_true', default=False, help='tune the kernel')
    args = parser.parse_args()
    B, S, H, D, dtype, causal, tune = args.batch, args.seq_len, args.heads, args.dim, args.dtype, args.causal, args.tune

    test_mha_kernel(B, S, H, D, dtype, causal, tune)
