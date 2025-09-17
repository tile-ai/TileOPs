# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
from tileops import MHADecodeKernel


def test_mha_decode_kernel(B, S, H, D, tune):
    kernel = MHADecodeKernel(B, H, S, D, tune=tune)
    kernel.check()
    kernel.profile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--seqlen_kv', type=int, default=8192, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    parser.add_argument('--tune', action='store_true', default=True, help='tune the kernel')
    args = parser.parse_args()
    B, S, H, D, tune = args.batch, args.seqlen_kv, args.heads, args.dim, args.tune

    test_mha_decode_kernel(B, S, H, D, tune)
