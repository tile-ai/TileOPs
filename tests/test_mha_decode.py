# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
from top import MHADecodeKernel


def test_mha_decode_kernel(B, S, H, D):
    kernel = MHADecodeKernel(B, H, S, D)
    kernel.check()
    kernel.profile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--seqlen_kv', type=int, default=8192, help='sequence length')
    parser.add_argument('--heads', type=int, default=32, help='num heads')
    parser.add_argument('--dim', type=int, default=128, help='head dim')
    args = parser.parse_args()
    B, S, H, D = args.batch, args.seqlen_kv, args.heads, args.dim

    test_mha_decode_kernel(B, S, H, D)
