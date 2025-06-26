# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
import torch
from top import MLAKernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--heads', type=int, default=128, help='q heads number')
    parser.add_argument('--kv_heads', type=int, default=1, help='kv heads number')
    parser.add_argument('--kv_ctx', type=int, default=4096, help='kv context length')
    parser.add_argument('--dim', type=int, default=256, help='head dim')
    parser.add_argument('--pe_dim', type=int, default=64, help='pe head dim')
    args = parser.parse_args()
    batch, heads, kv_heads, kv_ctx, dim, pe_dim = args.batch, args.heads, args.kv_heads, args.kv_ctx, args.dim, args.pe_dim

    BLOCK_N = 32
    BLOCK_H = 32
    num_split = 1

    mla = MLAKernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim, BLOCK_N, BLOCK_H, num_split)

    q = torch.randn(batch, heads, dim, device='cuda', dtype=torch.float16)
    q_pe = torch.randn(batch, heads, pe_dim, device='cuda', dtype=torch.float16)
    kv = torch.randn(batch, kv_ctx, kv_heads, dim, device='cuda', dtype=torch.float16)
    k_pe = torch.randn(batch, kv_ctx, kv_heads, pe_dim, device='cuda', dtype=torch.float16)

    o = mla(q, q_pe, kv, k_pe)
    print(o)
    latency = mla.profile()
    print(f"Latency: {latency:.4f} ms")
    mla.check(q, q_pe, kv, k_pe)


if __name__ == "__main__":
    main()
