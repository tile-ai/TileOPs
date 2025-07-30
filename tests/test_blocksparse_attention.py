# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
import torch
from top import BlockSparseAttentionKernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--h', type=int, default=32, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=1024, help='Context size')
    parser.add_argument('--d_head_qk', type=int, default=128, help='Head dimension for Q/K')
    parser.add_argument('--d_head_v', type=int, default=64, help='Head dimension for V')
    parser.add_argument('--causal', action='store_true', help='Causal flag')
    parser.add_argument('--groups', type=int, default=16, help='groups')
    args = parser.parse_args()
    BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, groups, causal = args.batch, args.h, args.n_ctx, args.d_head_qk, args.d_head_v, args.groups, args.causal

    BLOCK_M = 128
    BLOCK_N = 64
    Q = (
        torch.empty(BATCH, N_CTX, H, D_HEAD_QK, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())

    head_kv = H // groups
    K = (
        torch.empty(BATCH, N_CTX, head_kv, D_HEAD_QK, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())
    V = (
        torch.empty(BATCH, N_CTX, head_kv, D_HEAD_V, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())
    dO = (
        torch.empty(BATCH, N_CTX, H, D_HEAD_V, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())

    M_BLOCKS = N_CTX // BLOCK_M
    N_BLOCKS = N_CTX // BLOCK_N

    block_mask = torch.randint(
        low=0, high=2, size=(BATCH, H, M_BLOCKS, N_BLOCKS), dtype=torch.bool, device="cuda")

    attention = BlockSparseAttentionKernel(
        BATCH,
        H,
        N_CTX,
        D_HEAD_QK,
        D_HEAD_V,
        BLOCK_M,
        BLOCK_N,
        causal=causal,
        block_mask=block_mask,
        groups=groups,
        fwd_tune=True,
        bwd_tune=True)

    o = attention.backward(Q, K, V, dO)
    print(o)

    attention.profile()


if __name__ == "__main__":
    main()
