# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
import torch
from top import GQAKernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--h', type=int, default=32, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=1024, help='Context size')
    parser.add_argument('--d_head_qk', type=int, default=128, help='Head dimension for Q/K')
    parser.add_argument('--d_head_v', type=int, default=64, help='Head dimension for V')
    parser.add_argument('--causal', type=bool, default=False, help='Casual flag')
    parser.add_argument('--groups', type=int, default=16, help='groups')
    args = parser.parse_args()
    BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, groups, causal = args.batch, args.h, args.n_ctx, args.d_head_qk, args.d_head_v, args.groups, args.causal

    fwd_BLOCK_M = 128
    fwd_BLOCK_N = 32
    bwd_BLOCK_M = 64
    bwd_BLOCK_N = 32
    # q = torch.randn(batch, heads, dim, device='cuda', dtype=torch.float16)
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

    gqa = GQAKernel(
        BATCH,
        H,
        N_CTX,
        D_HEAD_QK,
        D_HEAD_V,
        fwd_BLOCK_M,
        fwd_BLOCK_N,
        bwd_BLOCK_M,
        bwd_BLOCK_N,
        fwd_tune=True,
        bwd_tune=True,
        causal=causal,
        groups=groups)
    o = gqa.backward(Q, K, V, dO)
    print(o)
    gqa.profile(Q, K, V, dO)
    gqa.check(Q, K, V, dO)
    # gqa.fwd_autotune(Q, K, V)
    # gqa.bwd_autotune(Q, K, V)
    # gqa.profile(Q, K, V, dO)


if __name__ == "__main__":
    main()
