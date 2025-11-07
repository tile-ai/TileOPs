import argparse
import torch
from top.utils import str2dtype
from top.layers.flash_attn import MHA


def test_mha_layer(B, S, H, D, causal, dtype):

    mha = MHA(B, H, S, D, causal, dtype)

    # create inputs
    Q = torch.randn(B, H, S, D, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, S, D, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, S, D, dtype=dtype, requires_grad=True)

    # forward pass (fwd)
    output = mha(Q, K, V)
    print(f"Output shape: {output.shape}")

    # compute loss and backpropagate (bwd)
    loss = output.sum()
    loss.backward()

    # check gradients
    print(f"Q.grad shape: {Q.grad.shape}")
    print(f"K.grad shape: {K.grad.shape}")
    print(f"V.grad shape: {V.grad.shape}")


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

    test_mha_layer(args.batch, args.seq_len, args.heads, args.dim, args.causal, str2dtype[args.dtype])
