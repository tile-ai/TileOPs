import argparse
from top import mha_fn
from top.utils import str2dtype
import torch


def test_mha_fn(B, S, H, D, causal, dtype):
    fn = mha_fn(B, H, S, D, causal, dtype)
    Q = torch.randn(B, S, H, D, dtype=dtype, device='cuda', requires_grad=True)
    K = torch.randn(B, S, H, D, dtype=dtype, device='cuda', requires_grad=True)
    V = torch.randn(B, S, H, D, dtype=dtype, device='cuda', requires_grad=True)
    O = fn(Q, K, V)
    print(O.shape)

    dO = torch.randn_like(O, device='cuda')
    O.backward(dO)
    print(Q.grad.shape)
    print(K.grad.shape)
    print(V.grad.shape)


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

    test_mha_fn(args.batch, args.seq_len, args.heads, args.dim, args.causal, str2dtype[args.dtype])