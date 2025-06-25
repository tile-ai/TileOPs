import argparse
import torch
from tla import GQA_decode_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--groups', type=int, default=8, help='groups')
    parser.add_argument('--kv_seqlen', type=int, default=8192, help='kv sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--num_split', type=int, default=8, help='num_split')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()
    batch, heads, groups, kv_seqlen, dim, num_split, tune = args.batch, args.heads, args.groups, args.kv_seqlen, args.dim, args.num_split, args.tune

    BLOCK_N = 128
    BLOCK_H = 64
    threads = 128

    Q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.float16)
    K = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.float16)
    V = torch.randn(batch, kv_seqlen, groups, dim, device="cuda", dtype=torch.float16)

    gqa_decode = GQA_decode_kernel(batch, heads, kv_seqlen, dim, BLOCK_N, BLOCK_H, threads,
                                   num_split, groups)
    if tune:
        gqa_decode.autotune()
    o = gqa_decode.decode(Q, K, V)
    print(o)

    latency = gqa_decode.profile()
    print(f"Latency: {latency:.4f} ms")

    gqa_decode.check(Q, K, V)

    gqa_decode.autotune()
    latency_ = gqa_decode.profile()
    print(f"Latency: {latency_:.4f} ms")


if __name__ == "__main__":
    main()
