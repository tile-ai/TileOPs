import argparse
import torch
from tla import MAMBA_CHUNK_SCAN_kernel
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=80, help='heads')
    parser.add_argument('--groups', type=int, default=1, help='groups')
    parser.add_argument('--seq_len',
                        type=int,
                        default=4096,
                        help='sequence length')
    parser.add_argument('--chunk_size',
                        type=int,
                        default=256,
                        help='chunk size')
    parser.add_argument('--dim', type=int, default=64, help='dim')
    parser.add_argument('--dstate', type=int, default=128, help='dstate')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()
    BATCH, HEADS, GROUPS, SEQ_LEN, CHUNK_SIZE, DIM, DSTATE = args.batch, args.heads, args.groups, args.seq_len, args.chunk_size, args.dim, args.dstate
    NCHUNKS = math.ceil(SEQ_LEN / CHUNK_SIZE)

    cb = torch.empty((BATCH, NCHUNKS, GROUPS, CHUNK_SIZE, CHUNK_SIZE),
                     dtype=torch.half).cuda().normal_(-1.0, 1.0)
    x = torch.empty((BATCH, SEQ_LEN, HEADS, DIM),
                    dtype=torch.half).cuda().normal_(-1.0, 1.0)
    dt = torch.empty((BATCH, HEADS, NCHUNKS, CHUNK_SIZE),
                     dtype=torch.half).cuda().normal_(-1.0, 1.0)
    dA_cumsum = torch.empty((BATCH, HEADS, NCHUNKS, CHUNK_SIZE),
                            dtype=torch.half).cuda().normal_(-1.0, 1.0)
    C = torch.empty((BATCH, SEQ_LEN, GROUPS, DSTATE),
                    dtype=torch.half).cuda().normal_(-1.0, 1.0)
    prev_states = torch.empty((BATCH, NCHUNKS, HEADS, DIM, DSTATE),
                              dtype=torch.half).cuda().normal_(-1.0, 1.0)
    D = torch.empty((HEADS, ), dtype=torch.half).cuda().normal_(-1.0, 1.0)

    tune = False
    if tune:
        mamba_chunk_scan = MAMBA_CHUNK_SCAN_kernel(BATCH,
                                                   HEADS,
                                                   GROUPS,
                                                   SEQ_LEN,
                                                   CHUNK_SIZE,
                                                   DIM,
                                                   DSTATE,
                                                   tune=True)
        o = mamba_chunk_scan(cb, x, dt, dA_cumsum, C, prev_states, D)
        print(o)
        latency = mamba_chunk_scan.profile()
        print(f"Latency: {latency:.4f} ms")
        mamba_chunk_scan.check(cb, x, dt, dA_cumsum, C, prev_states, D)

    else:
        block_M = 64
        block_N = 64
        block_K = 64
        block_Dstate = 128
        num_stages = 2
        threads = 128

        mamba_chunk_scan = MAMBA_CHUNK_SCAN_kernel(BATCH,
                                                   HEADS,
                                                   GROUPS,
                                                   SEQ_LEN,
                                                   CHUNK_SIZE,
                                                   DIM,
                                                   DSTATE,
                                                   block_M=block_M,
                                                   block_N=block_N,
                                                   block_K=block_K,
                                                   block_Dstate=block_Dstate,
                                                   num_stages=num_stages,
                                                   threads=threads)
        o = mamba_chunk_scan(cb, x, dt, dA_cumsum, C, prev_states, D)
        print(o)
        latency = mamba_chunk_scan.profile()
        print(f"Latency: {latency:.4f} ms")
        mamba_chunk_scan.check(cb, x, dt, dA_cumsum, C, prev_states, D)


if __name__ == "__main__":
    main()
