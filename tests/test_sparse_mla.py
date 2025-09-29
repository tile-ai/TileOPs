import argparse
import torch
from top import SparseMLAKernel


def test_sparsemla_kernel(batch, seq_len, seq_len_kv, q_start_index_s, heads, dim, tail_dim, topk,
                          kv_stride, kv_group, sm_scale, dtype):
    sparse_mla = SparseMLAKernel(
        batch=batch,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        q_start_index_s=q_start_index_s,
        heads=heads,
        dim=dim,
        tail_dim=tail_dim,
        topk=topk,
        kv_stride=kv_stride,
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_casual=True,
        dtype=dtype,
        device='cuda',
    )
    sparse_mla.check()
    latency = sparse_mla.profile()
    print(f"Latency: {latency:.4f} ms")
    print(f'fwd tflops = ',
          (batch * seq_len * (dim + tail_dim + dim) * topk * 2 * heads) / (latency * 1e-3) / 1e12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parser.add_argument('--seq_len_kv', type=int, default=2048, help='key/value sequence length')
    parser.add_argument('--heads', type=int, default=128, help='num heads')
    parser.add_argument('--dim', type=int, default=512, help='head dim')
    parser.add_argument('--tail_dim', type=int, default=64, help='tail dim')
    parser.add_argument('--topk', type=int, default=2048, help='topk')
    parser.add_argument('--kv_stride', type=int, default=1, help='kv_stride')
    parser.add_argument('--kv_group', type=int, default=1, help='kv_group')
    parser.add_argument('--sm_scale', type=float, default=None, help='softmax scaling factor')
    args = parser.parse_args()
    batch = args.batch
    seq_len = args.seq_len
    seq_len_kv = args.seq_len_kv
    heads = args.heads
    dim = args.dim
    tail_dim = args.tail_dim
    topk = args.topk
    kv_stride = args.kv_stride
    kv_group = args.kv_group
    sm_scale = args.sm_scale
    dtype = torch.bfloat16

    test_sparsemla_kernel(batch, seq_len, seq_len_kv, 1024, heads, dim, tail_dim, topk, kv_stride,
                          kv_group, sm_scale, dtype)
