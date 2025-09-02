import argparse
import torch

from top import VerticalSlashSparseAttentionKernel

from torch.utils.cpp_extension import load

from top.utils.utils import performance, partity

import os

current_dir = os.path.dirname(os.path.abspath(__file__))

sources = [
    os.path.join(current_dir, 'ops', 'kernels.cpp'),
    os.path.join(current_dir, 'ops', 'vertical_slash_index.cu')
]


ops = load(name='convert', sources=sources, verbose=False)
convert_vertical_slash_indexes = ops.convert_vertical_slash_indexes

def _sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device)  # Zero matrix used for padding
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)  # pads the matrix on left and right
    mat_strided = mat_padded.as_strided(
        (1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1))  # Change the strides
    sum_diags = torch.sum(mat_strided, 2)  # Sums the resulting matrix's columns
    return sum_diags[:, :, 1:]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--vertical_size", type=int, default=1000)
    parser.add_argument("--slash_size", type=int, default=200)

    args = parser.parse_args()

    BATCH, N_HEADS, SEQ_LEN, D_HEAD = args.batch, args.heads, args.seq_len, args.head_dim

    vertical_size, slash_size = args.vertical_size, args.slash_size

    torch.manual_seed(0)
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)

    q_len = SEQ_LEN

    vertical_size, slash_size = min(q_len, vertical_size), min(q_len, slash_size)
    last_q = 64
    qk = torch.einsum('bhmk, bhnk -> bhmn', q[:, :, -last_q:, :], k)
    arange = torch.arange(last_q, device="cuda")
    qk[:, :, :, -last_q:] = torch.where(arange[None, None, :, None] >= arange[None, None, None, :],
                                        qk[:, :, :, -last_q:], -torch.inf)
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    vertical = qk.sum(-2, keepdim=True)
    vertical[..., :30] = torch.inf
    vertical_topk = torch.topk(vertical, vertical_size, -1).indices

    slash = _sum_all_diagonal_matrix(qk)[..., :-last_q + 1]
    slash[..., -30:] = torch.inf

    slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

    batch_size, num_heads, context_size, head_dim = q.shape

    v_idx = vertical_topk.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(
        dim=-1, descending=False)[0]
    s_idx = slash.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(
        dim=-1, descending=True)[0]

    seqlens = torch.tensor([context_size], dtype=torch.int32, device=q.device)

    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        seqlens,
        v_idx,
        s_idx,
        context_size,
        64,
        64,
    )


    kernel = VerticalSlashSparseAttentionKernel(
        batch_size=BATCH,
        num_heads=N_HEADS,
        seq_len=SEQ_LEN,
        head_dim=D_HEAD,
    )

    partity(kernel, q, k, v, block_count=block_count, block_offset=block_offset, column_count=column_count, column_index=column_index)

    perf = performance(kernel, [kernel.ref_program], q, k, v, block_count=block_count, block_offset=block_offset, column_count=column_count, column_index=column_index)

    perf.print_performance()



if __name__ == "__main__":
    main()

