from benchmarks.benchmark import Benchmark
from top.ops import MultiHeadLatentAttentionDecodeOp
import torch
from torch.nn import functional as F
from einops import rearrange, einsum


class MultiHeadLatentAttentionDecodeBenchmark(Benchmark):

    op_type = MultiHeadLatentAttentionDecodeOp

    def __init__(self, batch, heads, kv_head_num, seq_len_kv, dim, pe_dim, dtype):
        self.batch = batch
        self.heads = heads
        self.kv_head_num = kv_head_num
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.pe_dim = pe_dim
        self.dtype = dtype

    @property
    def total_flops(self):
        # qk_flops = 2 * batch * heads *  seq_len_kv * (dim + pe_dim)
        # pv_flops = 2 * batch * heads *  seq_len_kv * dim
        qk_flops = 2 * self.batch * self.heads * self.seq_len_kv * (self.dim + self.pe_dim)
        pv_flops = 2 * self.batch * self.heads * self.seq_len_kv * self.dim
        flops = qk_flops + pv_flops
        return flops

    @property
    def total_memory(self):
        # Q: batch * heads * dim
        # Q_pe: batch * heads * pe_dim
        # K: batch * seq_len_kv * kv_heads_num * dim
        # K_pe: batch * seq_len_kv * kv_heads_num * pe_dim
        # Output: batch * heads * dim
        return self.batch * (self.heads + self.seq_len_kv * self.kv_head_num) * (
            self.dim + self.pe_dim) * self.dtype.itemsize

    def gen_inputs(self):
        Q = torch.randn(self.batch, self.heads, self.dim, device='cuda', dtype=self.dtype)
        Q_pe = torch.randn(self.batch, self.heads, self.pe_dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_head_num,
            self.dim,
            device='cuda',
            dtype=self.dtype)
        K_pe = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_head_num,
            self.pe_dim,
            device='cuda',
            dtype=self.dtype)
        return Q, Q_pe, K, K_pe

    def ref_program(self, Q: torch.Tensor, Q_pe: torch.Tensor, KV: torch.Tensor,
                    K_pe: torch.Tensor):
        #     """
        #     Inputs:
        #     - Q (Tensor): [batch, heads, dim]
        #     - Q_pe (Tensor): [batch, heads, pe_dim]
        #     - KV (Tensor): [batch, seqlen_kv, kv_head_num, dim]
        #     - K_pe (Tensor): [batch, seqlen_kv, kv_head_num, pe_dim]
        #     Outputs:
        #     - output (Tensor): [batch, heads, dim]
        #     """
        dim = Q.shape[-1]
        pe_dim = Q_pe.shape[-1]
        num_head_groups = Q.shape[1] // KV.shape[2]
        scale = (dim + pe_dim)**0.5
        Q = rearrange(
            Q, 'b (h g) d -> b g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

        Q_pe = rearrange(
            Q_pe, 'b (h g) d -> b g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, pe_dim]

        KV = rearrange(KV, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

        K_pe = rearrange(K_pe,
                         'b n h d -> b h n d')  # [batch_size, num_head_groups, groups, pe_dim]

        query = torch.concat([Q, Q_pe], dim=-1)
        key = torch.concat([KV, K_pe], dim=-1)

        scores = einsum(
            query, key,
            'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

        attention = F.softmax(
            scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

        out = einsum(attention, KV,
                     'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
        out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
        return out
