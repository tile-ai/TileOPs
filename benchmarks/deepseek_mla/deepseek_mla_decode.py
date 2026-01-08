from typing import Tuple

import torch
from einops import einsum, rearrange
from torch.nn import functional as F

from benchmarks.benchmark import Benchmark
from top.ops import MultiHeadLatentAttentionDecodeWithKVCacheOp


class MultiHeadLatentAttentionDecodeBenchmark(Benchmark):

    op_type = MultiHeadLatentAttentionDecodeWithKVCacheOp

    def __init__(self, batch: int, heads: int, head_num_kv: int, seq_len_kv: int, dim: int,
                 dim_pe: int, dtype: torch.dtype):
        self.batch = batch
        self.heads = heads
        self.head_num_kv = head_num_kv
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_pe = dim_pe
        self.dtype = dtype

    @property
    def total_flops(self) -> float:
        # qk_flops = 2 * batch * heads *  seq_len_kv * (dim + dim_pe)
        # pv_flops = 2 * batch * heads *  seq_len_kv * dim
        qk_flops = 2 * self.batch * self.heads * self.seq_len_kv * (self.dim + self.dim_pe)
        pv_flops = 2 * self.batch * self.heads * self.seq_len_kv * self.dim
        flops = qk_flops + pv_flops
        return flops

    @property
    def total_memory(self) -> float:
        # Q: batch * heads * dim
        # Q_pe: batch * heads * dim_pe
        # K: batch * seq_len_kv * kv_heads_num * dim
        # K_pe: batch * seq_len_kv * kv_heads_num * dim_pe
        # Output: batch * heads * dim
        return self.batch * (self.heads + self.seq_len_kv * self.head_num_kv) * (
            self.dim + self.dim_pe) * self.dtype.itemsize

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(self.batch, self.heads, self.dim, device='cuda', dtype=self.dtype)
        Q_pe = torch.randn(self.batch, self.heads, self.dim_pe, device='cuda', dtype=self.dtype)
        K = torch.randn(self.batch,
                        self.seq_len_kv,
                        self.head_num_kv,
                        self.dim,
                        device='cuda',
                        dtype=self.dtype)
        K_pe = torch.randn(self.batch,
                           self.seq_len_kv,
                           self.head_num_kv,
                           self.dim_pe,
                           device='cuda',
                           dtype=self.dtype)
        return Q, Q_pe, K, K_pe

    def ref_program(self, q: torch.Tensor, q_pe: torch.Tensor, kv: torch.Tensor,
                    k_pe: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
        - q (Tensor): [batch, heads, dim]
        - q_pe (Tensor): [batch, heads, dim_pe]
        - kv (Tensor): [batch, seqlen_kv, head_num_kv, dim]
        - k_pe (Tensor): [batch, seqlen_kv, head_num_kv, dim_pe]
        Outputs:
        - output (Tensor): [batch, heads, dim]
        """
        dim = q.shape[-1]
        dim_pe = q_pe.shape[-1]
        num_head_groups = q.shape[1] // kv.shape[2]
        scale = (dim + dim_pe)**0.5
        Q = rearrange(q, 'b (h g) d -> b g h d',
                      g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

        Q_pe = rearrange(q_pe, 'b (h g) d -> b g h d',
                         g=num_head_groups)  # [batch_size, num_head_groups, groups, dim_pe]

        KV = rearrange(kv, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

        K_pe = rearrange(k_pe,
                         'b n h d -> b h n d')  # [batch_size, num_head_groups, groups, dim_pe]

        query = torch.concat([Q, Q_pe], dim=-1)
        key = torch.concat([KV, K_pe], dim=-1)

        scores = einsum(
            query, key,
            'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

        attention = F.softmax(scores / scale,
                              dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

        out = einsum(attention, KV,
                     'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
        out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
        return out
