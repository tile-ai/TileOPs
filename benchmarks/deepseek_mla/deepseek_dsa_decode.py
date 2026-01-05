from typing import Tuple

import torch

from benchmarks.benchmark import Benchmark
from top.ops import DeepSeekSparseAttentionDecodeWithKVCacheOp


class DeepSeekSparseAttentionDecodeBenchmark(Benchmark):

    op_type = DeepSeekSparseAttentionDecodeWithKVCacheOp

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 seq_len_kv: int,
                 dim: int,
                 dim_tail: int,
                 topk: int,
                 stride_kv: int,
                 group_kv: int,
                 q_start_index_s: int,
                 sm_scale: float = None,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_tail = dim_tail
        self.topk = topk
        self.stride_kv = stride_kv
        self.group_kv = group_kv
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.dtype = dtype
        self.q_start_index_s = q_start_index_s

    @property
    def total_flops(self) -> float:
        flops = self.batch * self.seq_len * (2 * self.dim +
                                             self.dim_tail) * self.topk * 2 * self.heads
        return flops

    @property
    def total_memory(self) -> float:
        # Q: batch, seq_len, heads, dim + dim_tail
        # KV: batch, seq_len_kv, group_kv, dim + dim_tail
        # Indices: batch, seq_len, group_kv, topk
        # Output: batch, seq_len, heads, dim
        q_memory = self.batch * self.seq_len * self.heads * (self.dim +
                                                             self.dim_tail) * self.dtype.itemsize
        kv_memory = self.batch * self.seq_len_kv * self.group_kv * (
            self.dim + self.dim_tail) * self.dtype.itemsize
        indices_memory = self.batch * self.seq_len * self.group_kv * self.topk * 4  # int32
        output_memory = self.batch * self.seq_len * self.heads * self.dim * self.dtype.itemsize
        return q_memory + kv_memory + indices_memory + output_memory

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim + self.dim_tail,
            device='cuda',
            dtype=self.dtype)
        KV = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.group_kv,
            self.dim + self.dim_tail,
            device='cuda',
            dtype=self.dtype)
        Indices = torch.full((self.batch, self.seq_len, self.group_kv, self.topk),
                             self.seq_len_kv,
                             dtype=torch.int32,
                             device='cuda')
        for b in range(self.batch):
            for t in range(self.seq_len):
                for h in range(self.group_kv):
                    i_i = torch.randperm(
                        min(
                            max(1, ((t + int(self.q_start_index_s)) // self.stride_kv)),
                            self.seq_len_kv))[:self.topk]
                    Indices[b, t, h, :len(i_i)] = i_i
        return Q, KV, Indices

    def ref_program(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        q = q.float()
        kv = kv.float()
        indices = indices.transpose(1, 2)
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape
        q_start_index_s = self.q_start_index_s
        if self.q_start_index_s is None:
            q_start_index_s = sk * self.stride_kv - sq

        assert kv.shape[-1] == self.dim + self.dim_tail, 'you should assign dim otherwise'
        dim = self.dim
        k = kv
        v = kv[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        compressed_causal_mask = torch.arange(
            q_start_index_s, sq + q_start_index_s, dtype=torch.int32,
            device="cuda").view(-1, 1) >= torch.arange(
                self.stride_kv - 1,
                sk * self.stride_kv,
                self.stride_kv,
                dtype=torch.int32,
                device="cuda").view(1, -1)

        mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
        mask[:, :, :self.stride_kv - 1, 0] = True
        mask = mask.view(b, g_index, 1, sq, sk)

        q = q.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q, k)
        sm_scale = dim_q**-0.5 if self.sm_scale is None else self.sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        return o.to(torch.float16)
