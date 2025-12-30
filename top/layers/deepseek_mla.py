import torch
from torch import nn
from top.functions import MultiHeadLatentAttentionDecodeWithKVCacheFunc, DeepSeekSparseAttentionDecodeWithKVCacheFunc
from typing import Optional


class MultiHeadLatentAttentionDecodeLayer(nn.Module):

    def __init__(self,
                 batch_size: int,
                 heads: int,
                 kv_head_num: int,
                 seqlen_kv: int,
                 dim: int,
                 pe_dim: int,
                 dtype: torch.dtype,
                 tune: bool = False):
        super().__init__()

        self.batch_size = batch_size
        self.heads = heads
        self.kv_head_num = kv_head_num
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim
        self.dtype = dtype

        self.fn = MultiHeadLatentAttentionDecodeWithKVCacheFunc(
            batch_size, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype, tune=tune)

    def forward(self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor,
                k_pe: torch.Tensor) -> torch.Tensor:
        return self.fn(q, q_pe, k, k_pe)


class DeepSeekSparseAttentionDecodeLayer(nn.Module):

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 seq_len_kv: int,
                 dim: int,
                 tail_dim: int,
                 topk: int,
                 kv_stride: int,
                 kv_group: int,
                 q_start_index_s: int,
                 sm_scale: Optional[float] = None,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16,
                 tune: bool = False):
        super().__init__()

        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.tail_dim = tail_dim
        self.topk = topk
        self.kv_stride = kv_stride
        self.kv_group = kv_group
        self.sm_scale = sm_scale
        self.dtype = dtype
        self.is_causal = is_causal
        self.q_start_index_s = q_start_index_s

        self.fn = DeepSeekSparseAttentionDecodeWithKVCacheFunc(
            batch,
            heads,
            seq_len,
            seq_len_kv,
            dim,
            tail_dim,
            topk,
            kv_stride,
            kv_group,
            q_start_index_s,
            sm_scale,
            is_causal,
            dtype,
            tune=tune)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.fn(q, kv, indices)
