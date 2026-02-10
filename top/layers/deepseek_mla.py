from typing import Optional

import torch
from torch import nn

from top.functions import (DeepSeekSparseAttentionDecodeWithKVCacheFunc,
                           MultiHeadLatentAttentionDecodeWithKVCacheFunc, Fp8LightingIndexerFunc,
                           TopkSelectorFunc, Fp8QuantFunc)


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
                 dim_tail: int,
                 topk: int,
                 stride_kv: int,
                 group_kv: int,
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
        self.dim_tail = dim_tail
        self.topk = topk
        self.stride_kv = stride_kv
        self.group_kv = group_kv
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
            dim_tail,
            topk,
            stride_kv,
            group_kv,
            q_start_index_s,
            sm_scale,
            is_causal,
            dtype,
            tune=tune)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.fn(q, kv, indices)


class Fp8LightingIndexerDecodeLayer(nn.Module):

    def __init__(self,
                 seq_len,
                 heads,
                 index_dim,
                 seq_len_kv,
                 clean_logits=True,
                 config: Optional[dict] = None,
                 tune=False) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.clean_logits = clean_logits
        self.tune = tune
        self.fn = Fp8LightingIndexerFunc(
            seq_len, heads, index_dim, seq_len_kv, clean_logits, config, tune=tune)

    def forward(self, index_q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor,
                cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> torch.Tensor:
        return self.fn(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke)


class TopkSelectorLayer(nn.Module):

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 topk: int,
                 in_dtype: str,
                 out_dtype: str,
                 tune: bool = False):
        super().__init__()

        self.fn = TopkSelectorFunc(batch, seq_len, topk, in_dtype, out_dtype, tune=tune)

    def forward(self, index_scores: torch.Tensor, starts: torch.Tensor,
                ends: torch.Tensor) -> torch.Tensor:
        return self.fn(index_scores, starts, ends)


class Fp8QuantLayer(nn.Module):

    def __init__(self,
                 seq_len_kv: int,
                 index_dim: int,
                 in_dtype: torch.dtype = torch.float16,
                 tune: bool = False):
        super().__init__()
        self.fn = Fp8QuantFunc(seq_len_kv, index_dim, in_dtype)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fn(input_tensor)
