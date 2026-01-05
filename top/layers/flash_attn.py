import torch
from torch import nn

from top.functions import GroupQueryAttentionFunc, MultiHeadAttentionFunc


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, batch_size: int, heads: int, seq_len: int, dim: int, is_causal: bool,
                 dtype: torch.dtype):
        super().__init__()

        self.batch_size = batch_size
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.fn = MultiHeadAttentionFunc(batch_size, heads, seq_len, dim, is_causal, dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.fn(q, k, v)


class GroupQueryAttentionLayer(nn.Module):

    def __init__(self, batch_size: int, heads: int, heads_kv: int, seq_len: int, dim: int,
                 is_causal: bool, dtype: torch.dtype):
        super().__init__()

        self.batch_size = batch_size
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.fn = GroupQueryAttentionFunc(batch_size, heads, heads_kv, seq_len, dim, is_causal,
                                          dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.fn(q, k, v)
