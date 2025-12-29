import torch
from torch import nn
from top.functions import MultiHeadAttentionDecodeWithKVCacheFunc, GroupQueryAttentionDecodeWithKVCacheFunc


class MultiHeadAttentionDecodeLayer(nn.Module):

    def __init__(self, batch_size, heads, seqlen_q, seqlen_kv, dim, dtype):
        super().__init__()

        self.batch_size = batch_size
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.dtype = dtype

        self.fn = MultiHeadAttentionDecodeWithKVCacheFunc(batch_size, heads, seqlen_q, seqlen_kv,
                                                          dim, dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.fn(q, k, v)


class GroupQueryAttentionDecodeLayer(nn.Module):

    def __init__(self, batch_size, heads, groups, seqlen_kv, dim, dtype):
        super().__init__()

        self.batch_size = batch_size
        self.heads = heads
        self.groups = groups
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.dtype = dtype

        self.fn = GroupQueryAttentionDecodeWithKVCacheFunc(batch_size, heads, groups, seqlen_kv,
                                                           dim, dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.fn(q, k, v)