import torch
from torch import nn
from top import mha_decode_fn, gqa_decode_fn


class MHADecode(nn.Module):

    def __init__(self, batch_size, heads, seqlen_q, seqlen_kv, dim, dtype):
        super().__init__()

        self.batch_size = batch_size
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.dtype = dtype

        self.fn = mha_decode_fn(batch_size, heads, seqlen_q, seqlen_kv, dim, dtype)


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.fn(Q, K, V)
    

class GQADecode(nn.Module):

    def __init__(self, batch_size, heads, groups, seqlen_kv, dim, dtype):
        super().__init__()

        self.batch_size = batch_size
        self.heads = heads
        self.groups = groups
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.dtype = dtype

        self.fn = gqa_decode_fn(batch_size, heads, groups, seqlen_kv, dim, dtype)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fn(Q, K, V, mask)
