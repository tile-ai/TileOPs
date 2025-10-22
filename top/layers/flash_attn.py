import torch
from torch import nn
from top import mha_fn


class MHA(nn.Module):

    def __init__(self, batch_size, heads, seq_len, dim, is_causal, dtype):
        super().__init__()

        self.batch_size = batch_size
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.fn = mha_fn(batch_size, heads, seq_len, dim, is_causal, dtype)


    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.fn(Q, K, V)