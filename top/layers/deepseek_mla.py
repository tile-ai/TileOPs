import torch
from torch import nn
from top import mla_decode_fn


class MLADecode(nn.Module):

    def __init__(self, batch_size, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype, tune=False):
        super().__init__()

        self.batch_size = batch_size
        self.heads = heads
        self.kv_head_num = kv_head_num
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim
        self.dtype = dtype

        self.fn = mla_decode_fn(batch_size, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype, tune=tune)


    def forward(self, Q: torch.Tensor, Q_pe: torch.Tensor, K: torch.Tensor, K_pe: torch.Tensor) -> torch.Tensor:
        return self.fn(Q, Q_pe, K, K_pe)

    