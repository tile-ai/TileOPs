import torch
from torch import nn
from top.functions import mla_decode_fn, sparse_mla_fn


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

        self.fn = mla_decode_fn(
            batch_size, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype, tune=tune)

    def forward(self, Q: torch.Tensor, Q_pe: torch.Tensor, K: torch.Tensor,
                K_pe: torch.Tensor) -> torch.Tensor:
        return self.fn(Q, Q_pe, K, K_pe)


class SparseMLADecode(nn.Module):

    def __init__(self,
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
                 sm_scale=None,
                 is_causal=True,
                 dtype=torch.float16,
                 tune=False):
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

        self.fn = sparse_mla_fn(
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

    def forward(self, Q: torch.Tensor, KV: torch.Tensor, Indices: torch.Tensor) -> torch.Tensor:
        return self.fn(Q, KV, Indices)
