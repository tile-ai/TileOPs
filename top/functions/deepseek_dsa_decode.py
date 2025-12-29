import torch
from .function import Function
from top.ops import DeepSeekSparseAttentionDecodeWithKVCacheOp

__all__ = ['DeepSeekSparseAttentionDecodeWithKVCacheFunc']


class sparse_mla_ctx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, indices, fwd_op):
        o = fwd_op(q, kv, indices)
        return o

    @staticmethod
    def backward(ctx, do):
        raise RuntimeError("Inference-only op")


class DeepSeekSparseAttentionDecodeWithKVCacheFunc(Function):

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

        self.fwd_op = DeepSeekSparseAttentionDecodeWithKVCacheOp(
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

    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor):
        return sparse_mla_ctx.apply(q, kv, indices, self.fwd_op)