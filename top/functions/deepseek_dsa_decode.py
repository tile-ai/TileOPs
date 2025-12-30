import torch
from .function import Function
from top.ops import DeepSeekSparseAttentionDecodeWithKVCacheOp
from typing import Any

__all__ = ['DeepSeekSparseAttentionDecodeWithKVCacheFunc']


class sparse_mla_ctx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, fwd_op: Any):
        o = fwd_op(q, kv, indices)
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor) -> Any:
        raise RuntimeError("Inference-only op")


class DeepSeekSparseAttentionDecodeWithKVCacheFunc(Function):

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
                 sm_scale: Any = None,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16,
                 tune: bool = False):
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

    def forward(self, q: torch.Tensor, kv_cache: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        return sparse_mla_ctx.apply(q, kv_cache, indices, self.fwd_op)
