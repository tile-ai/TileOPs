import torch
from torch.autograd.function import FunctionCtx
from .function import Function
from top.ops import DeepSeekSparseAttentionDecodeWithKVCacheOp
from typing import Any

__all__ = ['DeepSeekSparseAttentionDecodeWithKVCacheFunc']


class DSADecodeCtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor,
                fwd_op: DeepSeekSparseAttentionDecodeWithKVCacheOp) -> torch.Tensor:
        o = fwd_op(q, kv, indices)
        return o

    @staticmethod
    def backward(ctx: FunctionCtx, do: torch.Tensor) -> Any:
        raise RuntimeError("Inference-only op")


class DeepSeekSparseAttentionDecodeWithKVCacheFunc(Function):

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
                 sm_scale: Any = None,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16,
                 tune: bool = False):
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

        self.fwd_op = DeepSeekSparseAttentionDecodeWithKVCacheOp(
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

    def forward(self, q: torch.Tensor, kv_cache: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        return DSADecodeCtx.apply(q, kv_cache, indices, self.fwd_op)
