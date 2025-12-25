import torch
from .function import Function
from top.ops import GroupQueryAttentionDecodeWithKVCacheOp

__all__ = ['GroupQueryAttentionDecodeWithKVCacheFunc']


class gqa_decode_ctx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, mask, fwd_op):
        O = fwd_op(Q, K, V, mask)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("Backward pass is not implemented for gqa_decode.")


class GroupQueryAttentionDecodeWithKVCacheFunc(Function):

    def __init__(self, batch, heads, groups, seqlen_kv, dim, dtype=torch.float16, tune=False):
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype

        self.fwd_op = GroupQueryAttentionDecodeWithKVCacheOp(
            batch, heads, groups, seqlen_kv, dim, dtype, tune=tune)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return gqa_decode_ctx.apply(Q, K, V, mask, self.fwd_op)
