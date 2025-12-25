import torch
from .function import Function
from top.ops import GroupQueryAttentionFwdOp, GroupQueryAttentionBwdOp

__all__ = ['GroupQueryAttentionFunc', 'group_query_attention_func']


class gqa_ctx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, fwd_op, bwd_op):
        O, lse = fwd_op(Q, K, V)

        ctx.save_for_backward(Q, K, V, O, lse)
        ctx.bwd_op = bwd_op

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, lse = ctx.saved_tensors

        dQ, dK, dV = ctx.bwd_op(Q, K, V, O, dO, lse)

        return dQ, dK, dV, None, None


class GroupQueryAttentionFunc(Function):

    def __init__(self,
                 batch,
                 heads,
                 heads_kv,
                 seq_len,
                 dim,
                 is_causal,
                 dtype=torch.float16,
                 tune=False):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.fwd_op = GroupQueryAttentionFwdOp(
            batch, heads, heads_kv, seq_len, dim, is_causal, dtype, tune=tune)
        self.bwd_op = GroupQueryAttentionBwdOp(
            batch, heads, heads_kv, seq_len, dim, is_causal, dtype, tune=tune)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return gqa_ctx.apply(Q, K, V, self.fwd_op, self.bwd_op)


def group_query_attention_func(Q: torch.Tensor,
                               K: torch.Tensor,
                               V: torch.Tensor,
                               is_causal: bool = False,
                               tune: bool = False) -> torch.Tensor:

    # Validate that Q, K, V are 4-dimensional tensors
    if Q.dim() != 4:
        raise ValueError(f"Q must be 4-dimensional, but got {Q.dim()} dimensions")
    if K.dim() != 4:
        raise ValueError(f"K must be 4-dimensional, but got {K.dim()} dimensions")
    if V.dim() != 4:
        raise ValueError(f"V must be 4-dimensional, but got {V.dim()} dimensions")

    # Validate that dimensions are consistent (B, H, S, D)
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError(f"Q, K, V must have the same shape, "
                         f"but got Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

    # Validate that dtypes are consistent
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError(f"Q, K, V must have the same dtype, "
                         f"but got Q: {Q.dtype}, K: {K.dtype}, V: {V.dtype}")

    # Extract dimension information
    B, S, H, D = Q.shape

    B, S, H_kv, D = K.shape

    return GroupQueryAttentionFunc(B, H, H_kv, S, D, is_causal, Q.dtype, tune=tune).forward(Q, K, V)
