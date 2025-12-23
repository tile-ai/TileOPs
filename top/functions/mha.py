import torch
from .function import Function
from top.ops import MultiHeadAttentionFwdOp, MultiHeadAttentionBwdOp

__all__ = ['MultiHeadAttentionFunc']


class mhc_ctx(torch.autograd.Function):

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


class MultiHeadAttentionFunc(Function):

    def __init__(self, batch, heads, seq_len, dim, is_causal, dtype=torch.float16, tune=False):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.fwd_op = MultiHeadAttentionFwdOp(
            batch, heads, seq_len, dim, is_causal, dtype, tune=tune)
        self.bwd_op = MultiHeadAttentionBwdOp(
            batch, heads, seq_len, dim, is_causal, dtype, tune=tune)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return mhc_ctx.apply(Q, K, V, self.fwd_op, self.bwd_op)
