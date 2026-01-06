import torch
from top.functions.function import Function
from top.ops.deepseek_nsa import NativeSparseAttentionForwardOp

__all__ = ['NativeSparseAttentionFunc']


class nsa_decode_ctx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, BlockIndices, fwd_op):
        O = fwd_op(Q, K, V, BlockIndices)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("Backward pass is not implemented for nsa.")

    @staticmethod
    def decode(ctx, dO):
        raise NotImplementedError("Decode pass is not implemented for nsa.")


class NativeSparseAttentionFunc(Function):

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim,
                 is_causal,
                 scale=None,
                 block_size=64,
                 groups=1,
                 selected_blocks=16,
                 tune=False):

        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.scale = scale
        self.block_size = block_size
        self.groups = groups
        self.selected_blocks = selected_blocks
        self.tune = tune

        self.fwd_op = NativeSparseAttentionForwardOp(
            batch,
            heads,
            seq_len,
            dim,
            is_causal,
            scale,
            block_size,
            groups,
            selected_blocks,
            tune=tune)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                BlockIndices: torch.Tensor) -> torch.Tensor:
        return nsa_decode_ctx.apply(Q, K, V, BlockIndices, self.fwd_op)
