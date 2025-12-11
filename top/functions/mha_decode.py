import torch
from .function import Function
from top.ops.mha_decode import mha_decode

__all__ = ['mha_decode_fn']


class mha_decode_ctx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, fwd_op):
        O = fwd_op(Q, K, V)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("Backward pass is not implemented for mha_decode.")


class mha_decode_fn(Function):

    def __init__(self, batch, heads, seqlen_q, seqlen_kv, dim, dtype=torch.float16, tune=False):
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype

        self.fwd_op = mha_decode(batch, heads, seqlen_q, seqlen_kv, dim, dtype, tune=tune)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return mha_decode_ctx.apply(Q, K, V, self.fwd_op)
