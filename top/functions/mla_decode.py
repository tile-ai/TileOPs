import torch
from .function import Function
from top.ops.mla_decode import mla_decode


__all__ = ['mla_decode_fn']


class mla_decode_ctx(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, Q, Q_pe, K, K_pe, fwd_op):
        O = fwd_op(Q, Q_pe, K, K_pe)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("Backward pass is not implemented for mla_decode.")


class mla_decode_fn(Function):

    def __init__(self,
                 batch,
                 heads,
                 kv_head_num,
                 seqlen_kv,
                 dim,
                 pe_dim,
                 dtype=torch.float16,
                 tune=False):
        self.batch = batch
        self.heads = heads
        self.kv_head_num = kv_head_num
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim

        self.dtype = dtype

        self.fwd_op = mla_decode(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype, tune=tune)

    
    def forward(self, Q: torch.Tensor, Q_pe: torch.Tensor, K: torch.Tensor, K_pe: torch.Tensor) -> torch.Tensor:
        return mla_decode_ctx.apply(Q, Q_pe, K, K_pe, self.fwd_op)