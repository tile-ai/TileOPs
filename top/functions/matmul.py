import torch
from .function import Function
from top.ops.gemm import Gemm

__all__ = ['matmul']


class gemm_ctx(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, A, B, fwd_op, da_bwd_op, db_bwd_op):
        O = fwd_op(A, B)
        
        ctx.save_for_backward(A, B)
        ctx.da_bwd_op = da_bwd_op
        ctx.db_bwd_op = db_bwd_op
        
        return O
    
    @staticmethod
    def backward(ctx, dO):
        A, B = ctx.saved_tensors
        
        dO = dO.contiguous()
        dA = ctx.da_bwd_op(dO, B)
        dB = ctx.db_bwd_op(A, dO)
        
        return dA, dB, None, None, None

class matmul(Function):

    def __init__(
            self,
            M: int,
            N: int,
            K: int,
            dtype=torch.float16,
            tune=False,
    ):
        self.fwd_op = Gemm(M, N, K, dtype=dtype, tune=tune)
        self.da_bwd_op = Gemm(M, K, N, dtype=dtype, trans_B=False, tune=tune)
        self.db_bwd_op = Gemm(K, N, M, dtype=dtype, trans_A=False, tune=tune)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return gemm_ctx.apply(A, B, self.fwd_op, self.da_bwd_op, self.db_bwd_op)