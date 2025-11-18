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
        B_T = B.transpose(-2, -1).contiguous()
        A_T = A.transpose(-2, -1).contiguous()
        dA = ctx.da_bwd_op(dO, B_T)
        dB = ctx.db_bwd_op(A_T, dO)
        
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
        self.fwd_op = Gemm(M, N, K, dtype, tune=tune)
        self.da_bwd_op = Gemm(M, K, N, dtype, tune=tune)
        self.db_bwd_op = Gemm(K, N, M, dtype, tune=tune)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return gemm_ctx.apply(A, B, self.fwd_op, self.da_bwd_op, self.db_bwd_op)