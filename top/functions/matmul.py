import torch
from .function import Function
from top.ops import GemmOp
from typing import Tuple

__all__ = ['MatMulFunc', 'matmul']


class gemm_ctx(torch.autograd.Function):
    """Autograd function for GEMM operation.
    Handles forward and backward passes for general matrix multiplication.
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor, fwd_op: GemmOp, da_bwd_op: GemmOp,
                db_bwd_op: GemmOp) -> torch.Tensor:
        """Forward pass for GEMM operation.
        
        Args:
            ctx: Context object for saving tensors for backward pass
            A: Input tensor A of shape (M, K)
            B: Input tensor B of shape (K, N)
            fwd_op: Forward operation instance
            da_bwd_op: Backward operation instance for tensor A
            db_bwd_op: Backward operation instance for tensor B
            
        Returns:
            Output tensor of shape (M, N) after matrix multiplication
        """
        O = fwd_op(A, B)

        ctx.save_for_backward(A, B)
        ctx.da_bwd_op = da_bwd_op
        ctx.db_bwd_op = db_bwd_op

        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:
        """Backward pass for GEMM operation.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            dO: Gradient of the output tensor
            
        Returns:
            Gradients w.r.t. A, B and None for non-tensor parameters
        """
        A, B = ctx.saved_tensors

        dO = dO.contiguous()
        dA = ctx.da_bwd_op(dO, B)
        dB = ctx.db_bwd_op(A, dO)

        return dA, dB, None, None, None


class MatMulFunc(Function):
    """Function class for GEMM (General Matrix Multiplication) operation.
    
    This function performs general matrix multiplication with optimized forward and backward passes.
    """

    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        dtype=torch.float16,
        tune=False,
    ):
        """Initialize the function with configuration parameters.
        
        Args:
            M: First dimension of the output matrix (rows of A and output)
            N: Second dimension of the output matrix (columns of B and output)  
            K: Shared dimension of the input matrices (columns of A and rows of B)
            dtype: Data type, defaults to torch.float16
            tune: Whether to tune the operation, defaults to False
        """
        self.fwd_op = GemmOp(M, N, K, dtype=dtype, tune=tune)
        self.da_bwd_op = GemmOp(M, K, N, dtype=dtype, trans_B=False, tune=tune)
        self.db_bwd_op = GemmOp(K, N, M, dtype=dtype, trans_A=False, tune=tune)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return gemm_ctx.apply(A, B, self.fwd_op, self.da_bwd_op, self.db_bwd_op)


def matmul(A: torch.Tensor, B: torch.Tensor, tune: bool = False) -> torch.Tensor:
    """Perform matrix multiplication of two tensors A and B.

    This function computes the matrix multiplication of A and B using optimized GEMM operations.

    Args:
        A: Input tensor A of shape (M, K)
        B: Input tensor B of shape (K, N)
        tune: Whether to tune the operation for performance, defaults to False

    Returns:
        Output tensor of shape (M, N) after matrix multiplication

    Raises:
        ValueError: If input tensors are not 2-dimensional or have inconsistent shapes/dtypes
    """

    # Validate that A, B are 2-dimensional tensors
    if A.dim() != 2:
        raise ValueError(f"A must be 2-dimensional, but got {A.dim()} dimensions")
    if B.dim() != 2:
        raise ValueError(f"B must be 2-dimensional, but got {B.dim()} dimensions")

    # Validate that dimensions are consistent for matrix multiplication (A: MxK, B: KxN)
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"A and B dimensions are not consistent for matrix multiplication, "
                         f"A: {A.shape}, B: {B.shape}. The K dimension of A ({A.shape[1]}) "
                         f"must match the K dimension of B ({B.shape[0]})")

    # Validate that dtypes are consistent
    if A.dtype != B.dtype:
        raise ValueError(f"A and B must have the same dtype, "
                         f"but got A: {A.dtype}, B: {B.dtype}")

    # Extract dimension information
    M = A.shape[0]  # Rows of A and output
    K = A.shape[1]  # Columns of A and rows of B
    N = B.shape[1]  # Columns of B and output

    return MatMulFunc(M, N, K, A.dtype, tune=tune).forward(A=A, B=B)
