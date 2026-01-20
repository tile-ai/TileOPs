from typing import Tuple

import torch
from torch.autograd.function import FunctionCtx

from top.ops import GemmOp

from .function import Function

__all__ = ['MatMulFunc', 'matmul']


class GemmCtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, a: torch.Tensor, b: torch.Tensor, fwd_op: GemmOp,
                da_bwd_op: GemmOp, db_bwd_op: GemmOp) -> torch.Tensor:
        """Forward pass for GEMM operation.

        Args:
            ctx: Context object for saving tensors for backward pass
            a: Input tensor A of shape (M, K)
            b: Input tensor B of shape (K, N)
            fwd_op: Forward operation instance
            da_bwd_op: Backward operation instance for tensor A
            db_bwd_op: Backward operation instance for tensor B

        Returns:
            Output tensor of shape (M, N) after matrix multiplication
        """
        ctx.save_for_backward(a, b)
        ctx.da_bwd_op = da_bwd_op
        ctx.db_bwd_op = db_bwd_op

        return fwd_op(a, b)

    @staticmethod
    def backward(
            ctx: FunctionCtx,
            do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:  # noqa VNE002
        """Backward pass for GEMM operation.

        Args:
            ctx: Context object containing saved tensors from forward pass
            do: Gradient of the output tensor

        Returns:
            Gradients w.r.t. a, b and None for non-tensor parameters
        """
        a, b = ctx.saved_tensors

        do = do.contiguous()  # noqa VNE002
        da = ctx.da_bwd_op(do, b)
        db = ctx.db_bwd_op(a, do)
        return da, db, None, None, None


class MatMulFunc(Function):

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype = torch.float16,
        tune: bool = False,
    ):
        """Initialize the function with configuration parameters.

        Args:
            m: First dimension of the output matrix (rows of A and output)
            n: Second dimension of the output matrix (columns of B and output)
            k: Shared dimension of the input matrices (columns of A and rows of B)
            dtype: Data type, defaults to torch.float16
            tune: Whether to tune the operation, defaults to False
        """
        self.fwd_op = GemmOp(m, n, k, dtype=dtype, tune=tune)
        self.da_bwd_op = GemmOp(m, k, n, dtype=dtype, trans_b=False, tune=tune)
        self.db_bwd_op = GemmOp(k, n, m, dtype=dtype, trans_a=False, tune=tune)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return GemmCtx.apply(a, b, self.fwd_op, self.da_bwd_op, self.db_bwd_op)


def matmul(a: torch.Tensor, b: torch.Tensor, tune: bool = False) -> torch.Tensor:
    """Perform matrix multiplication of two tensors a and b.

    This function computes the matrix multiplication of a and b using optimized GEMM operations.

    Args:
        a: Input tensor a of shape (M, K)
        b: Input tensor b of shape (K, N)
        tune: Whether to tune the operation for performance, defaults to False

    Returns:
        Output tensor of shape (M, N) after matrix multiplication

    Raises:
        ValueError: If input tensors are not 2-dimensional or have inconsistent shapes/dtypes
    """
    # Validate that a, b are 2-dimensional tensors
    if a.dim() != 2:
        raise ValueError(f"a must be 2-dimensional, but got {a.dim()} dimensions")
    if b.dim() != 2:
        raise ValueError(f"b must be 2-dimensional, but got {b.dim()} dimensions")

    # Validate that dimensions are consistent for matrix multiplication (a: MxK, b: KxN)
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"a and b dimensions are not consistent for matrix multiplication, "
                         f"a: {a.shape}, b: {b.shape}. The K dimension of a ({a.shape[1]}) "
                         f"must match the K dimension of b ({b.shape[0]})")

    # Validate that dtypes are consistent
    if a.dtype != b.dtype:
        raise ValueError(f"a and b must have the same dtype, "
                         f"but got a: {a.dtype}, b: {b.dtype}")

    # Extract dimension information
    m = a.shape[0]  # Rows of a and output
    k = a.shape[1]  # Columns of a and rows of b
    n = b.shape[1]  # Columns of b and output

    return MatMulFunc(m, n, k, a.dtype, tune=tune).forward(a=a, b=b)
