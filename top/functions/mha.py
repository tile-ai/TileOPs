import torch
from .function import Function
from top.ops import MultiHeadAttentionFwdOp, MultiHeadAttentionBwdOp
from typing import Tuple

__all__ = ['MultiHeadAttentionFunc']


class mha_ctx(torch.autograd.Function):
    """
    Autograd function for multi-head attention operation.
    Handles forward and backward passes for multi-head attention.
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                fwd_op: MultiHeadAttentionFwdOp, bwd_op: MultiHeadAttentionBwdOp) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            Q: Query tensor of shape (B, H, S, D)
            K: Key tensor of shape (B, H, S, D)
            V: Value tensor of shape (B, H, S, D)
            fwd_op: Forward operation instance
            bwd_op: Backward operation instance
            
        Returns:
            Output tensor of the same shape as input Q
        """
        O, lse = fwd_op(Q, K, V)

        ctx.save_for_backward(Q, K, V, O, lse)
        ctx.bwd_op = bwd_op

        return O

    @staticmethod
    def backward(ctx,
                 dO: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        Backward pass for multi-head attention.
        
        Args:
            dO: Gradient of the output tensor
            
        Returns:
            Gradients w.r.t. Q, K, V
        """
        Q, K, V, O, lse = ctx.saved_tensors

        dQ, dK, dV = ctx.bwd_op(Q, K, V, O, dO, lse)

        return dQ, dK, dV, None, None


class MultiHeadAttentionFunc(Function):
    """
    Multi-Head Attention Function implementation.
    
    This class provides a multi-head attention function with forward and backward passes.
    It validates input tensors and returns the computed result directly.
    """

    def __new__(cls,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                is_causal: bool = False,
                tune: bool = False):
        """
        Create and compute the MultiHeadAttentionFunc, returning the result tensor directly.
        
        Args:
            Q: Query tensor of shape (B, H, S, D)
            K: Key tensor of shape (B, H, S, D)
            V: Value tensor of shape (B, H, S, D)
            is_causal: Whether to apply causal masking (default: False)
            tune: Whether to tune the kernel for performance (default: False)
            
        Returns:
            Output tensor of shape (B, H, S, D)
            
        Raises:
            ValueError: If input tensors are invalid
        """
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
        # Create forward and backward operations
        fwd_op = MultiHeadAttentionFwdOp(B, H, S, D, is_causal, Q.dtype, tune=tune)
        bwd_op = MultiHeadAttentionBwdOp(B, H, S, D, is_causal, Q.dtype, tune=tune)

        # Return the computed result directly
        return mha_ctx.apply(Q, K, V, fwd_op, bwd_op)
