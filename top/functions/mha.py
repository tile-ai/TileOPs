import torch
from .function import Function
from top.ops import MultiHeadAttentionFwdOp, MultiHeadAttentionBwdOp
from typing import Tuple

__all__ = ['MultiHeadAttentionFunc', 'multi_head_attention']


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
            Q: Query tensor of shape (B, S, H, D)
            K: Key tensor of shape (B, S, H, D)
            V: Value tensor of shape (B, S, H, D)
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
        return mha_ctx.apply(Q, K, V, self.fwd_op, self.bwd_op)


def multi_head_attention(Q: torch.Tensor,
                         K: torch.Tensor,
                         V: torch.Tensor,
                         is_causal: bool = False,
                         tune: bool = False) -> torch.Tensor:
    """Apply multi-head attention mechanism to input tensors.

    Args:
        Q: Query tensor of shape (B, S, H, D)
        K: Key tensor of shape (B, S, H, D)
        V: Value tensor of shape (B, S, H, D)
        is_causal: Whether to apply causal mask, defaults to False
        tune: Whether to tune the operation, defaults to False

    Returns:
        Output tensor of shape (B, S, H, D) after applying multi-head attention

    Raises:
        ValueError: If input tensors are not 4-dimensional or have inconsistent shapes/dtypes
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
    B = Q.shape[0]
    S = Q.shape[1]
    H = Q.shape[2]
    D = Q.shape[3]

    return MultiHeadAttentionFunc(B, H, S, D, is_causal, Q.dtype, tune=tune).forward(Q=Q, K=K, V=V)
