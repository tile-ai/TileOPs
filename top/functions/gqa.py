import torch
from .function import Function
from top.ops import GroupQueryAttentionFwdOp, GroupQueryAttentionBwdOp
from typing import Tuple

__all__ = ['GroupQueryAttentionFunc', 'group_query_attention', 'gqa']


class gqa_ctx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                fwd_op: GroupQueryAttentionFwdOp, bwd_op: GroupQueryAttentionBwdOp) -> torch.Tensor:
        """Forward pass for group query attention.
        
        Args:
            Q: Query tensor of shape (B, S, H, D)
            K: Key tensor of shape (B, S, H_kv, D)
            V: Value tensor of shape (B, S, H_kv, D)
            fwd_op: Forward operation object
            bwd_op: Backward operation object
            
        Returns:
            Output tensor of shape (B, S, H, D)
        """
        O, lse = fwd_op(Q, K, V)

        ctx.save_for_backward(Q, K, V, O, lse)
        ctx.bwd_op = bwd_op

        return O

    @staticmethod
    def backward(ctx,
                 dO: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """Backward pass for group query attention.
        
        Args:
            dO: Gradient of output tensor of shape (B, S, H, D)
            
        Returns:
            A tuple containing:
            - Gradient of Q tensor of shape (B, S, H, D)
            - Gradient of K tensor of shape (B, S, H_kv, D)
            - Gradient of V tensor of shape (B, S, H_kv, D)
            - None for fwd_op gradient
            - None for bwd_op gradient
        """
        Q, K, V, O, lse = ctx.saved_tensors

        dQ, dK, dV = ctx.bwd_op(Q, K, V, O, dO, lse)

        return dQ, dK, dV, None, None


class GroupQueryAttentionFunc(Function):

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype = torch.float16,
                 tune: bool = False):
        """Initialize the GroupQueryAttentionFunc with parameters.
        
        Args:
            batch: Batch size
            heads: Number of query heads
            heads_kv: Number of key-value heads
            seq_len: Sequence length
            dim: Dimension of the model
            is_causal: Whether to apply causal mask
            dtype: Data type of the tensors, defaults to torch.float16
            tune: Whether to tune the operation, defaults to False
        """
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.fwd_op = GroupQueryAttentionFwdOp(
            batch, heads, heads_kv, seq_len, dim, is_causal, dtype, tune=tune)
        self.bwd_op = GroupQueryAttentionBwdOp(
            batch, heads, heads_kv, seq_len, dim, is_causal, dtype, tune=tune)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Forward pass for group query attention.
        
        Args:
            Q: Query tensor of shape (B, S, H, D)
            K: Key tensor of shape (B, S, H_kv, D)
            V: Value tensor of shape (B, S, H_kv, D)
            
        Returns:
            Output tensor of shape (B, S, H, D)
        """
        return gqa_ctx.apply(Q, K, V, self.fwd_op, self.bwd_op)


def group_query_attention(Q: torch.Tensor,
                          K: torch.Tensor,
                          V: torch.Tensor,
                          is_causal: bool = False,
                          tune: bool = False) -> torch.Tensor:
    """Apply group query attention mechanism to input tensors.
    
    Args:
        Q: Query tensor of shape (B, S, H, D)
        K: Key tensor of shape (B, S, H_kv, D)
        V: Value tensor of shape (B, S, H_kv, D)
        is_causal: Whether to apply causal mask, defaults to False
        tune: Whether to tune the operation, defaults to False
        
    Returns:
        Output tensor of shape (B, S, H, D) after applying group query attention
        
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

    # Validate that dtypes are consistent
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError(f"Q, K, V must have the same dtype, "
                         f"but got Q: {Q.dtype}, K: {K.dtype}, V: {V.dtype}")

    # Extract dimension information
    B, S, H, D = Q.shape

    B, S, H_kv, D = K.shape

    if H % H_kv != 0:
        raise ValueError(
            f"The number of query heads H must be divisible by the number of key/value heads H_kv, "
            f"but got H: {H}, H_kv: {H_kv}")

    return GroupQueryAttentionFunc(B, H, H_kv, S, D, is_causal, Q.dtype, tune=tune).forward(Q, K, V)


gqa = group_query_attention
