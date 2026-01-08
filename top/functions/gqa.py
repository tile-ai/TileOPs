from typing import Tuple

import torch
from torch.autograd.function import FunctionCtx

from top.ops import GroupQueryAttentionBwdOp, GroupQueryAttentionFwdOp

from .function import Function

__all__ = ['GroupQueryAttentionFunc', 'group_query_attention', 'gqa']


class GQACtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                fwd_op: GroupQueryAttentionFwdOp, bwd_op: GroupQueryAttentionBwdOp) -> torch.Tensor:
        """Forward pass for group query attention.

        Args:
            q: Query tensor of shape (B, S, H, D)
            k: Key tensor of shape (B, S, H_kv, D)
            v: Value tensor of shape (B, S, H_kv, D)
            fwd_op: Forward operation object
            bwd_op: Backward operation object

        Returns:
            Output tensor of shape (B, S, H, D)
        """
        O, lse = fwd_op(q, k, v)

        ctx.save_for_backward(q, k, v, O, lse)
        ctx.bwd_op = bwd_op

        return O

    @staticmethod
    def backward(ctx: FunctionCtx,
                 do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """Backward pass for group query attention.

        Args:
            do: Gradient of output tensor of shape (B, S, H, D)

        Returns:
            A tuple containing:
            - Gradient of q tensor of shape (B, S, H, D)
            - Gradient of k tensor of shape (B, S, H_kv, D)
            - Gradient of v tensor of shape (B, S, H_kv, D)
            - None for fwd_op gradient
            - None for bwd_op gradient
        """
        q, k, v, O, lse = ctx.saved_tensors

        dq, dk, dv = ctx.bwd_op(q, k, v, O, do, lse)

        return dq, dk, dv, None, None


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

        self.fwd_op = GroupQueryAttentionFwdOp(batch,
                                               heads,
                                               heads_kv,
                                               seq_len,
                                               dim,
                                               is_causal,
                                               dtype,
                                               tune=tune)
        self.bwd_op = GroupQueryAttentionBwdOp(batch,
                                               heads,
                                               heads_kv,
                                               seq_len,
                                               dim,
                                               is_causal,
                                               dtype,
                                               tune=tune)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Forward pass for group query attention.

        Args:
            q: Query tensor of shape (B, S, H, D)
            k: Key tensor of shape (B, S, H_kv, D)
            v: Value tensor of shape (B, S, H_kv, D)

        Returns:
            Output tensor of shape (B, S, H, D)
        """
        return GQACtx.apply(q, k, v, self.fwd_op, self.bwd_op)


def group_query_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          is_causal: bool = False,
                          tune: bool = False) -> torch.Tensor:
    """Apply group query attention mechanism to input tensors.

    Args:
        q: Query tensor of shape (B, S, H, D)
        k: Key tensor of shape (B, S, H_kv, D)
        v: Value tensor of shape (B, S, H_kv, D)
        is_causal: Whether to apply causal mask, defaults to False
        tune: Whether to tune the operation, defaults to False

    Returns:
        Output tensor of shape (B, S, H, D) after applying group query attention

    Raises:
        ValueError: If input tensors are not 4-dimensional or have inconsistent shapes/dtypes
    """

    # Validate that q, k, v are 4-dimensional tensors
    if q.dim() != 4:
        raise ValueError(f"q must be 4-dimensional, but got {q.dim()} dimensions")
    if k.dim() != 4:
        raise ValueError(f"k must be 4-dimensional, but got {k.dim()} dimensions")
    if v.dim() != 4:
        raise ValueError(f"v must be 4-dimensional, but got {v.dim()} dimensions")

    # Validate that dtypes are consistent
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q, k, v must have the same dtype, "
                         f"but got q: {q.dtype}, k: {k.dtype}, v: {v.dtype}")

    # Extract dimension information
    B, S, H, D = q.shape

    B, S, H_kv, D = k.shape

    if H % H_kv != 0:
        raise ValueError(
            f"The number of query heads H must be divisible by the number of key/value heads H_kv, "
            f"but got H: {H}, H_kv: {H_kv}")

    return GroupQueryAttentionFunc(B, H, H_kv, S, D, is_causal, q.dtype, tune=tune).forward(q, k, v)


gqa = group_query_attention
