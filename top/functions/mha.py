from typing import Tuple

import torch
from torch.autograd.function import FunctionCtx

from top.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp

from .function import Function

__all__ = ['MultiHeadAttentionFunc', 'multi_head_attention', 'mha']


class MHACtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                fwd_op: MultiHeadAttentionFwdOp, bwd_op: MultiHeadAttentionBwdOp) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            q: Query tensor of shape (B, S, H, D)
            k: Key tensor of shape (B, S, H, D)
            v: Value tensor of shape (B, S, H, D)
            fwd_op: Forward operation instance
            bwd_op: Backward operation instance

        Returns:
            Output tensor of the same shape as input q
        """
        O, lse = fwd_op(q, k, v)

        ctx.save_for_backward(q, k, v, O, lse)
        ctx.bwd_op = bwd_op

        return O

    @staticmethod
    def backward(ctx: FunctionCtx,
                 do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        Backward pass for multi-head attention.

        Args:
            do: Gradient of the output tensor

        Returns:
            Gradients w.r.t. q, k, v
        """
        q, k, v, O, lse = ctx.saved_tensors

        dQ, dK, dV = ctx.bwd_op(q, k, v, O, do, lse)

        return dQ, dK, dV, None, None


class MultiHeadAttentionFunc(Function):

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype = torch.float16,
                 tune: bool = False):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.fwd_op = MultiHeadAttentionFwdOp(batch,
                                              heads,
                                              seq_len,
                                              dim,
                                              is_causal,
                                              dtype,
                                              tune=tune)
        self.bwd_op = MultiHeadAttentionBwdOp(batch,
                                              heads,
                                              seq_len,
                                              dim,
                                              is_causal,
                                              dtype,
                                              tune=tune)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return MHACtx.apply(q, k, v, self.fwd_op, self.bwd_op)


def multi_head_attention(q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         is_causal: bool = False,
                         tune: bool = False) -> torch.Tensor:
    """Apply multi-head attention mechanism to input tensors.

    Args:
        q: Query tensor of shape (B, S, H, D)
        k: Key tensor of shape (B, S, H, D)
        v: Value tensor of shape (B, S, H, D)
        is_causal: Whether to apply causal mask, defaults to False
        tune: Whether to tune the operation, defaults to False

    Returns:
        Output tensor of shape (B, S, H, D) after applying multi-head attention

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
    # Validate that dimensions are consistent (B, H, S, D)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must have the same shape, "
                         f"but got q: {q.shape}, k: {k.shape}, v: {v.shape}")

    # Validate that dtypes are consistent
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q, k, v must have the same dtype, "
                         f"but got q: {q.dtype}, k: {k.dtype}, v: {v.dtype}")

    # Extract dimension information
    B = q.shape[0]
    S = q.shape[1]
    H = q.shape[2]
    D = q.shape[3]

    return MultiHeadAttentionFunc(B, H, S, D, is_causal, q.dtype, tune=tune).forward(q=q, k=k, v=v)


mha = multi_head_attention
