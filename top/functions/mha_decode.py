import torch
from torch.autograd.function import FunctionCtx
from .function import Function
from top.ops import MultiHeadAttentionDecodeWithKVCacheOp
from typing import Any

__all__ = [
    'MultiHeadAttentionDecodeWithKVCacheFunc',
    'multi_head_attention_decode_with_kvcache',
    'mha_decode_with_kvcache',
]


class MHADecodeCtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                fwd_op: MultiHeadAttentionDecodeWithKVCacheOp) -> torch.Tensor:
        """Forward pass for multi-head attention with KV cache.
        
        Args:
            ctx: Context object for saving tensors for backward pass
            q: Query tensor of shape (B, S_q, H, D)
            k: Key tensor of shape (B, S_kv, H, D)
            v: Value tensor of shape (B, S_kv, H, D)
            fwd_op: Forward operation instance
            
        Returns:
            Output tensor of the same shape as input q
        """
        O = fwd_op(q, k, v)
        return O

    @staticmethod
    def backward(ctx: FunctionCtx, do: torch.Tensor) -> Any:
        """Backward pass for multi-head attention with KV cache.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            do: Gradient of the output tensor
            
        Raises:
            RuntimeError: Inference-only op
        """
        raise RuntimeError("Inference-only op")


class MultiHeadAttentionDecodeWithKVCacheFunc(Function):

    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen_q: int,
                 seqlen_kv: int,
                 dim: int,
                 dtype: torch.dtype = torch.float16,
                 tune: bool = False):
        """Initialize the function with configuration parameters.
        
        Args:
            batch: Batch size
            heads: Number of attention heads
            seqlen_q: Sequence length of query
            seqlen_kv: Sequence length of key-value
            dim: Head dimension
            dtype: Data type, defaults to torch.float16
            tune: Whether to tune the operation, defaults to False
        """
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype

        self.fwd_op = MultiHeadAttentionDecodeWithKVCacheOp(
            batch, heads, seqlen_q, seqlen_kv, dim, dtype, tune=tune)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return MHADecodeCtx.apply(q, k, v, self.fwd_op)


def multi_head_attention_decode_with_kvcache(q: torch.Tensor,
                                             k_cache: torch.Tensor,
                                             v_cache: torch.Tensor,
                                             tune: bool = False) -> torch.Tensor:
    """Apply multi-head attention decode with KV cache mechanism to input tensors.

    Args:
        q: Query tensor of shape (B, S_q, H, D)
        k_cache: Key tensor of shape (B, S_kv, H, D) - represents the cached keys
        v_cache: Value tensor of shape (B, S_kv, H, D) - represents the cached values
        tune: Whether to tune the operation, defaults to False

    Returns:
        Output tensor of shape (B, S_q, H, D) after applying multi-head attention with KV cache

    Raises:
        ValueError: If input tensors are not 4-dimensional or have inconsistent shapes/dtypes
    """

    # Validate that q, k, v are 4-dimensional tensors
    if q.dim() != 4:
        raise ValueError(f"q must be 4-dimensional, but got {q.dim()} dimensions")
    if k_cache.dim() != 4:
        raise ValueError(f"k_cache must be 4-dimensional, but got {k_cache.dim()} dimensions")
    if v_cache.dim() != 4:
        raise ValueError(f"v_cache must be 4-dimensional, but got {v_cache.dim()} dimensions")

    # Validate that dimensions are consistent (B, H, S, D)
    # B and H must be the same across q, k, v
    if q.shape[0] != k_cache.shape[0] or q.shape[0] != v_cache.shape[0]:
        raise ValueError(f"q, k_cache, v_cache must have the same batch size, "
                         f"but got q: {q.shape[0]}, k: {k_cache.shape[0]}, v: {v_cache.shape[0]}")
    if q.shape[2] != k_cache.shape[2] or q.shape[2] != v_cache.shape[2]:
        raise ValueError(f"q, k_cache, v_cache must have the same number of heads, "
                         f"but got q: {q.shape[2]}, k: {k_cache.shape[2]}, v: {v_cache.shape[2]}")
    if k_cache.shape[1] != v_cache.shape[1]:
        raise ValueError(f"k_cache and v_cache must have the same sequence length, "
                         f"but got k: {k_cache.shape[1]}, v: {v_cache.shape[1]}")

    # Check that the embedding dimension is consistent
    if q.shape[3] != k_cache.shape[3] or q.shape[3] != v_cache.shape[3]:
        raise ValueError(f"q, k_cache, v_cache must have the same embedding dimension, "
                         f"but got q: {q.shape[3]}, k: {k_cache.shape[3]}, v: {v_cache.shape[3]}")

    # Validate that dtypes are consistent
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError(f"q, k_cache, v_cache must have the same dtype, "
                         f"but got q: {q.dtype}, k: {k_cache.dtype}, v: {v_cache.dtype}")

    # Extract dimension information
    B = q.shape[0]
    S_q = q.shape[1]  # Sequence length of Query
    H = q.shape[2]  # Number of heads
    D = q.shape[3]  # Embedding dimension
    S_kv = k_cache.shape[1]  # Sequence length of KV cache

    return MultiHeadAttentionDecodeWithKVCacheFunc(
        B, H, S_q, S_kv, D, q.dtype, tune=tune).forward(
            q=q, k=k_cache, v=v_cache)


mha_decode_with_kvcache = multi_head_attention_decode_with_kvcache