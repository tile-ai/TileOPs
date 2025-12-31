import torch
from torch.autograd.function import FunctionCtx
from .function import Function
from top.ops import GroupQueryAttentionDecodeWithKVCacheOp
from typing import Any

__all__ = [
    'GroupQueryAttentionDecodeWithKVCacheFunc', 'group_query_attention_decode_with_kvcache',
    'gqa_decode_with_kvcache'
]


class GQADecodeCtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                fwd_op: GroupQueryAttentionDecodeWithKVCacheOp) -> torch.Tensor:
        """Forward pass for group query attention with KV cache.
        
        Args:
            ctx: Context object for saving tensors for backward pass
            q: Query tensor of shape (B, H, D)
            k: Key tensor of shape (B, S_kv, G, D)
            v: Value tensor of shape (B, S_kv, G, D)
            fwd_op: Forward operation instance
            
        Returns:
            Output tensor of the same shape as input q
        """
        O = fwd_op(q, k, v)
        return O

    @staticmethod
    def backward(ctx: FunctionCtx, do: torch.Tensor) -> Any:
        """Backward pass for group query attention with KV cache.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            do: Gradient of the output tensor
            
        Raises:
            RuntimeError: Inference-only op
        """
        raise RuntimeError("Inference-only op")


class GroupQueryAttentionDecodeWithKVCacheFunc(Function):

    def __init__(self,
                 batch: int,
                 heads: int,
                 groups: int,
                 seqlen_kv: int,
                 dim: int,
                 dtype: torch.dtype = torch.float16,
                 tune: bool = False):
        """Initialize the function with configuration parameters.
        
        Args:
            batch: Batch size
            heads: Number of attention heads
            groups: Number of key-value groups
            seqlen_kv: Sequence length of key-value
            dim: Head dimension
            dtype: Data type, defaults to torch.float16
            tune: Whether to tune the operation, defaults to False
        """
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype

        self.fwd_op = GroupQueryAttentionDecodeWithKVCacheOp(
            batch, heads, groups, seqlen_kv, dim, dtype, tune=tune)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return GQADecodeCtx.apply(q, k, v, self.fwd_op)


def group_query_attention_decode_with_kvcache(q: torch.Tensor,
                                              k_cache: torch.Tensor,
                                              v_cache: torch.Tensor,
                                              tune: bool = False) -> torch.Tensor:
    """Apply group query attention decode with KV cache mechanism to input tensors.

    This function performs group query attention computation where keys and values are cached,
    typically used in autoregressive decoding scenarios to avoid recomputing previous tokens.

    Args:
        q: Query tensor of shape (B, H, D), where B is batch size, 
           H is number of heads, D is head dimension
        k_cache: Key tensor of shape (B, S_kv, G, D) - represents the cached keys,
           where S_kv is key-value sequence length, G is number of key-value groups
        v_cache: Value tensor of shape (B, S_kv, G, D) - represents the cached values
        tune: Whether to tune the operation for performance, defaults to False

    Returns:
        Output tensor of shape (B, H, D) after applying group query attention with KV cache

    Raises:
        ValueError: If input tensors are not 4-dimensional or have inconsistent shapes/dtypes
    """

    # Validate that q, k_cache, v_cache are 4-dimensional tensors
    if q.dim() != 3:
        raise ValueError(f"q must be 3-dimensional, but got {q.dim()} dimensions")
    if k_cache.dim() != 4:
        raise ValueError(f"k_cache must be 4-dimensional, but got {k_cache.dim()} dimensions")
    if v_cache.dim() != 4:
        raise ValueError(f"v_cache must be 4-dimensional, but got {v_cache.dim()} dimensions")

    # Validate that dimensions are consistent (B, H/G, S, D)
    # B and D must be the same across q, k_cache, v_cache
    if q.shape[0] != k_cache.shape[0] or q.shape[0] != v_cache.shape[0]:
        raise ValueError(f"q, k_cache, v_cache must have the same batch size, "
                         f"but got q: {q.shape[0]}, k: {k_cache.shape[0]}, v: {v_cache.shape[0]}")
    if q.shape[-1] != k_cache.shape[-1] or q.shape[-1] != v_cache.shape[-1]:
        raise ValueError(
            f"q, k_cache, v_cache must have the same embedding dimension, "
            f"but got q: {q.shape[-1]}, k: {k_cache.shape[-1]}, v: {v_cache.shape[-1]}")
    if k_cache.shape[1] != v_cache.shape[1]:
        raise ValueError(f"k and v must have the same sequence length, "
                         f"but got k: {k_cache.shape[1]}, v: {v_cache.shape[1]}")

    # Validate that the number of heads (q.shape[1]) is a multiple of number of groups (k_cache.shape[2])
    if q.shape[1] % k_cache.shape[2] != 0:
        raise ValueError(f"Number of query heads must be a multiple of number of groups, "
                         f"but got q heads: {q.shape[2]}, k groups: {k_cache.shape[2]}")

    # Validate that dtypes are consistent
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError(f"q, k_cache, v_cache must have the same dtype, "
                         f"but got q: {q.dtype}, k: {k_cache.dtype}, v: {v_cache.dtype}")

    # Extract dimension information
    B = q.shape[0]  # Batch size
    H = q.shape[1]  # Number of heads
    D = q.shape[2]  # Head dimension
    S_kv = k_cache.shape[1]  # Sequence length of KV cache
    G = k_cache.shape[2]  # Number of groups

    return GroupQueryAttentionDecodeWithKVCacheFunc(
        B, H, G, S_kv, D, q.dtype, tune=tune).forward(
            q=q, k=k_cache, v=v_cache)


gqa_decode_with_kvcache = group_query_attention_decode_with_kvcache