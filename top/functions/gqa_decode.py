import torch
from .function import Function
from top.ops import GroupQueryAttentionDecodeWithKVCacheOp

__all__ = [
    'GroupQueryAttentionDecodeWithKVCacheFunc', 'group_query_attention_decode_with_kvcache',
    'gqa_decode_with_kvcache'
]


class gqa_decode_ctx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor,
                fwd_op: GroupQueryAttentionDecodeWithKVCacheOp) -> torch.Tensor:
        """Forward pass for group query attention with KV cache.
        
        Args:
            ctx: Context object for saving tensors for backward pass
            Q: Query tensor of shape (B, H, D)
            K: Key tensor of shape (B, S_kv, G, D)
            V: Value tensor of shape (B, S_kv, G, D)
            mask: Attention mask tensor
            fwd_op: Forward operation instance
            
        Returns:
            Output tensor of the same shape as input Q
        """
        O = fwd_op(Q, K, V, mask)
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        """Backward pass for group query attention with KV cache.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            dO: Gradient of the output tensor
            
        Raises:
            NotImplementedError: Backward pass is not implemented for gqa_decode
        """
        raise NotImplementedError("Backward pass is not implemented for gqa_decode.")


class GroupQueryAttentionDecodeWithKVCacheFunc(Function):

    def __init__(self,
                 batch: int,
                 heads: int,
                 groups: int,
                 seqlen_kv: int,
                 dim: int,
                 dtype=torch.float16,
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

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return gqa_decode_ctx.apply(Q, K, V, mask, self.fwd_op)


def group_query_attention_decode_with_kvcache(Q: torch.Tensor,
                                              K: torch.Tensor,
                                              V: torch.Tensor,
                                              mask: torch.Tensor,
                                              tune: bool = False) -> torch.Tensor:
    """Apply group query attention decode with KV cache mechanism to input tensors.

    This function performs group query attention computation where keys and values are cached,
    typically used in autoregressive decoding scenarios to avoid recomputing previous tokens.

    Args:
        Q: Query tensor of shape (B, H, D), where B is batch size, 
           H is number of heads, D is head dimension
        K: Key tensor of shape (B, S_kv, G, D) - represents the cached keys,
           where S_kv is key-value sequence length, G is number of key-value groups
        V: Value tensor of shape (B, S_kv, G, D) - represents the cached values
        mask: Attention mask tensor
        tune: Whether to tune the operation for performance, defaults to False

    Returns:
        Output tensor of shape (B, H, D) after applying group query attention with KV cache

    Raises:
        ValueError: If input tensors are not 4-dimensional or have inconsistent shapes/dtypes
    """

    # Validate that Q, K, V are 4-dimensional tensors
    if Q.dim() != 3:
        raise ValueError(f"Q must be 3-dimensional, but got {Q.dim()} dimensions")
    if K.dim() != 4:
        raise ValueError(f"K must be 4-dimensional, but got {K.dim()} dimensions")
    if V.dim() != 4:
        raise ValueError(f"V must be 4-dimensional, but got {V.dim()} dimensions")

    # Validate that dimensions are consistent (B, H/G, S, D)
    # B and D must be the same across Q, K, V
    if Q.shape[0] != K.shape[0] or Q.shape[0] != V.shape[0]:
        raise ValueError(f"Q, K, V must have the same batch size, "
                         f"but got Q: {Q.shape[0]}, K: {K.shape[0]}, V: {V.shape[0]}")
    if Q.shape[-1] != K.shape[-1] or Q.shape[-1] != V.shape[-1]:
        raise ValueError(f"Q, K, V must have the same embedding dimension, "
                         f"but got Q: {Q.shape[-1]}, K: {K.shape[-1]}, V: {V.shape[-1]}")
    if K.shape[1] != V.shape[1]:
        raise ValueError(f"K and V must have the same sequence length, "
                         f"but got K: {K.shape[1]}, V: {V.shape[1]}")

    # Validate that the number of heads (Q.shape[1]) is a multiple of number of groups (K.shape[2])
    if Q.shape[1] % K.shape[2] != 0:
        raise ValueError(f"Number of query heads must be a multiple of number of groups, "
                         f"but got Q heads: {Q.shape[2]}, K groups: {K.shape[2]}")

    # Validate that dtypes are consistent
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError(f"Q, K, V must have the same dtype, "
                         f"but got Q: {Q.dtype}, K: {K.dtype}, V: {V.dtype}")

    # Extract dimension information
    B = Q.shape[0]  # Batch size
    H = Q.shape[1]  # Number of heads
    D = Q.shape[2]  # Head dimension
    S_kv = K.shape[1]  # Sequence length of KV cache
    G = K.shape[2]  # Number of groups

    return GroupQueryAttentionDecodeWithKVCacheFunc(
        B, H, G, S_kv, D, Q.dtype, tune=tune).forward(
            Q=Q, K=K, V=V, mask=mask)


gqa_decode_with_kvcache = group_query_attention_decode_with_kvcache
