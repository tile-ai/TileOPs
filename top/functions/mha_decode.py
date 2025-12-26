import torch
from .function import Function
from top.ops import MultiHeadAttentionDecodeWithKVCacheOp

__all__ = ['MultiHeadAttentionDecodeWithKVCacheFunc', 'multi_head_attention_decode_with_kvcache']


class mha_decode_ctx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                fwd_op: MultiHeadAttentionDecodeWithKVCacheOp) -> torch.Tensor:
        """Forward pass for multi-head attention with KV cache.
        
        Args:
            ctx: Context object for saving tensors for backward pass
            Q: Query tensor of shape (B, S_q, H, D)
            K: Key tensor of shape (B, S_kv, H, D)
            V: Value tensor of shape (B, S_kv, H, D)
            fwd_op: Forward operation instance
            
        Returns:
            Output tensor of the same shape as input Q
        """
        O = fwd_op(Q, K, V)
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        """Backward pass for multi-head attention with KV cache.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            dO: Gradient of the output tensor
            
        Raises:
            NotImplementedError: Backward pass is not implemented for mha_decode
        """
        raise NotImplementedError("Backward pass is not implemented for mha_decode.")


class MultiHeadAttentionDecodeWithKVCacheFunc(Function):

    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen_q: int,
                 seqlen_kv: int,
                 dim: int,
                 dtype=torch.float16,
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

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return mha_decode_ctx.apply(Q, K, V, self.fwd_op)


def multi_head_attention_decode_with_kvcache(Q: torch.Tensor,
                                             K: torch.Tensor,
                                             V: torch.Tensor,
                                             tune: bool = False) -> torch.Tensor:
    """Apply multi-head attention decode with KV cache mechanism to input tensors.

    Args:
        Q: Query tensor of shape (B, S_q, H, D)
        K: Key tensor of shape (B, S_kv, H, D) - represents the cached keys
        V: Value tensor of shape (B, S_kv, H, D) - represents the cached values
        tune: Whether to tune the operation, defaults to False

    Returns:
        Output tensor of shape (B, S_q, H, D) after applying multi-head attention with KV cache

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
    # B and H must be the same across Q, K, V
    if Q.shape[0] != K.shape[0] or Q.shape[0] != V.shape[0]:
        raise ValueError(f"Q, K, V must have the same batch size, "
                         f"but got Q: {Q.shape[0]}, K: {K.shape[0]}, V: {V.shape[0]}")
    if Q.shape[2] != K.shape[2] or Q.shape[2] != V.shape[2]:
        raise ValueError(f"Q, K, V must have the same number of heads, "
                         f"but got Q: {Q.shape[2]}, K: {K.shape[2]}, V: {V.shape[2]}")
    if K.shape[1] != V.shape[1]:
        raise ValueError(f"K and V must have the same sequence length, "
                         f"but got K: {K.shape[1]}, V: {V.shape[1]}")

    # Check that the embedding dimension is consistent
    if Q.shape[3] != K.shape[3] or Q.shape[3] != V.shape[3]:
        raise ValueError(f"Q, K, V must have the same embedding dimension, "
                         f"but got Q: {Q.shape[3]}, K: {K.shape[3]}, V: {V.shape[3]}")

    # Validate that dtypes are consistent
    if Q.dtype != K.dtype or Q.dtype != V.dtype:
        raise ValueError(f"Q, K, V must have the same dtype, "
                         f"but got Q: {Q.dtype}, K: {K.dtype}, V: {V.dtype}")

    # Extract dimension information
    B = Q.shape[0]
    S_q = Q.shape[1]  # Sequence length of Query
    H = Q.shape[2]  # Number of heads
    D = Q.shape[3]  # Embedding dimension
    S_kv = K.shape[1]  # Sequence length of KV cache

    return MultiHeadAttentionDecodeWithKVCacheFunc(
        B, H, S_q, S_kv, D, Q.dtype, tune=tune).forward(
            Q=Q, K=K, V=V)
