import torch
from .function import Function
from top.ops import MultiHeadLatentAttentionDecodeWithKVCacheOp

__all__ = [
    'MultiHeadLatentAttentionDecodeWithKVCacheFunc',
    'multi_head_latent_attention_decode_with_kvcache'
]


class mla_decode_ctx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: torch.Tensor, Q_pe: torch.Tensor, K: torch.Tensor, K_pe: torch.Tensor,
                fwd_op: MultiHeadLatentAttentionDecodeWithKVCacheOp) -> torch.Tensor:
        """Forward pass for multi-head latent attention with KV cache.
        
        Args:
            ctx: Context object for saving tensors for backward pass
            Q: Query tensor of shape (B, H, D)
            Q_pe: Query position encoding tensor of shape (B, H, pe_dim)
            K: Key tensor of shape (B, S_kv, H_kv, D), where H_kv is number of kv heads
            K_pe: Key position encoding tensor of shape (B, S_kv, H_kv, pe_dim)
            fwd_op: Forward operation instance
            
        Returns:
            Output tensor of the same shape as input Q
        """
        O = fwd_op(Q, Q_pe, K, K_pe)
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        """Backward pass for multi-head latent attention with KV cache.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            dO: Gradient of the output tensor
            
        Raises:
            NotImplementedError: Backward pass is not implemented for mla_decode
        """
        raise NotImplementedError("Backward pass is not implemented for mla_decode.")


class MultiHeadLatentAttentionDecodeWithKVCacheFunc(Function):

    def __init__(self,
                 batch: int,
                 heads: int,
                 kv_head_num: int,
                 seqlen_kv: int,
                 dim: int,
                 pe_dim: int,
                 dtype=torch.float16,
                 tune: bool = False):
        """Initialize the function with configuration parameters.
        
        Args:
            batch: Batch size
            heads: Number of attention heads
            kv_head_num: Number of key-value heads
            seqlen_kv: Sequence length of key-value
            dim: Head dimension
            pe_dim: Position encoding dimension
            dtype: Data type, defaults to torch.float16
            tune: Whether to tune the operation, defaults to False
        """
        self.batch = batch
        self.heads = heads
        self.kv_head_num = kv_head_num
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim

        self.dtype = dtype

        self.fwd_op = MultiHeadLatentAttentionDecodeWithKVCacheOp(
            batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype, tune=tune)

    def forward(self, Q: torch.Tensor, Q_pe: torch.Tensor, K: torch.Tensor,
                K_pe: torch.Tensor) -> torch.Tensor:
        return mla_decode_ctx.apply(Q, Q_pe, K, K_pe, self.fwd_op)


def multi_head_latent_attention_decode_with_kvcache(Q: torch.Tensor,
                                                    Q_pe: torch.Tensor,
                                                    K: torch.Tensor,
                                                    K_pe: torch.Tensor,
                                                    tune: bool = False) -> torch.Tensor:
    """Apply multi-head latent attention decode with KV cache mechanism to input tensors.

    This function performs multi-head latent attention computation where keys and values are cached,
    typically used in autoregressive decoding scenarios to avoid recomputing previous tokens.

    Args:
        Q: Query tensor of shape (B, H, D), where B is batch size, 
           H is number of heads, D is head dimension
        Q_pe: Query position encoding tensor of shape (B, H, pe_dim)
        K: Key tensor of shape (B, S_kv, H_kv, D) - represents the cached keys,
           where S_kv is key-value sequence length, H_kv is number of kv heads
        K_pe: Key position encoding tensor of shape (B, S_kv, H_kv, pe_dim)
        tune: Whether to tune the operation for performance, defaults to False

    Returns:
        Output tensor of shape (B, H, D) after applying multi-head latent attention with KV cache

    Raises:
        ValueError: If input tensors are not 4-dimensional or have inconsistent shapes/dtypes
    """

    # Validate that Q, K, Q_pe, K_pe are 4-dimensional tensors
    if Q.dim() != 3:
        raise ValueError(f"Q must be 3-dimensional, but got {Q.dim()} dimensions")
    if K.dim() != 4:
        raise ValueError(f"K must be 4-dimensional, but got {K.dim()} dimensions")
    if Q_pe.dim() != 3:
        raise ValueError(f"Q_pe must be 3-dimensional, but got {Q_pe.dim()} dimensions")
    if K_pe.dim() != 4:
        raise ValueError(f"K_pe must be 4-dimensional, but got {K_pe.dim()} dimensions")

    # Validate that dimensions are consistent (B, H/G, S, D)
    # B and D must be the same across Q, K
    if Q.shape[0] != K.shape[0]:
        raise ValueError(f"Q and K must have the same batch size, "
                         f"but got Q: {Q.shape[0]}, K: {K.shape[0]}")
    if Q.shape[-1] != K.shape[3]:
        raise ValueError(f"Q and K must have the same embedding dimension, "
                         f"but got Q: {Q.shape[-1]}, K: {K.shape[3]}")

    # Validate that position encoding dimensions match
    if Q_pe.shape[0] != K_pe.shape[0]:
        raise ValueError(f"Q_pe and K_pe must have the same batch size, "
                         f"but got Q_pe: {Q_pe.shape[0]}, K_pe: {K_pe.shape[0]}")
    if Q_pe.shape[-1] != K_pe.shape[-1]:
        raise ValueError(f"Q_pe and K_pe must have the same embedding dimension, "
                         f"but got Q_pe: {Q_pe.shape[-1]}, K_pe: {K_pe.shape[-1]}")

    # Validate that dtypes are consistent
    if Q.dtype != K.dtype:
        raise ValueError(f"Q and K must have the same dtype, "
                         f"but got Q: {Q.dtype}, K: {K.dtype}")
    if Q_pe.dtype != K_pe.dtype:
        raise ValueError(f"Q_pe and K_pe must have the same dtype, "
                         f"but got Q_pe: {Q_pe.dtype}, K_pe: {K_pe.dtype}")
    if Q.dtype != Q_pe.dtype:
        raise ValueError(f"Q and Q_pe must have the same dtype, "
                         f"but got Q: {Q.dtype}, Q_pe: {Q_pe.dtype}")

    # Extract dimension information
    B = Q.shape[0]  # Batch size
    H = Q.shape[1]  # Number of query heads
    D = Q.shape[2]  # Head dimension
    S_kv = K.shape[1]  # Sequence length of KV cache
    H_kv = K.shape[2]  # Number of KV heads
    pe_dim = Q_pe.shape[2]  # Position encoding dimension
    return MultiHeadLatentAttentionDecodeWithKVCacheFunc(
        B, H, H_kv, S_kv, D, pe_dim, Q.dtype, tune=tune).forward(
            Q=Q, Q_pe=Q_pe, K=K, K_pe=K_pe)
