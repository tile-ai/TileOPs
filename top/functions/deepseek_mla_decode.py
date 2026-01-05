import torch
from torch.autograd.function import FunctionCtx
from .function import Function
from top.ops import MultiHeadLatentAttentionDecodeWithKVCacheOp
from typing import Any

__all__ = [
    'MultiHeadLatentAttentionDecodeWithKVCacheFunc',
    'multi_head_latent_attention_decode_with_kvcache', 'mla_decode_with_kvcache'
]


class MLADecodeCtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor,
                k_pe: torch.Tensor,
                fwd_op: MultiHeadLatentAttentionDecodeWithKVCacheOp) -> torch.Tensor:
        """Forward pass for multi-head latent attention with KV cache.

        Args:
            ctx: Context object for saving tensors for backward pass
            q: Query tensor of shape (B, H, D)
            q_pe: Query position encoding tensor of shape (B, H, pe_dim)
            k: Key tensor of shape (B, S_kv, H_kv, D), where H_kv is number of kv heads
            k_pe: Key position encoding tensor of shape (B, S_kv, H_kv, pe_dim)
            fwd_op: Forward operation instance

        Returns:
            Output tensor of the same shape as input q
        """
        o = fwd_op(q, q_pe, k, k_pe)
        return o

    @staticmethod
    def backward(ctx: FunctionCtx, do: torch.Tensor) -> Any:
        """Backward pass for multi-head latent attention with KV cache.

        Args:
            ctx: Context object containing saved tensors from forward pass
            do Gradient of the output tensor

        Raises:
            RuntimeError: Inference-only op
        """
        raise RuntimeError("Inference-only op")


class MultiHeadLatentAttentionDecodeWithKVCacheFunc(Function):

    def __init__(self,
                 batch: int,
                 heads: int,
                 kv_head_num: int,
                 seqlen_kv: int,
                 dim: int,
                 pe_dim: int,
                 dtype: torch.dtype = torch.float16,
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

    def forward(self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor,
                k_pe: torch.Tensor) -> torch.Tensor:
        return MLADecodeCtx.apply(q, q_pe, k, k_pe, self.fwd_op)


def multi_head_latent_attention_decode_with_kvcache(q: torch.Tensor,
                                                    q_pe: torch.Tensor,
                                                    k: torch.Tensor,
                                                    k_pe: torch.Tensor,
                                                    tune: bool = False) -> torch.Tensor:
    """Apply multi-head latent attention decode with KV cache mechanism to input tensors.

    This function performs multi-head latent attention computation where keys and values are cached,
    typically used in autoregressive decoding scenarios to avoid recomputing previous tokens.

    Args:
        q: Query tensor of shape (B, H, D), where B is batch size,
           H is number of heads, D is head dimension
        q_pe: Query position encoding tensor of shape (B, H, pe_dim)
        k: Key tensor of shape (B, S_kv, H_kv, D) - represents the cached keys,
           where S_kv is key-value sequence length, H_kv is number of kv heads
        k_pe: Key position encoding tensor of shape (B, S_kv, H_kv, pe_dim)
        tune: Whether to tune the operation for performance, defaults to False

    Returns:
        Output tensor of shape (B, H, D) after applying multi-head latent attention with KV cache

    Raises:
        ValueError: If input tensors are not 4-dimensional or have inconsistent shapes/dtypes
    """

    # Validate that q, k, q_pe, k_pe are 4-dimensional tensors
    if q.dim() != 3:
        raise ValueError(f"q must be 3-dimensional, but got {q.dim()} dimensions")
    if k.dim() != 4:
        raise ValueError(f"k must be 4-dimensional, but got {k.dim()} dimensions")
    if q_pe.dim() != 3:
        raise ValueError(f"q_pe must be 3-dimensional, but got {q_pe.dim()} dimensions")
    if k_pe.dim() != 4:
        raise ValueError(f"k_pe must be 4-dimensional, but got {k_pe.dim()} dimensions")

    # Validate that dimensions are consistent (B, H/G, S, D)
    # B and D must be the same across q, k
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"q and k must have the same batch size, "
                         f"but got q: {q.shape[0]}, k: {k.shape[0]}")
    if q.shape[-1] != k.shape[3]:
        raise ValueError(f"q and k must have the same embedding dimension, "
                         f"but got q: {q.shape[-1]}, k: {k.shape[3]}")

    # Validate that position encoding dimensions match
    if q_pe.shape[0] != k_pe.shape[0]:
        raise ValueError(f"q_pe and k_pe must have the same batch size, "
                         f"but got q_pe: {q_pe.shape[0]}, k_pe: {k_pe.shape[0]}")
    if q_pe.shape[-1] != k_pe.shape[-1]:
        raise ValueError(f"q_pe and k_pe must have the same embedding dimension, "
                         f"but got q_pe: {q_pe.shape[-1]}, k_pe: {k_pe.shape[-1]}")

    # Validate that dtypes are consistent
    if q.dtype != k.dtype:
        raise ValueError(f"q and k must have the same dtype, "
                         f"but got q: {q.dtype}, k: {k.dtype}")
    if q_pe.dtype != k_pe.dtype:
        raise ValueError(f"q_pe and k_pe must have the same dtype, "
                         f"but got q_pe: {q_pe.dtype}, k_pe: {k_pe.dtype}")
    if q.dtype != q_pe.dtype:
        raise ValueError(f"q and q_pe must have the same dtype, "
                         f"but got q: {q.dtype}, q_pe: {q_pe.dtype}")

    # Extract dimension information
    B = q.shape[0]  # Batch size
    H = q.shape[1]  # Number of query heads
    D = q.shape[2]  # Head dimension
    S_kv = k.shape[1]  # Sequence length of KV cache
    H_kv = k.shape[2]  # Number of KV heads
    pe_dim = q_pe.shape[2]  # Position encoding dimension
    return MultiHeadLatentAttentionDecodeWithKVCacheFunc(
        B, H, H_kv, S_kv, D, pe_dim, q.dtype, tune=tune).forward(
            q=q, q_pe=q_pe, k=k, k_pe=k_pe)


mla_decode_with_kvcache = multi_head_latent_attention_decode_with_kvcache
