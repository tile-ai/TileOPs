from typing import Dict, Optional

import torch

from top.kernels.deepseek_mla import sparse_mla_kernel
from top.kernels.kernel import Kernel

from .op import Op

__all__ = ["DeepSeekSparseAttentionDecodeWithKVCacheOp"]


class DeepSeekSparseAttentionDecodeWithKVCacheOp(Op):
    """
    Sparse Attention Decode Operation with Key-Value Cache for DeepSeek.

    This operation is part of a sparse attention mechanism, designed for use in decoding
    with key-value (KV) caching.

    The layout of the operation is BSHD.

    Args:
        batch (int): The batch size.
        heads (int): The number of attention heads.
        seq_len (int): The length of the input sequence.
        seq_len_kv (int): The length of the key-value sequence.
        dim (int): The dimension of the attention vectors.
        dim_tail (int): The dimension of the tail portion of the attention vectors.
        topk (int): The number of top elements to consider in sparse attention.
        stride_kv (int): The stride for the key-value sequence.
        group_kv (int): The number of key-value groups.
        q_start_index_s (int): The start index for queries in the sequence.
        sm_scale (Optional[float], default=None): Scaling factor for the softmax function.
        is_causal (bool, default=True): Whether the attention is causal
                    (True for causal, False for non-causal).
        dtype (torch.dtype, default=torch.float16): The data type for
                    the tensors used in the operation.
        kernel_map (Optional[Dict[str, Kernel]], default=None):
                    Optional mapping for custom kernels.
        tune (bool, default=False): Whether to enable kernel tuning.
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 seq_len_kv: int,
                 dim: int,
                 dim_tail: int,
                 topk: int,
                 stride_kv: int,
                 group_kv: int,
                 q_start_index_s: int,
                 sm_scale: Optional[float] = None,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_tail = dim_tail
        self.topk = topk
        self.stride_kv = stride_kv
        self.group_kv = group_kv
        self.sm_scale = sm_scale
        self.dtype = dtype
        self.is_causal = is_causal

        if q_start_index_s != 0:
            assert q_start_index_s > stride_kv, \
                "If it is because each cp has too short length, " \
                "you should fix the logic involving cp0 (cp_rank == 0),"\
                " to make sure q with pos < stride_kv - 1 is masked " \
                "(or you may just ignore how this is handled if nan in these q's Out"\
                "would not effect others, which is reported to be likely to happen by wangding)"

        cp0 = q_start_index_s == 0
        self.q_start_index_s = q_start_index_s

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["sparse_mla_kernel"](self.batch,
                                                           self.seq_len,
                                                           self.seq_len_kv,
                                                           self.heads,
                                                           self.dim,
                                                           self.dim_tail,
                                                           self.dtype,
                                                           self.topk,
                                                           self.stride_kv,
                                                           self.q_start_index_s,
                                                           self.group_kv,
                                                           self.sm_scale,
                                                           self.is_causal,
                                                           cp0,
                                                           tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        """
        Provides the default kernel map for the operation.

        Returns:
            Dict[str, Kernel]: A dictionary mapping kernel names to kernel functions.
            The default map includes the "sparse_mla_kernel".
        """
        return {"sparse_mla_kernel": sparse_mla_kernel}

    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the sparse attention operation.

        Args:
            q (torch.Tensor): The query tensor with shape
                        (batch, seq_len, heads, dim + dim_tail).
            kv (torch.Tensor): The key-value tensor with shape
                        (batch, seq_len_kv, group_kv, dim + dim_tail).
            indices (torch.Tensor): Indices tensor for sparse attention.

        Returns:
            torch.Tensor: The result of applying the sparse attention
                            operation on the input tensors.
        """
        return self.kernel(q, kv, indices)
