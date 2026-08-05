from typing import Dict, Optional

import torch

from tileops.kernels.attention import SparseMlaKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op

__all__ = ["DeepSeekSparseAttentionDecodeWithKVCacheFwdOp"]


class DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(Op):
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
        heads_kv (int): The number of key-value heads.
        q_start_index_s (int): The start index for queries in the sequence.
        sm_scale (Optional[float], default=None): Scaling factor for the softmax function.
        is_causal (bool, default=True): Whether the attention is causal
                    (True for causal, False for non-causal).
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
                 heads_kv: int,
                 q_start_index_s: int,
                 sm_scale: Optional[float] = None,
                 is_causal: bool = True,
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
        self.heads_kv = heads_kv
        self.sm_scale = sm_scale
        self.is_causal = is_causal

        if q_start_index_s != 0 and q_start_index_s <= stride_kv:
            raise ValueError(f"Invalid q_start_index_s={q_start_index_s}:"
                             f"must be > stride_kv={stride_kv}. "
                             "This indicates incorrect cp0 masking."
                             "Ensure queries with pos < stride_kv are masked "
                             "to avoid NaNs in early outputs.")

        cp0 = q_start_index_s == 0
        self.q_start_index_s = q_start_index_s

        self._cp0 = cp0
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[torch.dtype, Kernel] = {}

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        if dtype not in self._kernel_cache:
            self._kernel_cache[dtype] = self.kernel_map["sparse_mla_kernel"](
                self.batch,
                self.seq_len,
                self.seq_len_kv,
                self.heads,
                self.dim,
                self.dim_tail,
                dtype,
                self.topk,
                self.stride_kv,
                self.q_start_index_s,
                self.heads_kv,
                self.sm_scale,
                self.is_causal,
                self._cp0,
                tune=self.tune)
        return self._kernel_cache[dtype]

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        """
        Provides the default kernel map for the operation.

        Returns:
            Dict[str, Kernel]: A dictionary mapping kernel names to kernel functions.
            The default map includes the "sparse_mla_kernel".
        """
        return {"sparse_mla_kernel": SparseMlaKernel}

    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the sparse attention operation.

        Args:
            q (torch.Tensor): The query tensor with shape
                        (batch, seq_len, heads, dim + dim_tail).
            kv (torch.Tensor): The key-value tensor with shape
                        (batch, seq_len_kv, heads_kv, dim + dim_tail).
            indices (torch.Tensor): Indices tensor for sparse attention.

        Returns:
            torch.Tensor: The result of applying the sparse attention
                            operation on the input tensors.
        """
        self._validate_dtypes(q, kv, indices)
        self.dtype = q.dtype
        return self._get_kernel(q.dtype)(q, kv, indices)
