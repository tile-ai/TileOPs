from typing import Dict, Optional

import torch

from tileops.kernels.flash_decode.mha_decode_paged import (
    MHADecodePagedKernel,
    MHADecodePagedSplitKVKernel,
    MHADecodePagedCombineKernel,
)
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["MultiHeadAttentionDecodePagedWithKVCacheOp"]


class MultiHeadAttentionDecodePagedWithKVCacheOp(Op):
    """Paged MHA decode with dynamic KV cache. Layout: Q [batch, seqlen_q, heads, dim] (BSHD);
    K, V physical cache [seqlen_kv, heads, dim]; real_seqlen_kv [batch]; block_table [batch, num_pages].
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen_q: int,
                 seqlen_kv: int,
                 dim: int,
                 page_size: int,
                 is_causal: bool = False,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)

        # Construct the no-split kernel
        self.no_split_kernel = self.kernel_map["MHADecodePagedKernel"](
            batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, self.dtype, tune=tune)

        # Construct the split kernel
        self.split_kernel = self.kernel_map["MHADecodePagedSplitKVKernel"](
            batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, self.dtype, tune=tune)

        # Read num_split from the split kernel's autotuned/default config
        num_split = self.split_kernel.config["num_split"]

        # Construct the combine kernel with matching num_split
        self.combine_kernel = self.kernel_map["MHADecodePagedCombineKernel"](
            batch, heads, seqlen_q, num_split, dim, self.dtype, tune=tune)

        # Expose the split kernel as the primary kernel (for backward compat)
        self.kernel = self.split_kernel

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "MHADecodePagedKernel": MHADecodePagedKernel,
            "MHADecodePagedSplitKVKernel": MHADecodePagedSplitKVKernel,
            "MHADecodePagedCombineKernel": MHADecodePagedCombineKernel,
        }

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        # Dispatch: use no-split for short sequences
        num_split = self.split_kernel.config["num_split"]
        block_N = self.split_kernel.config["block_N"]
        threshold = num_split * block_N

        real_max = real_seqlen_kv.max().item() if real_seqlen_kv.dim() > 0 else real_seqlen_kv.item()

        if real_max < threshold:
            return self.no_split_kernel(q, k, v, real_seqlen_kv, block_table)

        # Split path: compute cumulative per-split lengths
        chunk_size = real_max // (num_split * block_N) * block_N
        split_length = torch.full((self.batch, num_split), chunk_size, dtype=torch.int32,
                                  device=q.device)
        split_length[:, -1] = int(real_max - (num_split - 1) * chunk_size)
        acc_split_length = torch.cumsum(split_length, dim=1).to(torch.int32)

        # Run split kernel (returns glse and Output_partial)
        glse, Output_partial = self.split_kernel(
            q, k, v, real_seqlen_kv, block_table, acc_split_length)

        # Run combine kernel
        return self.combine_kernel(glse, Output_partial)
