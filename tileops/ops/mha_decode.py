from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.flash_decode import MHADecodeCombineKernel, MHADecodeKernel, MHADecodeSplitKVKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["MultiHeadAttentionDecodeWithKVCacheOp"]


class MultiHeadAttentionDecodeWithKVCacheOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen_q: int,
                 seqlen_kv: int,
                 dim: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)

        # Construct the no-split kernel
        self.no_split_kernel = self.kernel_map["MHADecodeKernel"](
            batch, heads, seqlen_q, seqlen_kv, dim, False, self.dtype, tune=tune)

        # Construct the split kernel
        self.split_kernel = self.kernel_map["MHADecodeSplitKVKernel"](
            batch, heads, seqlen_q, seqlen_kv, dim, False, self.dtype, tune=tune)

        # Read num_split from the split kernel's autotuned/default config
        num_split = self.split_kernel.config["num_split"]

        # Construct the combine kernel with matching num_split
        self.combine_kernel = self.kernel_map["MHADecodeCombineKernel"](
            batch, heads, seqlen_q, num_split, dim, self.dtype, tune=tune)

        # Expose the split kernel as the primary kernel (for backward compat)
        self.kernel = self.split_kernel

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "MHADecodeKernel": MHADecodeKernel,
            "MHADecodeSplitKVKernel": MHADecodeSplitKVKernel,
            "MHADecodeCombineKernel": MHADecodeCombineKernel,
        }

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        real_seqlen_kv = k.shape[1]
        if real_seqlen_kv < self.seqlen_kv:
            k = F.pad(
                k, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)
            v = F.pad(
                v, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)

        # Dispatch: use no-split for short sequences
        num_split = self.split_kernel.config["num_split"]
        block_N = self.split_kernel.config["block_N"]
        threshold = num_split * block_N

        if real_seqlen_kv < threshold:
            return self.no_split_kernel(q, k, v, real_seqlen_kv)

        # Split path: compute per-split lengths
        base_len = real_seqlen_kv // (num_split * block_N) * block_N
        split_length = torch.full((num_split,), base_len, dtype=torch.int32, device=q.device)
        split_length[-1] = real_seqlen_kv - (num_split - 1) * base_len

        # Run split kernel (returns glse and Output_partial)
        glse, Output_partial = self.split_kernel(
            q, k, v, real_seqlen_kv, split_length)

        # Run combine kernel
        return self.combine_kernel(glse, Output_partial)
