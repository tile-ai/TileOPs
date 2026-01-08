from typing import Dict, Optional

import torch.nn.functional as F
import torch

from top.kernels.flash_decode import gqa_decode_kernel
from top.kernels.kernel import Kernel

from .op import Op

__all__ = ["GroupQueryAttentionDecodeWithKVCacheOp"]


class GroupQueryAttentionDecodeWithKVCacheOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 groups: int,
                 seqlen_kv: int,
                 dim: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_decode_kernel"](
            batch, heads, groups, seqlen_kv, dim, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_decode_kernel": gqa_decode_kernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        real_seqlen_kv = k.shape[1]
        if real_seqlen_kv < self.seqlen_kv:
            k = F.pad(k, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)
            v = F.pad(v, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)

        return self.kernel(q, k, v, real_seqlen_kv)