import torch
from .op import Op
from top.kernels.kernel import Kernel
from top.kernels.flash_decode import gqa_decode_kernel
from typing import Optional, Dict

__all__ = ["GroupQueryAttentionDecodeWithKVCacheOp"]


class GroupQueryAttentionDecodeWithKVCacheOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 heads,
                 groups,
                 seqlen_kv,
                 dim,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
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
    def default_kernel_map(self):
        return {"gqa_decode_kernel": gqa_decode_kernel}

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.kernel(Q, K, V)
