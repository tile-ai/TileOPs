import torch
from .op import Op
from top.kernels import mha_decode_kernel, Kernel
from typing import Optional, Dict

__all__ = ["MultiHeadAttentionDecodeOp"]


class MultiHeadAttentionDecodeOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 heads,
                 seqlen_q,
                 seqlen_kv,
                 dim,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["mha_decode_kernel"](
            batch, heads, seqlen_q, seqlen_kv, dim, False, self.dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {"mha_decode_kernel": mha_decode_kernel}

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.kernel(Q, K, V)
