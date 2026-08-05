from typing import Dict, Optional

import torch

from tileops.kernels.attention import MLADecodeWsKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op

__all__ = ["MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"]


class MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seqlen_kv: int,
                 dim: int,
                 pe_dim: int,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim

        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[torch.dtype, Kernel] = {}

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        if dtype not in self._kernel_cache:
            self._kernel_cache[dtype] = self.kernel_map["mla_decode_kernel"](
                self.batch, self.heads, self.heads_kv, self.seqlen_kv,
                self.dim, self.pe_dim, dtype, tune=self.tune)
        return self._kernel_cache[dtype]

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mla_decode_kernel": MLADecodeWsKernel}

    def forward(self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor,
                k_pe: torch.Tensor) -> torch.Tensor:
        self._validate_dtypes(q, q_pe, k, k_pe)
        self.dtype = q.dtype
        return self._get_kernel(q.dtype)(q, q_pe, k, k_pe)
