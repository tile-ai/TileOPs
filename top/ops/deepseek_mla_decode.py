import torch
from .op import Op
from top.kernels.deepseek_mla import mla_decode_kernel, mla_decode_ws_kernel
from top.kernels.kernel import Kernel
from top.utils import is_hopper
from typing import Optional, Dict

__all__ = ["MultiHeadLatentAttentionDecodeWithKVCacheOp"]


class MultiHeadLatentAttentionDecodeWithKVCacheOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 kv_head_num: int,
                 seqlen_kv: int,
                 dim: int,
                 pe_dim: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.kv_head_num = kv_head_num
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["mla_decode_kernel"](
            batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mla_decode_kernel": mla_decode_ws_kernel if is_hopper() else mla_decode_kernel}

    def forward(self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor,
                k_pe: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, q_pe, k, k_pe)
