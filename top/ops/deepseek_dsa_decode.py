import torch
from .op import Op
from top.kernels.deepseek_mla import sparse_mla_kernel
from top.kernels.kernel import Kernel
from typing import Optional, Dict

__all__ = ["DeepSeekSparseAttentionDecodeWithKVCacheOp"]


class DeepSeekSparseAttentionDecodeWithKVCacheOp(Op):
    """Layout: BSHD"""

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
            assert q_start_index_s > stride_kv, "If it is because each cp has too short length, " \
                "you should fix the logic involving CP0 (cp_rank == 0), to make sure q with pos < stride_kv - 1 is masked " \
                "(or you may just ignore how this is handled if nan in these q's Out would not effect others, which is reported to be likely to happen by wangding)"

        CP0 = q_start_index_s == 0
        self.q_start_index_s = q_start_index_s

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["sparse_mla_kernel"](
            self.batch,
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
            CP0,
            tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"sparse_mla_kernel": sparse_mla_kernel}

    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, kv, indices)
