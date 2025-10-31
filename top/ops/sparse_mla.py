import torch
from .op import Op
from top.kernels import sparse_mla_kernel, Kernel
from typing import Optional, Dict

__all__ = ["sparse_mla"]


class sparse_mla(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 seq_len_kv,
                 dim,
                 tail_dim,
                 topk,
                 kv_stride,
                 kv_group,
                 q_start_index_s,
                 sm_scale=None,
                 is_causal=True,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.tail_dim = tail_dim
        self.topk = topk
        self.kv_stride = kv_stride
        self.kv_group = kv_group
        self.sm_scale = sm_scale
        self.dtype = dtype
        self.is_causal = is_causal

        if q_start_index_s != 0:
            assert q_start_index_s > kv_stride, "If it is because each cp has too short length, "
            "you should fix the logic involving CP0 (cp_rank == 0), to make sure q with pos < KV_Stride - 1 is masked "
            "(or you may just ignore how this is handled if nan in these q's Out would not effect others, which is reported to be likely to happen by wangding)"
        
        CP0 = q_start_index_s == 0
        self.q_start_index_s = torch.tensor([q_start_index_s], dtype=torch.int32, device='cuda')

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["sparse_mla_kernel"](self.batch, self.seq_len, self.seq_len_kv, self.heads,
                                       self.dim, self.tail_dim, self.dtype, self.topk, self.kv_stride,
                                       self.kv_group, self.sm_scale, self.is_causal, CP0, tune=tune)

    @property
    def default_kernel_map(self):
        return {"sparse_mla_kernel": sparse_mla_kernel}

    def forward(self, Q: torch.Tensor, KV: torch.Tensor, Indices: torch.Tensor):
        return self.kernel(Q, KV, Indices, self.q_start_index_s)
