import torch
from torch.nn import functional as F
from .op import Op
from top.kernels import mha_fwd_kernel, mha_fwd_wgmma_pipelined_kernel
from top.utils import is_hopper


__all__ = ['mha_fwd']


class mha_fwd(Op):
    """Layout: BSHD"""

    def __init__(self, batch, heads, seq_len, dim, is_causal, dtype=torch.float16):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  #TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.input_shapes = [(batch, seq_len, heads, dim) for _ in range(3)]

        # TODO: dispatch to different kernels based on archs and input shapes
        mha_fwd_kernel_type = mha_fwd_wgmma_pipelined_kernel if is_hopper() else mha_fwd_kernel
        self.kernel = mha_fwd_kernel_type(batch, heads, seq_len, dim, is_causal, self.dtype)

    @property
    def total_flops(self):  # use property to support dynamic shape in the future
        flops_per_matmul = 2.0 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim
        return flops_per_matmul if self.is_causal else flops_per_matmul * 2
            
    @property
    def total_memory(self):
        return None

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.kernel(Q, K, V)

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        dim = Q.size(-1)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
        scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
        if self.is_causal:
            seq_len = Q.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
        return output, None  # do not check lse