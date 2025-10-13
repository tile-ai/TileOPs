import torch
from torch.nn import functional as F
from .function import Function
from top.kernels import mha_fwd_kernel, mha_fwd_wgmma_pipelined_kernel
from top.utils import is_hopper


__all__ = ['mha_fwd']


class mha_fwd(Function):
    """Layout: BSHD"""

    def __init__(self, batch, heads, seq_len, dim, is_causal):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  #TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = torch.float16  # TODO: support more dtypes

        self.input_shapes = [(batch, seq_len, heads, dim) for _ in range(3)]

        flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
        self.total_flops = 2 * flops_per_matmul
        if is_causal:
            self.total_flops *= 0.5

        # TODO: dispatch to different kernels based on archs and input shapes
        mha_fwd_kernel_type = mha_fwd_wgmma_pipelined_kernel if is_hopper() else mha_fwd_kernel
        self.kernel = mha_fwd_kernel_type(batch, heads, seq_len, dim, is_causal)

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

    # use default methods: gen_inputs(), check(), profile()