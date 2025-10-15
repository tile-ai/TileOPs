import torch
from torch.nn import functional as F
from .op import Op
from top.kernels import (mha_fwd_kernel, mha_fwd_wgmma_pipelined_kernel, mha_bwd_preprocess_kernel,
                         mha_bwd_kernel, mha_bwd_wgmma_pipelined_kernel, Kernel)
from top.utils import is_hopper
from typing import Optional, Dict

__all__ = ['mha_fwd', 'mha_bwd']


class mha_fwd(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim,
                 is_causal,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  #TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.input_shapes = [(batch, seq_len, heads, dim) for _ in range(3)]

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["mha_fwd_kernel"](
            batch, heads, seq_len, dim, is_causal, self.dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {"mha_fwd_kernel": mha_fwd_wgmma_pipelined_kernel if is_hopper() else mha_fwd_kernel}

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


class mha_bwd(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim,
                 is_causal,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  #TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        # Does not need to provide input shapes, as we override gen_inputs()

        self.dispatch_kernel(kernel_map)
        self.prep_kernel = self.kernel_map["mha_bwd_preprocess_kernel"](batch, heads, seq_len, dim,
                                                                        self.dtype)
        self.kernel = self.kernel_map["mha_bwd_kernel"](
            batch, heads, seq_len, dim, is_causal, self.dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {
            "mha_bwd_preprocess_kernel": mha_bwd_preprocess_kernel,
            "mha_bwd_kernel": mha_bwd_wgmma_pipelined_kernel if is_hopper() else mha_bwd_kernel,
        }

    @property
    def total_flops(self):  # use property to support dynamic shape in the future
        flops_per_matmul = 2.0 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim
        flops = flops_per_matmul * 5
        return flops / 2 if self.is_causal else flops

    @property
    def total_memory(self):
        return None

    def gen_inputs(self):
        Q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        K = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        V = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        dO = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device='cuda')

        fwd_op = mha_fwd(self.batch, self.heads, self.seq_len, self.dim, self.is_causal, self.dtype)
        with torch.no_grad():
            O, lse = fwd_op(Q, K, V)

        return Q, K, V, O, dO, lse

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
                dO: torch.Tensor, lse: torch.Tensor):
        delta = self.prep_kernel(O, dO)
        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK, dV = self.kernel(Q, K, V, dO, lse, delta, dQ)
        dQ = dQ.to(self.dtype)
        return dQ, dK, dV

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
                    dO: torch.Tensor, lse: torch.Tensor):
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

        output.backward(dO)
        return Q.grad, K.grad, V.grad
