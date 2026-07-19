import torch

from tileops.ops import MultiHeadAttentionFwdOp
from workloads.attention.gqa import _compute_gqa_square_lse
from workloads.workload_base import WorkloadBase


class MhaBwdTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool,
                 dtype: torch.dtype):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        k = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        v = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        grad_output = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device='cuda')

        fwd_op = MultiHeadAttentionFwdOp(self.batch, self.heads, self.seq_len, self.dim,
                                         self.is_causal, self.dtype)
        with torch.no_grad():
            result = fwd_op(q, k, v)
            o = result[0] if isinstance(result, tuple) else result
            lse = _compute_gqa_square_lse(
                q,
                k,
                heads=self.heads,
                heads_kv=self.heads,
                dim=self.dim,
                is_causal=self.is_causal,
            )

        return q, k, v, o, grad_output, lse


class MhaFwdTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool,
                 dtype: torch.dtype):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        k = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        v = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        return q, k, v
