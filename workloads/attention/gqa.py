import torch

from tileops.ops import GroupedQueryAttentionFwdOp
from workloads.workload_base import WorkloadBase


class GroupedQueryAttentionBwdTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len: int, dim: int,
                 is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
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
            self.heads_kv,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        v = torch.randn(
            self.batch,
            self.seq_len,
            self.heads_kv,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        grad_output = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device='cuda')

        fwd_op = GroupedQueryAttentionFwdOp(self.batch, self.heads, self.heads_kv, self.seq_len,
                                            self.dim, self.is_causal, self.dtype)
        with torch.no_grad():
            o, lse = fwd_op(q, k, v)

        return q, k, v, o, grad_output, lse

class GroupedQueryAttentionFwdTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len: int, dim: int,
                 is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        k = torch.randn(
            self.batch, self.seq_len, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        v = torch.randn(
            self.batch, self.seq_len, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        return q, k, v
