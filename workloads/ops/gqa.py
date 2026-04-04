from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tileops.ops import GroupQueryAttentionFwdOp
from workloads.base import WorkloadBase


class GqaBwdTest(WorkloadBase):

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        fwd_op = GroupQueryAttentionFwdOp(self.batch, self.heads, self.heads_kv, self.seq_len,
                                          self.dim, self.is_causal, self.dtype)
        with torch.no_grad():
            o, lse = fwd_op(q, k, v)

        return q, k, v, o, grad_output, lse

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                    grad_output: torch.Tensor,
                    lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


class GqaFwdTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len: int, dim: int,
                 is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def ref_program(self, q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output, None  # do not check lse
