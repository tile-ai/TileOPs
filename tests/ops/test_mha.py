from typing import Optional, Tuple

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import TestBase, FixtureBase
from tileops.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp


class MhaFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim, causal, dtype, tune", [
            (1, 1024, 8, 64, False, torch.float16, False),
            (16, 2048, 16, 128, False, torch.float16, False),
            (4, 4096, 16, 128, False, torch.bfloat16, True),
        ]),
    ]


class MhaBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim, causal, dtype, tune", [
            (1, 1024, 8, 64, False, torch.float16, False),
            (16, 2048, 16, 128, False, torch.float16, False),
            (4, 4096, 16, 128, False, torch.bfloat16, True),
        ]),
    ]


class MhaFwdTest(TestBase):

    def __init__(self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool,
                 dtype: torch.dtype):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        k = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        v = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        return q, k, v

    def ref_program(self, q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output, None  # do not check lse


class MhaBwdTest(TestBase):

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
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal)
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


@MhaFwdFixture
def test_mha_fwd(batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype,
                 tune: bool) -> None:
    test = MhaFwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@MhaBwdFixture
def test_mha_bwd(batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype,
                 tune: bool) -> None:
    test = MhaBwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
