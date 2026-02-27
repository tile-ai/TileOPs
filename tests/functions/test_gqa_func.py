from typing import Tuple, Union

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import TestBase, FixtureBase
from tileops.functions import GroupQueryAttentionFunc, gqa


class GqaFuncFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, heads_kv, dim, causal, dtype", [
            (8, 1024, 32, 8, 128, False, torch.float16),
        ]),
    ]


class GqaFuncTest(TestBase):
    """Combined fwd+bwd test for GQA function-level API."""

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
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device='cuda',
            requires_grad=True)
        k = torch.randn(
            self.batch, self.seq_len, self.heads_kv, self.dim, dtype=self.dtype, device='cuda',
            requires_grad=True)
        v = torch.randn(
            self.batch, self.seq_len, self.heads_kv, self.dim, dtype=self.dtype, device='cuda',
            requires_grad=True)
        return q, k, v

    def ref_program(self, q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> Union[torch.Tensor, tuple]:
        q_bhsd = q.transpose(1, 2)
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()

        loss = output.sum()
        loss.backward()
        return output, q.grad, k.grad, v.grad


@GqaFuncFixture
def test_gqa_fn(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                dtype: torch.dtype) -> None:
    test = GqaFuncTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    inputs = test.gen_inputs()

    print("=========Testing gqa function inference=========")
    test.check_fn(gqa, *inputs, atol=3e-4, rtol=1e-5)

    print("=========Testing gqa function class=========")
    fn = GroupQueryAttentionFunc(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    test.check_fn(fn, *inputs, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
