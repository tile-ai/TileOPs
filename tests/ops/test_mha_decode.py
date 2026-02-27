from typing import Tuple

import torch
import pytest
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import TestBase, FixtureBase
from tileops.ops import MultiHeadAttentionDecodeWithKVCacheOp


class MhaDecodeFixture(FixtureBase):
    PARAMS = [
        ("b, h, s_q, s_kv, d, dtype, tune", [
            (1, 32, 128, 8192, 128, torch.float16, False),
            (1, 32, 128, 8192, 128, torch.bfloat16, False),
            (1, 32, 128, 5, 128, torch.float16, False),
        ]),
    ]


class MhaDecodeTest(TestBase):

    def __init__(self, batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                 dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(
            self.batch, self.seq_len_q, self.heads, self.dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len_kv, self.heads, self.dim, device='cuda', dtype=self.dtype)
        V = torch.randn(
            self.batch, self.seq_len_kv, self.heads, self.dim, device='cuda', dtype=self.dtype)
        return Q, K, V

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.transpose(1, 2)  # [B, H, S_q, D]
        k_bhsd = k.transpose(1, 2)  # [B, H, S_kv, D]
        v_bhsd = v.transpose(1, 2)  # [B, H, S_kv, D]
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output


@MhaDecodeFixture
def test_mha_decode(b: int, h: int, s_q: int, s_kv: int, d: int, dtype: torch.dtype,
                    tune: bool) -> None:
    test = MhaDecodeTest(b, h, s_q, s_kv, d, dtype)
    op = MultiHeadAttentionDecodeWithKVCacheOp(b, h, s_q, s_kv, d, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=2e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
