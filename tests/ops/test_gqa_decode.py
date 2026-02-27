from typing import Tuple

import torch
import pytest
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import TestBase, FixtureBase
from tileops.ops import GroupQueryAttentionDecodeWithKVCacheOp


class GqaDecodeFixture(FixtureBase):
    PARAMS = [
        ("b, h, g, s_kv, d, dtype, tune", [
            (1, 32, 8, 8192, 128, torch.float16, False),
            (4, 32, 4, 4096, 128, torch.bfloat16, False),
            (8, 64, 16, 8192, 128, torch.float16, False),
        ]),
    ]


class GqaDecodeTest(TestBase):

    def __init__(self, batch: int, heads: int, groups: int, seq_len_kv: int, dim: int,
                 dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(self.batch, self.heads, self.dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len_kv, self.groups, self.dim, device='cuda', dtype=self.dtype)
        V = torch.randn(
            self.batch, self.seq_len_kv, self.groups, self.dim, device='cuda', dtype=self.dtype)
        return Q, K, V

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.unsqueeze(1).transpose(1, 2)  # [B, H, 1, D]
        k_bhsd = k.transpose(1, 2)  # [B, H, S_kv, D]
        v_bhsd = v.transpose(1, 2)  # [B, H, S_kv, D]
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_bhsd = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).squeeze(1).contiguous()
        return output


@GqaDecodeFixture
def test_gqa_decode(b: int, h: int, g: int, s_kv: int, d: int, dtype: torch.dtype,
                    tune: bool) -> None:
    test = GqaDecodeTest(b, h, g, s_kv, d, dtype)
    op = GroupQueryAttentionDecodeWithKVCacheOp(b, h, g, s_kv, d, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
