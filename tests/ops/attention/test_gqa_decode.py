
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GroupedQueryAttentionDecodeWithKVCacheFwdOp
from workloads.attention.gqa_decode import (
    GroupedQueryAttentionDecodeTest as _GroupedQueryAttentionDecodeTestWorkload,
)


class GroupedQueryAttentionDecodeTest(_GroupedQueryAttentionDecodeTestWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.unsqueeze(1).transpose(1, 2)  # [B, H, 1, D]
        k_bhsd = k.transpose(1, 2)  # [B, H, S_kv, D]
        v_bhsd = v.transpose(1, 2)  # [B, H, S_kv, D]
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_bhsd = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).squeeze(1).contiguous()
        return output


class GroupedQueryAttentionDecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, heads_kv, seq_len_kv, dim, dtype, tune", [
            pytest.param(1, 32, 8, 8192, 128, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 32, 8, 8192, 128, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(8, 64, 16, 8192, 128, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


@GroupedQueryAttentionDecodeFixture
def test_gqa_decode(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                    dtype: torch.dtype, tune: bool) -> None:
    pytest.skip("Temporarily skipping known GQA decode failures under TileLang 5f70374c (#999).")
    test = GroupedQueryAttentionDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dtype)
    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(batch, heads, heads_kv, seq_len_kv, dim, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
