
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GroupedQueryAttentionBwdOp, GroupedQueryAttentionFwdOp
from workloads.attention.gqa import (
    GroupedQueryAttentionBwdTest as _GroupedQueryAttentionBwdTestWorkload,
)
from workloads.attention.gqa import (
    GroupedQueryAttentionFwdTest as _GroupedQueryAttentionFwdTestWorkload,
)


class GroupedQueryAttentionBwdTest(_GroupedQueryAttentionBwdTestWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                    grad_output: torch.Tensor,
                    lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


class GroupedQueryAttentionFwdTest(_GroupedQueryAttentionFwdTestWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output, None  # do not check lse


class GroupedQueryAttentionFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, heads_kv, dim, causal, dtype, tune", [
            pytest.param(1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 1024, 8, 4, 64, False, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


class GroupedQueryAttentionBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, heads_kv, dim, causal, dtype, tune", [
            pytest.param(1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 1024, 8, 4, 64, False, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@GroupedQueryAttentionFwdFixture
def test_gqa_fwd(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                 dtype: torch.dtype, tune: bool) -> None:
    test = GroupedQueryAttentionFwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionFwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@GroupedQueryAttentionBwdFixture
def test_gqa_bwd(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                 dtype: torch.dtype, tune: bool) -> None:
    pytest.skip("Temporarily skipping known GQA backward failures under TileLang 0.1.9 (#1039).")
    test = GroupedQueryAttentionBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
