
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp
from workloads.attention.mha import MhaBwdTest as _MhaBwdTestWorkload
from workloads.attention.mha import MhaFwdTest as _MhaFwdTestWorkload


class MhaBwdTest(_MhaBwdTestWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                    grad_output: torch.Tensor,
                    lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal)
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


class MhaFwdTest(_MhaFwdTestWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output, None  # do not check lse


class MhaFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim, causal, dtype, tune", [
            pytest.param(
                1, 1024, 8, 64, False, torch.float16, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-fwd-fp16",
            ),
            pytest.param(
                1, 1024, 8, 64, False, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-fwd-bf16",
            ),
            pytest.param(
                16, 2048, 16, 128, False, torch.float16, False,
                marks=pytest.mark.full,
                id="full-fwd-fp16",
            ),
            pytest.param(
                4, 4096, 16, 128, False, torch.bfloat16, True,
                marks=pytest.mark.full,
                id="full-fwd-bf16-tuned",
            ),
        ]),
    ]


class MhaBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim, causal, dtype, tune", [
            pytest.param(
                1, 1024, 8, 64, False, torch.float16, False,
                marks=pytest.mark.smoke,
                id="smoke-bwd-fp16",
            ),
            pytest.param(
                1, 1024, 8, 64, False, torch.bfloat16, False,
                marks=pytest.mark.smoke,
                id="smoke-bwd-bf16",
            ),
            pytest.param(
                16, 2048, 16, 128, False, torch.float16, False,
                marks=pytest.mark.full,
                id="full-bwd-fp16-large",
            ),
            pytest.param(
                4, 4096, 16, 128, False, torch.bfloat16, True,
                marks=pytest.mark.full,
                id="full-bwd-bf16-tuned",
            ),
        ]),
    ]


@MhaFwdFixture
def test_mha_fwd(batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype,
                 tune: bool) -> None:
    test = MhaFwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@MhaBwdFixture
def test_mha_bwd(batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype,
                 tune: bool) -> None:
    pytest.skip("Temporarily skipping known MHA backward failures under TileLang 5f70374c (#999).")
    test = MhaBwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
