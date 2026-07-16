
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import tileops.ops.attention.gqa as gqa_module
from tests.test_base import FixtureBase, TestBase
from tileops.kernels.kernel_base import Kernel
from tileops.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp
from workloads.attention.mha import MhaBwdTest as _MhaBwdTestWorkload
from workloads.attention.mha import MhaFwdTest as _MhaFwdTestWorkload


class _FakeDenseKernel(Kernel):
    def forward(self, *args: object, **kwargs: object) -> object:
        return None


class _FakeSquareDenseKernel(Kernel):
    def forward(self, *args: object, **kwargs: object) -> object:
        return None


class _FakeLegacyMhaBwdKernel(Kernel):
    def __init__(self, batch: int, heads: int, seq_len: int, dim: int,
                 is_causal: bool, dtype: torch.dtype, tune: bool = False) -> None:
        super().__init__()

    def forward(self, *args: object, **kwargs: object) -> object:
        return None


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


@pytest.mark.smoke
def test_mha_fwd_dispatches_to_gqa_kernel() -> None:
    op = MultiHeadAttentionFwdOp(1, 8, 128, 64, False, torch.float16)
    assert op.kernel.__class__.__name__.startswith("GQA")


@pytest.mark.smoke
def test_mha_fwd_preserves_gqa_square_dense_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gqa_module, "is_h200", lambda: True)
    op = MultiHeadAttentionFwdOp(
        batch=4,
        heads=64,
        seq_len=512,
        dim=128,
        is_causal=True,
        dtype=torch.float16,
        kernel_map={
            "gqa_prefill_fwd_kernel": _FakeDenseKernel,
            "gqa_prefill_square_fwd_kernel": _FakeSquareDenseKernel,
        },
    )

    assert isinstance(op.kernel, _FakeSquareDenseKernel)


@pytest.mark.smoke
def test_mha_bwd_rejects_legacy_kernel_map_keys() -> None:
    with pytest.raises(ValueError, match="legacy MHA backward kernel_map keys"):
        MultiHeadAttentionBwdOp(
            batch=1,
            heads=8,
            seq_len=128,
            dim=64,
            is_causal=False,
            dtype=torch.float16,
            kernel_map={"mha_bwd_kernel": _FakeLegacyMhaBwdKernel},
        )


@MhaBwdFixture
def test_mha_bwd(batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype,
                 tune: bool) -> None:
    test = MhaBwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
