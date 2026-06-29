
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention.gqa_decode import GQADecodeKernel
from tileops.ops import GroupedQueryAttentionDecodeWithKVCacheFwdOp
from workloads.attention.gqa_decode import (
    GroupedQueryAttentionDecodeTest as _GroupedQueryAttentionDecodeTestWorkload,
)


class GroupedQueryAttentionDecodeTest(_GroupedQueryAttentionDecodeTestWorkload, TestBase):
    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.unsqueeze(1).transpose(1, 2)  # [B, H, 1, D]
        groups = self.heads // self.heads_kv
        k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        v_bhsd = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        scores = torch.matmul(q_bhsd.float(), k_bhsd.transpose(-2, -1)) * self.sm_scale
        if self.softcap > 0:
            scores = self.softcap * torch.tanh(scores / self.softcap)
        probs = torch.softmax(scores, dim=-1)
        output_bhsd = torch.matmul(probs, v_bhsd)
        return output_bhsd.transpose(1, 2).squeeze(1).to(q.dtype).contiguous()


class GroupedQueryAttentionDecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, heads_kv, seq_len_kv, dim, dtype, tune", [
            pytest.param(1, 32, 8, 8192, 128, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 32, 8, 8192, 128, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(1, 16, 2, 8192, 128, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(8, 64, 16, 8192, 128, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


@GroupedQueryAttentionDecodeFixture
def test_gqa_decode(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                    dtype: torch.dtype, tune: bool) -> None:
    test = GroupedQueryAttentionDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dtype)
    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(batch, heads, heads_kv, seq_len_kv, dim, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
@pytest.mark.parametrize("sm_scale, softcap", [
    pytest.param(0.25, None, id="custom-sm-scale"),
    pytest.param(None, 2.0, id="softcap"),
])
def test_gqa_decode_softmax_controls(sm_scale: float | None, softcap: float | None) -> None:
    batch, heads, heads_kv, seq_len_kv, dim = 1, 16, 4, 1024, 64
    dtype = torch.float16
    test = GroupedQueryAttentionDecodeTest(
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        dtype,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        dtype,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


@pytest.mark.smoke
def test_gqa_decode_default_split_policy() -> None:
    qwen_like = GQADecodeKernel(1, 16, 2, 8192, 128, dtype="float16")
    assert qwen_like.config["num_split"] == 32

    short_qwen_like = GQADecodeKernel(1, 16, 2, 8, 128, dtype="float16")
    assert short_qwen_like.config["num_split"] == 8

    llama_like = GQADecodeKernel(1, 40, 8, 8192, 128, dtype="float16")
    assert llama_like.config["num_split"] == 16

    batched_qwen_like = GQADecodeKernel(8, 16, 2, 8192, 128, dtype="float16")
    assert batched_qwen_like.config["num_split"] == 16


@pytest.mark.smoke
def test_gqa_decode_split_policy_filters_short_kv_autotune_configs() -> None:
    kernel = GQADecodeKernel(1, 16, 2, 8, 128, dtype="float16")
    assert {cfg["num_split"] for cfg in kernel.autotune_configs} == {2, 4, 8}

    tiny_kernel = GQADecodeKernel(1, 16, 2, 1, 128, dtype="float16")
    assert tiny_kernel.config["num_split"] == 1
    assert {cfg["num_split"] for cfg in tiny_kernel.autotune_configs} == {1}


@pytest.mark.smoke
def test_gqa_decode_rejects_non_positive_groups() -> None:
    with pytest.raises(ValueError, match="groups must be positive"):
        GQADecodeKernel(1, 16, 0, 8192, 128, dtype="float16")


@pytest.mark.smoke
def test_gqa_decode_rejects_non_divisible_heads() -> None:
    with pytest.raises(ValueError, match="heads must be divisible by groups"):
        GQADecodeKernel(1, 15, 2, 8192, 128, dtype="float16")


@pytest.mark.smoke
def test_gqa_decode_rejects_non_positive_seqlen_kv() -> None:
    with pytest.raises(ValueError, match="seqlen_kv must be positive"):
        GQADecodeKernel(1, 16, 2, 0, 128, dtype="float16")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
