
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import (
    GQAFwdKernel,
    GQAFwdWgmmaPipelinedKernel,
    GQAFwdWsPersistentCausalKernel,
    GQAFwdWsPersistentKernel,
)
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionFwdOp,
    GroupedQueryAttentionPrefillFwdOp,
    GroupedQueryAttentionPrefillWithKVCacheFwdOp,
)
from tileops.ops.attention.gqa import _select_gqa_fwd_kernel_cls
from workloads.attention.gqa import (
    GroupedQueryAttentionBwdTest as _GroupedQueryAttentionBwdTestWorkload,
)
from workloads.attention.gqa import (
    GroupedQueryAttentionFwdTest as _GroupedQueryAttentionFwdTestWorkload,
)

_PREFILL_TOLERANCE = {
    torch.float16: (5e-3, 1e-5),
    torch.bfloat16: (8e-2, 1e-2),
}


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


def _gqa_prefill_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    heads_kv: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    batch, seq_len_q, _, dim = q.shape
    seq_len_kv = k.shape[1]
    groups = heads // heads_kv
    q_bhsd = q.transpose(1, 2).float()
    k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    v_bhsd = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    scale = dim**-0.5 if sm_scale is None else sm_scale
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * scale
    if is_causal:
        offset = seq_len_kv - seq_len_q
        q_pos = torch.arange(seq_len_q, device=q.device)[:, None] + offset
        k_pos = torch.arange(seq_len_kv, device=q.device)[None, :]
        mask = k_pos <= q_pos
        scores = scores.masked_fill(~mask.view(1, 1, seq_len_q, seq_len_kv), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_bhsd)
    assert output.shape == (batch, heads, seq_len_q, dim)
    return output.transpose(1, 2).to(q.dtype).contiguous()


def _gqa_prefill_with_kv_cache_ref(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache_before: torch.Tensor,
    v_cache_before: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    heads: int,
    heads_kv: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    batch, seq_len_new, _, dim = q.shape
    groups = heads // heads_kv
    outputs = []
    for b in range(batch):
        old_len = int(cache_seqlens[b].item())
        k_all = torch.cat([k_cache_before[b, :old_len], k_new[b]], dim=0)
        v_all = torch.cat([v_cache_before[b, :old_len], v_new[b]], dim=0)
        q_bhsd = q[b].transpose(0, 1).float()
        k_bhsd = k_all.repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
        v_bhsd = v_all.repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
        scale = dim**-0.5 if sm_scale is None else sm_scale
        scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * scale
        if is_causal:
            q_pos = torch.arange(seq_len_new, device=q.device)[:, None] + old_len
            k_pos = torch.arange(old_len + seq_len_new, device=q.device)[None, :]
            mask = k_pos <= q_pos
            scores = scores.masked_fill(~mask.view(1, seq_len_new, old_len + seq_len_new),
                                        float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v_bhsd).transpose(0, 1).to(q.dtype).contiguous()
        outputs.append(output)
    return torch.stack(outputs, dim=0)


class GroupedQueryAttentionFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, heads_kv, dim, causal, dtype, tune", [
            pytest.param(1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 1024, 8, 4, 64, False, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(4, 512, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(4, 512, 64, 4, 128, True, torch.float16, False, marks=pytest.mark.smoke),
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


@pytest.mark.parametrize("batch, seq_len_q, seq_len_kv, heads, heads_kv, dim, causal, dtype", [
    pytest.param(1, 128, 128, 8, 2, 64, True, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_q_eq_kv_fp16"),
    pytest.param(1, 128, 384, 8, 2, 64, True, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_q_lt_kv_fp16"),
    pytest.param(2, 128, 256, 16, 4, 128, True, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_batch2_dim128_fp16"),
    pytest.param(1, 128, 256, 8, 2, 64, False, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_noncausal_fp16"),
    pytest.param(1, 128, 256, 8, 2, 64, True, torch.bfloat16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_bf16"),
])
def test_gqa_prefill_fwd(batch: int, seq_len_q: int, seq_len_kv: int, heads: int,
                         heads_kv: int, dim: int, causal: bool, dtype: torch.dtype) -> None:
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=dtype).contiguous()
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    v = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    ref = _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=causal)

    op = GroupedQueryAttentionPrefillFwdOp(
        batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, causal, dtype)
    output = op(q, k, v)

    atol, rtol = _PREFILL_TOLERANCE[dtype]
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_gqa_prefill_fwd_uses_bottom_right_causal_mask() -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 128, 256, 4, 2, 64
    q = torch.zeros(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.zeros(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.zeros(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
    q[..., 0] = 1
    k[..., 0] = 1
    v[:, :128, :, 0] = 1
    v[:, 128:, :, 0] = 100

    op = GroupedQueryAttentionPrefillFwdOp(
        batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, True, torch.float16)
    output = op(q.contiguous(), k.contiguous(), v.contiguous())

    assert output[0, 0, 0, 0] < 2
    assert output[0, -1, 0, 0] > 40


@pytest.mark.smoke
def test_gqa_prefill_fwd_respects_sm_scale() -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 128, 256, 8, 2, 64
    sm_scale = 0.125
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    v = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    ref = _gqa_prefill_ref(
        q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True, sm_scale=sm_scale)

    op = GroupedQueryAttentionPrefillFwdOp(
        batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, True, torch.float16,
        sm_scale=sm_scale)
    output = op(q, k, v)

    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.parametrize("batch, seq_len_new, seqlen_kv, heads, heads_kv, dim, causal, dtype", [
    pytest.param(1, 64, 256, 8, 2, 64, True, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_fp16"),
    pytest.param(2, 64, 320, 16, 4, 128, True, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_batch2_dim128_fp16"),
    pytest.param(1, 64, 256, 8, 2, 64, False, torch.float16, marks=pytest.mark.smoke,
                 id="gqa_noncausal_fp16"),
    pytest.param(1, 64, 256, 8, 1, 64, True, torch.float16, marks=pytest.mark.smoke,
                 id="mqa_fp16"),
    pytest.param(1, 64, 256, 8, 2, 64, True, torch.bfloat16, marks=pytest.mark.smoke,
                 id="gqa_ratio4_bf16"),
])
def test_gqa_prefill_with_kv_cache_fwd(batch: int, seq_len_new: int, seqlen_kv: int,
                                       heads: int, heads_kv: int, dim: int, causal: bool,
                                       dtype: torch.dtype) -> None:
    q = torch.randn(batch, seq_len_new, heads, dim, device="cuda", dtype=dtype).contiguous()
    k_new = torch.randn(batch, seq_len_new, heads_kv, dim, device="cuda",
                        dtype=dtype).contiguous()
    v_new = torch.randn(batch, seq_len_new, heads_kv, dim, device="cuda",
                        dtype=dtype).contiguous()
    k_cache = torch.randn(batch, seqlen_kv, heads_kv, dim, device="cuda",
                          dtype=dtype).contiguous()
    v_cache = torch.randn(batch, seqlen_kv, heads_kv, dim, device="cuda",
                          dtype=dtype).contiguous()
    old_lens = [65 + 37 * i for i in range(batch)]
    cache_seqlens = torch.tensor(old_lens, device="cuda", dtype=torch.int32)
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()

    ref = _gqa_prefill_with_kv_cache_ref(
        q,
        k_new,
        v_new,
        k_cache_before,
        v_cache_before,
        cache_seqlens,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=causal,
    )
    op = GroupedQueryAttentionPrefillWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_new, seqlen_kv, dim, causal, dtype)
    output = op(q, k_new, v_new, k_cache, v_cache, cache_seqlens)

    atol, rtol = _PREFILL_TOLERANCE[dtype]
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)
    for b, old_len in enumerate(old_lens):
        torch.testing.assert_close(k_cache[b, :old_len], k_cache_before[b, :old_len])
        torch.testing.assert_close(v_cache[b, :old_len], v_cache_before[b, :old_len])
        torch.testing.assert_close(k_cache[b, old_len:old_len + seq_len_new], k_new[b])
        torch.testing.assert_close(v_cache[b, old_len:old_len + seq_len_new], v_new[b])


@pytest.mark.smoke
def test_gqa_prefill_with_kv_cache_respects_sm_scale() -> None:
    batch, seq_len_new, seqlen_kv, heads, heads_kv, dim = 1, 64, 256, 8, 2, 64
    sm_scale = 0.125
    q = torch.randn(batch, seq_len_new, heads, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    k_new = torch.randn(batch, seq_len_new, heads_kv, dim, device="cuda",
                        dtype=torch.float16).contiguous()
    v_new = torch.randn_like(k_new)
    k_cache = torch.randn(batch, seqlen_kv, heads_kv, dim, device="cuda",
                          dtype=torch.float16).contiguous()
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.tensor([65], device="cuda", dtype=torch.int32)
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()
    ref = _gqa_prefill_with_kv_cache_ref(
        q,
        k_new,
        v_new,
        k_cache_before,
        v_cache_before,
        cache_seqlens,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
        sm_scale=sm_scale,
    )

    op = GroupedQueryAttentionPrefillWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_new, seqlen_kv, dim, True, torch.float16,
        sm_scale=sm_scale)
    output = op(q, k_new, v_new, k_cache, v_cache, cache_seqlens)

    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_prefill_with_kv_cache_rejects_capacity_overflow() -> None:
    batch, seq_len_new, seqlen_kv, heads, heads_kv, dim = 1, 64, 96, 8, 2, 64
    q = torch.randn(batch, seq_len_new, heads, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    k_new = torch.randn(batch, seq_len_new, heads_kv, dim, device="cuda",
                        dtype=torch.float16).contiguous()
    v_new = torch.randn_like(k_new)
    k_cache = torch.randn(batch, seqlen_kv, heads_kv, dim, device="cuda",
                          dtype=torch.float16).contiguous()
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.tensor([65], device="cuda", dtype=torch.int32)

    op = GroupedQueryAttentionPrefillWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_new, seqlen_kv, dim, True, torch.float16)
    with pytest.raises(ValueError, match="exceeds KV cache capacity"):
        op(q, k_new, v_new, k_cache, v_cache, cache_seqlens)


@pytest.mark.smoke
def test_gqa_prefill_with_kv_cache_rejects_bad_contract_inputs() -> None:
    batch, seq_len_new, seqlen_kv, heads, heads_kv, dim = 1, 64, 256, 8, 2, 64
    q = torch.randn(batch, seq_len_new, heads, dim, device="cuda",
                    dtype=torch.float16).contiguous()
    k_new = torch.randn(batch, seq_len_new, heads_kv, dim, device="cuda",
                        dtype=torch.float16).contiguous()
    v_new = torch.randn_like(k_new)
    k_cache = torch.randn(batch, seqlen_kv, heads_kv, dim, device="cuda",
                          dtype=torch.float16).contiguous()
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.tensor([65], device="cuda", dtype=torch.int32)

    op = GroupedQueryAttentionPrefillWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_new, seqlen_kv, dim, True, torch.float16)
    with pytest.raises(ValueError, match="Expected k_new shape"):
        op(q, k_new[:, :-1], v_new, k_cache, v_cache, cache_seqlens)
    with pytest.raises(ValueError, match="Expected cache_seqlens.dtype"):
        op(q, k_new, v_new, k_cache, v_cache, cache_seqlens.to(torch.int64))


@GroupedQueryAttentionBwdFixture
def test_gqa_bwd(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                 dtype: torch.dtype, tune: bool) -> None:
    pytest.skip("Temporarily skipping known GQA backward failures under TileLang 0.1.9 (#1039).")
    test = GroupedQueryAttentionBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_fwd_dispatch_selects_ws_noncausal_on_h200() -> None:
    kernel_cls = _select_gqa_fwd_kernel_cls(
        4, 64, 4, 512, 128, False, torch.float16, hopper=True, h200=True)
    assert kernel_cls is GQAFwdWsPersistentKernel


@pytest.mark.smoke
def test_gqa_fwd_dispatch_selects_ws_causal_on_h200() -> None:
    kernel_cls = _select_gqa_fwd_kernel_cls(
        4, 64, 4, 512, 128, True, torch.float16, hopper=True, h200=True)
    assert kernel_cls is GQAFwdWsPersistentCausalKernel


@pytest.mark.smoke
def test_gqa_fwd_dispatch_falls_back_for_small_causal_shape() -> None:
    kernel_cls = _select_gqa_fwd_kernel_cls(
        1, 32, 8, 1024, 128, True, torch.float16, hopper=True, h200=True)
    assert kernel_cls is GQAFwdWgmmaPipelinedKernel


@pytest.mark.smoke
def test_gqa_fwd_dispatch_falls_back_off_h200() -> None:
    hopper_cls = _select_gqa_fwd_kernel_cls(
        4, 64, 4, 512, 128, False, torch.float16, hopper=True, h200=False)
    assert hopper_cls is GQAFwdWgmmaPipelinedKernel

    non_hopper_cls = _select_gqa_fwd_kernel_cls(
        4, 64, 4, 512, 128, False, torch.float16, hopper=False, h200=False)
    assert non_hopper_cls is GQAFwdKernel


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
