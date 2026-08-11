"""Tests for fixed-shape GQA prefill with contiguous KV-cache append."""

import pytest
import torch

from tileops.kernels.attention import (
    GQAPrefillWithKVCacheFwdKernel,
    GQAPrefillWithKVCacheRopeAppendKernel,
    GQAPrefillWithKVCacheRopeFwdKernel,
)
from tileops.ops import GroupedQueryAttentionPrefillWithKVCacheFwdOp, RopeNeoxPositionIdsOp

pytestmark = pytest.mark.smoke


def _reference(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_lens: list[int],
    *,
    is_causal: bool,
    softcap: float | None = None,
) -> torch.Tensor:
    batch, seq_len_new, heads, dim = q.shape
    heads_kv = k_new.shape[2]
    groups = heads // heads_kv
    output = []
    for b in range(batch):
        old_len = cache_lens[b]
        total_len = old_len + seq_len_new
        k = torch.cat((k_cache[b, :old_len], k_new[b]), dim=0)
        v = torch.cat((v_cache[b, :old_len], v_new[b]), dim=0)
        q_bhsd = q[b].transpose(0, 1).float()
        k_bhsd = k.repeat_interleave(groups, dim=1).transpose(0, 1).float()
        v_bhsd = v.repeat_interleave(groups, dim=1).transpose(0, 1).float()
        scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * dim**-0.5
        if softcap is not None and softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        if is_causal:
            q_pos = torch.arange(seq_len_new, device=q.device)[:, None] + old_len
            kv_pos = torch.arange(total_len, device=q.device)[None, :]
            scores.masked_fill_(~(kv_pos <= q_pos).unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1).nan_to_num()
        output.append(torch.matmul(probs, v_bhsd).transpose(0, 1).to(q.dtype))
    return torch.stack(output).contiguous()


def _inputs(
    cache_lens: list[int],
    *,
    seq_len_new: int = 17,
    seqlen_kv: int = 128,
    heads: int = 8,
    heads_kv: int = 2,
    dim: int = 64,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, ...]:
    batch = len(cache_lens)
    q = torch.randn(batch, seq_len_new, heads, dim, device="cuda",
                    dtype=dtype).contiguous()
    k_new = torch.randn(batch, seq_len_new, heads_kv, dim, device="cuda",
                        dtype=dtype).contiguous()
    v_new = torch.randn_like(k_new)
    k_cache = torch.randn(batch, seqlen_kv, heads_kv, dim, device="cuda",
                          dtype=dtype).contiguous()
    v_cache = torch.randn_like(k_cache)
    lengths = torch.tensor(cache_lens, device="cuda", dtype=torch.int32)
    return q, k_new, v_new, k_cache, v_cache, lengths


@pytest.mark.parametrize("dtype, is_causal, softcap", [
    pytest.param(torch.float16, True, 2.0, id="fp16-causal-softcap"),
    pytest.param(torch.bfloat16, False, None, id="bf16-noncausal"),
])
def test_gqa_prefill_with_contiguous_kv_cache_append(
    dtype: torch.dtype,
    is_causal: bool,
    softcap: float | None,
) -> None:
    cache_lens = [37, 100]
    q, k_new, v_new, k_cache, v_cache, lengths = _inputs(cache_lens, dtype=dtype)
    k_before = k_cache.clone()
    v_before = v_cache.clone()
    expected = _reference(
        q, k_new, v_new, k_before, v_before, cache_lens,
        is_causal=is_causal, softcap=softcap)
    op = GroupedQueryAttentionPrefillWithKVCacheFwdOp(
        batch=2,
        heads=8,
        heads_kv=2,
        seq_len_new=17,
        seqlen_kv=128,
        dim=64,
        is_causal=is_causal,
        softcap=softcap,
    )

    actual = op(q, k_new, v_new, k_cache, v_cache, lengths)

    atol, rtol = (5e-3, 1e-5) if dtype == torch.float16 else (8e-2, 1e-2)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    for b, old_len in enumerate(cache_lens):
        torch.testing.assert_close(k_cache[b, :old_len], k_before[b, :old_len])
        torch.testing.assert_close(v_cache[b, :old_len], v_before[b, :old_len])
        torch.testing.assert_close(k_cache[b, old_len:old_len + 17], k_new[b])
        torch.testing.assert_close(v_cache[b, old_len:old_len + 17], v_new[b])
    built = op.built_kernels("gqa_prefill_with_kv_cache_fwd_kernel")
    assert isinstance(built[dtype], GQAPrefillWithKVCacheFwdKernel)


def test_gqa_prefill_with_contiguous_kv_cache_fused_rope() -> None:
    cache_lens = [37, 100]
    max_position = 128
    rotary_dim = 32
    q, k_new, v_new, k_cache, v_cache, lengths = _inputs(cache_lens)
    k_before = k_cache.clone()
    v_before = v_cache.clone()
    positions = torch.stack([
        torch.arange(old_len, old_len + 17, device="cuda", dtype=torch.int32)
        for old_len in cache_lens
    ])
    rope = RopeNeoxPositionIdsOp(max_position=max_position, rotary_dim=rotary_dim)
    q_rot = rope(q.view(-1, 8, 64), positions.flatten()).view_as(q)
    k_new_rot = rope(k_new.view(-1, 2, 64), positions.flatten()).view_as(k_new)
    expected = _reference(
        q_rot, k_new_rot, v_new, k_before, v_before, cache_lens, is_causal=True)
    op = GroupedQueryAttentionPrefillWithKVCacheFwdOp(
        batch=2,
        heads=8,
        heads_kv=2,
        seq_len_new=17,
        seqlen_kv=128,
        dim=64,
        fuse_rope=True,
        max_position=max_position,
        rotary_dim=rotary_dim,
    )

    actual = op(q, k_new, v_new, k_cache, v_cache, lengths)

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=1e-5)
    for b, old_len in enumerate(cache_lens):
        torch.testing.assert_close(k_cache[b, :old_len], k_before[b, :old_len])
        torch.testing.assert_close(v_cache[b, :old_len], v_before[b, :old_len])
        torch.testing.assert_close(k_cache[b, old_len:old_len + 17], k_new_rot[b])
        torch.testing.assert_close(v_cache[b, old_len:old_len + 17], v_new[b])
    built = op.built_kernels("gqa_prefill_with_kv_cache_rope_fwd_kernel")
    kernel = built[torch.float16]
    assert isinstance(kernel, GQAPrefillWithKVCacheRopeFwdKernel)
    assert isinstance(kernel._append, GQAPrefillWithKVCacheRopeAppendKernel)


def test_gqa_prefill_with_contiguous_kv_cache_rejects_capacity_overflow() -> None:
    inputs = _inputs([112], seqlen_kv=128)
    op = GroupedQueryAttentionPrefillWithKVCacheFwdOp(
        batch=1,
        heads=8,
        heads_kv=2,
        seq_len_new=17,
        seqlen_kv=128,
        dim=64,
    )

    with pytest.raises(ValueError, match="exceeds contiguous KV capacity"):
        op(*inputs)
