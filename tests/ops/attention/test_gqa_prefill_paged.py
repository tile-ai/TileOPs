"""Tests for packed GQA prefill with paged KV cache append."""

from itertools import accumulate

import pytest
import torch

from tileops.ops import GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp

_PREFILL_PAGED_TOLERANCE = {
    torch.float16: (5e-3, 1e-5),
    torch.bfloat16: (8e-2, 1e-2),
}


def _make_cu_seqlens(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)


def _physical_pos(block_table: torch.Tensor, batch_idx: int, logical_pos: int,
                  page_size: int) -> int:
    logical_page = logical_pos // page_size
    page_offset = logical_pos % page_size
    physical_page = int(block_table[batch_idx, logical_page].item())
    return physical_page * page_size + page_offset


def _make_block_table(batch: int, max_pages_per_req: int) -> torch.Tensor:
    rows = []
    for b in range(batch):
        start = b * max_pages_per_req
        pages = list(range(start, start + max_pages_per_req))
        rows.append(pages[::2] + pages[1::2])
    return torch.tensor(rows, device="cuda", dtype=torch.int32).contiguous()


def _fill_paged_cache_from_logical(
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    k_old: list[torch.Tensor],
    v_old: list[torch.Tensor],
    block_table: torch.Tensor,
    page_size: int,
) -> None:
    for b, (k_b, v_b) in enumerate(zip(k_old, v_old, strict=True)):
        for pos in range(k_b.shape[0]):
            physical_pos = _physical_pos(block_table, b, pos, page_size)
            k_pages[physical_pos].copy_(k_b[pos])
            v_pages[physical_pos].copy_(v_b[pos])


def _gqa_prefill_paged_ref(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_old: list[torch.Tensor],
    v_old: list[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    *,
    batch: int,
    heads: int,
    heads_kv: int,
    is_causal: bool,
) -> torch.Tensor:
    groups = heads // heads_kv
    dim = q.shape[-1]
    scale = dim**-0.5
    outputs = []
    for b in range(batch):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        q_b = q[q_start:q_end]
        k_all = torch.cat([k_old[b], k_new[q_start:q_end]], dim=0)
        v_all = torch.cat([v_old[b], v_new[q_start:q_end]], dim=0)
        q_len = q_end - q_start
        old_len = k_old[b].shape[0]
        total_len = old_len + q_len

        q_bhsd = q_b.transpose(0, 1).float()
        k_bhsd = k_all.repeat_interleave(groups, dim=1).transpose(0, 1).float()
        v_bhsd = v_all.repeat_interleave(groups, dim=1).transpose(0, 1).float()
        scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * scale
        if is_causal:
            q_pos = torch.arange(q_len, device=q.device)[:, None] + old_len
            kv_pos = torch.arange(total_len, device=q.device)[None, :]
            mask = kv_pos <= q_pos
            scores = scores.masked_fill(~mask.view(1, q_len, total_len), float("-inf"))
        probs = torch.softmax(scores, dim=-1).nan_to_num()
        outputs.append(torch.matmul(probs, v_bhsd).transpose(0, 1).to(q.dtype).contiguous())
    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("q_lens, old_lens, heads, heads_kv, dim, is_causal, dtype", [
    pytest.param([64, 96], [80, 128], 8, 2, 64, True, torch.float16,
                 marks=pytest.mark.smoke, id="gqa_ratio4_mixed_fp16"),
    pytest.param([17, 33], [37, 100], 8, 2, 64, True, torch.float16,
                 marks=pytest.mark.smoke, id="gqa_unaligned_old_len_fp16"),
    pytest.param([1], [511], 8, 2, 64, True, torch.float16,
                 marks=pytest.mark.smoke, id="gqa_decode_len_capacity_boundary_fp16"),
    pytest.param([64, 64], [64, 128], 8, 8, 64, True, torch.float16,
                 marks=pytest.mark.smoke, id="mha_fp16"),
    pytest.param([32, 64], [96, 160], 8, 1, 64, True, torch.float16,
                 marks=pytest.mark.smoke, id="mqa_fp16"),
    pytest.param([64, 96], [80, 128], 8, 2, 64, False, torch.float16,
                 marks=pytest.mark.smoke, id="gqa_noncausal_fp16"),
    pytest.param([64, 96], [80, 128], 8, 2, 64, True, torch.bfloat16,
                 marks=pytest.mark.smoke, id="gqa_ratio4_bf16"),
])
def test_gqa_prefill_paged_with_kv_cache_fwd(
    q_lens: list[int],
    old_lens: list[int],
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    dtype: torch.dtype,
) -> None:
    batch = len(q_lens)
    page_size = 64
    max_pages_per_req = 8
    num_pages = batch * max_pages_per_req
    total_q = sum(q_lens)
    block_table = _make_block_table(batch, max_pages_per_req)
    cu_seqlens_q = _make_cu_seqlens(q_lens)
    cache_seqlens = torch.tensor(old_lens, device="cuda", dtype=torch.int32)
    q = torch.randn(total_q, heads, dim, device="cuda", dtype=dtype).contiguous()
    k_new = torch.randn(total_q, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    v_new = torch.randn(total_q, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    k_pages = torch.zeros(num_pages * page_size, heads_kv, dim, device="cuda",
                          dtype=dtype).contiguous()
    v_pages = torch.zeros_like(k_pages)
    k_old = [
        torch.randn(old_len, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
        for old_len in old_lens
    ]
    v_old = [
        torch.randn(old_len, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
        for old_len in old_lens
    ]
    _fill_paged_cache_from_logical(k_pages, v_pages, k_old, v_old, block_table, page_size)
    k_pages_before = k_pages.clone()
    v_pages_before = v_pages.clone()
    ref = _gqa_prefill_paged_ref(
        q,
        k_new,
        v_new,
        k_old,
        v_old,
        cu_seqlens_q,
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=is_causal,
    )
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        max_pages_per_req=max_pages_per_req,
        page_size=page_size,
        dim=dim,
        is_causal=is_causal,
        dtype=dtype,
    )

    output = op(
        q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table,
        max(q_lens))
    assert isinstance(output, torch.Tensor)
    atol, rtol = _PREFILL_PAGED_TOLERANCE[dtype]
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)

    for b, (q_len, old_len) in enumerate(zip(q_lens, old_lens, strict=True)):
        q_start = int(cu_seqlens_q[b].item())
        for i in range(q_len):
            physical_pos = _physical_pos(block_table, b, old_len + i, page_size)
            torch.testing.assert_close(k_pages[physical_pos], k_new[q_start + i])
            torch.testing.assert_close(v_pages[physical_pos], v_new[q_start + i])

    for b, old_len in enumerate(old_lens):
        for pos in range(old_len):
            physical_pos = _physical_pos(block_table, b, pos, page_size)
            torch.testing.assert_close(k_pages[physical_pos], k_pages_before[physical_pos])
            torch.testing.assert_close(v_pages[physical_pos], v_pages_before[physical_pos])


@pytest.mark.smoke
def test_gqa_prefill_paged_with_kv_cache_validates_capacity() -> None:
    batch, heads, heads_kv, dim = 1, 8, 2, 64
    page_size, max_pages_per_req = 64, 2
    q_lens = [65]
    old_lens = [64]
    q = torch.randn(sum(q_lens), heads, dim, device="cuda", dtype=torch.float16).contiguous()
    k_new = torch.randn(sum(q_lens), heads_kv, dim, device="cuda",
                        dtype=torch.float16).contiguous()
    v_new = torch.randn_like(k_new)
    k_pages = torch.zeros(max_pages_per_req * page_size, heads_kv, dim, device="cuda",
                          dtype=torch.float16).contiguous()
    v_pages = torch.zeros_like(k_pages)
    block_table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        max_pages_per_req=max_pages_per_req,
        page_size=page_size,
        dim=dim,
        dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="capacity"):
        op(
            q, k_new, v_new, k_pages, v_pages, _make_cu_seqlens(q_lens),
            torch.tensor(old_lens, device="cuda", dtype=torch.int32), block_table, max(q_lens))


@pytest.mark.smoke
def test_gqa_prefill_paged_with_kv_cache_requires_power_of_two_page_size() -> None:
    with pytest.raises(ValueError, match="power of two"):
        GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
            batch=1,
            heads=8,
            heads_kv=2,
            max_pages_per_req=8,
            page_size=24,
            dim=64,
            dtype=torch.float16,
        )


@pytest.mark.parametrize("page_size", [
    pytest.param(16, marks=pytest.mark.smoke, id="page16_multi_page_per_block"),
    pytest.param(32, marks=pytest.mark.smoke, id="page32_multi_page_per_block"),
    pytest.param(128, marks=pytest.mark.smoke, id="page128_blocks_per_page"),
])
def test_gqa_prefill_paged_with_kv_cache_page_sizes(page_size: int) -> None:
    q_lens = [32, 64]
    old_lens = [48, 80]
    batch, heads, heads_kv, dim = 2, 8, 2, 64
    dtype = torch.float16
    max_pages_per_req = 16
    num_pages = batch * max_pages_per_req
    total_q = sum(q_lens)
    block_table = _make_block_table(batch, max_pages_per_req)
    cu_seqlens_q = _make_cu_seqlens(q_lens)
    cache_seqlens = torch.tensor(old_lens, device="cuda", dtype=torch.int32)
    q = torch.randn(total_q, heads, dim, device="cuda", dtype=dtype).contiguous()
    k_new = torch.randn(total_q, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    v_new = torch.randn(total_q, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
    k_pages = torch.zeros(num_pages * page_size, heads_kv, dim, device="cuda",
                          dtype=dtype).contiguous()
    v_pages = torch.zeros_like(k_pages)
    k_old = [
        torch.randn(old_len, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
        for old_len in old_lens
    ]
    v_old = [
        torch.randn(old_len, heads_kv, dim, device="cuda", dtype=dtype).contiguous()
        for old_len in old_lens
    ]
    _fill_paged_cache_from_logical(k_pages, v_pages, k_old, v_old, block_table, page_size)
    ref = _gqa_prefill_paged_ref(
        q,
        k_new,
        v_new,
        k_old,
        v_old,
        cu_seqlens_q,
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
    )
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        max_pages_per_req=max_pages_per_req,
        page_size=page_size,
        dim=dim,
        dtype=dtype,
    )

    output = op(
        q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table,
        max(q_lens))
    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)
