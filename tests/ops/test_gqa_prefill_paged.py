"""Tests for packed GQA prefill with paged KV cache append."""

import pytest
import torch

from tests.test_base import served_in_tree
from tileops.ops import GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp
from workloads.attention.gqa.prefill_paged_kv_append import GQAPrefillPagedWithKVCacheFwdWorkload
from workloads.attention.paged_kv_cache import (
    fill_paged_cache_from_logical,
    make_fragmented_block_table,
    make_interleaved_block_table,
    make_unit_cache_scales,
    paged_cache_row,
)
from workloads.device import run_device
from workloads.sequence_metadata import make_cu_seqlens

_PREFILL_PAGED_TOLERANCE = {
    torch.float16: (5e-3, 1e-5),
    torch.bfloat16: (8e-2, 1e-2),
}


@pytest.mark.parametrize(
    "q_lens, old_lens, heads, heads_kv, dim, is_causal, dtype",
    [
        pytest.param(
            [64, 96],
            [80, 128],
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_ratio4_mixed_fp16",
        ),
        pytest.param(
            [17, 33],
            [37, 100],
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_unaligned_old_len_fp16",
        ),
        pytest.param(
            [1],
            [511],
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_decode_len_capacity_boundary_fp16",
        ),
        pytest.param(
            [1, 17],
            [511, 37],
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_mixed_capacity_boundary_fp16",
        ),
        pytest.param(
            [64, 64],
            [64, 128],
            8,
            8,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="mha_fp16",
        ),
        pytest.param(
            [32, 64],
            [96, 160],
            8,
            1,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="mqa_fp16",
        ),
        pytest.param(
            [64, 96],
            [80, 128],
            8,
            2,
            64,
            False,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_noncausal_fp16",
        ),
        pytest.param(
            [64, 96],
            [80, 128],
            8,
            2,
            64,
            True,
            torch.bfloat16,
            marks=pytest.mark.smoke,
            id="gqa_ratio4_bf16",
        ),
    ],
)
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
    block_table = make_interleaved_block_table(batch, max_pages_per_req)
    cu_seqlens_q = make_cu_seqlens(q_lens)
    cache_seqlens = torch.tensor(old_lens, device=run_device(), dtype=torch.int32)
    q = torch.randn(total_q, heads, dim, device=run_device(), dtype=dtype).contiguous()
    k_new = torch.randn(total_q, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
    v_new = torch.randn(total_q, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
    k_pages = torch.zeros(
        num_pages * page_size, heads_kv, dim, device=run_device(), dtype=dtype
    ).contiguous()
    v_pages = torch.zeros_like(k_pages)
    k_old = [
        torch.randn(old_len, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
        for old_len in old_lens
    ]
    v_old = [
        torch.randn(old_len, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
        for old_len in old_lens
    ]
    fill_paged_cache_from_logical(k_pages, v_pages, k_old, v_old, block_table, page_size)
    k_pages_before = k_pages.clone()
    v_pages_before = v_pages.clone()
    k_scale, v_scale = make_unit_cache_scales()
    case = GQAPrefillPagedWithKVCacheFwdWorkload(
        batch, heads, heads_kv, q_lens, old_lens, page_size, dim, is_causal, dtype
    )
    ref = case.ref_program(
        q,
        k_new,
        v_new,
        k_pages.clone(),
        v_pages.clone(),
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
    )
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        page_size=page_size,
        max_seqlen_q=max(q_lens),
        is_causal=is_causal,
    )

    output = op(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
    )
    assert isinstance(output, torch.Tensor)
    atol, rtol = _PREFILL_PAGED_TOLERANCE[dtype]
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)

    for b, (q_len, old_len) in enumerate(zip(q_lens, old_lens, strict=True)):
        q_start = int(cu_seqlens_q[b].item())
        for i in range(q_len):
            row = paged_cache_row(block_table, b, old_len + i, page_size)
            torch.testing.assert_close(k_pages[row], k_new[q_start + i])
            torch.testing.assert_close(v_pages[row], v_new[q_start + i])

    for b, old_len in enumerate(old_lens):
        for pos in range(old_len):
            row = paged_cache_row(block_table, b, pos, page_size)
            torch.testing.assert_close(k_pages[row], k_pages_before[row])
            torch.testing.assert_close(v_pages[row], v_pages_before[row])


@pytest.mark.smoke
@pytest.mark.parametrize(
    "is_causal, softcap, dtype, page_size",
    [
        pytest.param(True, None, torch.float16, 64, id="causal-fp16-page64"),
        pytest.param(False, None, torch.float16, 64, id="noncausal-fp16-page64"),
        pytest.param(True, 2.0, torch.float16, 64, id="causal-softcap-fp16-page64"),
        pytest.param(True, None, torch.bfloat16, 64, id="causal-bf16-page64"),
        pytest.param(True, None, torch.float16, 16, id="causal-fp16-page16"),
        pytest.param(True, None, torch.float16, 128, id="causal-fp16-page128"),
    ],
)
def test_gqa_prefill_paged_with_fp8_kv_cache_fwd(
    is_causal: bool,
    softcap: float | None,
    dtype: torch.dtype,
    page_size: int,
) -> None:
    q_lens = [33, 48]
    old_lens = [67, 80]
    batch, heads, heads_kv, dim = 2, 8, 2, 64
    cache_dtype = torch.float8_e4m3fn
    max_pages_per_req = 8
    num_pages = batch * max_pages_per_req
    total_q = sum(q_lens)
    block_table = make_interleaved_block_table(batch, max_pages_per_req)
    cu_seqlens_q = make_cu_seqlens(q_lens)
    cache_seqlens = torch.tensor(old_lens, device=run_device(), dtype=torch.int32)
    k_scale = torch.tensor([0.02], device=run_device(), dtype=torch.float32)
    v_scale = torch.tensor([0.02], device=run_device(), dtype=torch.float32)

    q = torch.randn(total_q, heads, dim, device=run_device(), dtype=dtype).contiguous()
    k_new = (
        torch.randn(total_q, heads_kv, dim, device=run_device(), dtype=dtype) * 0.5
    ).contiguous()
    v_new = (
        torch.randn(total_q, heads_kv, dim, device=run_device(), dtype=dtype) * 0.5
    ).contiguous()
    k_pages = torch.zeros(
        num_pages * page_size, heads_kv, dim, device=run_device(), dtype=cache_dtype
    ).contiguous()
    v_pages = torch.zeros_like(k_pages)
    k_old = [
        (torch.randn(old_len, heads_kv, dim, device=run_device(), dtype=dtype) * 0.5).contiguous()
        for old_len in old_lens
    ]
    v_old = [
        (torch.randn(old_len, heads_kv, dim, device=run_device(), dtype=dtype) * 0.5).contiguous()
        for old_len in old_lens
    ]
    k_old_quant = [(k_b.float() / k_scale[0]).to(cache_dtype).contiguous() for k_b in k_old]
    v_old_quant = [(v_b.float() / v_scale[0]).to(cache_dtype).contiguous() for v_b in v_old]
    fill_paged_cache_from_logical(
        k_pages, v_pages, k_old_quant, v_old_quant, block_table, page_size
    )
    k_pages_before = k_pages.clone()
    v_pages_before = v_pages.clone()
    case = GQAPrefillPagedWithKVCacheFwdWorkload(
        batch, heads, heads_kv, q_lens, old_lens, page_size, dim, is_causal, dtype, softcap=softcap
    )
    ref = case.ref_program(
        q,
        k_new,
        v_new,
        k_pages.clone(),
        v_pages.clone(),
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
    )
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        page_size=page_size,
        max_seqlen_q=max(q_lens),
        is_causal=is_causal,
        cache_dtype=cache_dtype,
        softcap=softcap,
    )

    output = op(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
    )
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, ref, atol=8e-2, rtol=2e-2)

    for b, (q_len, old_len) in enumerate(zip(q_lens, old_lens, strict=True)):
        q_start = int(cu_seqlens_q[b].item())
        for i in range(q_len):
            row = paged_cache_row(block_table, b, old_len + i, page_size)
            expected_k = (k_new[q_start + i].float() / k_scale[0]).to(cache_dtype).float()
            expected_v = (v_new[q_start + i].float() / v_scale[0]).to(cache_dtype).float()
            torch.testing.assert_close(k_pages[row].float(), expected_k, atol=0, rtol=0)
            torch.testing.assert_close(v_pages[row].float(), expected_v, atol=0, rtol=0)

    for b, old_len in enumerate(old_lens):
        for pos in range(old_len):
            row = paged_cache_row(block_table, b, pos, page_size)
            torch.testing.assert_close(k_pages[row].float(), k_pages_before[row].float())
            torch.testing.assert_close(v_pages[row].float(), v_pages_before[row].float())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "scale_name,bad_value",
    [
        pytest.param("k_scale", 0.0, id="k_zero"),
        pytest.param("k_scale", -0.01, id="k_negative"),
        pytest.param("k_scale", float("inf"), id="k_inf"),
        pytest.param("v_scale", float("nan"), id="v_nan"),
    ],
)
def test_gqa_prefill_paged_with_fp8_kv_cache_rejects_invalid_scales(
    scale_name: str,
    bad_value: float,
) -> None:
    heads, heads_kv, dim = 8, 2, 64
    q_lens = [1]
    page_size, max_pages_per_req = 64, 1
    q = torch.randn(sum(q_lens), heads, dim, device=run_device(), dtype=torch.float16).contiguous()
    k_new = torch.randn(
        sum(q_lens), heads_kv, dim, device=run_device(), dtype=torch.float16
    ).contiguous()
    v_new = torch.randn_like(k_new)
    k_pages = torch.zeros(
        max_pages_per_req * page_size, heads_kv, dim, device=run_device(), dtype=torch.float8_e4m3fn
    ).contiguous()
    v_pages = torch.zeros_like(k_pages)
    k_scale = torch.tensor([0.02], device=run_device(), dtype=torch.float32)
    v_scale = torch.tensor([0.02], device=run_device(), dtype=torch.float32)
    if scale_name == "k_scale":
        k_scale = torch.tensor([bad_value], device=run_device(), dtype=torch.float32)
    else:
        v_scale = torch.tensor([bad_value], device=run_device(), dtype=torch.float32)
    block_table = torch.tensor([[0]], device=run_device(), dtype=torch.int32)
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        page_size=page_size,
        max_seqlen_q=max(q_lens),
        cache_dtype=torch.float8_e4m3fn,
    )

    with pytest.raises(ValueError, match=f"{scale_name}.*finite positive"):
        op(
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            v_scale,
            make_cu_seqlens(q_lens),
            torch.tensor([0], device=run_device(), dtype=torch.int32),
            block_table,
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "rotary_dim, is_causal, softcap",
    [
        pytest.param(None, True, None, id="full-causal"),
        pytest.param(32, True, None, id="partial-causal"),
        pytest.param(32, False, None, id="partial-noncausal"),
        pytest.param(32, True, 2.0, id="partial-causal-softcap"),
    ],
)
def test_gqa_prefill_paged_with_kv_cache_fused_rope(
    rotary_dim: int | None,
    is_causal: bool,
    softcap: float | None,
) -> None:
    q_lens = [48, 33]
    old_lens = [67, 100]
    batch, heads, heads_kv, dim = 2, 8, 2, 64
    dtype = torch.float16
    page_size = 64
    max_pages_per_req = 8
    num_pages = batch * max_pages_per_req
    total_q = sum(q_lens)
    max_position = max(old + new for old, new in zip(old_lens, q_lens, strict=True)) + 1
    block_table = make_interleaved_block_table(batch, max_pages_per_req)
    cu_seqlens_q = make_cu_seqlens(q_lens)
    cache_seqlens = torch.tensor(old_lens, device=run_device(), dtype=torch.int32)

    q_raw = torch.randn(total_q, heads, dim, device=run_device(), dtype=dtype).contiguous()
    k_new_raw = torch.randn(total_q, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
    v_new = torch.randn(total_q, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
    k_pages = torch.zeros(
        num_pages * page_size, heads_kv, dim, device=run_device(), dtype=dtype
    ).contiguous()
    v_pages = torch.zeros_like(k_pages)

    new_positions = torch.cat(
        [
            torch.arange(old_len, old_len + q_len, device=run_device(), dtype=torch.int32)
            for old_len, q_len in zip(old_lens, q_lens, strict=True)
        ]
    )
    old_positions = torch.cat(
        [torch.arange(old_len, device=run_device(), dtype=torch.int32) for old_len in old_lens]
    )
    case = GQAPrefillPagedWithKVCacheFwdWorkload(
        batch,
        heads,
        heads_kv,
        q_lens,
        old_lens,
        page_size,
        dim,
        is_causal,
        dtype,
        fuse_rope=True,
        rotary_dim=rotary_dim,
        softcap=softcap,
    )
    k_new_rot = case.rope(k_new_raw, new_positions)
    k_old_raw = [
        torch.randn(old_len, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
        for old_len in old_lens
    ]
    v_old = [
        torch.randn(old_len, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
        for old_len in old_lens
    ]
    k_old = list(
        torch.split(
            case.rope(torch.cat(k_old_raw, dim=0), old_positions),
            old_lens,
            dim=0,
        )
    )
    fill_paged_cache_from_logical(k_pages, v_pages, k_old, v_old, block_table, page_size)
    k_pages_before = k_pages.clone()
    v_pages_before = v_pages.clone()

    k_scale, v_scale = make_unit_cache_scales()
    ref = case.ref_program(
        q_raw,
        k_new_raw,
        v_new,
        k_pages.clone(),
        v_pages.clone(),
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
    )
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        page_size=page_size,
        max_seqlen_q=max(q_lens),
        is_causal=is_causal,
        softcap=softcap,
        fuse_rope=True,
        max_position=max_position,
        rotary_dim=rotary_dim,
    )

    output = op(
        q_raw,
        k_new_raw,
        v_new,
        k_pages,
        v_pages,
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
    )
    atol, rtol = _PREFILL_PAGED_TOLERANCE[dtype]
    torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)

    for b, (q_len, old_len) in enumerate(zip(q_lens, old_lens, strict=True)):
        q_start = int(cu_seqlens_q[b].item())
        for i in range(q_len):
            row = paged_cache_row(block_table, b, old_len + i, page_size)
            # Two rotations of the same input, not a copy: the kernel's against the reference.
            torch.testing.assert_close(k_pages[row], k_new_rot[q_start + i], atol=atol, rtol=rtol)
            torch.testing.assert_close(v_pages[row], v_new[q_start + i])
        for pos in range(old_len):
            row = paged_cache_row(block_table, b, pos, page_size)
            torch.testing.assert_close(k_pages[row], k_pages_before[row])
            torch.testing.assert_close(v_pages[row], v_pages_before[row])


@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gqa_prefill_paged_with_kv_cache_requires_power_of_two_page_size() -> None:
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(page_size=24, max_seqlen_q=16)
    q = torch.randn(2, 8, 64, device=run_device(), dtype=torch.float16)
    k_new = torch.randn(2, 2, 64, device=run_device(), dtype=torch.float16)
    k_pages = torch.zeros(48, 2, 64, device=run_device(), dtype=torch.float16)
    scale = torch.ones(1, device=run_device(), dtype=torch.float32)
    metadata = (
        torch.tensor([0, 2], device=run_device(), dtype=torch.int32),
        torch.tensor([0], device=run_device(), dtype=torch.int32),
        torch.tensor([[0]], device=run_device(), dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="requires a power-of-two page_size"):
        op(q, k_new, k_new.clone(), k_pages, k_pages.clone(), scale, scale.clone(), *metadata)


@pytest.mark.parametrize(
    "page_size",
    [
        pytest.param(16, marks=pytest.mark.smoke, id="page16_multi_page_per_block"),
        pytest.param(32, marks=pytest.mark.smoke, id="page32_multi_page_per_block"),
        pytest.param(128, marks=pytest.mark.smoke, id="page128_blocks_per_page"),
    ],
)
def test_gqa_prefill_paged_with_kv_cache_page_sizes(page_size: int) -> None:
    q_lens = [32, 64]
    old_lens = [48, 80]
    batch, heads, heads_kv, dim = 2, 8, 2, 64
    dtype = torch.float16
    max_pages_per_req = 16
    num_pages = batch * max_pages_per_req
    total_q = sum(q_lens)
    block_table = make_interleaved_block_table(batch, max_pages_per_req)
    cu_seqlens_q = make_cu_seqlens(q_lens)
    cache_seqlens = torch.tensor(old_lens, device=run_device(), dtype=torch.int32)
    q = torch.randn(total_q, heads, dim, device=run_device(), dtype=dtype).contiguous()
    k_new = torch.randn(total_q, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
    v_new = torch.randn(total_q, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
    k_pages = torch.zeros(
        num_pages * page_size, heads_kv, dim, device=run_device(), dtype=dtype
    ).contiguous()
    v_pages = torch.zeros_like(k_pages)
    k_old = [
        torch.randn(old_len, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
        for old_len in old_lens
    ]
    v_old = [
        torch.randn(old_len, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
        for old_len in old_lens
    ]
    fill_paged_cache_from_logical(k_pages, v_pages, k_old, v_old, block_table, page_size)
    k_scale, v_scale = make_unit_cache_scales()
    case = GQAPrefillPagedWithKVCacheFwdWorkload(
        batch, heads, heads_kv, q_lens, old_lens, page_size, dim, True, dtype
    )
    ref = case.ref_program(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
    )
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        page_size=page_size,
        max_seqlen_q=max(q_lens),
    )

    output = op(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        k_scale,
        v_scale,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
    )
    torch.testing.assert_close(output, ref, atol=5e-3, rtol=1e-5)


@pytest.mark.smoke
def test_gqa_prefill_paged_serves_two_dtypes_from_one_instance() -> None:
    """One instance answers both element types, each on its own kernel."""
    q_lens = [32, 64]
    old_lens = [48, 80]
    batch, heads, heads_kv, dim = 2, 8, 2, 64
    page_size, max_pages_per_req = 64, 8
    num_pages = batch * max_pages_per_req
    total_q = sum(q_lens)
    block_table = make_interleaved_block_table(batch, max_pages_per_req)
    cu_seqlens_q = make_cu_seqlens(q_lens)
    cache_seqlens = torch.tensor(old_lens, device=run_device(), dtype=torch.int32)
    k_scale, v_scale = make_unit_cache_scales()
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        page_size=page_size,
        max_seqlen_q=max(q_lens),
    )

    for dtype in (torch.float16, torch.bfloat16):
        q = torch.randn(total_q, heads, dim, device=run_device(), dtype=dtype).contiguous()
        k_new = torch.randn(total_q, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
        v_new = torch.randn(total_q, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
        k_pages = torch.zeros(
            num_pages * page_size, heads_kv, dim, device=run_device(), dtype=dtype
        ).contiguous()
        v_pages = torch.zeros_like(k_pages)
        k_old = [
            torch.randn(old_len, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
            for old_len in old_lens
        ]
        v_old = [
            torch.randn(old_len, heads_kv, dim, device=run_device(), dtype=dtype).contiguous()
            for old_len in old_lens
        ]
        fill_paged_cache_from_logical(k_pages, v_pages, k_old, v_old, block_table, page_size)
        case = GQAPrefillPagedWithKVCacheFwdWorkload(
            batch, heads, heads_kv, q_lens, old_lens, page_size, dim, True, dtype
        )
        ref = case.ref_program(
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
        )
        output = op(
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
        )
        assert output.dtype == dtype
        atol, rtol = _PREFILL_PAGED_TOLERANCE[dtype]
        torch.testing.assert_close(output, ref, atol=atol, rtol=rtol)

    if served_in_tree(op):
        built = op.built_kernels("gqa_prefill_paged")
        assert {kernel.dtype for kernel in built.values()} == {torch.float16, torch.bfloat16}


# ----------------------------------------------------------------------
# Paged workload helpers
# ----------------------------------------------------------------------


@pytest.mark.smoke
def test_paged_workloads_hand_out_a_fragmented_block_table() -> None:
    """A timed run walks a fragmented pool, so its number is not the best case."""
    batch, pages_per_req = 4, 8
    pool_pages = batch * pages_per_req

    disjoint = make_fragmented_block_table(batch, pages_per_req, pool_pages)
    assert disjoint.shape == (batch, pages_per_req)
    assert sorted(disjoint.flatten().tolist()) == list(range(pool_pages))
    assert not torch.equal(
        disjoint,
        torch.arange(pool_pages, dtype=torch.int32, device=run_device()).reshape(batch, -1),
    )

    shared = make_fragmented_block_table(batch, pages_per_req, pages_per_req)
    for row in shared.tolist():
        assert sorted(row) == list(range(pages_per_req))
    assert not all(row == sorted(row) for row in shared.tolist())

    assert torch.equal(disjoint, make_fragmented_block_table(batch, pages_per_req, pool_pages))

    case = GQAPrefillPagedWithKVCacheFwdWorkload(
        2, 16, 4, [64, 64], [128, 128], 64, 128, True, torch.float16
    )
    table = case.gen_inputs()[-1]
    pool = 2 * case.max_pages_per_req
    assert sorted(table.flatten().tolist()) == list(range(pool))
