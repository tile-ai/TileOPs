"""Kernel-selection coverage for the remaining paged attention ops."""

import pytest
import torch

from tileops.ops import (
    GroupedQueryAttentionPagedFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="attention selection reads the device architecture"
    ),
    pytest.mark.cuda_only,
]


def _prefill_call_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``q``, ``k_new`` and ``block_table`` of a two-request fp16 paged prefill call."""
    q = torch.empty(16, 32, 128, dtype=torch.float16, device="cuda")
    k_new = torch.empty(16, 8, 128, dtype=torch.float16, device="cuda")
    block_table = torch.empty(2, 8, dtype=torch.int32, device="cuda")
    return q, k_new, block_table


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("ctor", "dtype", "expected"),
    [
        pytest.param({}, torch.float16, "GQADecodePagedBs1Kernel", id="bs1-fp16"),
        pytest.param({}, torch.bfloat16, "GQADecodePagedKernel", id="bf16-falls-back"),
        pytest.param({"batch": 2}, torch.float16, "GQADecodePagedKernel", id="batched"),
        pytest.param({"dim": 64}, torch.float16, "GQADecodePagedKernel", id="head-dim"),
        pytest.param(
            {"page_size": 16, "pages": 512}, torch.float16, "GQADecodePagedKernel", id="small-page"
        ),
        pytest.param(
            {"page_size": 192, "pages": 42},
            torch.float16,
            "GQADecodePagedKernel",
            id="page-tile",
        ),
        pytest.param({"softcap": 2.0}, torch.float16, "GQADecodePagedKernel", id="softcap"),
    ],
)
def test_paged_decode_dispatch_is_unchanged(ctor: dict, dtype: torch.dtype, expected: str) -> None:
    """Paged decode keeps its batch-1 fast path and its page-tile guard."""
    extents = {"batch": 1, "pages": 32, "page_size": 256, "dim": 128, "softcap": None}
    extents.update(ctor)
    batch, dim = extents["batch"], extents["dim"]
    op = GroupedQueryAttentionPagedFwdOp(softcap=extents["softcap"])
    q = torch.empty(batch, 32, dim, dtype=dtype, device="cuda")
    k_pages = torch.empty(
        extents["pages"], extents["page_size"], 4, dim, dtype=dtype, device="cuda"
    )
    page_table = torch.empty(batch, extents["pages"], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    call = op.paged_call(q, k_pages, page_table, cu_seqlens_q)
    assert op.kernel_map[op.select_implementation("gqa_paged", call)].__name__ == expected


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("ctor", "expected"),
    [
        pytest.param({}, "GQAPrefillPagedWithKVCacheFwdKernel", id="plain-cache"),
        pytest.param(
            {"fuse_rope": True, "max_position": 4096},
            "GQAPrefillPagedWithKVCacheRopeFwdKernel",
            id="fused-rope",
        ),
    ],
)
def test_paged_prefill_dispatch_is_unchanged(ctor: dict, expected: str) -> None:
    """Paged prefill keeps its plain and fused-RoPE regions."""
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(page_size=256, max_seqlen_q=512, **ctor)
    call = op.attention_call(*_prefill_call_tensors())
    assert op.kernel_map[op.select_implementation("gqa_prefill_paged", call)].__name__ == expected


@pytest.mark.smoke
def test_paged_prefill_fp8_cache_dispatch_is_unchanged() -> None:
    """An FP8 KV cache still selects the FP8-cache kernel."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch build has no float8_e4m3fn")
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        page_size=256, max_seqlen_q=512, cache_dtype=torch.float8_e4m3fn
    )
    call = op.attention_call(*_prefill_call_tensors())
    key = op.select_implementation("gqa_prefill_paged", call)
    assert op.kernel_map[key].__name__ == "GQAPrefillPagedWithFP8KVCacheFwdKernel"
