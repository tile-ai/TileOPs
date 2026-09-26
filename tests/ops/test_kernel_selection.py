"""Kernel-selection coverage for the remaining paged attention ops."""

import pytest
import torch

from tileops.ops import (
    GroupedQueryAttentionDecodePagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="attention selection reads the device architecture",
)


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
        pytest.param(
            {"page_size": 192, "seqlen_kv": 8064},
            torch.float16,
            "GQADecodePagedKernel",
            id="page-tile",
        ),
    ],
)
def test_paged_decode_dispatch_is_unchanged(ctor: dict, dtype: torch.dtype, expected: str) -> None:
    """Paged decode keeps its batch-1 fast path and its page-tile guard."""
    extents = {"batch": 1, "seqlen_kv": 8192, "page_size": 256}
    extents.update(ctor)
    op = GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(extents["page_size"])
    q = torch.empty(extents["batch"], 32, 128, dtype=dtype, device="cuda")
    k = torch.empty(extents["seqlen_kv"], 4, 128, dtype=dtype, device="cuda")
    candidate = op.select_kernel(op.attention_call(q, k)).__name__
    assert candidate == expected


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
    candidate = op.select_kernel(op.attention_call(*_prefill_call_tensors())).__name__
    assert candidate == expected


@pytest.mark.smoke
def test_paged_prefill_fp8_cache_dispatch_is_unchanged() -> None:
    """An FP8 KV cache still selects the FP8-cache kernel."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch build has no float8_e4m3fn")
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        page_size=256, max_seqlen_q=512, cache_dtype=torch.float8_e4m3fn
    )
    candidate = op.select_kernel(op.attention_call(*_prefill_call_tensors())).__name__
    assert candidate == "GQAPrefillPagedWithFP8KVCacheFwdKernel"
