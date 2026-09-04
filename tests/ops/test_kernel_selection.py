"""Kernel-selection coverage for the remaining paged attention ops."""

import pytest
import torch

from tileops.ops import (
    GroupedQueryAttentionDecodePagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
)
from tileops.ops.attention.selection import PAGED_DECODE_KEYS, PAGED_PREFILL_KEYS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="attention selection reads the device architecture",
)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("ctor", "dtype", "expected"),
    [
        pytest.param({}, torch.float16, "gqa_decode_paged_bs1_kernel", id="bs1-fp16"),
        pytest.param({}, torch.bfloat16, "gqa_decode_paged_kernel", id="bf16-falls-back"),
        pytest.param({"batch": 2}, torch.float16, "gqa_decode_paged_kernel", id="batched"),
        pytest.param(
            {"page_size": 192, "seqlen_kv": 8064},
            torch.float16,
            "gqa_decode_paged_kernel",
            id="page-tile",
        ),
    ],
)
def test_paged_decode_dispatch_is_unchanged(ctor: dict, dtype: torch.dtype, expected: str) -> None:
    """Paged decode keeps its batch-1 fast path and its page-tile guard."""
    kwargs = {
        "batch": 1,
        "heads": 32,
        "heads_kv": 4,
        "seqlen_kv": 8192,
        "dim": 128,
        "page_size": 256,
    }
    kwargs.update(ctor)
    op = GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(**kwargs)
    candidate = op.select_kernel_key(PAGED_DECODE_KEYS, op.attention_call(dtype))
    assert candidate == expected


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("ctor", "expected"),
    [
        pytest.param({}, "gqa_prefill_paged_with_kv_cache_fwd_kernel", id="plain-cache"),
        pytest.param(
            {"fuse_rope": True, "max_position": 4096},
            "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
            id="fused-rope",
        ),
    ],
)
def test_paged_prefill_dispatch_is_unchanged(ctor: dict, expected: str) -> None:
    """Paged prefill keeps its plain and fused-RoPE regions."""
    kwargs = {
        "batch": 2,
        "heads": 32,
        "heads_kv": 8,
        "max_pages_per_req": 8,
        "page_size": 256,
        "dim": 128,
    }
    kwargs.update(ctor)
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(**kwargs)
    candidate = op.select_kernel_key(PAGED_PREFILL_KEYS, op.attention_call(torch.float16))
    assert candidate == expected


@pytest.mark.smoke
def test_paged_prefill_fp8_cache_dispatch_is_unchanged() -> None:
    """An FP8 KV cache still selects the FP8-cache kernel."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch build has no float8_e4m3fn")
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        batch=2,
        heads=32,
        heads_kv=8,
        max_pages_per_req=8,
        page_size=256,
        dim=128,
        cache_dtype=torch.float8_e4m3fn,
    )
    candidate = op.select_kernel_key(PAGED_PREFILL_KEYS, op.attention_call(torch.float16))
    assert candidate == "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel"
