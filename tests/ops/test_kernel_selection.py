"""Kernel-selection coverage for the remaining paged attention ops."""

from itertools import accumulate

import pytest
import torch

from tileops.ops import (
    GQAPagedFwdOp,
    GQAPrefillPagedWithKVCacheFwdOp,
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
        pytest.param(
            {}, torch.float16, "GQADecodePagedBs1Kernel", id="bs1-fp16", marks=pytest.mark.sm90
        ),
        pytest.param({}, torch.bfloat16, "GQAPagedVarlenFwdKernel", id="bf16"),
        pytest.param({"batch": 2}, torch.float16, "GQAPagedVarlenFwdKernel", id="batched"),
        pytest.param({"dim": 64}, torch.float16, "GQAPagedVarlenFwdKernel", id="head-dim"),
        pytest.param(
            {"page_size": 16, "pages": 512},
            torch.float16,
            "GQAPagedVarlenFwdKernel",
            id="small-page",
        ),
        pytest.param(
            {"page_size": 192, "pages": 42},
            torch.float16,
            "GQAPagedVarlenFwdKernel",
            id="page-tile",
        ),
        pytest.param({"softcap": 2.0}, torch.float16, "GQAPagedVarlenFwdKernel", id="softcap"),
        pytest.param(
            {"batch": 2, "q_lens": [0, 2]},
            torch.float16,
            "GQAPagedVarlenFwdKernel",
            id="ragged-lengths",
        ),
        pytest.param(
            {"window_size_left": 128}, torch.float16, "GQAPagedVarlenFwdKernel", id="window"
        ),
        pytest.param(
            {"page_size": 65, "pages": 64},
            torch.float16,
            "GQAPagedVarlenFwdKernel",
            id="page-no-tile-fits",
        ),
    ],
)
def test_paged_dispatch_regions(ctor: dict, dtype: torch.dtype, expected: str) -> None:
    """The packed kernel serves the 16-bit contract; the batch-1 kernel wins its own shape."""
    extents = {
        "batch": 1,
        "pages": 32,
        "page_size": 256,
        "dim": 128,
        "softcap": None,
        "window_size_left": -1,
        "q_lens": None,
    }
    extents.update(ctor)
    batch, dim = extents["batch"], extents["dim"]
    op = GQAPagedFwdOp(softcap=extents["softcap"], window_size_left=extents["window_size_left"])
    q_lens = extents["q_lens"] or [1] * batch
    q = torch.empty(sum(q_lens), 32, dim, dtype=dtype, device="cuda")
    k_pages = torch.empty(
        extents["pages"], extents["page_size"], 4, dim, dtype=dtype, device="cuda"
    )
    page_table = torch.empty(batch, extents["pages"], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.tensor([0, *accumulate(q_lens)], dtype=torch.int32, device="cuda")
    call = op.paged_call(q, k_pages, page_table, cu_seqlens_q)
    assert op.kernel_map[op.select_implementation("gqa_paged", call)].__name__ == expected


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("ctor", "expected"),
    [
        pytest.param({}, "GQAPrefillPagedWithKVCacheFwdKernel", id="plain-cache"),
        pytest.param(
            {"fuse_rope": True, "max_position": 4096},
            "GQAPrefillPagedWithKVCacheRoPEFwdKernel",
            id="fused-rope",
        ),
    ],
)
def test_paged_prefill_dispatch_is_unchanged(ctor: dict, expected: str) -> None:
    """Paged prefill keeps its plain and fused-RoPE regions."""
    op = GQAPrefillPagedWithKVCacheFwdOp(page_size=256, max_seqlen_q=512, **ctor)
    call = op.attention_call(*_prefill_call_tensors())
    assert op.kernel_map[op.select_implementation("gqa_prefill_paged", call)].__name__ == expected


@pytest.mark.smoke
def test_paged_prefill_fp8_cache_dispatch_is_unchanged() -> None:
    """An FP8 KV cache still selects the FP8-cache kernel."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch build has no float8_e4m3fn")
    op = GQAPrefillPagedWithKVCacheFwdOp(
        page_size=256, max_seqlen_q=512, cache_dtype=torch.float8_e4m3fn
    )
    call = op.attention_call(*_prefill_call_tensors())
    key = op.select_implementation("gqa_prefill_paged", call)
    assert op.kernel_map[key].__name__ == "GQAPrefillPagedWithFP8KVCacheFwdKernel"
