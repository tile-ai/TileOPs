"""Tests for GqaSlidingWindowVarlenFwdOp against a pure-PyTorch reference."""
from typing import List

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GqaSlidingWindowVarlenFwdOp
from workloads.ops.gqa_sliding_window_varlen_fwd import (
    GqaSlidingWindowVarlenFwdTest as _GqaSlidingWindowVarlenFwdTestWorkload,
)


class GqaSlidingWindowVarlenFwdTest(_GqaSlidingWindowVarlenFwdTestWorkload, TestBase):
    pass


class GqaSlidingWindowVarlenFwdFixture(FixtureBase):
    # Parameters: (batch, seqlens_q, seqlens_k, heads, heads_kv, dim,
    #              is_causal, wl, wr, dtype, tune)
    PARAMS = [
        ("batch, seqlens_q, seqlens_k, heads, heads_kv, dim,"
         " is_causal, wl, wr, dtype, tune", [
             # ── Prefill: seqlen_q == seqlen_k (offset=0) ─────────────────────
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, True,  -1,  -1, torch.float16,  False, marks=pytest.mark.smoke),   # causal
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, True, 128,  -1, torch.float16,  False, marks=pytest.mark.full),    # causal + wl
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, False, -1,  -1, torch.float16,  False, marks=pytest.mark.full),    # bidirectional
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, False, 64,  64, torch.float16,  False, marks=pytest.mark.full),    # window
             # ── KV-cache: seqlen_k > seqlen_q (offset > 0) ───────────────────
             pytest.param(2, [64, 128],  [256, 512], 8, 2, 64, True,  -1,  -1, torch.float16,  False, marks=pytest.mark.full),    # causal kvcache
             pytest.param(2, [64, 128],  [256, 512], 8, 2, 64, True, 128,  -1, torch.float16,  False, marks=pytest.mark.full),    # causal+wl kvcache
             pytest.param(2, [64, 128],  [256, 512], 8, 2, 64, False, 64,  64, torch.float16,  False, marks=pytest.mark.full),    # window kvcache
             # ── bfloat16 ─────────────────────────────────────────────────────
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, True,  -1,  -1, torch.bfloat16, False, marks=pytest.mark.full),    # causal bf16
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, False, 64,  64, torch.bfloat16, False, marks=pytest.mark.full),    # window bf16
             # ── GQA ratios ───────────────────────────────────────────────────
             pytest.param(2, [256, 512], [256, 512], 8, 8, 64, True,  -1,  -1, torch.float16,  False, marks=pytest.mark.full),    # MHA 1:1
             pytest.param(2, [256, 512], [256, 512], 16, 1, 64, True, -1,  -1, torch.float16,  False, marks=pytest.mark.full),    # ratio 16:1
             # ── Mixed lengths within batch ────────────────────────────────────
             pytest.param(3, [128, 256, 384], [128, 256, 384], 8, 2, 64, True, -1, -1, torch.float16, False, marks=pytest.mark.full),
             # ── Right window only ─────────────────────────────────────────────
             pytest.param(2, [256, 512], [256, 512], 8, 2, 64, False, -1,  64, torch.float16,  False, marks=pytest.mark.full),    # right window
             # ── wl=0 boundary ────────────────────────────────────────────────
             pytest.param(2, [128, 256], [128, 256], 8, 2, 64, True,   0,  -1, torch.float16,  False, marks=pytest.mark.full),    # wl=0
         ]),
    ]


@GqaSlidingWindowVarlenFwdFixture
def test_gqa_sliding_window_varlen_fwd_op(
    batch: int,
    seqlens_q: List[int],
    seqlens_k: List[int],
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    wl: int,
    wr: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GqaSlidingWindowVarlenFwdTest(
        batch, seqlens_q, seqlens_k, heads, heads_kv, dim,
        is_causal, wl, wr, dtype)
    op = GqaSlidingWindowVarlenFwdOp(
        batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
        is_causal=is_causal, window_size_left=wl, window_size_right=wr,
        dtype=dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
