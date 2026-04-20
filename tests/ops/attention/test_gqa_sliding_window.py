"""Tests for GroupedQueryAttentionSlidingWindowFwdOp against a pure-PyTorch reference."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GroupedQueryAttentionSlidingWindowFwdOp
from workloads.attention.gqa_sliding_window import (
    GroupedQueryAttentionSlidingWindowFwdTest as _GroupedQueryAttentionSlidingWindowFwdTestWorkload,
)


class GroupedQueryAttentionSlidingWindowFwdTest(_GroupedQueryAttentionSlidingWindowFwdTestWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch reference: expand KV heads, compute masked softmax attention."""
        groups = self.heads // self.heads_kv
        scale = self.dim ** -0.5

        # Expand KV to match Q heads: [B, S, H, D]
        k_exp = k.repeat_interleave(groups, dim=2).float()
        v_exp = v.repeat_interleave(groups, dim=2).float()

        # [B, H, S, S]
        scores = torch.matmul(
            q.float().transpose(1, 2),
            k_exp.transpose(1, 2).transpose(-2, -1),
        ) * scale

        # Build attention mask
        S = self.seq
        q_pos = torch.arange(S, device=q.device).unsqueeze(1)  # [S, 1]
        k_pos = torch.arange(S, device=q.device).unsqueeze(0)  # [1, S]
        mask = torch.zeros(S, S, dtype=torch.bool, device=q.device)
        if self.is_causal:
            mask = mask | (q_pos < k_pos)
        if self.wl >= 0:
            mask = mask | (k_pos < q_pos - self.wl)
        if self.wr >= 0:
            mask = mask | (k_pos > q_pos + self.wr)

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v_exp.transpose(1, 2))  # [B, H, S, D]
        return output.transpose(1, 2).to(q.dtype)            # [B, S, H, D]


class GroupedQueryAttentionSlidingWindowFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, heads, heads_kv, dim, is_causal, wl, wr, dtype, tune", [
            # ── Basic correctness ─────────────────────────────────────────────
            pytest.param(2, 512,  8, 2,  64, True,  -1,  -1, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 512,  8, 2,  64, True,  -1,  -1, torch.bfloat16, False, marks=pytest.mark.smoke),
        ]),
    ]


@GroupedQueryAttentionSlidingWindowFwdFixture
def test_gqa_sliding_window_fwd_op(
    batch: int,
    seq: int,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    wl: int,
    wr: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionSlidingWindowFwdTest(batch, seq, heads, heads_kv, dim, is_causal, wl, wr, dtype)
    op = GroupedQueryAttentionSlidingWindowFwdOp(
        batch=batch, heads=heads, heads_kv=heads_kv, seq_len=seq, dim=dim,
        is_causal=is_causal, window_size_left=wl, window_size_right=wr,
        dtype=dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
