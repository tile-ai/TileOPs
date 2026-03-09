"""Tests for GqaSlidingWindowFwdOp against a pure-PyTorch reference."""
from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GqaSlidingWindowFwdOp


class GqaSlidingWindowFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, heads, heads_kv, dim, is_causal, wl, wr, dtype, tune", [
            # ── Basic correctness ─────────────────────────────────────────────
            (2, 512,  8, 2,  64, True,  -1,  -1, torch.float16, False),  # causal full
            (2, 512,  8, 2,  64, True,  128, -1, torch.float16, False),  # causal + left window
            (2, 512,  8, 2,  64, False, -1,  -1, torch.float16, False),  # bidirectional full
            (2, 512,  8, 2,  64, False, 64,  64, torch.float16, False),  # bidirectional window
            (2, 128,  8, 1, 128, True,   1,  -1, torch.float16, False),  # tiny left window
            # ── dtype ─────────────────────────────────────────────────────────
            (2, 512,  8, 2,  64, True,  -1,  -1, torch.bfloat16, False),  # bfloat16 causal
            (2, 512,  8, 2,  64, False, 64,  64, torch.bfloat16, False),  # bfloat16 window
            # ── GQA ratio ─────────────────────────────────────────────────────
            (2, 512,  8, 8,  64, True,  -1,  -1, torch.float16, False),  # MHA (ratio 1:1)
            (2, 512, 16, 1,  64, True,  -1,  -1, torch.float16, False),  # ratio 16:1
            # ── Non-power-of-2 sequence lengths ───────────────────────────────
            (2, 384,  8, 2,  64, True,  -1,  -1, torch.float16, False),  # seq=384
            (2, 768,  8, 2,  64, False, 256, -1, torch.float16, False),  # seq=768 + left window
            # ── Large sequence ─────────────────────────────────────────────────
            (1, 2048, 8, 2,  64, True,  512, -1, torch.float16, False),  # long causal + window
            # ── Right window only ──────────────────────────────────────────────
            (2, 512,  8, 2,  64, False, -1,  64, torch.float16, False),  # right window only
            # ── wl=0 boundary: only current-position left context ─────────────
            (2, 256,  8, 2,  64, True,   0,  -1, torch.float16, False),  # causal + wl=0
        ]),
    ]


class GqaSlidingWindowFwdTest(TestBase):

    def __init__(
        self,
        batch: int,
        seq: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        wl: int,
        wr: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.seq = seq
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.wl = wl
        self.wr = wr
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(self.batch, self.seq, self.heads,    self.dim,
                        dtype=self.dtype, device="cuda") * 0.1
        k = torch.randn(self.batch, self.seq, self.heads_kv, self.dim,
                        dtype=self.dtype, device="cuda") * 0.1
        v = torch.randn(self.batch, self.seq, self.heads_kv, self.dim,
                        dtype=self.dtype, device="cuda") * 0.1
        return q, k, v

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


@GqaSlidingWindowFwdFixture
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
    test = GqaSlidingWindowFwdTest(batch, seq, heads, heads_kv, dim, is_causal, wl, wr, dtype)
    op = GqaSlidingWindowFwdOp(
        batch=batch, heads=heads, heads_kv=heads_kv, seq_len=seq, dim=dim,
        is_causal=is_causal, window_size_left=wl, window_size_right=wr,
        dtype=dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=1e-2, rtol=1e-2)


class TestGqaSlidingWindowFwdOpMetrics:
    """Unit tests for total_flops and total_memory correctness."""

    def test_total_flops_causal_wl0(self):
        """is_causal=True, wl=0: every query attends only to itself (eff_kv=1)."""
        B, S, H, Hkv, D = 2, 256, 8, 2, 64
        op = GqaSlidingWindowFwdOp(
            batch=B, heads=H, heads_kv=Hkv, seq_len=S, dim=D,
            is_causal=True, window_size_left=0)
        expected = 4 * B * H * S * 1 * D   # total_attended = S * 1
        assert op.total_flops == expected, f"got {op.total_flops}, expected {expected}"

    def test_total_flops_causal_finite_window(self):
        """is_causal=True, wl=128, S=512: window limits left context, not S//2."""
        B, S, H, Hkv, D = 1, 512, 8, 2, 64
        wl = 128
        op = GqaSlidingWindowFwdOp(
            batch=B, heads=H, heads_kv=Hkv, seq_len=S, dim=D,
            is_causal=True, window_size_left=wl)
        total_attended = sum(min(wl + 1, q + 1) for q in range(S))
        expected = 4 * B * H * total_attended * D
        assert op.total_flops == expected, f"got {op.total_flops}, expected {expected}"

    def test_total_memory_gqa(self):
        """For GQA (heads > heads_kv), Q and O must use heads, not heads_kv."""
        B, S, H, Hkv, D = 2, 512, 8, 2, 64
        op = GqaSlidingWindowFwdOp(
            batch=B, heads=H, heads_kv=Hkv, seq_len=S, dim=D,
            is_causal=True)
        elem = torch.tensor([], dtype=torch.float16).element_size()
        # Q read + O write: heads each; K read + V read: heads_kv each
        expected = 2 * B * S * (H + Hkv) * D * elem
        assert op.total_memory == expected, f"got {op.total_memory}, expected {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
