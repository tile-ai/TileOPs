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
            pytest.param(2, 512,  8, 2,  64, True,  128, -1, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 512,  8, 2,  64, False, -1,  -1, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 512,  8, 2,  64, False, 64,  64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 128,  8, 1, 128, True,   1,  -1, torch.float16, False, marks=pytest.mark.full),
            # ── dtype ─────────────────────────────────────────────────────────
            pytest.param(2, 512,  8, 2,  64, False, 64,  64, torch.bfloat16, False, marks=pytest.mark.full),
            # ── GQA ratio ─────────────────────────────────────────────────────
            pytest.param(2, 512,  8, 8,  64, True,  -1,  -1, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 512, 16, 1,  64, True,  -1,  -1, torch.float16, False, marks=pytest.mark.full),
            # ── Non-power-of-2 sequence lengths ───────────────────────────────
            pytest.param(2, 384,  8, 2,  64, True,  -1,  -1, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 768,  8, 2,  64, False, 256, -1, torch.float16, False, marks=pytest.mark.full),
            # ── Large sequence ─────────────────────────────────────────────────
            pytest.param(1, 2048, 8, 2,  64, True,  512, -1, torch.float16, False, marks=pytest.mark.full),
            # ── Right window only ──────────────────────────────────────────────
            pytest.param(2, 512,  8, 2,  64, False, -1,  64, torch.float16, False, marks=pytest.mark.full),
            # ── wl=0 boundary: only current-position left context ─────────────
            pytest.param(2, 256,  8, 2,  64, True,   0,  -1, torch.float16, False, marks=pytest.mark.full),
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


class TestGroupedQueryAttentionSlidingWindowFwdOpMetrics:
    """Unit tests for total_flops and total_memory correctness."""

    @pytest.mark.full
    def test_total_flops_causal_wl0(self):
        """is_causal=True, wl=0: every query attends only to itself (eff_kv=1)."""
        B, S, H, Hkv, D = 2, 256, 8, 2, 64
        op = GroupedQueryAttentionSlidingWindowFwdOp(
            batch=B, heads=H, heads_kv=Hkv, seq_len=S, dim=D,
            is_causal=True, window_size_left=0)
        expected = 4 * B * H * S * 1 * D   # total_attended = S * 1
        assert op.total_flops == expected, f"got {op.total_flops}, expected {expected}"

    @pytest.mark.full
    def test_total_flops_causal_finite_window(self):
        """is_causal=True, wl=128, S=512: window limits left context, not S//2."""
        B, S, H, Hkv, D = 1, 512, 8, 2, 64
        wl = 128
        op = GroupedQueryAttentionSlidingWindowFwdOp(
            batch=B, heads=H, heads_kv=Hkv, seq_len=S, dim=D,
            is_causal=True, window_size_left=wl)
        total_attended = sum(min(wl + 1, q + 1) for q in range(S))
        expected = 4 * B * H * total_attended * D
        assert op.total_flops == expected, f"got {op.total_flops}, expected {expected}"

    @pytest.mark.full
    def test_total_memory_gqa(self):
        """For GQA (heads > heads_kv), Q and O must use heads, not heads_kv."""
        B, S, H, Hkv, D = 2, 512, 8, 2, 64
        op = GroupedQueryAttentionSlidingWindowFwdOp(
            batch=B, heads=H, heads_kv=Hkv, seq_len=S, dim=D,
            is_causal=True)
        elem = torch.tensor([], dtype=torch.float16).element_size()
        # Q read + O write: heads each; K read + V read: heads_kv each
        expected = 2 * B * S * (H + Hkv) * D * elem
        assert op.total_memory == expected, f"got {op.total_memory}, expected {expected}"


class TestGroupedQueryAttentionSlidingWindowFwdOpValidation:
    """Early-error validation: __init__ and forward guard-clauses."""

    # ── window_size validation (caught in __init__, no GPU kernel needed) ─────

    @pytest.mark.full
    def test_invalid_window_size_left_raises(self):
        with pytest.raises(ValueError, match="window_size_left"):
            GroupedQueryAttentionSlidingWindowFwdOp(
                batch=1, heads=4, heads_kv=2, seq_len=64, dim=64,
                is_causal=True, window_size_left=-2)

    @pytest.mark.full
    def test_invalid_window_size_right_raises(self):
        with pytest.raises(ValueError, match="window_size_right"):
            GroupedQueryAttentionSlidingWindowFwdOp(
                batch=1, heads=4, heads_kv=2, seq_len=64, dim=64,
                is_causal=True, window_size_right=-3)

    # ── dtype / device validation (caught in forward) ─────────────────────────

    @pytest.fixture(scope="class")
    def float16_op(self):
        return GroupedQueryAttentionSlidingWindowFwdOp(
            batch=1, heads=4, heads_kv=2, seq_len=64, dim=64,
            is_causal=True, dtype=torch.float16)

    @pytest.mark.full
    def test_dtype_mismatch_raises(self, float16_op):
        q = torch.randn(1, 64, 4, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 64, 2, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, 64, 2, 64, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="dtype"):
            float16_op.forward(q, k, v)

    @pytest.mark.full
    def test_cpu_tensor_raises(self, float16_op):
        q = torch.randn(1, 64, 4, 64, dtype=torch.float16)   # CPU
        k = torch.randn(1, 64, 2, 64, dtype=torch.float16)
        v = torch.randn(1, 64, 2, 64, dtype=torch.float16)
        with pytest.raises(ValueError, match="cuda"):
            float16_op.forward(q, k, v)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
