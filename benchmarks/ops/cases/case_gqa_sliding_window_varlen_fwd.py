"""Tests for GqaSlidingWindowVarlenFwdOp against a pure-PyTorch reference."""
from typing import List, Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GqaSlidingWindowVarlenFwdOp


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


class GqaSlidingWindowVarlenFwdTest(TestBase):

    def __init__(
        self,
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
    ) -> None:
        self.batch = batch
        self.seqlens_q = seqlens_q
        self.seqlens_k = seqlens_k
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.wl = wl
        self.wr = wr
        self.dtype = dtype

    def gen_inputs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, int]:
        total_q = sum(self.seqlens_q)
        total_k = sum(self.seqlens_k)
        q = torch.randn(total_q, self.heads, self.dim,
                        dtype=self.dtype, device="cuda") * 0.1
        k = torch.randn(total_k, self.heads_kv, self.dim,
                        dtype=self.dtype, device="cuda") * 0.1
        v = torch.randn(total_k, self.heads_kv, self.dim,
                        dtype=self.dtype, device="cuda") * 0.1

        cu_seqlens_q = torch.tensor(
            [0] + list(torch.cumsum(
                torch.tensor(self.seqlens_q), 0).tolist()),
            dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.tensor(
            [0] + list(torch.cumsum(
                torch.tensor(self.seqlens_k), 0).tolist()),
            dtype=torch.int32, device="cuda")
        max_seqlen_q = max(self.seqlens_q)

        return q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """Pure-PyTorch reference: per-sample masked softmax attention.

        offset = seqlen_k - seqlen_q aligns the causal mask bottom-right
        (FA3 convention).  When seqlen_q == seqlen_k, offset=0 and the mask
        reduces to the standard causal mask.
        """
        groups = self.heads // self.heads_kv
        scale = self.dim ** -0.5
        outputs = []

        for i in range(self.batch):
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i + 1].item()
            kv_start = cu_seqlens_k[i].item()
            kv_end = cu_seqlens_k[i + 1].item()

            q_i = q[q_start:q_end]          # [seqlen_q, heads,    dim]
            k_i = k[kv_start:kv_end]        # [seqlen_k, heads_kv, dim]
            v_i = v[kv_start:kv_end]

            seqlen_q = q_end - q_start
            seqlen_k = kv_end - kv_start
            # offset: aligns causal mask to bottom-right corner
            offset = seqlen_k - seqlen_q

            # Expand KV for GQA
            k_exp = k_i.repeat_interleave(groups, dim=1).float()  # [sk, H, D]
            v_exp = v_i.repeat_interleave(groups, dim=1).float()

            # [H, seqlen_q, seqlen_k]
            scores = torch.matmul(
                q_i.float().transpose(0, 1),              # [H, sq, D]
                k_exp.transpose(0, 1).transpose(-2, -1),  # [H, D, sk]
            ) * scale

            # Build attention mask
            q_pos = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
            k_pos = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
            mask = torch.zeros(seqlen_q, seqlen_k,
                               dtype=torch.bool, device=q.device)
            if self.is_causal:
                mask = mask | (k_pos > q_pos + offset)
            if self.wl >= 0:
                mask = mask | (k_pos < q_pos + offset - self.wl)
            if self.wr >= 0:
                mask = mask | (k_pos > q_pos + offset + self.wr)

            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
            probs = torch.softmax(scores, dim=-1).nan_to_num()
            out_i = torch.matmul(probs, v_exp.transpose(0, 1))  # [H, sq, D]
            outputs.append(out_i.transpose(0, 1).to(q.dtype))   # [sq, H, D]

        return torch.cat(outputs, dim=0)  # [total_q, H, D]


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
